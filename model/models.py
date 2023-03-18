import torch.nn
from pytorch_lightning import LightningModule
from pl_bolts.optimizers import lr_scheduler as pl_lr_scheduler
from torch.optim import lr_scheduler

from torch import nn
from torch.nn import functional as F

from model.config import cfg
import math

import torch
from model.basemodel_1D import TemporalConvNet
import numpy as np
import copy
from torch.autograd import Variable

from torchvision.ops import sigmoid_focal_loss
from torch import nn


def CCCLoss(y_hat, y, scale_factor=1., num_classes=2):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes))

    yhat_mean = torch.mean(y_hat_fl, dim=0, keepdim=True)
    y_mean = torch.mean(y_fl, dim=0, keepdim=True)

    sxy = torch.mean(torch.mul(y_fl - y_mean, y_hat_fl - yhat_mean), dim=0)
    rhoc = torch.div(2 * sxy,
                     torch.var(y_fl, dim=0) + torch.var(y_hat_fl, dim=0) + torch.square(y_mean - yhat_mean) + 1e-8)

    return 1 - torch.mean(rhoc)


def BCEwithLogitsLoss(y_hat, y, scale_factor=1, num_classes=12, pos_weights=None):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes)) * 1.0

    return F.binary_cross_entropy_with_logits(y_hat_fl, y_fl, reduction='mean', pos_weight=pos_weights)


def CEFocalLoss(y_hat, y, scale_factor=1, num_classes=8, label_smoothing=0., class_weights=None, alpha=0.25, gamma=2.,
                distillation_loss=True):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1,))  # Class indices

    ce_loss = F.cross_entropy(y_hat_fl, y_fl, label_smoothing=label_smoothing, reduction='none')
    target_one_hot = F.one_hot(y_fl, num_classes=num_classes)
    p = F.softmax(y_hat_fl, dim=1)

    if distillation_loss:
        target_one_hot_smooth = target_one_hot * (1 - label_smoothing) + label_smoothing / num_classes
        dist_loss = F.kl_div(F.log_softmax(y_hat_fl, dim=1), target_one_hot_smooth, reduction='batchmean')
    else:
        dist_loss = 0.

    p_t = torch.sum(p * target_one_hot, dim=1)  # + (1 - p) * (1 - target_one_hot)
    loss = ce_loss * torch.pow(1 - p_t, gamma)
    if alpha > 0.:
        alpha_t = torch.sum(alpha * target_one_hot, dim=1)
        loss = alpha_t * loss

    if distillation_loss:
        dist_loss_coeff = 0.2
        return loss.mean() * (1 - dist_loss_coeff) + dist_loss_coeff * dist_loss
    else:
        return loss.mean()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features)).to(device)
        self.b_2 = nn.Parameter(torch.zeros(features)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class Encoder(nn.Module):

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = layer
        self.norm = LayerNorm(layer[0].size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class MultiModalEncoder(nn.Module):

    def __init__(self, layer, N, modal_num):
        super(MultiModalEncoder, self).__init__()
        self.modal_num = modal_num
        self.layers = layer
        self.norm = nn.ModuleList()
        for i in range(self.modal_num):
            self.norm.append(LayerNorm(layer[0].size))


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))

        x = torch.cat(_x_list, dim=-1)

        return x


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))


class MultiModalSublayerConnection(nn.Module):

    def __init__(self, size, modal_num, dropout):
        super(MultiModalSublayerConnection, self).__init__()
        self.modal_num = modal_num

        self.norm = nn.ModuleList()
        for i in range(self.modal_num):
            self.norm.append(LayerNorm(size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        residual = x

        _x_list = []
        _x = torch.chunk(x, self.modal_num, -1)
        for i in range(self.modal_num):
            _x_list.append(self.norm[i](_x[i]))
        x = torch.cat(_x_list, dim=-1)

        return self.dropout(sublayer(x)) + residual


class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList()
        self.sublayer.append(SublayerConnection(size, dropout))
        self.sublayer.append(SublayerConnection(size, dropout))

        self.size = size

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class MultiModalEncoderLayer(nn.Module):

    def __init__(self, size, modal_num, mm_atten, mt_atten, feed_forward, dropout):
        super(MultiModalEncoderLayer, self).__init__()
        self.modal_num = modal_num

        self.mm_atten = mm_atten
        self.mt_atten = mt_atten
        self.feed_forward = feed_forward

        mm_sublayer = MultiModalSublayerConnection(size, modal_num, dropout)
        mt_sublayer = nn.ModuleList()
        for i in range(modal_num):
            mt_sublayer.append(SublayerConnection(size, dropout))
        ff_sublayer = nn.ModuleList()
        for i in range(modal_num):
            ff_sublayer.append(SublayerConnection(size, dropout))

        self.sublayer = nn.ModuleList()
        self.sublayer.append(mm_sublayer)
        self.sublayer.append(mt_sublayer)
        self.sublayer.append(ff_sublayer)

        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.mm_atten(x, x, x))

        _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            # feature = self.sublayer[1][i](_x[i], lambda x: self.mt_atten[i](x, x, x, mask[i]))
            feature = self.sublayer[1][i](_x[i], lambda x: self.mt_atten[i](x, x, x, mask=None))
            feature = self.sublayer[2][i](feature, self.feed_forward[i])
            _x_list.append(feature)
        x = torch.cat(_x_list, dim=-1)

        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, _ = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiModalAttention(nn.Module):
    def __init__(self, h, d_model, modal_num, dropout=0.1):

        super(MultiModalAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.modal_num = modal_num
        self.mm_linears = nn.ModuleList()
        for i in range(self.modal_num):
            linears = clones(nn.Linear(d_model, d_model), 4)
            self.mm_linears.append(linears)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        query = torch.chunk(query, self.modal_num, dim=-1)
        key   = torch.chunk(key, self.modal_num, dim=-1)
        value = torch.chunk(value, self.modal_num, dim=-1)

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query[0].size(0)

        _query_list = []
        _key_list = []
        _value_list = []
        for i in range(self.modal_num):
            _query_list.append(self.mm_linears[i][0](query[i]).view(nbatches, -1, self.h, self.d_k))
            _key_list.append(self.mm_linears[i][1](key[i]).view(nbatches, -1, self.h, self.d_k))
            _value_list.append(self.mm_linears[i][2](value[i]).view(nbatches, -1, self.h, self.d_k))

        mm_query = torch.stack(_query_list, dim=-2)
        mm_key = torch.stack(_key_list, dim=-2)
        mm_value = torch.stack(_value_list, dim=-2)
        x, _ = attention(mm_query, mm_key, mm_value, mask=mask, dropout=self.dropout)

        x = x.transpose(-2, -3).contiguous().view(nbatches, -1, self.modal_num, self.h * self.d_k)
        _x = torch.chunk(x, self.modal_num, dim=-2)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.mm_linears[i][-1](_x[i].squeeze()))
        x = torch.cat(_x_list, dim=-1)

        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SEmbeddings(nn.Module):
    def __init__(self, d_model, dim):
        super(SEmbeddings, self).__init__()
        self.lut = nn.Linear(dim, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x)
        x = x * math.sqrt(self.d_model)
        return x


class TEmbeddings(nn.Module):
    def __init__(self, dim):
        super(TEmbeddings, self).__init__()
        self.levels = 5
        self.ksize = 3
        self.d_model = 128
        self.dropout = 0.2

        self.channel_sizes = [self.d_model] * self.levels
        self.lut = TemporalConvNet(dim, self.channel_sizes, kernel_size=self.ksize, dropout=self.dropout)

    def forward(self, x):
        x = self.lut(x.transpose(1, 2)).transpose(1, 2) * math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        v = torch.arange(0, d_model, 2).type(torch.float)
        v = v * -(math.log(1000.0) / d_model)
        div_term = torch.exp(v)
        pe[:, 0::2] = torch.sin(position.type(torch.float) * div_term)
        pe[:, 1::2] = torch.cos(position.type(torch.float) * div_term)
        pe = pe.unsqueeze(0).to(device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class ProcessInput(nn.Module):
    def __init__(self, d_model, dim):
        super(ProcessInput, self).__init__()

        # if opts.embed == 'spatial':
        if False:
            self.Embeddings = SEmbeddings(opts.d_model, dim)
        # elif opts.embed == 'temporal':
        elif True:
            self.Embeddings = TEmbeddings(dim)
        self.PositionEncoding = PositionalEncoding(d_model, 0.2, max_len=5000)

    def forward(self, x):
        return self.PositionEncoding(self.Embeddings(x))


class ABAW5Model(LightningModule):

    def _reset_parameters(self) -> None:
        # Performs ResNet-style weight initialization
        for m_name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def __init__(self):
        super(ABAW5Model, self).__init__()

        self.modal_num = len([512])

        self.num_features = [512]
        self.N = 6
        self.dropout_mmatten = 0.5
        self.dropout_mtatten = 0.5
        self.dropout_ff = 0.2
        self.dropout_subconnect = 0.2
        self.h = 4
        self.h_mma = 4
        self.d_model = 128
        self.d_ff = 256

        self.input = nn.ModuleList()
        for i in range(self.modal_num):
            self.input.append(ProcessInput(self.d_model, 512))
        self.dropout_embed = nn.Dropout(p=0.2)

        multimodal_encoder_layer = nn.ModuleList()
        for i in range(self.N):
            mm_atten = MultiModalAttention(self.h_mma, self.d_model, self.modal_num, self.dropout_mmatten)
            mt_atten = nn.ModuleList()
            ff = nn.ModuleList()
            for j in range(self.modal_num):
                mt_atten.append(MultiHeadedAttention(self.h, self.d_model, self.dropout_mtatten))
                ff.append(PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout_ff))
            multimodal_encoder_layer.append(MultiModalEncoderLayer(self.d_model, self.modal_num, mm_atten, mt_atten, ff, self.dropout_subconnect))

        self.temma = MultiModalEncoder(multimodal_encoder_layer, self.N, self.modal_num)

        self.regress = nn.Linear(self.d_model, 2)
        self.drop = nn.Dropout(0.5)
        for p in self.temma.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.input.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for p in self.regress.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


        self.seq_len = cfg.DATA_LOADER.SEQ_LEN
        self.task = cfg.TASK
        self.scale_factor = 1.
        self.threshold = 0.5
        self.learning_rate = cfg.OPTIM.BASE_LR
        if self.task == 'EXPR':
            # Classification
            self.num_outputs = 8
            self.label_smoothing = cfg.TRAIN.LABEL_SMOOTHING
            # Class weights
            self.cls_weights = nn.Parameter(torch.tensor(
                [0.42715146, 5.79871879, 6.67582676, 4.19317243, 1.01682121, 1.38816715, 2.87961987, 0.32818288],
                requires_grad=False), requires_grad=False) if cfg.TRAIN.LOSS_WEIGHTS else None
            self.loss_func = partial(CEFocalLoss, scale_factor=self.scale_factor, num_classes=self.num_outputs,
                                     label_smoothing=self.label_smoothing,
                                     alpha=cfg.OPTIM.FOCAL_ALPHA, gamma=cfg.OPTIM.FOCAL_GAMMA)

            self.train_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs, average='macro')
            self.val_metric = F1Score(threshold=self.threshold, num_classes=self.num_outputs, average='macro')
        elif self.task == 'VA':
            # Classification
            self.num_outputs = 2
            self.label_smoothing = cfg.TRAIN.LABEL_SMOOTHING
            # Class weights
            self.loss_func = partial(CCCLoss, scale_factor=self.scale_factor, num_classes=self.num_outputs)

            self.train_metric = ConcordanceCorrCoef(num_outputs=2)
            self.val_metric = ConcordanceCorrCoef(num_outputs=2)
        else:
            raise ValueError('Do not know {}'.format(self.task))

        # self._reset_parameters()
        self.model_name = cfg.MODEL_NAME

#         return out, out_aux
    def forward(self, batch):
        # _x = torch.chunk(x, self.modal_num, dim=-1)
        _x_list = []
        for i in range(self.modal_num):
            _x_list.append(self.input[i](batch['feature']))
        x = torch.cat(_x_list, dim=-1)

        x = self.dropout_embed(x)
        x = self.temma(x, mask=None)
        out = self.drop(x)
        out = self.regress(x)
        return out, None

    def _shared_eval(self, batch, batch_idx, cal_loss=False):
        out, out_aux = self(batch)

        loss = None
        loss_aux_coeff = 0.2
        if cal_loss:
            if self.task != 'MTL':
                loss = self.loss_func(out, batch[self.task])
                if out_aux is not None:
                    loss = loss_aux_coeff * self.loss_func(out, batch[self.task]) + (1 - loss_aux_coeff) * loss

        return out, loss

    def update_metric(self, out, y, is_train=True):
        if self.task == 'EXPR':
            y = torch.reshape(y, (-1,))
            # out = F.softmax(out, dim=1)
        elif self.task == 'AU':
            out = torch.sigmoid(out)
            y = torch.reshape(y, (-1, self.num_outputs))

        elif self.task == 'VA':
            y = torch.reshape(y, (-1, self.num_outputs))

        out = torch.reshape(out, (-1, self.num_outputs))

        if is_train:
            self.train_metric(out, y)
        else:
            self.val_metric(out, y)
            # self.val_metric(out[:, 1:].reshape(-1), y[:, 1:].reshape(-1))

    def training_step(self, batch, batch_idx):

        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=True)

        self.log_dict({'train_metric': self.train_metric}, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=cfg.TRAIN.BATCH_SIZE)

        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=True)
        if self.task != 'MTL':
            self.update_metric(out, batch[self.task], is_train=False)

        # self.log_dict({'val_metric': self.val_metric, 'val_valence_metric': self.train_valence_metric, 'val_arousal_metric': self.train_arousal_metric, 'val_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
        #               batch_size=cfg.TEST.BATCH_SIZE)

        self.log_dict({'val_metric': self.val_metric}, on_step=False, on_epoch=True, prog_bar=True,
                       batch_size=cfg.TEST.BATCH_SIZE)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        out, loss = self._shared_eval(batch, batch_idx, cal_loss=False)

        if self.task != 'MTL':
            if self.task == 'EXPR':
                out = torch.argmax(F.softmax(out, dim=-1), dim=-1)
            elif self.task == 'AU':
                out = torch.sigmoid(out)

            with open('VA.txt', 'a+') as fd:
                batch_size = out.shape[0]
                seq_len = out.shape[1]
                for i in range(0, batch_size):
                    for j in range(0, seq_len):
                        frame_name = '{}/{:05d}.jpg'.format(batch['video_id'][i], batch['index'][i, j])
                        test = out[i, j][0]
                        row = [frame_name, str(out[i, j][0].item()), str(out[i, j][1].item())]
                        fd.write(','.join(row) + '\n')

            return out, batch[self.task], batch['index'], batch['video_id']
        else:
            raise ValueError('Do not implement MTL task.')

    def test_step(self, batch, batch_idx):
        # Copy from validation step
        out, loss = self._shared_eval(batch, batch_idx)
        # if self.task != 'MTL':
        #     self.update_metric(out, batch[self.task], is_train=False)

        # self.log_dict({'test_metric': self.val_metric, 'test_loss': loss}, on_step=False, on_epoch=True, prog_bar=True,
        #               batch_size=cfg.TEST.BATCH_SIZE)

    def configure_optimizers(self):

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_training_steps = 336
        if cfg.OPTIM.NAME == 'adam':
            print('Adam optimization ', self.learning_rate)
            opt = torch.optim.Adam(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        elif cfg.OPTIM.NAME == 'adamw':
            print('AdamW optimization ', self.learning_rate)
            opt = torch.optim.AdamW(model_parameters, lr=self.learning_rate, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        else:
            print('SGD optimization ', self.learning_rate)
            opt = torch.optim.SGD(model_parameters, lr=self.learning_rate, momentum=cfg.OPTIM.MOMENTUM,
                                  dampening=cfg.OPTIM.DAMPENING, weight_decay=cfg.OPTIM.WEIGHT_DECAY)

        opt_lr_dict = {'optimizer': opt}
        lr_policy = cfg.OPTIM.LR_POLICY
        if lr_policy == 'cos':
            warmup_start_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            scheduler = pl_lr_scheduler.LinearWarmupCosineAnnealingLR(opt, warmup_epochs=cfg.OPTIM.WARMUP_EPOCHS,
                                                                      max_epochs=cfg.OPTIM.MAX_EPOCH,
                                                                      warmup_start_lr=warmup_start_lr,
                                                                      eta_min=cfg.OPTIM.MIN_LR)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched'}})

        elif lr_policy == 'cos-restart':
            min_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            t_0 = cfg.OPTIM.WARMUP_EPOCHS * self.num_training_steps
            print('Number of training steps: ', t_0 // cfg.OPTIM.WARMUP_EPOCHS)
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=t_0, T_mult=2,
                                                                 eta_min=min_lr)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'cyclic':
            base_lr = cfg.OPTIM.BASE_LR * cfg.OPTIM.WARMUP_FACTOR
            step_size_up = self.num_training_steps * cfg.OPTIM.WARMUP_EPOCHS // 2
            mode = 'triangular'  # triangular, triangular2, exp_range
            scheduler = lr_scheduler.CyclicLR(opt, base_lr=base_lr, max_lr=self.learning_rate,
                                              step_size_up=step_size_up, mode=mode, gamma=1., cycle_momentum=False)
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'name': 'lr_sched'}})

        elif lr_policy == 'reducelrMetric':
            scheduler = lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=10, min_lr=1e-7, mode='max')
            opt_lr_dict.update({'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'name': 'lr_sched',
                                                 "monitor": "val_metric"}})
        else:
            # TODO: add 'exp', 'lin', 'steps' lr scheduler
            pass
        return opt_lr_dict

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass
######################################################################################################################



