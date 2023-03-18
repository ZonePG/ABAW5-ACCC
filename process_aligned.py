import csv

csv_reader = csv.reader(open('VA.txt'))
csv_reader_example = csv.reader(open('VA_test_set_example.txt'))

orgin = []
example = []

result = []

for line in csv_reader:
    if len(orgin) == 0 or line[0] != orgin[-1][0]:
        orgin.append(line)

for line in csv_reader_example:
    example.append(line)

i = 0
j = 0
while i < len(example):
    if j == len(orgin):
        result.append([example[i][0], result[-1][1], result[-1][2]])
        i += 1
    elif example[i][0] == orgin[j][0]:
        # print(i, example[i], orgin[j])
        result.append(orgin[j])
        i += 1
        j += 1
    elif (int(example[i][0][-9:-4]) < int(orgin[j][0][-9:-4])) and (example[i][0][:-9] == orgin[j][0][:-9]):
        result.append([example[i][0], orgin[j][1], orgin[j][2]])
        i += 1
    elif (int(example[i][0][-9:-4]) > int(orgin[j][0][-9:-4])) and (example[i][0][:-9] != orgin[j][0][:-9]):
        result.append([example[i][0], result[-1][1], result[-1][2]])
        i += 1
    elif (int(example[i][0][-9:-4]) < int(orgin[j][0][-9:-4])) and (example[i][0][:-9] != orgin[j][0][:-9]):
        j += 1
    # print(i, j)
    # print(int(example[i][0][-9:-4]), int(orgin[j][0][-9:-4]))
    # print(example[i][0][:-9], orgin[j][0][:-9])
    # elif (int(example[i][0][-9:-4]) < int(orgin[j][0][-9:-4])) and (example[i][0][-9] != orgin[j][0][:-9]):
    #     # print(i, j)
    #     # print(i, example[i], orgin[j])
    #     # print(result[-1][1], result[-1][2])
    #     result.append([example[i][0], result[-1][1], result[-1][2]])
    #     # print(result[-1][1], result[-1][2])
    #     i += 1


with open('VA_summit.txt', 'a+') as fd:
    for line in result:
        fd.write(','.join(line) + '\n')
