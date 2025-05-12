label =[]
list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
for i in range(10):
    for j in range(10):
        letter = list[j]
        label.append((i, letter))
# print(label)\
import numpy as np
label = np.array(label)
print(label[0][0])
print(label[0][1])
print(label[1][0])
print(label[:][0])
