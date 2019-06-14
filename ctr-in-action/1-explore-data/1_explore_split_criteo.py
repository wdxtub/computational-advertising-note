import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# https://github.com/PnYuan/Practice-of-Machine-Learning/blob/master/code/Kaggle_CTR/data_pre.py
# https://github.com/Hourout/CTR-keras/blob/master/CTR/LR.py

header = ""
with open("../data/criteo_sample.txt") as f:
    header = f.readline()

print("Dataset Header")
print(header, end='')

train_size = 200000
print("Generate Train Dataset, size %d" % train_size)
with open("../data/criteo_train.txt") as f:
    with open("../data/criteo_train_small.txt", "w") as out:
        out.write(header)
        index = 0
        while index < train_size:
            line = f.readline()
            out.write(line.replace("\t", ","))
            index = index + 1
            if index % 10000 == 0:
                print("processing %d" % index)

test_size = 40000
print("Generate Test Dataset, size %d" % test_size)
with open("../data/criteo_test.txt") as f:
    with open("../data/criteo_test_small.txt", "w") as out:
        out.write(header)
        index = 0
        while index < test_size:
            line = f.readline()
            out.write(line.replace("\t", ","))
            index = index + 1
            if index % 10000 == 0:
                print("processing %d" % index)

print("ALL Done")