import csv
from matplotlib import pyplot as plt
import numpy as np


with open('ResNet10_2_accuracy.csv', newline='') as f:
    reader = csv.reader(f)
    ResNet10Data = list(np.float_(list(reader)))

with open('ResNet34_accuracy.csv', newline='') as f:
    reader = csv.reader(f)
    ResNet34Data = list(np.float_(list(reader)))



plt.plot(ResNet10Data, label="ResNet10")
plt.plot(ResNet34Data, label="ResNet34")
plt.legend()
plt.show()