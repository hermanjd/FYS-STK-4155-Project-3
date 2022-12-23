import json
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.utils import image_dataset_from_directory
import pandas as pd
from keras import layers, Input
from keras.models import Model
from ResNet import *
from livelossplot import PlotLossesKeras
from sklearn.metrics import r2_score

image_size = (227, 227)
batch_size = 32

train_ds, val_ds = image_dataset_from_directory(
    "./archive/classes",
    validation_split=0.1,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


import matplotlib.pyplot as plt

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
plt.show()