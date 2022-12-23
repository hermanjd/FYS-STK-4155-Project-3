import json
import keras
from keras.utils import image_dataset_from_directory
from keras import layers
from keras.models import Model
from ResNet import *



image_size = (227, 227) # Choose image size to work with
batch_size = 32 # Choose batchsize 

# Retrieve the images presorted into folder (if you want to sort the images into folders you can use the sortingFaces.py script to do so)
train_ds, val_ds = image_dataset_from_directory(
    "./archive/classes",
    validation_split=0.1,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ]
)

batch_end_loss = list()
batch_end_accuracy = list()

class SaveBatchLoss(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None): 
        batch_end_loss.append(logs['loss'])
        batch_end_accuracy.append(logs['accuracy'])


input_shape = (227, 227, 3)

inputs = keras.Input(shape=input_shape)
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

x = ResNet10(x, inputs)

x = layers.Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs, x)

epochs = 2

callbacks = [
    keras.callbacks.ModelCheckpoint("ResNet10_2_at_{epoch}.keras"),
    SaveBatchLoss()
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

hist = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

with open('ResNet10_2.json', 'w') as f:
    json.dump(hist.history, f)

with open('ResNet10_2_loss.csv', 'w') as f:
    for batch in batch_end_loss:
        f.write(f"{batch}\n")

with open('ResNet10_2_accuracy.csv', 'w') as f:
    for batch in batch_end_accuracy:
        f.write(f"{batch}\n")