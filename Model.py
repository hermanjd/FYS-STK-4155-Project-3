import json
import keras
from keras.utils import image_dataset_from_directory
from keras import layers
from keras.models import Model
from ResNet import *

image_size = (227, 227) # (227, 227) is chosen as image size as thats what the model needs
batch_size = 32 # Batch size is set to 32 as its a vell used size

# Retrieve the images presorted into folder (if you want to sort the images into folders you can use the sortingFaces.py script to do so)
train_ds, val_ds = image_dataset_from_directory(
    "./archive/classes",
    validation_split=0.1,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Data augmentation randomly flips images horizontally and rotates images 
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
    ]
)


# Lists to keep track of accuracy and loss for each batch
batch_end_loss = list()
batch_end_accuracy = list()


# Callback class for appending loss and accuracy
class SaveBatchLoss(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None): 
        batch_end_loss.append(logs['loss'])
        batch_end_accuracy.append(logs['accuracy'])


# Define input shape (227,227) images as we defined earlier with 3 color channels
input_shape = (227, 227, 3)

# Initialize input object
inputs = keras.Input(shape=input_shape)

# First stage in layers add the data augmentation
x = data_augmentation(inputs)

# Scale the images to instead og 0-255 to me 0 - 1
x = layers.Rescaling(1./255)(x)


# initialize all the convolutional layers based on desired Resnet 
x = ResNet10(x, inputs)

# Add the last dense layer with one output using the sigmoid function
x = layers.Dense(1, activation='sigmoid')(x)

# Create the model
model = Model(inputs, x)

# Define how many epoch
epochs = 2

# Add the callback functions for saving model weights and add the callback class for saving batxh loss and accuracy
callbacks = [
    keras.callbacks.ModelCheckpoint("ResNet10_2_at_{epoch}.keras"),
    SaveBatchLoss()
]

# Compile the model, add adam as a optimizer and set learning rate to 0.001, accuracy will also be added as a metric
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

# Save epoch history
with open('ResNet10_2.json', 'w') as f:
    json.dump(hist.history, f)

# Save batch loss history
with open('ResNet10_2_loss.csv', 'w') as f:
    for batch in batch_end_loss:
        f.write(f"{batch}\n")

# Save batch accuracy  history
with open('ResNet10_2_accuracy.csv', 'w') as f:
    for batch in batch_end_accuracy:
        f.write(f"{batch}\n")