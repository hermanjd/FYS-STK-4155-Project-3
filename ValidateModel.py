import keras
from keras.utils import image_dataset_from_directory
from keras import layers
from keras.models import Model
import keras.backend as K

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

input_shape = (227, 227, 3)

inputs = keras.Input(shape=input_shape)

model = keras.models.load_model('./TrainedModels/ResNet10_at_2.keras')

penultimate_layer = model.layers[-1]  # layer that you want to connect your new FC layer to 
new_top_layer = layers.Lambda(lambda x: K.round(x))(penultimate_layer.output)
new_model = Model(model.input, new_top_layer)  # define your new model

new_model.summary()

score = model.evaluate(val_ds, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
