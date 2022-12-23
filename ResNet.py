from keras import layers

#This two stack residual block is what forms the basis for creating residual 
def TwoStackResidualBlock(x, kernals, strides):
    # Save a copy of input to be used in skip connection
    skipConnection = x

    #Add First convolutional layer with stride specified and runds the matrix through the relu activation function
    x = layers.Conv2D(kernals, kernel_size=1, strides=strides, padding='same', activation='relu')(x)

    #Then preforme bachnormalization
    x = layers.BatchNormalization()(x)

    #Add the second convolution layer
    x = layers.Conv2D(kernals, kernel_size=3, strides=1, padding='same', activation='relu')(x)

    #Then preform another bach normalization
    x = layers.BatchNormalization()(x)  

    #We now check if the skip layer has the same shape as the outout of the second convolution layer because we can not add a siklayer of different shape to the output
    if (strides != 1 or kernals != skipConnection.shape[-1]):
        #Apply a convolution with the same number of kernals and stride so that it fits the output of convolution number 2
        skipConnection = layers.Conv2D(kernals, kernel_size=1, strides=strides, padding='same')(skipConnection)

        #Apply a batch normalization to the skipconnection
        skipConnection = layers.BatchNormalization()(skipConnection)

    #Add the skip connection to the output and run it through the relu activation function
    x = layers.add([x, skipConnection])
    x = layers.Activation('relu')(x)

    return x

def ResNet10(x, inputs):
    # Add the first convolutional layer of the model
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)

    # Apply batch normalization
    x = layers.BatchNormalization()(x)

    # Add a maxpool layer
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # Add residual blocks in accordance to the ResNet10 architechture
    x = TwoStackResidualBlock(x, 64, 1)
    x = TwoStackResidualBlock(x, 128, 2)
    x = TwoStackResidualBlock(x, 256, 2)
    x = TwoStackResidualBlock(x, 512, 2)

    # Apply Global averageing 
    x = layers.GlobalAveragePooling2D()(x)
    return x

def ResNet18(x, inputs):
    # Add the first convolutional layer of the model
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)

    # Apply batch normalization
    x = layers.BatchNormalization()(x)

    # Add a maxpool layer
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # Add residual blocks in accordance to the ResNet18 architechture
    x = TwoStackResidualBlock(x, 64, 1)
    x = TwoStackResidualBlock(x, 64, 1)
    x = TwoStackResidualBlock(x, 128, 2)
    x = TwoStackResidualBlock(x, 128, 1)
    x = TwoStackResidualBlock(x, 256, 2)
    x = TwoStackResidualBlock(x, 256, 1)
    x = TwoStackResidualBlock(x, 512, 2)
    x = TwoStackResidualBlock(x, 512, 1)

    # Apply Global averageing 
    x = layers.GlobalAveragePooling2D()(x)
    return x

def ResNet34(x, inputs):
    # Add the first convolutional layer of the model
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
    
    # Apply batch normalization
    x = layers.BatchNormalization()(x)

    # Add a maxpool layer
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # Add residual blocks in accordance to the ResNet34 architechture
    x = TwoStackResidualBlock(x, 64, 1)
    x = TwoStackResidualBlock(x, 64, 1)
    x = TwoStackResidualBlock(x, 64, 1)
    x = TwoStackResidualBlock(x, 128, 2)
    x = TwoStackResidualBlock(x, 128, 2)
    x = TwoStackResidualBlock(x, 128, 2)
    x = TwoStackResidualBlock(x, 128, 2)
    x = TwoStackResidualBlock(x, 256, 2)
    x = TwoStackResidualBlock(x, 256, 2)
    x = TwoStackResidualBlock(x, 256, 2)
    x = TwoStackResidualBlock(x, 256, 2)
    x = TwoStackResidualBlock(x, 512, 2)
    x = TwoStackResidualBlock(x, 512, 2)
    x = TwoStackResidualBlock(x, 512, 2)
    
    # Apply Global averageing 
    x = layers.GlobalAveragePooling2D()(x)
    return x
