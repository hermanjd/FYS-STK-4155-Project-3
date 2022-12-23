# Detecting facial expressions using residual convolutional neural networks

This repo contains the code for detecting facial expressions using residual convolutional neural networks.

## Getting Started

To use this project you will first need to download the dataset referanced in the paper.

You can then use the ResNet.py script to sort smiling and not smiling images into folders

Once that is done you may use the Model.py script to start training the model. You can change the layer depth of the model by changing the Resnet10() function with other functions found in ResNet.py.

For validating the model you can use ValidateModel.py


### Dependencies
Pip packages
* Keras installed
* Pandas

## Results
To have a look at raw data from the result you can find this in the Results folder

## Results
To have a look at raw data from the result you can find this in the Results folder

## FromScratchAttempt
There was made an attempt to program the entire network from scratch, but timeconstrains put a limit to it. There is still one convolutional neural network made from scratch working in this folder, but its not a bout the celeb database and can be ignored. 
