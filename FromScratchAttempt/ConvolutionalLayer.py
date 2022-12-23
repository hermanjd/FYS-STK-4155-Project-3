import numpy as np
from scipy import signal

class ConvolutionalLayer():
    def __init__(self, inputShape, filterSize, filterCount, weightScaler):
        self.inputShape = inputShape
        self.outputShape = (filterCount, inputShape[1] - filterSize + 1, inputShape[2] - filterSize + 1)
        self.weights = weightScaler * np.random.randn(filterCount, inputShape[0], filterSize, filterSize)
        self.biases = np.zeros(self.outputShape)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.weights.shape[0]):
            for j in range(self.inputShape[0]):
                self.output += signal.correlate2d(self.input[j], self.weights[i,j], "valid")
    
    def backward(self, doutput):
        dfilters = np.zeros(self.weights.shape)
        dinput = np.zeros(self.inputShape)
        for i in range(self.weights.shape[0]):
            for j in range(self.inputShape[0]):
                dfilters[i,j] = signal.correlate2d(self.input[j], doutput[i], "valid")
                dinput[j] += signal.convolve2d(doutput[i], self.weights[i,j], "full")
        self.dweights = dfilters
        self.dbiases = doutput
        self.dinputs = dinput
