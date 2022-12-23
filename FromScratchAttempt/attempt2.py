import numpy as np
from Layers import *
from ConvolutionalLayer import *
import gzip
from sklearn.model_selection import train_test_split


num_images = 1000
image_size = 28
epochs = 2
decay=5e-7
learning_rate = 0.002
momentum=0.2
weightscalar= np.sqrt(1/1000);

images_file_stream = gzip.open('images/train-images-idx3-ubyte.gz','r')
labels_file_stream = gzip.open('images/train-labels-idx1-ubyte.gz','r')
labels_file_stream.read(8)
buf = labels_file_stream.read(num_images)
labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
b = np.zeros((labels.size, labels.max() + 1))
b[np.arange(labels.size), labels] = 1

images_file_stream.read(16)
buf = images_file_stream.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
data = data.reshape(num_images, 1, image_size, image_size)
data = data / data.max()

X_train, X_test, y_train, y_test = train_test_split(data, b, test_size=0.2, random_state=42)


con1 = ConvolutionalLayer((1, image_size, image_size),3,10, weightscalar)
act1 = ReluActivationLayer()
con2 = ConvolutionalLayer((10, 26, 26),3,15, 4)
act2 = ReluActivationLayer()
con3 = ConvolutionalLayer((15, 24, 24),3,5, 4)
act3 = ReluActivationLayer()


con1.forward(X_train[0])
con2.forward(con1.output)
con3.forward(con2.output)
print(con3.output.shape)




"""
con1 = ConvolutionalLayer((1, image_size, image_size),3,10, weightscalar)
act1 = ReluActivationLayer()
res1 = ReshapeLayer((10,26,26),(-1, (10*26*26)))
den1 = DenseLayer((10*26*26), 16, weightscalar, weight_regularizer_l2=5e-2, bias_regularizer_l2=5e-2)
act2 = ReluActivationLayer()
den2 = DenseLayer(16, 10, weightscalar, weight_regularizer_l2=5e-2, bias_regularizer_l2=5e-2)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(decay=decay, learning_rate=learning_rate, momentum=momentum)

accuracy_train = []
accuracy_test = []


for epoch in range(epochs):
    print(epoch)
    for i in range(len(X_train)):
        con1.forward(X_train[i])
        act1.forward(con1.output)
        res1.forward(act1.output)
        den1.forward(res1.output)
        act2.forward(den1.output)
        den2.forward(act2.output)
        data_loss = loss_activation.forward(den2.output, y_train[i])
        accuracy_train.append(sum(den1.dweights))
        predictions = np.argmax(loss_activation.output, axis=1)
        print(data_loss)
        accuracy = np.mean(predictions==y_train[i])
        
        # Calculate overall loss
        loss = data_loss

        if(loss > 14):
            break
        # Backward pass
        loss_activation.backward(loss_activation.output, y_train[i])
        den2.backward(loss_activation.dinputs)
        act2.backward(den2.dinputs)
        den1.backward(act2.dinputs)
        res1.backward(den1.dinputs)
        act1.backward(res1.dinputs)
        con1.backward(act1.dinputs)

    
        #print(con1.dweights[0])
        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(con1)
        optimizer.update_params(den1)
        optimizer.update_params(den2)
        optimizer.post_update_params()



#txt = f's_{num_images}_n_{neurons}_l_2_bs_{batch_size}_SGDS_(d=5e-7_lr=0.02_m=0.9)'
#print(txt)
#plt.title(txt)
plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.show()

"""








"""
print(data.shape)



image = img.imread("./images/hint.jpg");

print(image.shape)

image = image/255



lol = np.rot90(image, 1, axes=(2, 0))



con1 = Layers.ConvolutionalLayer(lol.shape,3,10)
act1 = Layers.ReluActivationLayer()


con1.forward(lol)
act1.forward(con1.output)
print(con1.output.shape)


print(act1.output.shape)
plt.imshow(con1.output[0])
plt.show()
plt.imshow(act1.output[0])
plt.show()
"""