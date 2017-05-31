import os
import struct
import numpy as np
# import matplotlib.pyplot as plt

# myPyNN
from myPyNN import *


# Keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

# Read training images
fname_img = os.path.join('.', 'train-images-idx3-ubyte')
fname_lbl = os.path.join('.', 'train-labels-idx1-ubyte')

with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    trainLbls = np.fromfile(flbl, dtype=np.int8)

with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    trainImages = np.fromfile(fimg, dtype=np.uint8).reshape(len(trainLbls), rows*cols)

# Read test images
fname_img = os.path.join('.', 't10k-images-idx3-ubyte')
with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    testImages = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows*cols)

# Make it float
trainImages = trainImages.astype('float32')/255.0
testImages = testImages.astype('float32')/255.0

# Encode trainY as nx10 np array
trainY = np.zeros((len(trainLbls), 10))
for i in range(len(trainLbls)):
    trainY[i, trainLbls[i]] = 1

# fig = plt.figure(figsize=(10, 2))
# for i in range(20):
#     ax1 = fig.add_subplot(2, 10, i+1)
#     ax1.imshow(np.reshape(trainImages[i], (28, 28)), cmap='gray');
#     ax1.axis('off')

# USING myPyNN
# Got 0.94460 accuracy on 10k test data

# Initialize network
layers = [784, 64, 64, 10]
weights = initializeWeights(layers)

# Set options of mini-batch gradient descent
minibatchSize = 10
nEpochs = 60
learningRate = 1.0
mu = 0.9

# Split Train and Val data
valSplit = 0.2      # Use this much for validation 
fullIdx = list(range(len(trainY)))
np.random.shuffle(fullIdx)
trainIdx = fullIdx[int(valSplit*len(trainY)):]
valIdx = fullIdx[:int(valSplit*len(trainY))]

# Train
trainUsingMinibatchGD(weights, trainImages[trainIdx], trainY[trainIdx],
                      minibatchSize, nEpochs, learningRate, decay=None, optimizer='nag', mu=mu,
                      testX=trainImages[valIdx], testY=trainY[valIdx])

# Test
yPreds = np.argmax(forwardProp(testImages, weights)[-1], axis=1)

# Save as csv
yTest = np.zeros((len(yPreds), 2))
yTest[:, 0] = np.array(list(range(len(yPreds)))) + 1
yTest[:, 1] = yPreds
np.savetxt("yTest_64_64.csv", yTest, fmt='%i',
           delimiter=',', header="Id,Category", comments='')

# USING MY CONV2D
# 99.4% accuracy on 28k test data

# Model
inputA = Input(shape=(28, 28, 1))
x = Conv2D(256, 7, strides=1, padding='valid', activation='relu',  input_shape=(28, 28, 1))(inputA)
x = Conv2D(256, 7, strides=1, padding='valid', activation='relu')(x)
x = Flatten()(x)
preds = Dense(10, activation='softmax')(x)
model = Model(inputA, preds)

model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

# Model Checkpoint
filepath = "C256C256f10-val0.2-epoch{epoch:02d}-l{loss:.4f}-a{acc:.4f}-vl{val_loss:.4f}-va{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True)

# Train
model.fit(np.reshape(trainImages, (len(trainImages), 28, 28, 1)), trainY,
          batch_size=128, epochs=100, verbose=1, callbacks=[checkpoint],
          validation_split=valSplit)

# Test
yPreds = np.argmax(model.predict(np.reshape(testImages, (len(testImages), 28, 28, 1))), axis=1)
yTest = np.zeros((len(yPreds), 2))
yTest[:, 0] = np.array(list(range(len(yPreds)))) + 1
yTest[:, 1] = yPreds
np.savetxt("yTest10k_C256C256f10_epoch06.csv", yTest, fmt='%i',
           delimiter=',', header="Id,Category", comments='')

# TESTING ON 28k MNIST TEST
MNISTtest = np.loadtxt("MNIST_test.csv", delimiter=",", skiprows=1).astype(int)
model.load_weights(
    "C256C256f10-val0.2-epoch04-l0.0131-a0.9960-vl0.0456-va0.9879.hdf5")
yPreds = np.argmax(model.predict(np.reshape(MNISTtest.astype('float32')/255.0, (len(MNISTtest), 28, 28, 1))), axis=1)
yTest = np.zeros((len(yPreds), 2))
yTest[:, 0] = np.array(list(range(len(yPreds)))) + 1
yTest[:, 1] = yPreds
np.savetxt("yTest28k_C256-7-C256-7-f10_epoch06.csv", yTest, fmt='%i',
           delimiter=',', header="ImageId,Label", comments='')

# USING PYTORCH EXAMPLE
# https://github.com/pytorch/examples/blob/master/mnist/main.py
# After 100 epochs: loss: 0.0503 - acc: 0.9844 - val_loss: 0.0306 - val_acc: 0.9915
inputA = Input(shape=(28, 28, 1))
x = Conv2D(10, 5, strides=1, padding='valid', activation='relu', input_shape=(28, 28, 1))(inputA)
x = MaxPooling2D(pool_size=2, strides=2)(x)
x = Conv2D(20, 5, strides=1, padding='valid', activation='relu')(x)
x = Dropout(0.5)(x)
x = MaxPooling2D(pool_size=2, strides=2)(x)
x = Flatten()(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(10, activation='softmax')(x)
model = Model(inputA, preds)

sgd = SGD(lr=0.01, momentum=0.5, decay=0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Model Checkpoint
filepath = "C10-5-P2-C20-5-do0.5-P2-f50-do0.5-f10-val0.2-epoch{epoch:02d}-l{loss:.4f}-a{acc:.4f}-vl{val_loss:.4f}-va{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, verbose=1, save_best_only=False, save_weights_only=True)

model.fit(np.reshape(trainImages, (len(trainImages), 28, 28, 1)), trainY,
                    batch_size=64, epochs=100, verbose=1,
                    callbacks=[checkpoint], validation_split=validationSplit,
                    initial_epoch=0)

# Test
yPreds = np.argmax(model.predict(np.reshape(testImages, (len(testImages), 28, 28, 1))), axis=1)
yTest = np.zeros((len(yPreds), 2))
yTest[:, 0] = np.array(list(range(len(yPreds)))) + 1
yTest[:, 1] = yPreds
np.savetxt("yTest10k_PyTorchEg_epoch09.csv", yTest, fmt='%i',
           delimiter=',', header="Id,Category", comments='')
