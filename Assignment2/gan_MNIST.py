# http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
# http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
# https://github.com/osh/KerasGAN/blob/master/mnist_gan.py

import os
import numpy as np
import struct
# import cv2
# from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

# with tf.device('/gpu:0'):
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.objectives import binary_crossentropy
# from keras.callbacks import LearningRateScheduler

# Read training images
fname_img = os.path.join('.', 'train-images-idx3-ubyte')
with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    trainImages = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows * cols)

# Read training labels
fname_lbl = os.path.join('.', 'train-labels-idx1-ubyte')
with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    trainLbls = np.fromfile(flbl, dtype=np.int8)

# Read test images
fname_img = os.path.join('.', 't10k-images-idx3-ubyte')
with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    testImages = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows * cols)

# Read test labels
fname_lbl = os.path.join('.', 't10k-labels-idx1-ubyte')
with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    testLbls = np.fromfile(flbl, dtype=np.int8)

# Make it float and 0-centered
trainImages = (trainImages.astype('float32')- 127.5) / 255.
testImages = (testImages.astype('float32') - 127.5) / 255.

# NETWORK

minibatchSize = 128
fitBatchSize = 128
nEpochs = 10
discImproveSteps = 1


discHiddenDim = 128
latentDim = 100
genHiddenDim = 128

# Discriminator Net
discInput = Input(shape=(totalPixels,))
discHidden = Dense(discHiddenDim, activation='relu')(discInput)
discOutput = Dense(1, activation='tanh')(discHidden)
discriminator = Model(discInput, discOutput)

# Discriminator Loss


def discLoss(y_true, y_pred):
    # Categorical cross-entropy
    y_pred = K.clip(y_pred, 0.000001, 0.9999999)
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

# Compile discriminator
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# Generator Net
genInput = Input(shape=(latentDim,))
genHidden = Dense(genHiddenDim, activation='relu')(genInput)
genOutput = Dense(totalPixels, activation='sigmoid')(genHidden)
generator = Model(genInput, genOutput)

# Actual generator loss


def actualGenLoss(y_true, y_pred):
    return -K.mean(K.log(discriminator.predict(y_pred)))

# BUT we are not going to use this, since "discriminator.predict()" is somehow not usable in this context (Keras)
# Instead, we shall just use binary_crossentropy, and train the generator as part of GAN,
# while keeping discriminator untrainable
# Dummy generator loss


def dummyGenLoss(y_true, y_pred):
    # Binary crossentropy
    return -K.mean(K.log(y_pred))

# Compile generator
generator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN

# Enable/disable training in a network


def makeTrainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

# Freeze weights in the discriminator while training generator as part of GAN
makeTrainable(discriminator, False)

ganInput = Input(shape=[latentDim])
ganHidden = generator(ganInput)
ganOutput = discriminator(ganHidden)
GAN = Model(ganInput, ganOutput)

GAN.compile(optimizer='adam', loss='binary_crossentropy')
# GAN.summary()

# PRE-TRAIN DISCRIMINATOR ONCE

preTrainNum = 10000
# Select a random minibatch of real images
realImagesMinibatch = trainImages[
    np.random.randint(0, len(trainImages), size=preTrainNum)]

# Sample a minibatch-size of latent variables from noise prior (Gaussian)
latentVars = np.random.random((preTrainNum, latentDim))

# Generate images from noise using the generator
fakeImagesMinibatch = generator.predict(latentVars)

# Full list of training images input to discriminator
discTrainInputs = np.concatenate((realImagesMinibatch, fakeImagesMinibatch))

# List of training image labels
discTrainLabels = np.zeros((2 * preTrainNum,))

# Set label of real images as 2st column
discTrainLabels[:preTrainNum] = 1

# Shuffle the inputs
discInputIdx = list(range(len(discTrainInputs)))
np.random.shuffle(discInputIdx)
discTrainInputs = discTrainInputs[discInputIdx]
discTrainLabels = discTrainLabels[discInputIdx]

# Make Discriminator trainable
makeTrainable(discriminator, True)

# Pre-train discriminator once
discriminator.fit(discTrainInputs, discTrainLabels,
                  batch_size=fitBatchSize, epochs=1)

# Check accuracy of discriminator
discPreds = discriminator.predict(discTrainInputs)
acc = (np.round(discPreds) - np.reshape(discTrainLabels,
                                        (len(discTrainLabels), 1)) == 0).sum() * 100. / len(discTrainLabels)
print(acc)

# TO PLOT LOSSES
losses = {"d": [], "g": []}


def plotLosses(losses):
    plt.figure(figsize=(10, 8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    plt.show()


def plotGen(n=16, dim=(4, 4), figsize=(10, 10)):
    latentVars = np.random.random((n, latentDim))
    generatedImages = generator.predict(latentVars)
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = np.reshape(generatedImages[i], (imageWidth, imageWidth))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# TRAIN
trainImagesFullIdx = list(range(len(trainImages)))

def trainForNEpochs(nEpochs, pltFreq=25):
    # For each epoch
    for e in tqdm(range(nEpochs)):

        # TRAIN THE DISCRIMINATOR


        # Select a random minibatch of real images
        realImagesMinibatch = trainImages[np.random.randint(
            0, len(trainImages), size=minibatchSize)]

        # Sample a minibatch-size of latent variables from noise prior
        # (Gaussian)
        latentVars = np.random.random((minibatchSize, latentDim))

        # Generate images from noise using the generator
        fakeImagesMinibatch = generator.predict(latentVars)

        # Full list of training images input to discriminator
        discTrainInputs = np.concatenate(
            (realImagesMinibatch, fakeImagesMinibatch))

        # List of training image labels
        discTrainLabels = np.zeros((2 * minibatchSize,))

        # Set label of real images as 2nd column
        discTrainLabels[:minibatchSize] = 1

        # Shuffle the training images and labels
        discTrainMiniIdx = list(range(2 * minibatchSize))
        np.random.shuffle(discTrainMiniIdx)
        discTrainInputs = discTrainInputs[discTrainMiniIdx]
        discTrainLabels = discTrainLabels[discTrainMiniIdx]

        # Train the discriminator
        makeTrainable(discriminator, True)
        # discriminator.fit(discTrainInputs, discTrainLabels, batch_size=fitBatchSize, epochs=1)
        dLoss = discriminator.train_on_batch(
            discTrainInputs, discTrainLabels)
        losses["d"].append(dLoss)

        # TRAIN THE GENERATOR via GAN

        # Sample a minibatch-size of latent variables from noise prior
        # (Gaussian)
        latentVars = np.random.random((minibatchSize, latentDim))

        # Fake image labels)
        fakeImageLabels = np.ones((minibatchSize,))

        # Train the Generato via GANr
        makeTrainable(discriminator, False)
        # GAN.fit(fakeImagesMinibatch, fakeImageLabels, batch_size=fitBatchSize, epochs=1)
        gLoss = GAN.train_on_batch(latentVars, fakeImageLabels)
        losses["g"].append(gLoss)

        if e % pltFreq == pltFreq - 1:
            plotLosses(losses)
            plotGen()

trainForNEpochs(10000, 1000)
plotLosses(losses)
plotGen()
