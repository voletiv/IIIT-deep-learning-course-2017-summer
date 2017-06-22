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
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, LeakyReLU
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.optimizers import Adam
# from keras.callbacks import LearningRateScheduler

totalPixels = 784
imageWidth = 28

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

# Make it float
trainImages = np.reshape(trainImages.astype(
    'float32') / 255., (len(trainImages), imageWidth, imageWidth, 1))
testImages = np.reshape(testImages.astype('float32') /
                        255., (len(testImages), imageWidth, imageWidth, 1))

# NETWORK

minibatchSize = 32
fitBatchSize = 128
nEpochs = 10
discImproveSteps = 1
latentDim = 100
dropoutRate = 0.25

# Discriminator Net
discInput = Input(shape=(imageWidth, imageWidth, 1))
H = Conv2D(256, (5, 5), strides=(2, 2), padding='same',
           activation='relu')(discInput)
H = LeakyReLU(0.2)(H)
H = Dropout(dropoutRate)(H)
H = Conv2D(512, (5, 5), strides=(2, 2), padding='same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropoutRate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropoutRate)(H)
discOutput = Dense(1, activation='sigmoid')(H)
discriminator = Model(discInput, discOutput)


# Discriminator Loss
def discLoss(y_true, y_pred):
    # Categorical cross-entropy
    y_pred = K.clip(y_pred, 0.000001, 0.9999999)
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

# Compile discriminator
dOpt = Adam(lr=1e-3)
discriminator.compile(optimizer=dOpt, loss=discLoss)

# Generator Net
genInput = Input(shape=(latentDim,))                                                                 #100
H = Dense(14*14*200, activation='relu', init='glorot_normal')(genInput)                              #39200
H = BatchNormalization()(H)                                                                          #39200
H = Reshape((14, 14, 200))(H)                                                                        #14x14x200
H = UpSampling2D(size=(2, 2))(H)                                                                     #28x28x200
H = Conv2D(100, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(H)   #28x28x100
H = BatchNormalization()(H)                                                                          #28x28x100
H = Conv2D(50, (3, 3), padding='same', activation='relu', kernel_initializer='glorot_uniform')(H)    #28x28x50
H = BatchNormalization()(H)                                                                          #28x28x50
genOutput = Conv2D(1, (1, 1), padding='same', activation='sigmoid', kernel_initializer='glorot_uniform')(H)
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
gOpt = Adam(lr=1e-4)
generator.compile(optimizer=gOpt, loss=dummyGenLoss)

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

GAN.compile(optimizer=gOpt, loss='binary_crossentropy')
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

# Set label of real images as 1
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
discPreds = discriminator.predict(discTrainInputs, verbose=1)
acc = (np.round(discPreds) - np.reshape(discTrainLabels,
                                        (len(discTrainLabels), 1)) == 0).sum() * 100. / len(discTrainLabels)

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


def trainForNEpochs(nEpochs, pltFreq=25):
    # For each epoch
    for e in tqdm(range(nEpochs)):

        # TRAIN THE DISCRIMINATOR

        # For the number of steps of training discriminator per generator train
        for k in range(discImproveSteps):

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

            # Set label of real images as 1
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

        # Fake image labels (really doesn't matter, generator loss does not
        # include y_true)
        fakeImageLabels = np.zeros((minibatchSize,))

        # Train the Generator
        makeTrainable(discriminator, False)
        # GAN.fit(fakeImagesMinibatch, fakeImageLabels, batch_size=fitBatchSize, epochs=1)
        gLoss = GAN.train_on_batch(latentVars, fakeImageLabels)
        losses["g"].append(gLoss)

        if e % pltFreq == pltFreq - 1:
            plotLosses(losses)
            plotGen()

# Train for 6000 epochs at original learning rates
trainForNEpochs(6000, 500)

# Train for 2000 epochs at reduced learning rates
gOpt.lr.set_value(1e-5)
dOpt.lr.set_value(1e-4)
train_for_n(2000, 500)

# Train for 2000 epochs at reduced learning rates
gOpt.lr.set_value(1e-6)
dOpt.lr.set_value(1e-5)
train_for_n(2000, 500)

