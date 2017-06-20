# http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
# http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
# https://github.com/osh/KerasGAN/blob/master/mnist_gan.py

import os
import numpy as np
import struct
# import cv2
# from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# with tf.device('/gpu:0'):
from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.objectives import binary_crossentropy
# from keras.callbacks import LearningRateScheduler

images = []
folder = 'img_align_celeba'
totalImages = len(os.listdir(folder))

# Read training images
fname_img = os.path.join('.', 'train-images-idx3-ubyte')
with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    trainImages = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows*cols)

# Read training labels
fname_lbl = os.path.join('.', 'train-labels-idx1-ubyte')
with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    trainLbls = np.fromfile(flbl, dtype=np.int8)

# Read test images
fname_img = os.path.join('.', 't10k-images-idx3-ubyte')
with open(fname_img, 'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    testImages = np.fromfile(fimg, dtype=np.uint8).reshape(num, rows*cols)

# Read test labels
fname_lbl = os.path.join('.', 't10k-labels-idx1-ubyte')
with open(fname_lbl, 'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    testLbls = np.fromfile(flbl, dtype=np.int8)

# Make it float
trainImages = trainImages.astype('float32')/255.
testImages = testImages.astype('float32')/255.

# NETWORK

minibatchSize = 50
fitBatchSize = 128
nEpochs = 10
discImproveSteps = 1

totalPixels = 784
imageWidth = 28

discHiddenDim = 128
latentDim = 100
genHiddenDim = 128

# Discriminator Net
discInput = Input(shape=(totalPixels,))
discHidden = Dense(discHiddenDim, activation='relu')(discInput)
discOutput = Dense(1, activation='sigmoid')(discHidden)
discriminator = Model(discInput, discOutput)

# Discriminator Loss
def discLoss(y_true, y_pred):
    # Categorical cross-entropy
    return -K.mean(y_true*K.log(y_pred) + (1 - y_true)*K.log(1 - y_pred))

# Compile discriminator
discriminator.compile(optimizer='adam', loss=discLoss)

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
generator.compile(optimizer='adam', loss=dummyGenLoss)

# GAN

# Enable/disable training in a network
def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

# Freeze weights in the discriminator while training generator as part of GAN
make_trainable(discriminator, False)

ganInput = Input(shape=[latentDim])
ganHidden = generator(ganInput)
ganOutput = discriminator(ganHidden)
GAN = Model(ganInput, ganOutput)

GAN.compile(optimizer='adam', loss='binary_crossentropy')
# GAN.summary()

# PRE-TRAIN DISCRIMINATOR ONCE

preTrainNum = 10000
# Select a random minibatch of real images
realImagesMinibatch = trainImages[np.random.randint(0, len(trainImages),size=preTrainNum)]

# Sample a minibatch-size of latent variables from noise prior (Gaussian)
latentVars = np.random.random((preTrainNum, latentDim))

# Generate images from noise using the generator
fakeImagesMinibatch = generator.predict(latentVars)

# Full list of training images input to discriminator
discTrainInputs = np.concatenate((realImagesMinibatch, fakeImagesMinibatch))

# List of training image labels
discTrainLabels = np.zeros((2*preTrainNum,))

# Set label of real images as 1
discTrainLabels[:preTrainNum] = 1

# Make Discriminator trainable
make_trainable(discriminator,True)

# Pre-train discriminator once
discriminator.fit(discTrainInputs, discTrainLabels, batch_size=fitBatchSize, epochs=1)

# TRAIN

def trainForNEpochs(nEpochs):
    # For each epoch
    for e in range(nEpochs):

        # TRAIN THE DISCRIMINATOR

        # For the number of discriminator improvement steps
        for k in range(discImproveSteps):

            # Select a random minibatch of real images
            realImagesMinibatch = trainImages[np.random.randint(0, len(trainImages),size=minibatchSize)]

            # Sample a minibatch-size of latent variables from noise prior (Gaussian)
            latentVars = np.random.random((minibatchSize, latentDim))

            # Generate images from noise using the generator
            fakeImagesMinibatch = generator.predict(latentVars)

            # Full list of training images input to discriminator
            discTrainInputs = np.concatenate((realImagesMinibatch, fakeImagesMinibatch))

            # List of training image labels
            discTrainLabels = np.zeros((2*minibatchSize,))

            # Set label of real images as 1
            discTrainLabels[:minibatchSize] = 1

            # Shuffle the training images and labels
            discTrainMiniIdx = list(range(2*minibatchSize))
            np.random.shuffle(discTrainMiniIdx)
            discTrainInputs = discTrainInputs[discTrainMiniIdx]
            discTrainLabels = discTrainLabels[discTrainMiniIdx]

            # Train the discriminator
            discriminator.fit(discTrainInputs, discTrainLabels, batch_size=fitBatchSize, epochs=1)

        # TRAIN THE GENERATOR via GAN

        # Sample a minibatch-size of latent variables from noise prior (Gaussian)
        latentVars = np.random.random((minibatchSize, latentDim))

        # Generate images from noise using the generator
        fakeImagesMinibatch = generator.predict(latentVars)

        # Fake image labels (really doesn't matter, generator loss does not include y_true)
        fakeImageLabels = np.zeros((minibatchSize,))

        # Train the generator
        GAN.fit(fakeImagesMinibatch, fakeImageLabels, batch_size=fitBatchSize, epochs=1)








