# http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
# http://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/
# https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py

import os
import numpy as np
import struct
# import cv2
# from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf

with tf.device('/gpu:0'):
    from keras import backend as K
    from keras.layers import Input, Flatten, Dense, Lambda, Reshape
    from keras.layers.core import Dropout, Activation
    from keras.layers import Conv2D, Conv2DTranspose
    from keras.layers.advanced_activations import LeakyReLU
    from keras.models import Model
    from keras.objectives import binary_crossentropy
    from keras.optimizers import Adam
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
trainImages = (trainImages.astype('float32') - 127.5) / 255.
testImages = (testImages.astype('float32') - 127.5) / 255.

# Make it 2D
imageWidth = 28
trainImages = np.reshape(
    trainImages, (len(trainImages), imageWidth, imageWidth, 1))
testImages = np.reshape(
    testImages, (len(testImages), imageWidth, imageWidth, 1))

# NETWORK

minibatchSize = 128
latentDim = 100
genNumOfImages = 100

# Optimizer
myAdam = Adam(lr=0.0002, beta_1=0.5)

# Discriminator Net
discInput = Input(shape=(imageWidth, imageWidth, 1))
discHidden = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(discInput)
discHidden = LeakyReLU(0.2)(discHidden)
# discHidden = Dropout(0.3)(discHidden)
discHidden = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(discHidden)
discHidden = LeakyReLU(0.2)(discHidden)
# discHidden = Dropout(0.3)(discHidden)
discHidden = Flatten()(discHidden)
discOutput = Dense(1, activation='sigmoid')(discHidden)
discriminator = Model(discInput, discOutput)

# Compile discriminator
discriminator.compile(optimizer=myAdam, loss='binary_crossentropy')

# Generator Net
genInput = Input(shape=(latentDim,))
genHidden = Dense(7 * 7 * 128)(genInput)
genHidden = LeakyReLU(0.2)(genHidden)
genHidden = Reshape((7, 7, 128))(genHidden)
genHidden = Conv2DTranspose(
    64, (5, 5), strides=(2, 2), padding='same')(genHidden)
genHidden = LeakyReLU(0.2)(genHidden)
genOutput = Conv2DTranspose(1, (5, 5), strides=(
    2, 2), padding='same', activation='tanh')(genHidden)
generator = Model(genInput, genOutput)

# Compile generator
generator.compile(optimizer=myAdam, loss='binary_crossentropy')

# GAN
ganInput = Input(shape=[latentDim])
ganHidden = generator(ganInput)
discriminator.trainable = False
ganOutput = discriminator(ganHidden)
GAN = Model(ganInput, ganOutput)

# GAN compile
GAN.compile(optimizer=myAdam, loss='binary_crossentropy')

# TO PLOT LOSSES
losses = {"d": [], "g": []}


def plotLosses(losses, showIm=True):
    if showIm:
        plt.figure(figsize=(10, 8))
        plt.plot(losses["d"], label='discriminitive loss')
        plt.plot(losses["g"], label='generative loss')
        plt.legend()
        plt.show()

# TO PLOT GENERATED IMAGES


def plotGen(n=100, dim=(10, 10), figsize=(10, 10), showIm=True, saveIm=True, epoch=-1, genLatentVars=None):
    if genLatentVars is None:
        genLatentVars = np.zeros((n, latentDim))
        for i in range(n):
            genLatentVars[i] = np.random.uniform(-1, 1, latentDim)
    generatedImages = generator.predict(genLatentVars)
    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        img = np.reshape(generatedImages[i], (imageWidth, imageWidth))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    if(saveIm):
        plt.savefig('gan_CNN_MNIST_MP_epoch%02d.png' % epoch)
    if(showIm):
        plt.show()

genLatentVars = np.zeros((100, latentDim))
for i in range(100):
    genLatentVars[i] = np.random.uniform(-1, 1, latentDim)

plotGen(showIm=False, saveIm=True, genLatentVars=genLatentVars)

# TRAIN
trainImagesFullIdx = list(range(len(trainImages)))
discTrainMiniIdx = list(range(2 * minibatchSize))
latentVars = np.zeros((minibatchSize, latentDim))
nOfMinibatches = int(len(trainImages) / minibatchSize)


def trainForNEpochs(nEpochs, pltFreq=25, showIm=True, saveIm=True, initEpoch=0):
    # For each epoch
    for e in tqdm(range(initEpoch, nEpochs)):

        # Randomly shuffle the trainIdices
        np.random.shuffle(trainImagesFullIdx)

        # For each minibatch
        for m in tqdm(range(nOfMinibatches)):

            # TRAIN THE DISCRIMINATOR ONCE

            # Select a minibatch of real images
            realImagesMinibatch = trainImages[trainImagesFullIdx[
                m * minibatchSize:(m + 1) * minibatchSize]]

            # Sample a minibatch-size of latent variables from noise prior
            # (Uniform)
            for i in range(minibatchSize):
                latentVars[i] = np.random.uniform(-1, 1, latentDim)

            # Generate images from noise using the generator
            fakeImagesMinibatch = generator.predict(latentVars)

            # Full list of training images input to discriminator
            discTrainInputs = np.concatenate(
                (realImagesMinibatch, fakeImagesMinibatch))

            # List of training image labels
            discTrainLabels = np.zeros((2 * minibatchSize,))

            # Set label of real images as 1
            discTrainLabels[:minibatchSize] = 0.9

            # Shuffle the training images and labels
            np.random.shuffle(discTrainMiniIdx)
            discTrainInputs = discTrainInputs[discTrainMiniIdx]
            discTrainLabels = discTrainLabels[discTrainMiniIdx]

            # Train the Discriminator
            discriminator.trainable = True
            dLoss = discriminator.train_on_batch(
                discTrainInputs, discTrainLabels)
            losses["d"].append(dLoss)

            # TRAIN THE GENERATOR via GAN

            # Sample a minibatch-size of latent variables from noise prior
            # (Uniform)
            for i in range(minibatchSize):
                latentVars[i] = np.random.uniform(-1, 1, latentDim)

            # Fake image labels - to be said to be real by discriminator, so 1
            fakeImageLabels = np.ones((minibatchSize,))

            # Train the Generato via GAN
            discriminator.trainable = False
            gLoss = GAN.train_on_batch(latentVars, fakeImageLabels)
            losses["g"].append(gLoss)

        if e == 0 or e == 1 or e == 2 or e % pltFreq == pltFreq - 1:
            plotLosses(losses, showIm=showIm)
            plotGen(showIm=showIm, saveIm=saveIm, epoch=e, genLatentVars=genLatentVars)

trainForNEpochs(nEpochs=50, pltFreq=5, showIm=False, saveIm=True, initEpoch=0)
