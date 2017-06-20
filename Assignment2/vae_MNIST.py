# http://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

minibatchSize = 50
imageDim = 784
hiddenDim = 512
zDims = 2
nEpochs = 10

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
testImages = testImages.astype('float32')/255.0

# # Encode trainY as nx10 np array
# trainY = np.zeros((len(trainLbls), 10))
# for i in range(len(trainLbls)):
#     trainY[i, trainLbls[i]] = 1

# Q(z|X) -- encoder
x = Input(shape=(imageDim,))
h = Dense(hiddenDim, activation='relu')(x)
zMean = Dense(zDims, activation='linear')(h)
zLogSigmaSq = Dense(zDims, activation='linear')(h)

# To sample z
def sampleZ(args):
    zMean, zLogSigmaSq = args
    eps = K.random_normal(shape=(minibatchSize, zDims), mean=0., stddev=1.)
    return zMean + K.exp(zLogSigmaSq / 2) * eps

# Sample z ~ Q(z|X)
z = Lambda(sampleZ, output_shape=(zDims,))([zMean, zLogSigmaSq])

# P(X|z) -- decoder
decoder_hidden = Dense(hiddenDim, activation='relu')
decoder_out = Dense(imageDim, activation='sigmoid')

hP = decoder_hidden(z)
decoded = decoder_out(hP)

# Overall VAE model, for reconstruction and training
vae = Model(x, decoded)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(x, zMean)

# Generator model, generate new data given latent variable z
genInput = Input(shape=(zDims,))
genH = decoder_hidden(genInput)
genOutput = decoder_out(genH)
generator = Model(genInput, genOutput)

def vaeLoss(y_true, y_pred):
    # E[log P(X|z)]
    xent_loss = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl_loss = 0.5 * K.sum(K.exp(zLogSigmaSq) + K.square(zMean) - 1. - zLogSigmaSq, axis=1)
    return xent_loss + kl_loss

# Compile
vae.compile(optimizer='adam', loss=vaeLoss)

# Fit
vae.fit(trainImages, trainImages, batch_size=minibatchSize, epochs=nEpochs)

# Plot trainImages latent dim
trainImagesEncoded = encoder.predict(trainImages, batch_size=minibatchSize)
plt.figure(figsize=(6, 6))
plt.scatter(trainImagesEncoded[:, 0], trainImagesEncoded[:, 1], c=trainLbls)
plt.colorbar()
plt.show()

# Plot testImages latent dim
testImagesEncoded = encoder.predict(testImages, batch_size=minibatchSize)
plt.figure(figsize=(6, 6))
plt.scatter(testImagesEncoded[:, 0], testImagesEncoded[:, 1], c=testLbls)
plt.colorbar()
plt.show()

# Plot generated images
digitSize = 28
n = 15
figure = np.zeros((digitSize * n, digitSize * n))
grid_x = np.linspace(-6, 6, n)
grid_y = np.linspace(-6, 6, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digitSize, digitSize)
        figure[i * digitSize: (i + 1) * digitSize,
               j * digitSize: (j + 1) * digitSize] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='gray')
plt.show()

