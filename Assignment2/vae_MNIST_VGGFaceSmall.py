# http://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# https://blog.keras.io/building-autoencoders-in-keras.html

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


# CNN - VGG Face SMALL

minibatchSize = 50
totalPixels = 784
imageWidth = 28
hiddenDim = 512
zDims = 2
nEpochs = 10

# Q(z|X) -- encoder
inputImg = Input(shape=(totalPixels,)) #image data format #28x28
x = Reshape((imageWidth, imageWidth, 1))(inputImg)                                 #28x28
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)    #28x28x64
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          #14x14x64
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #14x14x128
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          #7x7x128
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #7x7x256
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          #3x3x256
x = Flatten()(x)                                                                #2304
x = Dense(128, activation='relu')(x)                                            #128

zMean = Dense(zDims, activation='linear')(x)
zLogSigmaSq = Dense(zDims, activation='linear')(x)

# Encoder model, to encode input into latent variable
# We use the mean as the output as it is the center point, the representative of the gaussian
encoder = Model(inputImg, zMean)

# To sample z
def sampleZ(args):
    zMean, zLogSigmaSq = args
    eps = K.random_normal(shape=(minibatchSize, zDims), mean=0., stddev=1.)
    return zMean + K.exp(zLogSigmaSq / 2) * eps

# Sample z ~ Q(z|X)
z = Lambda(sampleZ, output_shape=(zDims,))([zMean, zLogSigmaSq])

# VGG Face SMALL
x = Dense(128, activation='relu')(z)                                            #128
x = Dense(4096, activation='relu')(x)                                           #4096
x = Reshape((4, 4, 256))(x)                                                     #4x4x256
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #4x4x256
x = UpSampling2D(size=(2, 2))(x)                                                #8x8x256
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #8x8x128
x = UpSampling2D(size=(2, 2))(x)                                                #16x16x128
x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(x)   #14x14x64
x = UpSampling2D(size=(2, 2))(x)                                                #28x28x64
x = Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='linear')(x)   #28x28
decodedImg = Flatten()(x)                                                        #784

# Overall VAE model, for reconstruction and training
vae = Model(inputImg, decodedImg)

# Generator model, generate new data given latent variable z
genInput = Input(shape=(zDims,))
x = Dense(128, activation='relu')(genInput)                                     #128
x = Dense(4096, activation='relu')(x)                                           #4096
x = Reshape((4, 4, 256))(x)                                                     #4x4x256
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #4x4x256
x = UpSampling2D(size=(2, 2))(x)                                                #8x8x256
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #8x8x128
x = UpSampling2D(size=(2, 2))(x)                                                #16x16x128
x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(x)   #14x14x64
x = UpSampling2D(size=(2, 2))(x)                                                #28x28x64
x = Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='linear')(x)   #28x28
genOutput = Flatten()(x)                                                        #784

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
history = vae.fit(trainImages, trainImages, batch_size=minibatchSize, epochs=nEpochs)


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
