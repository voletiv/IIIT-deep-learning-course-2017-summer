# http://wiseodd.github.io/techblog/2016/12/10/variational-autoencoder/
# https://blog.keras.io/building-autoencoders-in-keras.html

from keras import backend as K
from keras.layers import Input, Flatten, Dense, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.callbacks import LearningRateScheduler

import os
import numpy as np
import struct
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

# images = []
# folder = 'img_align_celeba'
# totalImages = len(os.listdir(folder))

# print("Reading CelebFaces")
# imagesList = os.listdir(folder)
# img = np.array(Image.open(os.path.join(folder, imagesList[0])))
# imRows = img.shape[0]
# imCols = img.shape[1]
# imChs = img.shape[2]
# desiredNumOfImages = 1000
# trainImages = np.zeros((desiredNumOfImages, imRows, imCols, imChs))
# for i, filename in enumerate(os.listdir(folder)):
#         if i>=desiredNumOfImages:
#             break
#         print(float(i)/desiredNumOfImages, end='\r')
#         # img = cv2.imread(os.path.join(folder,filename))/255.
#         img = np.array(Image.open(os.path.join(folder, filename)))/255.
#         if img is not None:
#             trainImages[i] = img

# print("Finished reading CelebFaces")

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

# CNN - VGG Face SMALL

minibatchSize = 10
imageDim = 784
hiddenDim = 512
zDims = 2
nEpochs = 10

# Q(z|X) -- encoder
inputImg = Input(shape=(imRows, imCols, imChs,)) #image data format #218x178x3
x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputImg)     #218x178x64
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          #109x89x64
x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #109x89x128
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          #54x44x128
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #54x44x256
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          #27x22x256
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #27x22x512
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)          #13x11x512
x = Flatten()(x)                                                                #73216
x = Dense(512, activation='relu')(x)                                            #512
# z = Dense(2*zDims, activation='linear')(x)

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



x = Dense(512, activation='relu')(z)                                            #512
x = Dense(86016, activation='relu')(x)                                          #86016
x = Reshape((14, 12, 512))(x)                                                   #14x12x512
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #14x12x512
x = UpSampling2D(size=(2, 2))(x)                                                #28x24x512
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #28x24x256
x = UpSampling2D(size=(2, 2))(x)                                                #56x48x256
x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu')(x)  #54x46x128
x = ZeroPadding2D(padding=(1, 0))(x)                                            #56x46x256
x = UpSampling2D(size=(2, 2))(x)                                                #112x92x128
x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(x)   #110x90x64
x = UpSampling2D(size=(2, 2))(x)                                                #220x180x64
decodedImg = Conv2D(3, (3, 3), strides=(1, 1), padding='valid', activation='linear')(x)      #218x178x3

# Overall VAE model, for reconstruction and training
vae = Model(inputImg, decodedImg)

# Generator model, generate new data given latent variable z
genInput = Input(shape=(zDims,))
x = Dense(512, activation='relu')(genInput)                                     #512
x = Dense(86016, activation='relu')(x)                                          #86016
x = Reshape((14, 12, 512))(x)                                                   #14x12x512
x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #14x12x512
x = UpSampling2D(size=(2, 2))(x)                                                #28x24x512
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)   #28x24x256
x = UpSampling2D(size=(2, 2))(x)                                                #56x48x256
x = Conv2D(128, (3, 3), strides=(1, 1), padding='valid', activation='relu')(x)  #54x46x128
x = ZeroPadding2D(padding=(1, 0))(x)                                            #56x46x256
x = UpSampling2D(size=(2, 2))(x)                                                #112x92x128
x = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu')(x)   #110x90x64
x = UpSampling2D(size=(2, 2))(x)                                                #220x180x64
genOutput = Conv2D(3, (3, 3), strides=(1, 1), padding='valid', activation='linear')(x)      #218x178x3

generator = Model(genInput, genOutput)


# Plot gen
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
        plt.savefig('vae_MNIST_epoch%02d.png' % epoch)
    if(showIm):
        plt.show()

genLatentVars = np.zeros((100, latentDim))
for i in range(100):
    genLatentVars[i] = np.random.uniform(-1, 1, latentDim)

plotGen(showIm=False, saveIm=True, genLatentVars=genLatentVars)


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



