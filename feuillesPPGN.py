import keras
import cv2
import os, h5py
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from matplotlib import pylab as plt

import NoiselessJointPPGN as PPGN

from sklearn.preprocessing import Normalizer, StandardScaler

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, UpSampling2D, Conv2DTranspose,  Reshape

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Convolution2D

from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras_adversarial.legacy import l1l2
from keras_adversarial.legacy import Dense, BatchNormalization, AveragePooling2D
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from image_utils import dim_ordering_shape

import matplotlib.pyplot as plt

import sys

import numpy as np

from keras.applications import VGG16

## Definbe custom GAN training procedure based on http://www.nada.kth.se/~ann/exjobb/hesam_pakdaman.pdf
#Do 5 disc iterations for one gan iteration. Except for the 500 first epoch and every 500 subsequent epochs
#where disc is trained 100 times
#Based on implementation found in https://github.com/hesampakdaman/ppgn-disc/blob/master/src/vanilla.py
def customGANTrain(x_train, h1_train, batch_size, disc_model, gan_model, epochID):
    disc_train = 100 if (epochID < 25 or epochID % 500) else 5
    disc_loss, disc_pred = [], []
    #train disc
    for i in range(disc_train):
        idX = np.random.randint(0, x_train.shape[0], batch_size)

        valid = x_train[idX]
        fake  = gan_model.predict(x_train[idX])[0]
        x_disc = np.concatenate((valid, fake), axis=0)
        y_disc = np.concatenate((np.ones((batch_size)), np.zeros((batch_size))))

        disc_loss.append(disc_model.train_on_batch(x_disc, y_disc))
        #disc_pred.append(100 * np.mean(disc_model.predict(x_disc) == y_disc))
        disc_pred.append(disc_model.predict(x_disc))

    print('GAN/Disc {:.2f} +/- {:.2f}'.format(np.mean(disc_pred), np.std(disc_pred)))
    #train gen
    x_gan = x_train[idX][-1:]
    y_gan = np.ones((1))
    h1_gan = h1_train[idX][-1:]

    gan_loss = gan_model.train_on_batch(x_gan, [x_gan, y_gan, h1_gan])

    return (np.mean(disc_loss), gan_loss)

def simpleGANTrain(x_train, h1_train, batch_size, disc_model, gan_model, epochID):
    #train disc
    idX = np.random.randint(0, x_train.shape[0], batch_size)

    valid = x_train[idX]
    fake  = gan_model.predict(x_train[idX])[0]
    x_disc = np.concatenate((valid, fake), axis=0)
    y_disc = np.concatenate((np.zeros((batch_size)), np.ones((batch_size))))

    disc_loss = disc_model.train_on_batch(x_disc, y_disc)
    #disc_pred.append(100 * np.mean(disc_model.predict(x_disc) == y_disc))
    disc_pred = disc_model.predict(x_disc).T[0]
    print('GAN/Disc {:.2f}'.format(100 * np.mean(np.round(disc_pred) == y_disc)))
    #train gen
    x_gan = x_train[idX]
    y_gan = np.ones((len(idX)))
    h1_gan = h1_train[idX]

    gan_loss = gan_model.train_on_batch(x_gan, [x_gan, y_gan, h1_gan])

    return (disc_loss, gan_loss)


#Test on dataset les Feuilles
batch_size = 64
num_classes = 14
epochs = 15
# input image dimensions
img_rows, img_cols = 64, 64
data_path = '/home/romain/Projects/cda_bn2018/data/h5py/'
fname = '/gray_%ix%i.hdf5' %(img_rows, img_cols)
#Get the data and reshape/convert/normalize
data = h5py.File(data_path + fname, "r")
n_train = len(data['x_train'])
idx = np.arange(n_train)
np.random.shuffle(idx)
x_train = np.array(data['x_train'])[idx]
y_train = np.array(data['y_train'])[idx]
n_test = len(data['x_test'])
idx = np.arange(n_test)
np.random.shuffle(idx)
x_test = np.array(data['x_test'])[idx]
y_test = np.array(data['y_test'])[idx]
n_train, n_test = y_train.shape[0], y_test.shape[0]
print('train images: %i' %n_train)
print('test images: %i' %n_test)

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

img_mean = x_train.mean()
img_scale = x_train.std()

# sc = StandardScaler()
# sc.fit(x_train.reshape(x_train.shape[0], -1))
# img_mean  = sc.mean_.reshape(img_rows, img_cols, 1)
# img_scale = sc.scale_.reshape(img_rows, img_cols, 1)

x_train = (x_train - img_mean) / img_scale
x_test = (x_test - img_mean) / img_scale

# or take the min/max over each channel and normalize
# Here it just means we divide by 256
#x_train = ((x_train/255) - 0.5)*2
#x_test =  ((x_test/255) - 0.5)*2
#x_train = ((x_train/255) - 0.5)*2
#x_test =  ((x_test/255) - 0.5)*2

#categorical y
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Classifier
# model = Sequential()
# model.add(Conv2D(64, (7,7), activation='relu', input_shape=input_shape, padding='valid'))
# model.add(Conv2D(128, (7,7), activation='relu', padding='valid'))
# model.add(MaxPooling2D((2,2)))
# model.add(Conv2D(256, (7,7), activation='relu', padding='valid'))
# model.add(MaxPooling2D((2,2)))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
# model.trainable=True
def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras
    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = vgg16_model(img_rows, img_cols, channel=1, num_classes=num_classes)

def model_generator():
    model = Sequential()
    input_shape = 1024
    nch = 32
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)#l1l2(l1=1e-4, l2=1e-4)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=input_shape, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape((4, 4, nch)))
    model.add(Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 4), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 8), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 16), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 32), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    #model.add(Activation('sigmoid'))
    #model.add(Activation('tanh'))
    return model

g_gen = model_generator()

def model_discriminator():
    nch = 256
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)#l1l2(l1=1e-7, l2=1e-7)

    c1 = Convolution2D(int(nch / 4), (h, h), padding='same',
                       kernel_regularizer=reg(), input_shape=(64, 64, 1))
    c2 = Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg())
    c3 = Convolution2D(nch, (h, h), padding='same', kernel_regularizer=reg())
    c4 = Convolution2D(1, (h, h), padding='same', kernel_regularizer=reg())

    model = Sequential()
    model.add(c1)
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c2)
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c3)
    model.add(MaxPooling2D(pool_size=(4, 4), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c4)
    model.add(MaxPooling2D(pool_size=(4, 4), data_format='channels_last'))#, border_mode='valid')
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    #model.add(Dense(1, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    return model

g_disc = model_discriminator()

#Create ppgn BEFORE assigning loaded weights
ppgn = PPGN.NoiselessJointPPGN(model, 22, 34, 37, verbose=3,
                               #gan_generator='Default', gan_discriminator='Default')
                               gan_generator=g_gen, gan_discriminator=g_disc)

#Load weights and skip fit if possible
skipFitClf=False
skipFitGAN=False
if skipFitClf and 'clf_feuilles.h5' in os.listdir('weights/'):
    model.load_weights('weights/vgg16_feuilles.h5')
    skipFitClf=True
    print('Loaded CLF weights from existing file, will skip training')
if skipFitGAN and 'g_gen_feuilles.h5' in os.listdir('weights/') and 'g_disc_feuilles.h5' in os.listdir('weights/'):
    g_gen.load_weights('weights/g_gen_dcgan_feuilles.h5')
    g_disc.load_weights('weights/g_disc_dcgan_feuilles.h5')
    skipFitGAN=True
    print('Loaded GAN weights from existing file, will skip training')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[1, 2, 1e-1])

if not skipFitClf:
    print('Fitting classifier')
    ppgn.fit_classifier(x_train, y_train, validation_data=[x_test, y_test], epochs=20)
    ppgn.classifier.save_weights('weights/vgg16_feuilles.h5')

if not skipFitGAN:
    print('Fitting GAN')
    src, gen = ppgn.fit_gan(x_train, batch_size=32, epochs=5000,
                report_freq=10, train_procedure=simpleGANTrain)
    ppgn.g_gen.save_weights('weights/g_gen_dcgan_feuilles.h5')
    ppgn.g_disc.save_weights('weights/g_disc_dcgan_feuilles.h5')

    #Plot some GAN metrics computed during fit
    plt.ion()
    plt.figure()
    plt.plot(np.array(ppgn.g_disc_loss))
    plt.plot(np.array(ppgn.gan_loss)[:, 2])
    plt.legend(['disc_loss', 'gen_loss'])
    plt.figure()
    plt.plot(np.array(ppgn.gan_loss))
    plt.legend(['total loss', 'img loss', 'gan loss', 'h loss'])

    for i in range(len(src)):
        src_img = np.concatenate((src[i] * img_scale + img_mean), axis=0)
        gen_img = np.concatenate((255 * (gen[i] - gen[i].min()) / (gen[i].max() - gen[i].min())), axis=0)
        img = np.concatenate((src_img, gen_img), axis=1)
        #src[i] = np.concatenate((src[i]), axis=0)
        #gen[i] = np.concatenate((gen[i]), axis=0)
        #img = (np.concatenate((src[i], gen[i]), axis=1)+1)*255/2
        img[img < 0  ] = 0
        img[img > 255] = 255
        cv2.imwrite('img/feuilles64x64_gan{}.bmp'.format(i), np.uint8(img))

h2_base = ppgn.enc2.predict(ppgn.enc1.predict(x_test[0:1]))
# h2_base=None
for i in range(num_classes):
    samples, h2 = ppgn.sample(i, nbSamples=100,
                              h2_start=h2_base,
                              epsilons=(1e-6, 1, 1e-15),
                              lr=1e-2, lr_end=1e-2, use_lr=True)
    h2_base = None#h2[-1]
    gen = np.array(samples)
    img = np.concatenate((255 * (gen - gen.min()) / (gen.max() - gen.min())), axis=0)
    #img = (np.concatenate((samples), axis=0)+1)*255/2
    print("min/max generated: {}/{}".format(img.min(), img.max()))
    img[img < 0  ] = 0
    img[img > 255] = 255
    # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
    fname = 'img/feuilles_{}x{}samples{}.bmp'.format(input_shape[0], input_shape[1], i)
    cv2.imwrite(fname, img)
