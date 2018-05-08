import keras
import cv2
import sys, os, h5py
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
import mido

from keras.applications import VGG16

#Test on dataset les Feuilles
batch_size = 64
num_classes = 14
epochs = 15
# input image dimensions
img_rows, img_cols = 64, 64

#Classifier
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

ppgn.classifier.load_weights('weights/vgg16_feuilles.h5')
ppgn.g_gen.load_weights('weights/g_gen_dcgan_feuilles_365.h5')
ppgn.g_disc.load_weights('weights/g_disc_dcgan_feuilles_365.h5')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[1, 2, 1e-1])

h2_base = None
class_range = np.linspace(0, num_classes, 128)
eps_range = np.logspace(-15, 1, 128)
lr_range = np.logspace(-5, 5, 128)
epsilons = [1e2, 1, 1e-15]
neuronId = 0
lr_value = 1e2
h_diff = 0

if len(sys.argv) > 1:
    portname = sys.argv[1]
else:
    portname = 'Midi Fighter Twister:Midi Fighter Twister MIDI 1 24:0'

samples = [np.zeros([img_rows, img_cols])]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(samples[-1])
try:
    with mido.open_input(portname) as port:
        print('Using {}'.format(port))
        while True:
            for message in port.iter_pending():
                if message.control < 3:
                    eps_val = eps_range[message.value]
                    epsilons[message.control] = eps_val
                    print('setting ', neuronId, epsilons)
                elif message.control == 3:
                    neuronId = int(class_range[message.value])
                    print('setting ', neuronId, epsilons)
                elif message.control == 4:
                    print(h2_base)
                    h2_base=None
                elif message.control == 5:
                    lr_value = lr_range[message.value]
                    print('setting ', neuronId, lr_value)

            old_samples = samples
            samples, h2 = ppgn.sample(neuronId, nbSamples=2,
                                      h2_start=h2_base,
                                      epsilons=epsilons,
                                      lr=lr_value, lr_end=lr_value, use_lr=True)

            if h2_base is not None:
                h_diff = np.linalg.norm(h2_base - h2[-1])
                s_diff = np.abs(old_samples[-1]-samples[-1]).sum()
                print(h_diff, s_diff)
            h2_base = h2[-1]
            if np.isnan(samples[-1]).sum() == 0:
                plt.imshow(samples[-1][:, :, 0])#, vmin=0, vmax=16)
                plt.title('class id=%i' %neuronId + ' diff norm=%.3e' %h_diff)
                plt.pause(0.5)
except KeyboardInterrupt:
    pass
