import mido
import keras
import os, sys, time
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from matplotlib import pylab as plt
plt.ion()

import NoiselessJointPPGN as PPGN

from sklearn.preprocessing import Normalizer

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, UpSampling2D, Conv2DTranspose,  Reshape

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
num_classes = 10

midin = mido.open_input('Midi Fighter Twister')

#Classifier
model = Sequential()
model.add(Conv2D(64, (7,7), activation='relu', input_shape=input_shape, padding='valid'))
model.add(Conv2D(128, (7,7), activation='relu', padding='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(256, (7,7), activation='relu', padding='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.trainable=True

#Create ppgn BEFORE assigning loaded weights
ppgn = PPGN.NoiselessJointPPGN(model, 6, 7, 8, verbose=2,
                               gan_generator='Default', gan_discriminator='Default')

ppgn.classifier.load_weights('weights/clf_mnist.h5')
ppgn.g_gen.load_weights('weights/g_gen_mnist.h5')
ppgn.g_disc.load_weights('weights/g_disc_mnist.h5')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[1, 2, 1e-1])

h2_base=None
class_range = np.linspace(0, num_classes, 128)
eps_range = np.logspace(-15, 1, 128)
epsilons = [1e-2, 1, 1e-15]
neuronId = 0
if len(sys.argv) > 1:
    portname = sys.argv[1]
else:
    portname = 'Midi Fighter Twister'

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
                    h2_base=None

            samples, h2 = ppgn.sample(neuronId, nbSamples=1,
                                      h2_start=h2_base,
                                      epsilons=epsilons,
                                      lr=2, lr_end=2, use_lr=True)
            h2_base = h2[-1]#None#

            plt.imshow(samples[-1][:, :, 0])
            plt.title('class id=%i' %neuronId)
            plt.pause(0.5)
except KeyboardInterrupt:
    pass
