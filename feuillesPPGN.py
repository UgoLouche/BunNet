import keras
import cv2, gc
import sys, os, h5py
import numpy as np
import keras.backend as K
from keras.datasets import mnist
from matplotlib import pylab as plt

import NoiselessJointPPGN as PPGN
from models import vgg16_model, dcgan_generator, dcgan_discriminator
from training import customGANTrain, deepSimTrain

from sklearn.preprocessing import Normalizer, StandardScaler
import matplotlib.pyplot as plt

#Test on dataset les Feuilles
batch_size = 64
num_classes = 15
n_epochs = 30
# input image dimensions
img_rows, img_cols = 64, 64
data_path = '/home/romain/Projects/cda_bn2018/data/h5py/'
# data_path = '/media/romain/data/BainNumeriques/'
fname = '/rgb_%ix%i.hdf5' %(img_rows, img_cols)
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

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

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

# Classifier
model = vgg16_model(img_rows, img_cols, channel=3, num_classes=num_classes)
# GAN definition
g_gen = dcgan_generator()
g_disc = dcgan_discriminator(channel=3)

#Create ppgn BEFORE assigning loaded weights
ppgn = PPGN.NoiselessJointPPGN(model, 32, 34, 37, verbose=3,
                               #gan_generator='Default', gan_discriminator='Default')
                               gan_generator=g_gen, gan_discriminator=g_disc)

#Load weights and skip fit if possible
skipFitClf=True
skipFitGAN=False
if skipFitClf and 'vgg16_feuilles.h5' in os.listdir('weights/'):
    model.load_weights('weights/vgg16_rgb64_feuilles_20epo.h5')
    skipFitClf=True
    print('Loaded CLF weights from existing file, will skip training')
if skipFitGAN and 'g_gen_feuilles.h5' in os.listdir('weights/') and 'g_disc_feuilles.h5' in os.listdir('weights/'):
    g_gen.load_weights('weights/g_gen_dcgan_rbg64_feuilles_1400.h5')
    g_disc.load_weights('weights/g_disc_dcgan_rbg64_feuilles_1400.h5')
    skipFitGAN=True
    print('Loaded GAN weights from existing file, will skip training')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[10, 2, 1e-1]) #[10, 1e-1, 1])

if not skipFitClf:
    print('Fitting classifier')
    ppgn.fit_classifier(x_train, y_train, validation_data=[x_test, y_test], epochs=n_epochs)
    ppgn.classifier.save_weights('weights/vgg16_rgb64_feuilles_%iepo.h5' %n_epochs)

if not skipFitGAN:
    print('Fitting GAN')
    src, gen = ppgn.fit_gan(x_train, batch_size=64, epochs=2000,
                save_freq=100, report_freq=10, train_procedure=deepSimTrain)
                #save_freq=100, report_freq=10, train_procedure=customGANTrain)

    # Plot some GAN metrics computed during fit
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
        #gen_img = np.concatenate((255 * (gen[i] - gen[i].min()) / (gen[i].max() - gen[i].min())), axis=0)
        gen_img = np.concatenate((gen[i] * img_scale + img_mean), axis=0)
        img = np.concatenate((src_img, gen_img), axis=1)
        #src[i] = np.concatenate((src[i]), axis=0)
        #gen[i] = np.concatenate((gen[i]), axis=0)
        #img = (np.concatenate((src[i], gen[i]), axis=1)+1)*255/2
        img[img < 0  ] = 0
        img[img > 255] = 255
        cv2.imwrite('img/rgb_feuilles64x64_gan{}.bmp'.format(i), np.uint8(img))

h2_base = ppgn.enc2.predict(ppgn.enc1.predict(x_test[0:1]))
# h2_base=None
for i in range(num_classes):
    ppgn.sampler_init(i)
    samples, h2 = ppgn.sample(i, nbSamples=100,
                              h2_start=h2_base,
                              epsilons=(1e2, 1, 1e-15),
                              lr=1e-1, lr_end=1e-1, use_lr=True)
    h2_base = None#h2[-1]
    gen = np.array(samples)
    img = np.concatenate((255 * (gen - gen.min()) / (gen.max() - gen.min())), axis=0)
    #img = np.concatenate((gen[i] * img_scale + img_mean), axis=0)
    #img = (np.concatenate((samples), axis=0)+1)*255/2
    print("min/max generated: {}/{}".format(img.min(), img.max()))
    img[img < 0  ] = 0
    img[img > 255] = 255
    # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
    fname = 'img/feuilles_{}x{}samples{}.bmp'.format(input_shape[0], input_shape[1], i)
    cv2.imwrite(fname, img)

i_class = 0
n_samples = 1000
h2_data = np.zeros([n_samples, 2048])
img_data = np.zeros([n_samples, 64, 64, 3])
for i in range(n_samples):
    samples, h2 = ppgn.sample(i_class, nbSamples=1,
                              h2_start=h2_base,
                              epsilons=(1e2, 1, 1e-15),
                              lr=1e-2, lr_end=1e-2, use_lr=True)
    h2_base = None#h2[-1]
    h2_data[i] = h2[-1]
    img_data[i] = samples[-1]
