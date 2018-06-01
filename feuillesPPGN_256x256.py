import keras
import cv2, gc
import sys, os, h5py
import numpy as np
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix
from keras.preprocessing.image import ImageDataGenerator

from matplotlib import pylab as plt

import NoiselessJointPPGN as PPGN
from models import vgg16_model, customCNN, customCNN_ultralight
from models import dcgan_128generator, dcgan_256generator
from models import dcgan_disc_light, dcgan_discriminator
from training import customGANTrain, deepSimTrain

from sklearn.preprocessing import Normalizer, StandardScaler
import matplotlib.pyplot as plt

# Data augmentation parameters
zoom_range = 0.2
rotation_range = 30
#Test on dataset les Feuilles
batch_size = 64
num_classes = 15
n_epochs = 10
# input image dimensions
img_rows, img_cols = 256, 256
#data_path = '/home/romain/Projects/cda_bn2018/data/h5py/'
data_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Augmented_Enghien/'
gan_gen = ImageDataGenerator(rescale=1./255, channel_shift_range=0.0, rotation_range=rotation_range,
                             width_shift_range=0, height_shift_range=0,
                             horizontal_flip=True, vertical_flip=False,
                             shear_range=0.1, zoom_range=zoom_range)

train_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Enghein/train/'
train_gen = ImageDataGenerator(rescale=1./255, channel_shift_range=0.0, rotation_range=rotation_range,
                              width_shift_range=0, height_shift_range=0,
                              horizontal_flip=True, vertical_flip=False,
                              shear_range=0.1, zoom_range=zoom_range)

test_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Enghein/validation/'
test_gen = ImageDataGenerator(rescale=1./255, channel_shift_range=0.0, rotation_range=rotation_range,
                              width_shift_range=0, height_shift_range=0,
                              horizontal_flip=True, vertical_flip=False,
                              shear_range=0.1, zoom_range=zoom_range)

output_name = 'rgb256_aug_rot%i_zoom%i' %(rotation_range, int(100*zoom_range))

# Classifier
# model = vgg16_model(img_rows, img_cols, channel=3, num_classes=num_classes)
model = customCNN_ultralight(img_rows, img_cols, channel=3, num_classes=num_classes)
# GAN definition
g_gen = dcgan_256generator()
g_disc = dcgan_discriminator(channel=3, input_shape=(img_rows, img_cols, 3))

# Create ppgn BEFORE assigning loaded weights
# VGG16 indexes = 32, 34, 37
# customCNN indexes = 18, 20, 23
ppgn = PPGN.NoiselessJointPPGN(model, 19, 20, 23, verbose=3,
            gan_generator=g_gen, gan_discriminator=g_disc)

#Load weights and skip fit if possible
skipFitClf=False
skipFitGAN=False
if skipFitClf and 'vgg16_feuilles.h5' in os.listdir('weights/'):
    model.load_weights('weights/cnn7_ultralight_' + output_name + '_%iepo.h5' %(n_epochs))
    #rgb256_aug_%iepo.h5' %n_epochs)
    skipFitClf=True
    print('Loaded CLF weights from existing file, will skip training')
if skipFitGAN and 'g_gen_feuilles.h5' in os.listdir('weights/') and 'g_disc_feuilles.h5' in os.listdir('weights/'):
    g_gen.load_weights('weights/g_gen_dcgan_rbg256_deepsim_noisy_045000.h5')
    g_disc.load_weights('weights/g_disc_dcgan_rbg256_deepsim_noisy_045000.h5')
    skipFitGAN=True
    print('Loaded GAN weights from existing file, will skip training')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[10, 1e-1, 1]) #[10, 2, 1e-1]) #[10, 1e-1, 1])

if not skipFitClf:
    print('Fitting classifier')
    # ppgn.fit_classifier(x_train, y_train, validation_data=[x_test, y_test], epochs=n_epochs)
    ppgn.fit_classifier_from_directory(train_gen, train_path,
                        test_gen, test_path, epochs=n_epochs, batch_size=64)
    ppgn.classifier.save_weights('weights/cnn7_ultralight_' + output_name + '_%iepo.h5' %(n_epochs))

if not skipFitGAN:
    print('Fitting GAN')
    #src, gen = ppgn.fit_gan(x_train, batch_size=64, epochs=15000, starting_epoch=2000,
    #            save_freq=500, report_freq=10, train_procedure=deepSimTrain)
                #save_freq=100, report_freq=10, train_procedure=customGANTrain)
    src, gen = ppgn.fit_gan_from_directory(gan_gen, data_path, fname=output_name,
                target_size=(img_rows, img_cols), batch_size=64, epochs=30000,
                save_freq=500, report_freq=100, train_procedure=deepSimTrain)#, starting_epoch=45000)#, save_img_dir='augmented')
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
        src_img = np.concatenate((src[i]*255), axis=0)
        #src_img = np.concatenate(((src[i] +1)*255/2), axis=0)
        #src_img = np.concatenate((src[i] * img_scale + img_mean), axis=0)
        #gen_img = np.concatenate((255 * (gen[i] - gen[i].min()) / (gen[i].max() - gen[i].min())), axis=0)
        #gen_img = np.concatenate((gen[i] * img_scale + img_mean), axis=0)
        #gen_img = np.concatenate(((gen[i] +1)*255/2), axis=0)
        gen_img = np.concatenate((gen[i]*255), axis=0)
        img = np.concatenate((src_img, gen_img), axis=1)
        #src[i] = np.concatenate((src[i]), axis=0)
        #gen[i] = np.concatenate((gen[i]), axis=0)
        #img = (np.concatenate((src[i], gen[i]), axis=1)+1)*255/2
        img[img < 0  ] = 0
        img[img > 255] = 255
        cv2.imwrite('img/rgb_feuilles256x256_gan{}.bmp'.format(i), np.uint8(img))

h2_base = ppgn.enc2.predict(ppgn.enc1.predict(x_test[0:1]))
h2_base=None
for i in range(num_classes):
    ppgn.sampler_init(i)
    samples, h2 = ppgn.sample(i, nbSamples=100,
                              h2_start=h2_base,
                              epsilons=(1e-2, 1, 1e-15),
                              lr=5e-6, lr_end=5e-6, use_lr=True)
    h2_base = None#h2[-1]
    gen = np.array(samples)
    #img = np.concatenate((255 * (gen - gen.min()) / (gen.max() - gen.min())), axis=0)
    img = (np.concatenate((samples), axis=0)+1)*255/2
    #img = np.concatenate((gen[i] * img_scale + img_mean), axis=0)
    print("min/max generated: {}/{}".format(img.min(), img.max()))
    img[img < 0  ] = 0
    img[img > 255] = 255
    # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
    fname = 'img/feuilles_{}x{}samples{}.bmp'.format(input_shape[0], input_shape[1], i)
    cv2.imwrite(fname, img)

# i_class = 0
# n_samples = 1000
# h2_data = np.zeros([n_samples, 2048])
# img_data = np.zeros([n_samples, 64, 64, 3])
# for i in range(n_samples):
#     samples, h2 = ppgn.sample(i_class, nbSamples=1,
#                               h2_start=h2_base,
#                               epsilons=(1e2, 1, 1e-15),
#                               lr=1e-2, lr_end=1e-2, use_lr=True)
#     h2_base = None#h2[-1]
#     h2_data[i] = h2[-1]
#     img_data[i] = samples[-1]
