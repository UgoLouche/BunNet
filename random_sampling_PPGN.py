import keras
import random, cv2, gc, time
import sys, os, h5py
import numpy as np
import keras.backend as K
from PIL import Image

import NoiselessJointPPGN as PPGN
from models import vgg16_model, customCNN, customCNN_ultralight
from models import dcgan_generator, dcgan_discriminator, dcgan_256generator
from sampling import online_sampling

import matplotlib.pyplot as plt
plt.ion()

#Test on dataset les Feuilles
batch_size = 64
num_classes = 16
n_epochs = 10
# input image dimensions
img_rows, img_cols = 256, 256

# Classifier
# model = vgg16_model(img_rows, img_cols, channel=3, num_classes=num_classes)
# model = customCNN(img_rows, img_cols, channel=3, num_classes=num_classes)
model = customCNN_ultralight(img_rows, img_cols, channel=3, num_classes=num_classes)
# GAN definition
g_gen = dcgan_256generator()
# model = customCNN_ultralight(img_rows, img_cols, channel=3, num_classes=num_classes)
# # GAN definition
# g_gen = dcgan_256generator()
g_disc = dcgan_discriminator(channel=3, input_shape=(img_rows, img_cols, 3))

# Create ppgn BEFORE assigning loaded weights
# VGG16 indexes = 32, 34, 37
ppgn = PPGN.NoiselessJointPPGN(model, 19, 20, 23, verbose=0,
            gan_generator=g_gen, gan_discriminator=g_disc)

#Load weights
#model.load_weights('weights/cnn7_ultralight_rgb256_aug_10epo.h5')
#g_gen.load_weights('weights/g_gen_dcgan_rbg256_deepsim_noisy_068000.h5')
#g_disc.load_weights('weights/g_disc_dcgan_rbg256_deepsim_noisy_068000.h5')
# model.load_weights('weights/cnn7_ultralight_rgb256_aug_rot30_zoom20_10epo.h5')
# g_gen.load_weights('weights/g_gen_rgb256_aug_rot30_zoom20_029000.h5')
# g_disc.load_weights('weights/g_disc_rgb256_aug_rot30_zoom20_029000.h5')

# ppgn.classifier.load_weights('weights/cnn7_ultralight_rgb256_aug_10epo.h5')
# ppgn.g_gen.load_weights('weights/g_gen_dcgan_rbg256_deepsim_noisy_068000.h5')
# ppgn.g_disc.load_weights('weights/g_disc_dcgan_rbg256_deepsim_noisy_068000.h5')
model.load_weights('weights/cnn7_ultralight_rgb256_aug_rot0_zoom20_10epo.h5')
g_gen.load_weights('weights/g_gen_rgb256_aug_rot0_zoom20_068000.h5')
g_disc.load_weights('weights/g_disc_rgb256_aug_rot0_zoom20_068000.h5')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[10, 2, 1e-1]) #[10, 1e-1, 1])

with open('class_names.txt') as f:
    class_names = f.readlines()

# epsilons = (0.5, 1, 1e-5)
# epsilons = (0.5, 1, 5e-4)
epsilons = (0.5, 2, 1e-2)
lr = 1e-2
img_rows, img_cols = 256, 256#64, 64 #
h2_base = None
map_count = 0

s = 'lr%.2e_' %lr + 'eps_%.2e_%.2e_%.2e_' %epsilons + time.ctime()
save_dir = 'prod/' + s.replace(' ','_').replace(':','_')
os.mkdir(save_dir)
os.mkdir(save_dir + '/img')
os.mkdir(save_dir + '/code')

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros([img_rows, img_cols, 3]))

online_sampling(ppgn, im=im, epsilons=epsilons, lr=lr,
            img_rows=256, img_cols=256, save_dir=save_dir, class_names=class_names)
