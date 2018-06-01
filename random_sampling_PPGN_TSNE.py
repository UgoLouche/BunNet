import keras
import random, cv2, gc
import sys, os, h5py
import numpy as np
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix

from matplotlib import pylab as plt
from PIL import Image

import NoiselessJointPPGN as PPGN
from models import vgg16_model, customCNN, customCNN_ultralight
from models import dcgan_generator, dcgan_discriminator, dcgan_256generator
from training import customGANTrain, deepSimTrain

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
plt.ion()

def random_sampling(n_img, ppgn, num_classes=15, img_rows=64, img_cols=64, h_len=2048):
    samples = np.zeros([n_img, img_rows, img_cols, 3], dtype=np.uint8)
    h2_values = np.zeros([n_img, h_len])
    i_img, i_class = 0, 0
    h2_base = None
    while i_img < n_img:
        if random.randint(0, 1000) > 990:
            i_class = random.randint(0, num_classes-1)
            ppgn.sampler_init(i_class)

        sample, h2 = ppgn.sample(i_class, nbSamples=1,
                                 h2_start=h2_base,
                                 epsilons=(1, 1, 1e-15),
                                 lr=1e-1, lr_end=1e-1, use_lr=True)

        if np.isnan(h2[-1][0]):
            h2_base = None
        else:
            img = (sample[-1] * 255).astype(np.uint8)
            samples[i_img] = cv2.resize(img, (img_rows, img_cols))
            h2_values[i_img] = h2[-1]
            h2_base = h2[-1]
            i_img += 1

    return samples, h2_values

#Test on dataset les Feuilles
batch_size = 64
num_classes = 15
n_epochs = 20
# input image dimensions
img_rows, img_cols = 256, 256
data_path = '/home/romain/Projects/cda_bn2018/data/h5py/'

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
ppgn = PPGN.NoiselessJointPPGN(model, 19, 20, 23, verbose=3,
            gan_generator=g_gen, gan_discriminator=g_disc)

#Load weights
#model.load_weights('weights/cnn7_ultralight_rgb256_aug_10epo.h5')
#g_gen.load_weights('weights/g_gen_dcgan_rbg256_deepsim_noisy_068000.h5')
#g_disc.load_weights('weights/g_disc_dcgan_rbg256_deepsim_noisy_068000.h5')
model.load_weights('weights/cnn7_ultralight_rgb256_aug_rot30_zoom20_10epo.h5')
g_gen.load_weights('weights/g_gen_rgb256_aug_rot30_zoom20_029000.h5')
g_disc.load_weights('weights/g_disc_rgb256_aug_rot30_zoom20_029000.h5')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[10, 2, 1e-1]) #[10, 1e-1, 1])

sc = StandardScaler()
tsne = TSNE(n_components=2, random_state=42)
mds = MDS(n_components=2, n_jobs=4)
with open('class_names.txt') as f:
    class_names = f.readlines()

# go back to 64x64 for vizualisation
img_rows, img_cols = 64, 64 #256, 256#
n_img, i_img = 5000, 0
i_class = 0
h2_base = None
map_count = 0

while True:
    samples, h2_values = random_sampling(n_img, ppgn)#, img_rows=256, img_cols=256)
    print('fitting a TSNE representation of h2...')
    duplicat_count = 0
    try:
        h2_trans = tsne.fit_transform(h2_values) * 2
        hmin, hmax = h2_trans.min(), h2_trans.max()
        width = np.ceil(hmax-hmin).astype(int)
        wall = np.zeros([img_rows*width, img_cols*width, 3], dtype=np.uint8)
        for s in range(len(samples)):
            xy = np.int_(h2_trans[s]-hmin)
            x, y = xy[0]*img_rows, xy[1]*img_cols
            if wall[x:x+img_rows, y:y+img_cols].sum() > 0:
                duplicat_count += 1
            else:
                wall[x:x+img_rows, y:y+img_cols] = samples[s]

        wall[wall < 0  ] = 0
        wall[wall > 255] = 255
        print('found %i masked images' %duplicat_count)
        # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
        fname = 'img/tsne_wall%i_%ifeuilles.png' %(map_count, n_img)
        im = Image.fromarray(wall)
        print('saving map in ' + fname)
        im.save(fname, optimize=True)
    except MemoryError:
        print('Memory Error, skipping TSNE')

    map_count += 1
    gc.collect()

stop


print('fitting a MDS representation of h2...')
try:
    h2_trans = mds.fit_transform(h2_values)
    hmin, hmax = h2_trans.min(), h2_trans.max()
    width = np.ceil(hmax-hmin).astype(int)
    wall = np.zeros([img_rows*width, img_cols*width, 3], dtype=np.uint8)
    for s in range(len(samples)):
        xy = np.int_(h2_trans[s]-hmin)
        x, y = xy[0]*img_rows, xy[1]*img_cols
        wall[x:x+img_rows, y:y+img_cols] = samples[s]

    wall[wall < 0  ] = 0
    wall[wall > 255] = 255
    # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
    fname = 'img/mds_wall%i_feuilles%i.png' %(n_img, map_count)
    im = Image.fromarray(wall)
    print('saving map in ' + fname)
    im.save(fname, optimize=True)
except MemoryError:
    print('Memory Error, skipping MDS')

img_all = np.concatenate(spl_list)
h2_all = tsne.fit_transform(np.concatenate(h2_list, axis=0))
hmin, hmax = h2_all.min(), h2_all.max()
width = np.ceil(hmax-hmin).astype(int)
wall = np.zeros([64*width, 64*width, 3])
for s in range(len(samples)):
    x, y = np.int_(h2_all[s]-hmin)*64
    wall[x:x+64, y:y+64] = (samples[s]+1)*255/2

wall[wall < 0  ] = 0
wall[wall > 255] = 255

from matplotlib import cm

cstart = cm.Greens_r(np.linspace(0, 1, 255))
cstop  = cm.Reds_r(np.linspace(0, 1, 255))

n_iter = 5000
cmaps = np.zeros([int(n_iter/10), 255, 3])
for i in range(255):
    for j in range(3):
        cmaps[:, i, j] = np.linspace(cstart[i, j], cstop[i, j], int(n_iter/10))


h2_list, spl_list = [], []
for i_class in range(num_classes):
    ppgn.sampler_init(i_class)
    samples, h2 = ppgn.sample(i_class, nbSamples=5000,
                              h2_start=None,
                              epsilons=(5e-2, 1, 1e-15),
                              lr=1e-2, lr_end=1e-2, use_lr=True)
    print('fitting a TSNE representation of h2...')
    h2_trans = tsne.fit_transform(h2)
    hmin, hmax = h2_trans.min(), h2_trans.max()
    width = np.ceil(hmax-hmin).astype(int)
    wall = np.zeros([64*width, 64*width, 3])
    for s in range(len(samples)):
        x, y = np.int_(h2_trans[s]-hmin)*64
        img = ((samples[s]+1)*255/2).mean(-1).astype(int)
        cimg = np.zeros([64, 64, 3])
        for i in range(64):
            for j in range(64):
                cimg[i, j] = cmaps[int(s/10)][img[i, j]]
        wall[x:x+64, y:y+64] = (cimg * 255).astype(int)

    wall[wall < 0  ] = 0
    wall[wall > 255] = 255
    # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
    fname = 'img/feuilles_' + class_names[i_class].strip().replace(' ', '_')
    fname += '_tsne_cmap_{}x{}wall.jpg'.format(input_shape[0], input_shape[1])
    cv2.imwrite(fname, wall)

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
