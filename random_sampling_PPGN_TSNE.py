import keras
import random, cv2, gc, time
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

def random_sampling_old(n_img, ppgn, num_classes=16, img_rows=64, img_cols=64, h_len=2048):
    samples = np.zeros([n_img, img_rows, img_cols, 3], dtype=np.uint8)
    h2_values = np.zeros([n_img, h_len])
    i_img, i_class, counter = 0, 0, 1000
    h2_base = None
    change_class = False
    while i_img < n_img:
        if counter > 100:
            i_class = random.randint(0, num_classes-1)
            print('producing class %i' %i_class)
            ppgn.sampler_init(i_class)
            counter = 0

        sample, h2 = ppgn.sample(i_class, nbSamples=1,
                                 h2_start=h2_base,
                                 epsilons=(2, 1, 1e-15),
                                 lr=1e-2, lr_end=1e-2, use_lr=True)

        if np.isnan(h2[-1][0]):
            h2_base = None
        else:
            prob = ppgn.classifier.predict(sample)[0]
            print(prob[i_class])
            if prob[i_class] > 0.9999:
                counter += 1
            img = (sample[-1] * 255).astype(np.uint8)
            samples[i_img] = cv2.resize(img, (img_rows, img_cols))
            h2_values[i_img] = h2[-1]
            h2_base = h2[-1]
            i_img += 1

    return samples, h2_values

def categorical_sampling(n_img, neuronId, ppgn, im=None,
        epsilons=(0.5, 1, 1e-15), lr=1e-1, reset_category=False,
        num_classes=16, img_rows=64, img_cols=64, h_len=2048):
    samples = np.zeros([n_img, img_rows, img_cols, 3], dtype=np.uint8)
    h2_values = np.zeros([n_img, h_len])
    ppgn.sampler_init(neuronId)
    i_img, i_class, counter = 0, 0, 1000
    h2_base = None
    change_class = False
    while i_img < n_img:
        if counter > 100:
            h2_base = None
            if reset_category:
                neuronId = random.randint(0, num_classes-1)
                print('setting class %i' %neuronId)
                ppgn.sampler_init(neuronId)
            else:
                print('-----')
            counter = 0

        sample, h2 = ppgn.sample(neuronId, nbSamples=1,
                                 h2_start=h2_base,
                                 epsilons=epsilons,
                                 lr=lr, lr_end=lr, use_lr=True)

        if np.isnan(h2[-1][0]):
            h2_base = None
        else:
            prob = ppgn.classifier.predict(sample)[0]
            print(prob[neuronId])
            if prob[neuronId] > 0.99:
                counter += 1
            img = (sample[-1] * 255).astype(np.uint8)
            samples[i_img] = cv2.resize(img, (img_rows, img_cols))
            h2_values[i_img] = h2[-1]
            h2_base = h2[-1]
            i_img += 1
            if im is not None:
                im.set_data(img)#[:, :, 0])#, vmin=0, vmax=16)
                plt.title('counter=%04d, prob=%.3f' %(i_img, prob[neuronId]))
                plt.pause(0.01)

    return samples, h2_values


def random_sampling(n_img, ppgn, im=None, save_dir=None, class_names=None,
        epsilons=(0.5, 1, 1e-15), lr=1e-1, reset_category=False,
        num_classes=16, img_rows=64, img_cols=64, h_len=2048):
    samples = np.zeros([n_img, img_rows, img_cols, 3], dtype=np.uint8)
    h2_values = np.zeros([n_img, h_len])
    i_img, neuronId, counter = 0, 0, 0
    h2_base = None
    ppgn.sampler_init(neuronId)
    while i_img < n_img:
        h2_base = None
        if neuronId != int(np.floor(i_img/(n_img/num_classes))):
            neuronId = int(np.floor(i_img/(n_img/num_classes)))
            ppgn.sampler_init(neuronId)
        iter_count = 0
        tstart = time.time()
        while counter < 30:
            iter_count += 1
            sample, h2 = ppgn.sample(neuronId, nbSamples=1,
                                 h2_start=h2_base,
                                 epsilons=epsilons,
                                 lr=lr, lr_end=lr, use_lr=True)

            if np.isnan(h2[-1][0]):
                h2_base = None
            else:
                prob = ppgn.classifier.predict(sample)[0]
                img = (sample[-1] * 255).astype(np.uint8)
                h2_base = h2[-1]
                if prob[neuronId] > 0.99:
                    counter += 1
                if im is not None:
                    im.set_data(img)#[:, :, 0])#, vmin=0, vmax=16)
                    plt.title('counter=%04d, prob=%.3f' %(i_img, prob[neuronId]))
                    plt.pause(0.01)

        print('sampling class %i in %.3fs (%i iterations) ' %(neuronId, (time.time()-tstart), iter_count))
        samples[i_img] = cv2.resize(img, (img_rows, img_cols))
        h2_values[i_img] = h2[-1]
        if save_dir is not None:
            if class_names is not None:
                cname = class_names[neuronId].replace(' ', '_')[:-1]
                fname = save_dir + '/img/feuille_%s_eps%i_%04d.png' %(cname, np.log10(epsilons[-1]), i_img)
                cname = save_dir + '/code/feuille_%s_eps%i_%04d.txt' %(cname, np.log10(epsilons[-1]), i_img)
            else:
                fname = save_dir + '/img/feuille%i_eps%i_%04d.png' %(neuronId, np.log10(epsilons[-1]), i_img)
                cname = save_dir + '/code/feuille%i_eps%i_%04d.txt' %(neuronId, np.log10(epsilons[-1]), i_img)
            # im.save(fname, optimize=True)
            # cv2.imwrite(fname, samples[i_img])
            img2 = Image.fromarray(samples[i_img])
            print('saving map in ' + fname)
            img2.save(fname, optimize=True)
            np.savetxt(cname, h2_values[i_img], delimiter=';')

        gc.collect()
        i_img += 1
        counter = 0

    return samples, h2_values

#Test on dataset les Feuilles
batch_size = 64
num_classes = 16
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

sc = StandardScaler()
tsne = TSNE(n_components=2, random_state=42)
mds = MDS(n_components=2, n_jobs=4)
with open('class_names.txt') as f:
    class_names = f.readlines()

# go back to 64x64 for vizualisation
epsilons = (0.5, 1, 1e-5)
img_rows, img_cols = 256, 256#64, 64 #
n_img, i_img = 2000, 0
i_class = 0
h2_base = None
# map_count = 0
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# im = ax.imshow(np.zeros([img_rows, img_cols, 3]))
#
# #while True:
# #    samples, h2_values = random_sampling(n_img, ppgn)#, img_rows=256, img_cols=256)
# #    samples, h2_values = categorical_sampling(n_img, 10, ppgn, im)
# for i_class in range(num_classes):
#     samples, h2_values = categorical_sampling(n_img, i_class, ppgn, im,
#                             epsilons=epsilons, lr=1e-2, img_rows=256, img_cols=256)
#     print('fitting a TSNE representation of h2...')
#     duplicat_count = 0
#     try:
#         h2_trans = tsne.fit_transform(h2_values)
#         hmin, hmax = h2_trans.min(), h2_trans.max()
#         width = np.ceil(hmax-hmin).astype(int)
#         wall = np.zeros([img_rows*width, img_cols*width, 3], dtype=np.uint8)
#         for s in range(len(samples)):
#             xy = np.int_(h2_trans[s]-hmin)
#             x, y = xy[0]*img_rows, xy[1]*img_cols
#             if wall[x:x+img_rows, y:y+img_cols].sum() > 0:
#                 duplicat_count += 1
#             else:
#                 wall[x:x+img_rows, y:y+img_cols] = samples[s]
#
#         wall[wall < 0  ] = 0
#         wall[wall > 255] = 255
#         print('found %i masked images' %duplicat_count)
#         # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
#         cname = class_names[i_class].replace(' ', '_')[:-1]
#         fname = 'img/tsne_wall_%s_eps%i_%ifeuilles.png' %(cname, np.log10(epsilons[-1]), n_img)
#         print('saving map in ' + fname)
#         # im.save(fname, optimize=True)
#         cv2.imwrite(fname, wall)
#     except MemoryError:
#         print('Memory Error, skipping TSNE')
#
#     map_count += 1
#     gc.collect()
#
# stop

save_dir = 'prod/' + time.ctime()
os.mkdir(save_dir)
os.mkdir(save_dir + '/img')
os.mkdir(save_dir + '/code')
n_img = 2000
epsilons = (0.5, 1, 5e-4)

samples, h2_values = random_sampling(n_img, ppgn, epsilons=epsilons,
                            lr=1e-2, img_rows=256, img_cols=256,
                            save_dir=save_dir, class_names=class_names)

np.save(save_dir + '/samples_values.npy', samples)
np.save(save_dir + '/h2_values.npy', h2_values)


save_dir = '/home/romain/Projects/cda_bn2018/PPGN_Keras/prod/'
samples1 = np.load(save_dir + 'Thu Jun  7 02:41:21 2018/samples_values.npy')
h2_values1 = np.load(save_dir + 'Thu Jun  7 02:41:21 2018/h2_values.npy')

save_dir = '/home/romain/Projects/cda_bn2018/PPGN_Keras/prod/'
samples2 = np.load(save_dir + '/Thu Jun  7 09:26:48 2018/samples_values.npy')
h2_values2 = np.load(save_dir + '/Thu Jun  7 09:26:48 2018/h2_values.npy')

samples = np.vstack((samples1, samples2))
h2_values = np.vstack((h2_values1, h2_values2))

del samples1, samples2
del h2_values1, h2_values2
gc.collect()

h2_trans = tsne.fit_transform(h2_values)
# h2_min = np.array([-55, -65])
# h2_max = np.array([68, 65])
# sel = np.sum((h2_trans > h2_min) & (h2_trans < h2_max), axis=1) == 2
# idx = sel.nonzero()[0]

h2_trans = tsne.fit_transform(h2_values)
h2_min = np.array([-45, -56])
h2_max = np.array([60, 66])
sel = np.sum((h2_trans > h2_min) & (h2_trans < h2_max), axis=1) == 2
idx = sel.nonzero()[0]
print('rejecting %.3f' %(100-100*np.mean(sel)))
h2_trans = h2_trans[idx, :]
samples = samples[idx]

img_rows, img_cols = 256, 256 #64, 64 #256, 256
map_size, im_size = 68, 4
duplicat_count = 0

h2_norm = (h2_trans - h2_trans.min()) / (h2_trans.max() - h2_trans.min())
h2_norm = (h2_norm - 0.5) * (map_size / im_size)

hmin, hmax = h2_norm.min(), h2_norm.max()
width = np.ceil(hmax-hmin).astype(int) + 1
wall = np.zeros([img_rows*width, img_cols*width, 3], dtype=np.uint8)
for s in range(len(samples)):
    xy = np.int_(h2_norm[s]-hmin)#[::-1]
    x, y = xy[0]*img_rows, xy[1]*img_cols
    if wall[x:x+img_rows, y:y+img_cols].sum() > 0:
        duplicat_count += 1
    else:
        wall[x:x+img_rows, y:y+img_cols] = cv2.resize(samples[s], (img_rows, img_cols))

wall[wall < 0  ] = 0
wall[wall > 255] = 255
print('found %i masked images' %duplicat_count)
fname = save_dir + '/tsne_wall_%ix%i_%icm_ppgn_%icm.png' %(img_rows, img_cols, map_size, im_size)
cv2.imwrite(fname, wall)

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
