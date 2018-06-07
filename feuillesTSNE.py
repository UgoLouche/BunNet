import keras
import random, cv2, gc
import sys, os, h5py
import numpy as np
import keras.backend as K
from keras.models import Model
from matplotlib import pylab as plt
from PIL import Image

from models import vgg16_model, customCNN, customCNN_ultralight

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
plt.ion()

#Test on dataset les Feuilles
batch_size = 64
num_classes = 16
n_epochs = 20
# input image dimensions
img_rows, img_cols = 256, 256
data_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Database/'
# data_path = 'C:\\Users\\Romain\\Projects\\cda_bn2018\\data\\BASE_IMAGE\\Database\\'
dir_names = os.listdir(data_path)
dir_names.sort()
# Classifier
# model = vgg16_model(img_rows, img_cols, channel=3, num_classes=num_classes)
# model = customCNN(img_rows, img_cols, channel=3, num_classes=num_classes)
model = customCNN_ultralight(img_rows, img_cols, channel=3, num_classes=num_classes)
#Load weights
model.load_weights('weights/cnn7_ultralight_rgb256_aug_rot0_zoom20_10epo.h5')
# with a Sequential model
layer_name = model.get_layer(index=19).name
h2_layer_model = Model(inputs=model.input,
                       outputs=model.get_layer(layer_name).output)

sc = StandardScaler()
tsne = TSNE(n_components=2, random_state=42)
mds = MDS(n_components=2, n_jobs=4)
with open('class_names.txt') as f:
    class_names = f.readlines()

# go back to 64x64 for vizualisation
n_img = 7227
img_rows, img_cols = 256, 256#
samples = np.zeros([n_img, img_rows, img_cols, 3])
h2_values = np.zeros([n_img, 2048])
counter = 0
for class_dir in dir_names:
    print('Processing ' + class_dir)
    fnames = os.listdir(data_path + class_dir)
    for fname in fnames:
        im = Image.open(data_path + class_dir + '/' + fname)
        img = np.array(im.resize((img_rows, img_cols)))/255.
        h2_value = h2_layer_model.predict(np.array([img]))[0]
        h2_values[counter] = h2_value
        samples[counter] = img
        counter += 1

# print('fitting a TSNE representation of h2...')
# duplicat_count = 0
# img_rows, img_cols = 64, 64 #256, 256
# try:
#     h2_trans = tsne.fit_transform(h2_values)
#     hmin, hmax = h2_trans.min(), h2_trans.max()
#     width = np.ceil(hmax-hmin).astype(int)
#     wall = np.zeros([img_rows*width, img_cols*width, 3], dtype=np.uint8)
#     for s in range(len(samples)):
#         xy = np.int_(h2_trans[s]-hmin)
#         x, y = xy[0]*img_rows, xy[1]*img_cols
#         if wall[x:x+img_rows, y:y+img_cols].sum() > 0:
#             duplicat_count += 1
#         else:
#             wall[x:x+img_rows, y:y+img_cols] = cv2.resize(samples[s], (img_rows, img_cols)) * 255
#
#     wall[wall < 0  ] = 0
#     wall[wall > 255] = 255
#     print('found %i masked images' %duplicat_count)
#     # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
#     fname = 'img/tsne_wall_orig_feuilles_final.png'
#     im = Image.fromarray(wall)
#     print('saving map in ' + fname)
#     im.save(fname, optimize=True)
# except MemoryError:
#     print('Memory Error, skipping TSNE')
#
# map_count += 1
# gc.collect()
#
# img_rows, img_cols = 256, 256
# h2_trans = tsne.fit_transform(h2_values)

del h2_values, model
gc.collect()

h2_trans = np.load('h2_tsne.npy')
img_rows, img_cols = 256, 256 #64, 64
#sel = (h2_trans[:, 0] < 0) & (h2_trans[:, 1] < 0)
#h2_trans = h2_trans[sel]
#samples = samples[sel]

map_size, im_size = 68, 1
duplicat_count = 0

for map_size in [68]:
    for im_size in [1, 2, 4]:
        h2_norm = (h2_trans - h2_trans.min()) / (h2_trans.max() - h2_trans.min())
        h2_norm = (h2_norm - 0.5) * (map_size / im_size)
        gc.collect()
        duplicat_count = 0
        hmin, hmax = h2_norm.min(), h2_norm.max()
        width = np.ceil(hmax-hmin).astype(int) + 1
        wall = np.zeros([img_rows*width, img_cols*width, 3], dtype=np.uint8)
        for s in range(h2_norm.shape[0]):
            xy = np.int_(h2_norm[s]-hmin)
            x, y = xy[0]*img_rows, xy[1]*img_cols
            if wall[x:x+img_rows, y:y+img_cols].sum() > 0:
                duplicat_count += 1
            else:
                wall[x:x+img_rows, y:y+img_cols] = cv2.resize(samples[s], (img_rows, img_cols)) * 255

        wall[wall < 0  ] = 0
        wall[wall > 255] = 255
        print('found %i masked images' %duplicat_count)
        # img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
        fname = 'prod/tsne_wall_%icm_feuilles_%icm.png' %(map_size, im_size)
        img = Image.fromarray(wall)
        print('saving map in ' + fname)
        img.save(fname, optimize=True)
        # cv2.imwrite(fname, wall)
        del wall

del samples, h2_trans
gc.collect()
# im = Image.fromarray(wall)
print('saving map in ' + fname)
# im.save(fname, optimize=True)
cv2.imwrite(fname, wall)


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
