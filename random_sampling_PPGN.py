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

# go back to 64x64 for vizualisation
epsilons = (0.5, 1, 1e-5)
img_rows, img_cols = 256, 256#64, 64 #
n_img, i_img = 2000, 0
i_class = 0
h2_base = None
map_count = 0

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros([img_rows, img_cols, 3]))

online_sampling(ppgn, im, epsilons=epsilons, lr=1e-2, img_rows=256, img_cols=256)
#while True:
#    samples, h2_values = random_sampling(n_img, ppgn)#, img_rows=256, img_cols=256)
#    samples, h2_values = categorical_sampling(n_img, 10, ppgn, im)
for i_class in range(num_classes):
    samples, h2_values = categorical_sampling(n_img, i_class, ppgn, im,
                            epsilons=epsilons, lr=1e-2, img_rows=256, img_cols=256)
    print('fitting a TSNE representation of h2...')
    duplicat_count = 0
    try:
        h2_trans = tsne.fit_transform(h2_values)
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
        cname = class_names[i_class].replace(' ', '_')[:-1]
        fname = 'img/tsne_wall_%s_eps%i_%ifeuilles.png' %(cname, np.log10(epsilons[-1]), n_img)
        print('saving map in ' + fname)
        # im.save(fname, optimize=True)
        cv2.imwrite(fname, wall)
    except MemoryError:
        print('Memory Error, skipping TSNE')

    map_count += 1
    gc.collect()
