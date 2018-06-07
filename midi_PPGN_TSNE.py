import keras
import cv2, gc
import sys, os, h5py
import numpy as np
import keras.backend as K
from keras.utils.io_utils import HDF5Matrix

from matplotlib import pylab as plt

import NoiselessJointPPGN as PPGN
from models import vgg16_model, customCNN, customCNN_ultralight
from models import dcgan_generator, dcgan_discriminator, dcgan_256generator
from training import customGANTrain, deepSimTrain

from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import mido
plt.ion()

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
model.load_weights('weights/cnn7_ultralight_rgb256_aug_10epo.h5')
g_gen.load_weights('weights/g_gen_dcgan_rbg256_deepsim_noisy_068000.h5')
g_disc.load_weights('weights/g_disc_dcgan_rbg256_deepsim_noisy_068000.h5')
# model.load_weights('weights/cnn7_ultralight_rgb256_augmented_20epo.h5')
# g_gen.load_weights('weights/g_gen_dcgan_rbg256_deepsim_noisy_048000.h5')
# g_disc.load_weights('weights/g_disc_dcgan_rbg256_deepsim_noisy_048000.h5')

ppgn.compile(clf_metrics=['accuracy'],
             gan_loss_weight=[10, 2, 1e-1]) #[10, 1e-1, 1])

sc = StandardScaler()
tsne = TSNE(n_components=2)

h2_base = None
class_range = np.linspace(0, num_classes-1, 128)
eps_range = np.logspace(-15, 15, 128)
lr_range = np.logspace(-5, 5, 128)
epsilons = [1e2, 1, 1e-15]
neuronId = 0
lr_value = 1e2
h_diff = 0
n_img = 1000
if len(sys.argv) > 1:
    portname = sys.argv[1]
else:
    portname = 'Midi Fighter Twister:Midi Fighter Twister MIDI 1 24:0'

with open('class_names.txt') as f:
    class_names = f.readlines()

samples = np.zeros([n_img, img_rows, img_cols, 3])
h2_values = np.zeros([n_img, 2048])
ppgn.sampler_init(neuronId)
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(samples[-1])
try:
    with mido.open_input(portname) as port:
        print('Using {}'.format(port))
        for i in range(n_img):
            for message in port.iter_pending():
                if message.control < 3:
                    eps_val = eps_range[message.value]
                    epsilons[message.control] = eps_val
                    print('setting ', neuronId, epsilons)
                elif message.control == 3:
                    if neuronId != int(class_range[message.value]):
                        neuronId = int(class_range[message.value])
                        ppgn.sampler_init(neuronId)
                        print('setting ', neuronId, epsilons)
                elif message.control == 4:
                    print(h2_base)
                    h2_base=None
                elif message.control == 5:
                    lr_value = lr_range[message.value]
                    print('setting ', neuronId, lr_value)

            spl, h2 = ppgn.sample(neuronId, nbSamples=1,
                                      h2_start=h2_base,
                                      epsilons=epsilons,
                                      lr=lr_value, lr_end=lr_value, use_lr=True)
            samples[i] = spl[0]
            h2_values[i] = h2
            if h2_base is not None:
                h_diff = np.linalg.norm(h2_base - h2[-1])

            if np.isnan(h2[-1]).sum() == 0:
                h2_base = h2[-1]
                # sample = (samples[i] - samples[i].min()) / (samples[i].max() - samples[i].min())
                sample = np.uint8(samples[i]*255)
                im.set_data(sample)#[:, :, 0])#, vmin=0, vmax=16)
                plt.title(class_names[neuronId].strip())
                plt.pause(0.01)
            else:
                h2_base = None

except KeyboardInterrupt:
    pass

img_rows, img_cols = 64, 64
print('fitting a TSNE representation of h2...')
h2_trans = tsne.fit_transform(h2_values)
hmin, hmax = h2_trans.min(), h2_trans.max()
width = np.ceil(hmax-hmin).astype(int)
wall = np.zeros([img_rows*width, img_cols*width, 3])
for s in range(len(samples)):
    xy = np.int_(h2_trans[s]-hmin)
    x, y = xy[0]*img_rows, xy[1]*img_cols
    # wall[x:x+img_rows, y:y+img_cols] = (samples[s]+1)*255/2
    wall[x:x+img_rows, y:y+img_cols] = cv2.resize(samples[s]*255, (img_rows, img_cols))

wall[wall < 0  ] = 0
wall[wall > 255] = 255
# img_grid = img.reshape(input_shape[0]*10, input_shape[1]*10, 1)
fname = 'img/feuilles_midi'
fname += '_tsne_cmap_{}x{}wall.jpg'.format(img_rows, img_cols)
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
