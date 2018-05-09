import keras
import cv2, gc, time
import sys, os, h5py
import numpy as np
import keras.backend as K
import mido
from matplotlib import pylab as plt

import NoiselessJointPPGN as PPGN
from models import vgg16_model, dcgan_generator, dcgan_discriminator

#Test on dataset les Feuilles
batch_size = 64
num_classes = 15
epochs = 15
# input image dimensions
img_rows, img_cols = 64, 64
# Classifier
model = vgg16_model(img_rows, img_cols, channel=3, num_classes=num_classes)
# GAN definition
g_gen = dcgan_generator()
g_disc = dcgan_discriminator(channel=3)

#Create ppgn BEFORE assigning loaded weights
ppgn = PPGN.NoiselessJointPPGN(model, 25, 34, 37, verbose=3,
                               #gan_generator='Default', gan_discriminator='Default')
                               gan_generator=g_gen, gan_discriminator=g_disc)

ppgn.classifier.load_weights('weights/vgg16_rgb64_feuilles_10epo.h5')
ppgn.g_gen.load_weights('weights/g_gen_dcgan_rbg64_feuilles_4900.h5')
ppgn.g_disc.load_weights('weights/g_disc_dcgan_rbg64_feuilles_4900.h5')

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

samples = [np.zeros([img_rows, img_cols, 3])]

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
            tstart = time.time()
            samples, h2 = ppgn.sample(neuronId, nbSamples=1,
                                      h2_start=h2_base,
                                      epsilons=epsilons,
                                      lr=lr_value, lr_end=lr_value, use_lr=True)

            print('time: %f' %(time.time()-tstart))
            # if h2_base is not None:
            #     h_diff = np.linalg.norm(h2_base - h2[-1])
            #     s_diff = np.abs(old_samples[-1]-samples[-1]).sum()
            #     print(h_diff, s_diff)
            # h2_base = h2[-1]
            # if np.isnan(samples[-1]).sum() == 0:
            #     sample = (samples[-1] - samples[-1].min()) / (samples[-1].max() - samples[-1].min())
            #     plt.imshow(sample)#[:, :, 0])#, vmin=0, vmax=16)
            #     plt.title('class id=%i' %neuronId + ' diff norm=%.3e' %h_diff)
            #     plt.pause(0.5)
except KeyboardInterrupt:
    pass
