import keras
import random, cv2, gc, time
import sys, os, h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def online_sampling(ppgn, im=None, save_dir=None, class_names=None,
        epsilons=(0.5, 1, 1e-15), lr=1e-1, reset_category=False,
        num_classes=16, img_rows=64, img_cols=64, h_len=2048):

    i_img, neuronId, counter = 0, 0, 0
    h2_base = None
    ppgn.sampler_init(num_classes)
    while True:
        # h2_base = None
        neuronId = random.randint(0, num_classes-1)
        iter_count = 0
        tstart = time.time()
        while counter < 10:
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
                    if class_names is not None:
                        cname = class_names[neuronId].replace(' ', '_')[:-1]
                    else:
                        cname = 'class=%i'
                    plt.title(cname + ' prob=%.3f' %(prob[neuronId]))
                    plt.pause(0.01)

        if save_dir is not None:
            if class_names is not None:
                class_name = class_names[neuronId].replace(' ', '_')[:-1]
                fname = save_dir + '/img/feuille_%s_eps%i_%04d.png' %(class_name, np.log10(epsilons[-1]), i_img)
                cname = save_dir + '/code/feuille_%s_eps%i_%04d.txt' %(class_name, np.log10(epsilons[-1]), i_img)
                print('sampling leave from class %s in %.3fs (%i iterations) ' %(class_name, (time.time()-tstart), iter_count))
            else:
                fname = save_dir + '/img/feuille%i_eps%i_%04d.png' %(neuronId, np.log10(epsilons[-1]), i_img)
                cname = save_dir + '/code/feuille%i_eps%i_%04d.txt' %(neuronId, np.log10(epsilons[-1]), i_img)
                print('sampling leave from class %i in %.3fs (%i iterations) ' %(neuronId, (time.time()-tstart), iter_count))
            # im.save(fname, optimize=True)
            # cv2.imwrite(fname, samples[i_img])
            Image.fromarray(img).save(fname, optimize=True)
            np.savetxt(cname, h2[-1], delimiter=';')

        gc.collect()
        i_img += 1
        counter = 0

    return False

def random_sampling(n_img, ppgn, im=None, save_dir=None, class_names=None,
        epsilons=(0.5, 1, 1e-15), lr=1e-1, reset_category=False,
        num_classes=16, img_rows=64, img_cols=64, h_len=2048):
    samples = np.zeros([n_img, img_rows, img_cols, 3], dtype=np.uint8)
    h2_values = np.zeros([n_img, h_len])
    i_img, neuronId, counter = 0, 0, 0
    h2_base = None
    ppgn.sampler_init(num_classes)
    while i_img < n_img:
        h2_base = None
        if neuronId != int(np.floor(i_img/(n_img/num_classes))):
            neuronId = int(np.floor(i_img/(n_img/num_classes)))
            # ppgn.sampler_init(neuronId)
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
                    if class_names is not None:
                        cname = class_names[neuronId].replace(' ', '_')[:-1]
                    else:
                        cname = 'class=%i'
                    plt.title(cname + ' prob=%.3f' %(prob[neuronId]))
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
            print('saving map in ' + fname)
            Image.fromarray(samples[i_img]).save(fname, optimize=True)
            np.savetxt(cname, h2_values[i_img], delimiter=';')

        gc.collect()
        i_img += 1
        counter = 0

    return samples, h2_values

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
    ppgn.sampler_init(num_classes)
    i_img, i_class, counter = 0, 0, 1000
    h2_base = None
    change_class = False
    while i_img < n_img:
        if counter > 100:
            h2_base = None
            if reset_category:
                neuronId = random.randint(0, num_classes-1)
                print('setting class %i' %neuronId)
                #ppgn.sampler_init(neuronId)
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
