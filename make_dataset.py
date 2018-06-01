import os, sys, h5py
from skimage import io
from skimage.color import rgb2grey
from skimage.transform import resize

out_shape = [64, 64]
test_size = 600

input_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Augmented_Enghien/'
output_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Enghein/'
dir_names = os.listdir(input_path) #['00_Citron/']

# remove fruits (used for testing purposes)
# dir_names.remove('00_Citron')
# dir_names.remove('03_Kiwi')
# dir_names.remove('06_Orange')

n_train_files, n_test_files = 0, 0
for dir_name in os.listdir(input_path):
    fnames = os.listdir(input_path + dir_name)
    print('found %i images in ' %len(fnames) + dir_name)
    n_train_files += len(fnames) - test_size
    n_test_files += test_size

feuilles = h5py.File("feuilles_%ix%i.hdf5" %(out_shape[0], out_shape[1]), "w")
train_shape = (n_train_files, out_shape[0], out_shape[1], 3)
test_shape = (n_test_files, out_shape[0], out_shape[1], 3)
x_train = feuilles.create_dataset("x_train", train_shape, dtype='i8')
y_train = feuilles.create_dataset("y_train", (n_train_files, ), dtype='i8')
x_test = feuilles.create_dataset("x_test", test_shape, dtype='i8')
y_test = feuilles.create_dataset("y_test", (n_test_files, ), dtype='i8')

train_counter, test_counter = 0, 0
for i_class, dir_name in enumerate(dir_names):
    print(dir_name)
    fnames = os.listdir(os.path.join(input_path, dir_name))
    for fname in fnames[:-test_size]:
        im = io.imread(os.path.join(input_path, dir_name, fname))
        # gray = rgb2grey(im) * 255
        res = resize(im, out_shape)
        x_train[train_counter] = res * 255
        y_train[train_counter] = i_class
        train_counter += 1

    for fname in fnames[-test_size:]:
        im = io.imread(os.path.join(input_path, dir_name, fname))
        # gray = rgb2grey(im) * 255
        res = resize(im, out_shape)
        x_test[test_counter] = res * 255
        y_test[test_counter] = i_class
        test_counter += 1

feuilles.close()
