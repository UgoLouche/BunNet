import os, sys, shutil

out_shape = [64, 64]
test_size = 600

input_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Augmented_Enghien/'
output_path = '/home/romain/Projects/cda_bn2018/data/BASE_IMAGE/Enghein/'
dir_names = os.listdir(input_path) #['00_Citron/']

# remove fruits (used for testing purposes)
# dir_names.remove('00_Citron')
# dir_names.remove('03_Kiwi')
# dir_names.remove('06_Orange')

for dir_name in os.listdir(input_path):
    fnames = os.listdir(input_path + dir_name)
    print('copy %i images from ' %len(fnames) + dir_name)
    os.mkdir(output_path + 'train/' + dir_name)
    os.mkdir(output_path + 'validation/' + dir_name)
    for train_fname in fnames[:-test_size]:
        shutil.copy(input_path + dir_name + '/' + train_fname,
                    output_path + 'train/' + dir_name + '/' + train_fname)

    for test_fname in fnames[-test_size:]:
        shutil.copy(input_path + dir_name + '/' + test_fname,
                    output_path + 'validation/' + dir_name + '/' + test_fname)
