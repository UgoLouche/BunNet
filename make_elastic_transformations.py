import os, cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.


    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]),
                          np.arange(shape[1]),
                          np.arange(shape[2]), indexing='ij')

    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)


#base_dir = '/Users/trachel.r/Documents/projects/florian/movimenta/BASE_IMAGE/FEUILLES'
base_path = '/media/romain/windows/Users/Romain/Projects/cda_bn2018/BASE_IMAGE/'
base_dir = base_path + 'Database/'
res_dir = base_path + 'Augmented_Enghien/'
image_dir = os.listdir(base_dir)

w = 256
amin, amax = 1000, 1400
alphas = np.arange(amin, amax, 50)
sigma = 8
alpha = 1500
state = np.random.RandomState(42)

# log parameters in the image
# text font of alpha and sigma
font = cv2.FONT_HERSHEY_PLAIN
pos = 3 * w / 4

for curdir in image_dir:
    # load an image
    img_path = base_dir + curdir
    img_list = os.listdir(img_path)
    if not os.path.exists(res_dir + curdir):
        os.mkdir(res_dir + curdir)
    print 'generating ' + curdir
    for img_name in img_list:
        image = cv2.imread(img_path + '/' + img_name)
        # resize the image to the network input
        image = cv2.resize(image, (w, w))
        for j in range(10):
            res = elastic_transform(image, alpha, sigma, state)
            fname = '/%03d_' %j + img_name
            cv2.imwrite(res_dir + curdir + fname, res.astype(np.uint8))
