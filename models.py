from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Reshape, AveragePooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Convolution2D

from keras.optimizers import Adam
from keras.applications import VGG16

from keras_adversarial.legacy import l1l2
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling
from image_utils import dim_ordering_shape

def dcgan_discriminator(channel=1):
    nch = 256
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)#l1l2(l1=1e-7, l2=1e-7)

    c1 = Convolution2D(int(nch / 4), (h, h), padding='same',
                       kernel_regularizer=reg(), input_shape=(64, 64, channel))
    c2 = Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg())
    c3 = Convolution2D(nch, (h, h), padding='same', kernel_regularizer=reg())
    c4 = Convolution2D(nch, (h, h), padding='same', kernel_regularizer=reg())

    model = Sequential()
    model.add(c1)
    model.add(AveragePooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c2)
    model.add(AveragePooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c3)
    model.add(AveragePooling2D(pool_size=(4, 4), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c4)
    model.add(AveragePooling2D(pool_size=(4, 4), data_format='channels_last'))#, border_mode='valid')
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    #model.add(Dense(1, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def dcgan_generator():
    model = Sequential()
    input_shape = 2048
    nch = 64
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)#l1l2(l1=1e-4, l2=1e-4)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=input_shape, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape((4, 4, nch)))
    model.add(Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 4), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 4), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(3, (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    #model.add(Activation('sigmoid'))
    model.add(Activation('tanh'))
    return model

def dcgan_gray_generator():
    model = Sequential()
    input_shape = 1024
    nch = 32
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)#l1l2(l1=1e-4, l2=1e-4)
    h = 5
    model.add(Dense(nch * 4 * 4, input_dim=input_shape, W_regularizer=reg()))
    model.add(BatchNormalization(mode=0))
    model.add(Reshape((4, 4, nch)))
    model.add(Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 4), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 8), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 16), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(int(nch / 32), (h, h), padding='same', kernel_regularizer=reg()))
    model.add(BatchNormalization(mode=0, axis=1))
    model.add(LeakyReLU(0.2))
    #model.add(Activation('sigmoid'))
    #model.add(Activation('tanh'))
    return model

def dcgan_gray_discriminator():
    nch = 256
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)#l1l2(l1=1e-7, l2=1e-7)

    c1 = Convolution2D(int(nch / 4), (h, h), padding='same',
                       kernel_regularizer=reg(), input_shape=(64, 64, 1))
    c2 = Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg())
    c3 = Convolution2D(nch, (h, h), padding='same', kernel_regularizer=reg())
    c4 = Convolution2D(1, (h, h), padding='same', kernel_regularizer=reg())

    model = Sequential()
    model.add(c1)
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c2)
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c3)
    model.add(MaxPooling2D(pool_size=(4, 4), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c4)
    model.add(MaxPooling2D(pool_size=(4, 4), data_format='channels_last'))#, border_mode='valid')
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    #model.add(Dense(1, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def dcgan_discriminator_max_pool(channel=1):
    nch = 256
    h = 5
    reg = lambda: l1l2(l1=1e-7, l2=1e-7)#l1l2(l1=1e-7, l2=1e-7)

    c1 = Convolution2D(int(nch / 4), (h, h), padding='same',
                       kernel_regularizer=reg(), input_shape=(64, 64, channel))
    c2 = Convolution2D(int(nch / 2), (h, h), padding='same', kernel_regularizer=reg())
    c3 = Convolution2D(nch, (h, h), padding='same', kernel_regularizer=reg())
    c4 = Convolution2D(nch, (h, h), padding='same', kernel_regularizer=reg())

    model = Sequential()
    model.add(c1)
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c2)
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c3)
    model.add(MaxPooling2D(pool_size=(4, 4), data_format='channels_last'))
    model.add(LeakyReLU(0.2))
    model.add(c4)
    model.add(MaxPooling2D(pool_size=(4, 4), data_format='channels_last'))#, border_mode='valid')
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    #model.add(Dense(1, activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    return model

def vgg16_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Model for Keras
    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))
    return model

def customCNN(img_rows, img_cols, channel=1, num_classes=None, depth=4):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (5, 5), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (5, 5), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (5, 5), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model
