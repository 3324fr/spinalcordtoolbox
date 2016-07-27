#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Camille Van Assel
#
# License: see the LICENSE.TXT
# ==========================================================================================

import sys
import os
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils
from msct_image import Image
import numpy as np
import sct_utils as sct
from keras import backend as K
from keras.optimizers import Adam

from matplotlib import pyplot as plt


IMAGE_SIZE = 128
TRAINING_SOURCE_DATA = '/Users/cavan/data/machine_learning/train/in/'
TEST_SOURCE_DATA = '/Users/cavan/data/machine_learning/test/label_1/'
learning_rate = 0.001

def load_data(path_data, verbose=1):
    """
    Extract the images into a 4D tensor [image index, channels , y, x].
    """
    from sys import stdout
    if verbose == 1:
        sct.printv('Extracting ' + path_data)
    cr = '\r'

    list_data = []
    for root, dirs, files in os.walk(path_data):
        for file in files:
            if file.endswith('.nii.gz') and file.find('_seg') == -1 and file.find('_gmseg') == -1 and file.find('.DS_Store') == -1:
                list_data.append(root + '/' + file)

    nx, ny, nz, nt, px, py, pz, pt = Image(list_data[0]).dim
    size = (nx, ny)

    num_data = len(list_data)
    data = np.empty((num_data, 1, nx, ny), dtype="float32")

    if list_data is not None:
        for i, fname_im in enumerate(list_data):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(num_data))
            im_data = Image(fname_im)

            data_image = np.asarray(im_data.data, dtype="float32")
            data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image))  # output between 0 and 1
            data[i, :, :] = data_image[:, :]

    return data, size, list_data


def load_labels(path_data, verbose = 1):
    """
        Extract the labels into a 4D tensor [image index, channels, y, x].
        """
    from sys import stdout
    cr = '\r'
    if verbose == 1:
        sct.printv('Extracting ' + path_data)

    list_labels = []
    for root, dirs, files in os.walk(path_data):
        for file in files:
            if file.endswith('.nii.gz') and file.find('_seg') != -1 and file.find('_pred') == -1:
                list_labels.append(root + '/' + file)
    nx, ny, nz, nt, px, py, pz, pt = Image(list_labels[0]).dim
    size = (nx, ny)

    num_data = len(list_labels)
    data = np.empty((num_data, 1, nx, ny), dtype="float32")

    if list_labels is not None:
        for i, fname_im in enumerate(list_labels):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(num_data))
            im_data = Image(fname_im)
            data_image = np.asarray(im_data.data, dtype="float32")
            data[i, :, :] = data_image[:, :]

    return data, size


def cost(y_true, y_pred):
    c = K.categorical_crossentropy(y_true, y_pred)
    return c


def get_model(img_rows, img_cols):

    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='lecun_uniform', dim_ordering = 'th')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', init='lecun_uniform')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid', init='lecun_uniform', dim_ordering='th')(conv9)

    model = Model(input=inputs, output=conv10)

    sgd = SGD(lr=0.01, momentum=0.99, decay=0.5, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def model2(img_rows, img_cols):
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape=(1,img_rows, img_cols)))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128,3,3,activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation= 'relu', input_shape = (256,256)))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation= 'relu', input_shape = (4096,256)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax', input_shape = (4096,256)))

    sgd=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def train_and_predict():
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)

    IMAGE_SIZE = 128
    SOURCE_DATA = '/Users/cavan/data/machine_learning/train/in/'
    nb_classes = 2
    batch_size = 1
    nb_epoch = 1

    data, size, list_data = load_data(SOURCE_DATA)
    labels, size_labels = load_labels(SOURCE_DATA)
    data = data.astype('float32')
    labels = labels.astype('int')

    data_train = data[0:int(0.6 * len(list_data)), :, :, :]
    data_validate = data[int(0.6 * len(list_data)): int(0.8 * len(list_data)), :, :, :]

    labels_train = labels[0:int(0.6 * len(list_data)), :, :, :]
    labels_validate = labels[int(0.6 * len(list_data)): int(0.8 * len(list_data)), :, :, :]

    # convert class vectors to binary class matrices
    # labels_train = np_utils.to_categorical(labels_train, nb_classes)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)
    model = get_model(IMAGE_SIZE, IMAGE_SIZE)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(data_train, labels_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

    print('-' * 30)
    print('Save the model')
    print('-' * 30)
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)

    # convert class vectors to binary class matrices
    # labels_test = np_utils.to_categorical(labels_test, nb_classes)

    score = model.evaluate(data_validate, labels_validate, batch_size=16)
    print "Test score : " + str(score)


def visualisation():
    batch_size = 1
    path_out = '/Users/cavan/data/machine_learning/result/'

    from keras.models import model_from_json
    model = model_from_json(open('my_model_architecture.json').read())
    model.load_weights('my_model_weights.h5')
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    IMAGE_TEST_DATA = '/Users/cavan/data/machine_learning/train/in/'
    data, size, list_data = load_data(IMAGE_TEST_DATA)
    data = data.astype('float32')

    data_test = data[int(0.8 * len(list_data)):, :, :, :]
    list_data = list_data[int(0.8 * len(list_data)):]

    out = model.predict(data_test[0:5, :, :, :], batch_size=batch_size, verbose=1)
    for i in range(0, 5):
        path, file, ext = sct.extract_fname(list_data[i])
        absolute_path = path_out + file + '_seg_pred.nii.gz'
        data_out = np.asarray(out[i,0,:,:])
        im = Image(data_out)
        im.setFileName(absolute_path)
        im.save()

        new_path = path_out + file + ext
        im_o = Image(list_data[i])
        im_c = im_o.copy()
        im_c.setFileName(new_path)
        im_c.save()

if __name__ == '__main__':
    #train_and_predict()
    visualisation()
