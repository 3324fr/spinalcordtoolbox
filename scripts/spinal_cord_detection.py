#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
#
# License: see the LICENSE.TXT
# ==========================================================================================



# import keras necessary classes
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import os
from msct_image import Image
import numpy as np
import sct_utils as sct

IMAGE_SIZE = 128
TRAINING_SOURCE_DATA = '/Users/cavan/data/machine_learning/train/in/'
TEST_SOURCE_DATA = '/Users/cavan/data/machine_learning/test/label_1/'


def extract_data(path_data, verbose=1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    """

    if verbose == 1:
        sct.printv('Extracting '+ path_data)

    list_data = []
    data = []
    for root, dirs, files in os.walk(path_data):
        for file in files:
            if file.endswith('.nii.gz') and file.find('_seg') == -1 and file.find('_gmseg') == -1 and file.find(
                    '.DS_Store') == -1:
                list_data.append(root + '/' + file)

    for i, fname in enumerate(list_data):
        im_data = Image(fname).data
        data.append(im_data)

    return data


def extract_label(path_data, segmentation_image_size=0, verbose=1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    """
    from sys import stdout
    cr = '\r'
    number_pixel = segmentation_image_size * segmentation_image_size
    ignore_list = ['.DS_Store']
    if verbose == 1:
        sct.printv('Extracting' + path_data)

    data, weights = [], []
    list_data = []
    for root, dirs, files in os.walk(path_data):
        for file in files:
            if file.endswith('.nii.gz') and file.find('_seg') != -1 :
                list_data.append(root + '/' + file)

    if verbose == 1:
        stdout.write(cr)
        sct.printv('Done.        ')

    for i, fname in enumerate(list_data):
        data_im = Image(fname).data
        data_im[data_im > 0] = 1
        im = Image(data_im)
        im.setFileName(fname)
        im.save()
        number_of_segpixels = np.count_nonzero(data_im)
        weights_image = data_im * number_of_segpixels / number_pixel + (1 - data_im) * (number_pixel - number_of_segpixels) / number_pixel
        data.append(np.expand_dims(data_im, axis=0))
        weights.append(np.expand_dims(weights_image, axis=0))

    data = np.concatenate(data, axis=0)
    weights_stack = np.concatenate(weights, axis=0)
    #data = np.expand_dims(data_stack, axis=3)
    #data = np.concatenate((1-data,data), axis=3)

    return data, weights_stack

img_rows = 128
img_cols = 128
nb_classes = 2


def load_data(path_data, verbose = 1):

    if verbose == 1:
        sct.printv('Extracting ' + path_data)

    list_data = []
    list_labels = []

    for root, dirs, files in os.walk(path_data):
        for file in files:
            if file.endswith('.nii.gz') and file.find('_seg') == -1 and file.find('_gmseg') == -1 and file.find('.DS_Store') == -1:
                list_data.append(root + '/' + file)
                path, name, ext = sct.extract_fname(root + '/' + file)
                file_seg = name + '_seg' + ext
                list_data.append(file)
                list_labels.append(file_seg)


    size_list = len(list_data)
    list_train = (list_data[:int(0.6*size_list)],list_labels[:int(0.6 * size_list)])
    list_validation = (list_data[int(0.6*size_list)+1:int(0.8*size_list)], list_labels[int(0.6 * size_list) + 1:int(0.8 * size_list)])
    list_test = (list_data[int(0.8*size_list):], list_labels[int(0.8 * size_list):])

    return list_train, list_validation, list_test

(X_train, y_train), (X_validation, y_validation), (X_test, y_test) = load_data(TRAINING_SOURCE_DATA)
X_train = np.asarray(X_train).reshape(np.asarray(X_train).shape[0], 1, img_rows, img_cols)
X_test = np.asarray(X_test).reshape(np.asarray(X_test).shape[0], 1, img_rows, img_cols)
X_train = np.asarray(X_train).astype('float32')
X_test = np.asarray(X_test).astype('float32')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

batch_size = 128
nb_epoch = 1

print X_train

# Creating the model which consists of 3 convolution layers followed by 2 fully connected layers
print('creating the model')

# Sequential wrapper model
model = Sequential()

# first convolutional layer
model.add(Convolution2D(16,3,3, input_shape = (128,128,1), dim_ordering = 'tf'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

# second convolutional layer
model.add(Convolution2D(32,3,3,border_mode='same', dim_ordering = 'tf'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#model.add(Convolution2D(8,3,3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(UpSampling2D((2,2)))
#model.add(Dropout(0.5))

#model.add(Convolution2D(8,3,3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(UpSampling2D((2,2)))
#model.add(Dropout(0.5))

#model.add(Convolution2D(16,3,3, border_mode='same'))
#model.add(Activation('relu'))
#model.add(UpSampling2D((2,2)))
#model.add(Dropout(0.5))

#model.add(Convolution2D(1,3,3, activation='sigmoid', border_mode='same'))

# setting sgd optimizer parameters
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

# save the model
# save as JSON
json_string = model.to_json()
open('my_model_architecture.json', 'w').write(json_string)
model.save_weights('my_model_weights.h5')
