#!/usr/bin/env python
# ==========================================================================================
#
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
#
# License: see the LICENSE.TXT
# ==========================================================================================


#from __future__ import absolute_import
#from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
import math

import os
import numpy as np

import sct_utils as sct
from msct_image import Image

THEANO_FLAGS='optimizer_excluding=conv_dnn, optimizer_excluding=conv_gemm'


def load_data(path_data, label_value, verbose=1):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
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
    label = np.zeros(num_data, dtype="uint8")

    if list_data is not None:
        for i, fname_im in enumerate(list_data):
            if verbose == 1:
                stdout.write(cr)
                stdout.write(str(i) + '/' + str(num_data))
            im_data = Image(fname_im)

            data_image = np.asarray(im_data.data, dtype="float32")
            data_image = (data_image - np.min(data_image)) / (np.max(data_image) - np.min(data_image))  # output between 0 and 1
            data[i, :, :] = data_image[:, :]
            label[i] = label_value

    return data, label, size


def proba_seg (path_data, patch_size, step, model):
    list_data = []
    for file in os.listdir(path_data):
        if file.find('.DS_Store') == -1:
            list_data.append(path_data + file)

    for iter in range(0, 1):
        im_file = Image(list_data[iter])
        path, file, ext = sct.extract_fname(list_data[iter])
        nx, ny, nz, nt, px, py, pz, pt = im_file.dim
        prob = np.zeros((nx, ny, 2))
        array_x = np.zeros((patch_size/2 , ny))
        array_y = np.zeros((nx + patch_size, patch_size/2))
        data_array = np.squeeze(np.asarray(im_file.data))
        data_array = np.concatenate((array_x, data_array, array_x), axis=0)
        data_array = np.concatenate((array_y, data_array, array_y), axis=1)
        im_file.setFileName(path +'temp_file/'+ file +'_rs'  + ext)
        im_file.data = data_array
        im_file.save()
        list_data[iter] = path + file +'_rs'  + ext

        data_prob, label_prob, size_prob = load_data(path_data='/Users/cavan/data/machine_learning/train/test_patch/temp_file', label_value=1)
        for i in range(0, nx/step):
            for j in range(0, ny/step):
                prob[step*i:step*(i+1),step*j:step*(j+1),:] = model.predict_proba(data_prob[:,:,step*i:step*i + patch_size, step*j:step*j + patch_size], batch_size = patch_size, verbose=1)
        print prob

        im_prob = Image(prob[:,:,1])
        im_prob.setFileName(path + file + '_prob' + ext)
        im_prob.save()

train_model = True
if train_model:
    # Load data
    data_in, label_in, size_in = load_data(path_data='/Users/cavan/data/machine_learning/train/in/', label_value=1)
    print 'Number of samples IN =', data_in.shape[0]
    data_out, label_out, size_out = load_data(path_data='/Users/cavan/data/machine_learning/train/out/', label_value=0)
    print 'Number of samples IN =', data_out.shape[0]
    data = np.concatenate((data_in, data_out), axis=0)
    label = np.concatenate((label_in, label_out), axis=0)
    print 'Total number of samples =', data.shape[0]

    # label 0 to 9 of 10 categories, keras requested format is binary class matrices, transforming it, directly call this function keras provided
    label = np_utils.to_categorical(label, 2)

    batch_size = 128
    nb_epoch = 1

    model = Sequential()

    #contraction
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, 128, 128)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, input_dim=128))
    model.add(Activation('softmax'))

    # let's train the model using SGD + momentum (how original).
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(data, label, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

    # save the model
    # save as JSON
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')

evaluate_model = False
load_model = True

if load_model:
    from keras.models import model_from_json
    model = model_from_json(open('my_model_architecture.json').read())
    model.load_weights('my_model_weights.h5')
    model.compile(optimizer='adagrad', loss='mse')

if evaluate_model:
    data_test_in, label_test_in, size_test_in = load_data(path_data='/Users/cavan/data/machine_learning/test/label_1/', label_value=1)
    data_test_out, label_test_out, size_test_out = load_data(path_data='/Users/cavan/data/machine_learning/test/label_0/', label_value=0)
    X_test = np.concatenate((data_test_in, data_test_out), axis = 0)
    y_test = np.concatenate((label_test_in, label_test_out), axis = 0)
    print 'Total number of test sample = ', X_test.shape[0]

    y_test = np_utils.to_categorical(y_test, 2)

    score = model.evaluate(X_test, y_test, batch_size=16)

    print('Test score:', score)

#Test
path_data = '/Users/cavan/data/machine_learning/train/test_patch/'
patch_size = 128
proba_seg(path_data, patch_size, 5, model)







