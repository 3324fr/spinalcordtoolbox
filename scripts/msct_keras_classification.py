#!/usr/bin/env python
#########################################################################################
#
# This module contains some functions and algorithm for image classification and segmentation
# using supervised machine learning
# GPU run command:
#    THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python msct_keras_classification.py
#    THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python msct_keras_classification.py
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2016-07-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import os
import sct_utils as sct
from msct_image import Image
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    import cPickle as pickle
except:
    import pickle

# import numpy
import numpy as np

# import keras necessary classes
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta
from keras.utils import np_utils


def extract_slices_from_image(fname_im, fname_seg=None):
    im_data = Image(fname_im)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    if fname_seg:
        im_seg = Image(fname_seg)
        im_seg.data = im_seg.data.astype(int)

    #im_data.data = (im_data.data - np.mean(im_data.data)) / np.abs(np.percentile(im_data.data, 1) - np.percentile(im_data.data, 99))
    im_data.data = 255.0 * im_data.data / np.abs(np.percentile(im_data.data, 1) - np.percentile(im_data.data, 99))

    data_im = []
    data_seg = []
    for k in range(nz):
        data_im.append(im_data.data[:, :, k])
        if fname_seg:
            slice_seg = im_seg.data[:, :, k]
            data_seg.append(slice_seg)

    if fname_seg:
        return data_im, data_seg
    else:
        return data_im

def extract_all_positive_patches_from_slice(slice_im, slice_seg, patch_size=32):
    data_to_patch = np.stack((slice_im, slice_seg), axis=2)
    indices_positive = np.where(slice_seg == 1)

    result = []
    for k in range(len(indices_positive[0])):
        ind = [indices_positive[0][k], indices_positive[1][k]]
        patch = data_to_patch[ind[0] - patch_size / 2:ind[0] + patch_size / 2, ind[1] - patch_size / 2:ind[1] + patch_size / 2, :]
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            result.append(np.expand_dims(patch, axis=0))
    if len(result) != 0:
        return np.concatenate(result, axis=0)
    else:
        return None


def extract_patch_from_slice(slice_im, slice_seg=None, patch_size=32, max_patches_factor=1):
    if slice_seg is not None:
        data_to_patch = np.stack((slice_im, slice_seg), axis=2)
    else:
        data_to_patch = slice_im
    max_patches = int(data_to_patch.shape[0] * data_to_patch.shape[1] / max_patches_factor)
    rng = np.random.RandomState(0)
    return extract_patches_2d(data_to_patch, (patch_size, patch_size), max_patches=max_patches, random_state=rng)


def extract_list_file_from_path(path_data):
    from sys import stdout
    ignore_list = ['.DS_Store']
    sct.printv('Extracting ' + path_data)
    cr = '\r'

    data = []
    list_data = []
    # images_folder = os.listdir(path_data)
    for root, dirs, files in os.walk(path_data):
        for fname_im in files:
            if fname_im in ignore_list:
                continue
            if 'seg' in fname_im or 'gmseg' in fname_im:
                continue
            f_seg = None
            for fname_seg in files:
                if fname_im[:-7] in fname_seg:
                    f_seg = fname_seg
            list_data.append([root + '/' + fname_im, root + '/' + f_seg])

    return list_data


def stream_images(list_data, patch_size, max_patches_factor, nb_epochs):
    for e in range(nb_epochs):
        if nb_epochs != 1:
            print 'Epoch ' + str(e + 1) + '/' + str(nb_epochs)
        for i, fname in enumerate(list_data):
            #print fname
            data_im, data_seg = extract_slices_from_image(list_data[i][0], list_data[i][1])
            number_of_slices = len(data_im)

            arr = range(number_of_slices)
            np.random.shuffle(arr)
            for k in arr:
                # slice-by-slice intensity normalization
                #normalized_slice = (data_im[k] - np.mean(data_im[k])) / np.abs(np.percentile(data_im[k], 1) - np.percentile(data_im[k], 99))
                #print normalized_slice.shape, data_seg[k].shape

                patches_pos = extract_all_positive_patches_from_slice(data_im[k], data_seg[k], patch_size)
                patches = extract_patch_from_slice(data_im[k], data_seg[k], patch_size, max_patches_factor)
                if patches_pos is not None:
                    patches = np.concatenate((patches_pos, patches), axis=0)
                number_of_patches = patches.shape[0]
                # print k, number_of_slices, number_of_patches
                arr_p = range(number_of_patches)
                np.random.shuffle(arr_p)
                for j in arr_p:
                    patch_im = patches[j, :, :, 0]
                    patch_seg = patches[j, :, :, 1]

                    result = {}
                    result['epoch'] = e+1
                    result['patch'] = patch_im
                    """plt.figure()
                    plt.subplot(2, 1, 1)
                    plt.imshow(patch_im)
                    plt.subplot(2, 1, 2)
                    plt.imshow(patch_seg)
                    plt.savefig('foo.png')
                    #plt.show()"""
                    if patch_seg[int(patch_size / 2), int(patch_size / 2)] == 1:
                        result['class'] = 1
                    else:
                        result['class'] = 0
                    yield result


def get_minibatch(patch_iter, size):
    """Extract a minibatch of examples, return a tuple X_text, y.

    Note: size is before excluding invalid docs with no topics assigned.

    """
    data = [(patch['patch'], patch['class']) for patch in itertools.islice(patch_iter, size)]
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)

    X, y = zip(*data)
    X, y = np.asarray(X, dtype=int), np.asarray(y, dtype=int)
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    return X, y


def iter_minibatches(patch_iter, minibatch_size):
    """Generator of minibatches."""
    X, y = get_minibatch(patch_iter, minibatch_size)
    while len(X):
        yield X, y
        X, y = get_minibatch(patch_iter, minibatch_size)


# Creating the model which consists of 3 conv layers followed by
# 2 fully conntected layers
print('creating the model')

patch_size = 32
test_ratio = 0.1
nb_epochs = 500
minibatch_size = 1000
max_patches_factor = 10
evaluation_factor = 5000


def modelA():
    model = Sequential()
    # input: 32x32 images with 1 channels -> (1, 32, 32) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, patch_size, patch_size)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model

def modelB():
    model = Sequential()
    # input: 32x32 images with 1 channels -> (1, 32, 32) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, patch_size, patch_size)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    return model

def modelC():
    model = Sequential()
    # input: 32x32 images with 1 channels -> (1, 32, 32) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, patch_size, patch_size)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    # Note: Keras does automatic shape inference.
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=ada)
    return model

model = modelC()

list_data = extract_list_file_from_path('/home/neuropoly/data/large_nobrain_nopad')
#list_data = extract_list_file_from_path('/Users/benjamindeleener/data/data_augmentation/small_nobrain_nopad')
np.random.shuffle(list_data)
nb_images = len(list_data)
nb_test = int(round(test_ratio * nb_images))
nb_train = nb_images - nb_test
list_test_data = list_data[:nb_test]
list_train_data = list_data[nb_test:]
print 'Number of images', nb_images
print 'Number of test images=', len(list_test_data)
print 'Number of train images=', len(list_train_data)

data_stream_test = stream_images(list_test_data, patch_size, max_patches_factor, 1)
data_stream_train = stream_images(list_train_data, patch_size, max_patches_factor, nb_epochs)

# test data statistics
test_stats = {'n_test': 0, 'n_test_pos': 0}

# First we hold out a number of examples to estimate accuracy
minibatch_iterator_test = iter_minibatches(data_stream_test, minibatch_size)
for i, (X_test, y_test) in enumerate(minibatch_iterator_test):
    test_stats['n_test'] += len(y_test)
    test_stats['n_test_pos'] += sum(y_test)
print("Test set is %d patches (%d positive)" % (test_stats['n_test'], test_stats['n_test_pos']))
weight_class = [test_stats['n_test_pos'] / float(test_stats['n_test']), 1.0]
print 100.0 * weight_class[0], '% positive'

minibatch_iterators = iter_minibatches(data_stream_train, minibatch_size)
total_vect_time = 0.0

stats = {'n_train': 0, 'n_train_pos': 0,
         'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'fscore': 0.0,
         'accuracy_history': [(0, 0)], 'precision_history': [(0, 0)], 'recall_history': [(0, 0)],
         'fscore_history': [(0, 0)],
         't0': time.time(),
         'runtime_history': [(0, 0)], 'total_fit_time': 0.0}
cls_stats = stats

def progress(stats):
    """Report progress information, return a string."""
    duration = time.time() - stats['t0']
    s = str(stats['n_train']) + " train samples (" + str(stats['n_train_pos']) + " positive)\n"
    s += str(test_stats['n_test']) + " test samples (" + str(test_stats['n_test_pos']) + " positive)\n"
    s += "accuracy: " + str(stats['accuracy']) + "\n"
    s += "precision: " + str(stats['precision']) + "\n"
    s += "recall: " + str(stats['recall']) + "\n"
    s += "fscore: " + str(stats['fscore']) + "\n"
    s += "in " + str(duration) + "s (" + str(stats['n_train'] / duration) + " samples/sec)"
    return s

# Main loop : iterate on mini-batchs of examples
print 'start training'
for i, (X_train, y_train) in enumerate(minibatch_iterators):
    number_of_positive = sum(y_train)
    #if number_of_positive == 0:
    #    print i, 'No positive sample...'

    tick = time.time()

    # update estimator with examples in the current mini-batch
    y_train = np_utils.to_categorical(y_train, nb_classes=2)
    model.train_on_batch(X_train, y_train, class_weight=weight_class)
    cls_stats['total_fit_time'] += time.time() - tick
    cls_stats['n_train'] += X_train.shape[0]
    cls_stats['n_train_pos'] += sum(y_train)

    if i % evaluation_factor == 0 and i != 0:
        print 'Iteration', i
        print '~' + str(cls_stats['n_train']*test_ratio/test_stats['n_test']) + ' epoch(s)'
        cls_stats['prediction_time'] = 0
        y_pred_sk, y_test_sk = [], []
        data_stream_test = stream_images(list_test_data, patch_size, max_patches_factor, 1)
        minibatch_iterator_test = iter_minibatches(data_stream_test, minibatch_size)
        for j, (X_test, y_test) in enumerate(minibatch_iterator_test):

            # accumulate test accuracy stats
            tick = time.time()
            y_pred = model.predict(X_test, batch_size=32)
            cls_stats['prediction_time'] += time.time() - tick
            y_pred_sk.extend(np.argmax(y_pred, axis=1).tolist())
            y_test_sk.extend(y_test)
        y_test_sk = np.array(y_test_sk)
        y_pred_sk = np.array(y_pred_sk)
        cls_stats['accuracy'] = accuracy_score(y_test_sk, y_pred_sk)
        cls_stats['precision'] = precision_score(y_test_sk, y_pred_sk)
        cls_stats['recall'] = recall_score(y_test_sk, y_pred_sk)
        cls_stats['fscore'] = f1_score(y_test_sk, y_pred_sk)

        acc_history = (cls_stats['accuracy'],
                       cls_stats['n_train'])
        cls_stats['accuracy_history'].append(acc_history)
        precision_history = (cls_stats['precision'],
                             cls_stats['n_train'])
        cls_stats['precision_history'].append(precision_history)
        recall_history = (cls_stats['recall'],
                          cls_stats['n_train'])
        cls_stats['recall_history'].append(recall_history)
        fscore_history = (cls_stats['fscore'],
                          cls_stats['n_train'])
        cls_stats['fscore_history'].append(fscore_history)
        run_history = (cls_stats['accuracy'],
                       total_vect_time + cls_stats['total_fit_time'])
        cls_stats['runtime_history'].append(run_history)

        pickle.dump(cls_stats, open("/home/neuropoly/data/result_large_nobrain_nopad/cnn_results_it"+str(i)+".p", "wb"))
        model.save('/home/neuropoly/data/result_large_nobrain_nopad/model_cnn_it'+str(i)+'.h5')

        print(progress(cls_stats))
        print('\n')

pickle.dump(cls_stats, open("/home/neuropoly/data/cnn_results.p", "wb"))
model.save('/home/neuropoly/data/model_cnn.h5')

