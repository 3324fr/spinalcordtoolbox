#!/usr/bin/env python
#########################################################################################
#
# This module contains some functions and algorithm for image classification and segmentation
# using supervised machine learning
# python msct_keras_results.py
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2016 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2016-07-08
#
# About the license: see the file LICENSE.TXT
#########################################################################################


import sys
from msct_image import Image
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

fname_im = sys.argv[1:]

from keras.models import load_model

model = load_model('/Users/benjamindeleener/data/machine_learning/results_detection/model_cnn_it195000.h5')

patch_size = 32

def extract_slices_from_image(fname_im, fname_seg=None):
    im_data = Image(fname_im)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim
    if fname_seg:
        im_seg = Image(fname_seg)

    data_im = []
    data_seg = []
    for k in range(nz):
        data_im.append(im_data.data[:, :, k])
        if fname_seg:
            data_seg.append(im_seg.data[:, :, k])

    if fname_seg:
        return data_im, data_seg
    else:
        return data_im

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(windowSize/2, image.shape[0]-windowSize/2, stepSize):
        for x in xrange(windowSize/2, image.shape[1]-windowSize/2, stepSize):
            # yield the current window
            yield (x, y, image[y-windowSize/2:y+windowSize/2, x-windowSize/2:x+windowSize/2])


# parameters
nb_patches_batch = 100000
stepSize = 1

image_slices = extract_slices_from_image(fname_im[0])
im = Image(fname_im[0])
im.data *= 0
patches, indexes, results = [], [], []

indexes_it = []
nb_patches = 0

nx, ny, nz, nt, px, py, pz, pt = im.dim
total_patches = (nx-patch_size/2) * (ny-patch_size/2) * nz / stepSize ^ 3

import time
tstart = time.time()
for z, slice in enumerate(image_slices):
    for (x, y, window) in sliding_window(slice, stepSize=stepSize, windowSize=patch_size):
        if window.shape[0] == patch_size and window.shape[1] == patch_size:
            patches.append(np.expand_dims(window, axis=0))
            indexes_it.append([y, x, z])
            if len(indexes_it) >= nb_patches_batch:
                patches = np.concatenate(patches, axis=0)
                patches = patches.reshape(patches.shape[0], 1, patches.shape[1], patches.shape[2])
                nb_patches += patches.shape[0]
                print 'Number of patches', 100 * nb_patches / float(total_patches), '%'
                predictions = model.predict(patches)
                results.extend(np.argmax(predictions, axis=1))
                indexes.extend(indexes_it)

                patches, indexes_it = [], []

print time.time() - tstart

for ind in np.where(results == 1)[0]:
    #print indexes[ind]
    im.data[indexes[ind][0], indexes[ind][1], indexes[ind][2]] = 1

im.setFileName('test.nii.gz')
im.save()
