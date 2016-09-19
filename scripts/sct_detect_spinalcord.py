#!/usr/bin/env python
#
# This program takes as input an anatomic image and the centerline or segmentation of its spinal cord (that you can get
# using sct_get_centerline.py or sct_segmentation_propagation) and returns the anatomic image where the spinal
# cord was straightened.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Julien Cohen-Adad, Geoffrey Leveque, Julien Touati
# Modified: 2014-09-01
#
# License: see the LICENSE.TXT
# ======================================================================================================================
# check if needed Python libraries are already installed or not

import os
import time
import sys
from msct_parser import Parser
import sct_utils as sct
import numpy as np
from msct_image import Image

try:
    import cPickle as pickle
except:
    import pickle

# import keras necessary classes
import theano
theano.config.floatX = 'float32'

from keras.models import load_model


def extract_slices_from_image(fname_im):
    im_data = Image(fname_im)
    nx, ny, nz, nt, px, py, pz, pt = im_data.dim

    #im_data.data = (im_data.data - np.mean(im_data.data)) / np.abs(np.percentile(im_data.data, 1) - np.percentile(im_data.data, 99))
    #im_data.data = 255.0 * im_data.data / np.abs(np.percentile(im_data.data, 1) - np.percentile(im_data.data, 99))
    im_data.data = 255.0 * (im_data.data - np.percentile(im_data.data, 0)) / np.abs(np.percentile(im_data.data, 0) - np.percentile(im_data.data, 100))

    data_im = []
    for k in range(nz):
        data_im.append(im_data.data[:, :, k])

    return data_im


def extract_patches_from_slice(slice_im, patches_coordinates, patch_size=32):
    result = []
    for k in range(len(patches_coordinates[0])):
        ind = [patches_coordinates[0][k], patches_coordinates[1][k]]
        patch = slice_im[ind[0] - patch_size / 2:ind[0] + patch_size / 2, ind[1] - patch_size / 2:ind[1] + patch_size / 2]
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            result.append(np.expand_dims(patch, axis=0))
            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots()
            #cax = ax.imshow(patch, cmap='gray')
            #cbar = fig.colorbar(cax, ticks=[0, 255])
            #plt.show()
    if len(result) != 0:
        return np.concatenate(result, axis=0)
    else:
        return None


def predict(fname_input, fname_model, initial_resolution):
    patch_size = 32
    model = load_model(fname_model)
    input_slices = extract_slices_from_image(fname_input)

    # first round of patch prediction
    initial_coordinates_x = range(patch_size/2, input_slices[0].shape[0] - patch_size/2, initial_resolution[0])
    initial_coordinates_y = range(patch_size/2, input_slices[0].shape[1] - patch_size/2, initial_resolution[1])
    X, Y = np.meshgrid(initial_coordinates_x, initial_coordinates_y)
    initial_coordinates = np.vstack([X.ravel(), Y.ravel()])

    coord_positive = []

    for slice_number in range(0, len(input_slices), initial_resolution[2]):
        current_slice = input_slices[slice_number]
        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #cax = ax.imshow(current_slice, cmap='gray')
        #cbar = fig.colorbar(cax, ticks=[0, 255])
        #plt.show()
        slice_shape = current_slice.shape
        print slice_number, slice_shape
        patches = extract_patches_from_slice(current_slice, initial_coordinates, patch_size=patch_size)
        patches = np.asarray(patches, dtype=int)
        patches = patches.reshape(patches.shape[0], 1, patches.shape[1], patches.shape[2])
        #print patches.shape
        y_pred = model.predict_classes(patches, batch_size=32, verbose=0)
        classes_predictions = np.where(y_pred == 1)[0].tolist()
        coord_positive.extend([[initial_coordinates[0][coord], initial_coordinates[1][coord], slice_number] for coord in classes_predictions])

    # write results
    input_image = Image(fname_input).copy()
    input_image.data *= 0
    for coord in coord_positive:
        input_image.data[coord[0], coord[1], coord[2]] = 1

    input_image.setFileName('/Users/benjamindeleener/data/machine_learning/test_detection/test.nii.gz')
    input_image.save()


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("This program takes as input an anatomic image and outputs a binary image with the spinal cord centerline/segmentation.")
    parser.add_option(name="-i",
                      type_value="image_nifti",
                      description="input image.",
                      mandatory=True,
                      example="t2.nii.gz")

    parser.add_option(name="-c",
                      type_value="multiple_choice",
                      description="type of image contrast, t2: cord dark / CSF bright ; t1: cord bright / CSF dark.",
                      mandatory=False,
                      example=['t1', 't2'])

    parser.add_option(name="-o",
                      type_value="file_output",
                      description="centerline/segmentation filename.",
                      mandatory=False,
                      default_value='',
                      example="data_straight.nii.gz")

    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="remove temporary files.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing, 1: basic, 2: extended.",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    fname_model = '/Users/benjamindeleener/data/machine_learning/results_detection/2016-08-16_nopad_HCandCSM/model_cnn_it240000.h5'
    #fname_model = '/Users/benjamindeleener/data/machine_learning/results_detection/2016-08-17_HCandCSM_4conv/model_cnn_it530000.h5'
    initial_resolution = [3, 3, 5]

    predict(arguments['-i'], fname_model, initial_resolution)
