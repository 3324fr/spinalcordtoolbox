#!/usr/bin/env python
#########################################################################################
#
# This module contains some functions and algorithm for image classification and segmentation
# using supervised machine learning
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

def extract_data(path_data, verbose=1):
    """
    Extract the images into a (samples, feature) matrix.
    """
    from sys import stdout
    ignore_list = ['.DS_Store']
    if verbose == 1:
        sct.printv('Extracting '+ path_data)
    cr = '\r'

    data = []
    list_data = []
    images_folder = os.listdir(path_data)
    for i, fname_im in enumerate(images_folder):
        if fname_im in ignore_list:
            continue
        list_data.append(fname_im)

    for i, fname_im in enumerate(list_data):
        if verbose == 1:
            stdout.write(cr)
            stdout.write(str(i) + '/' + str(len(list_data)))
        im_data = Image(path_data + fname_im)
        data.append(np.expand_dims(im_data.data.flatten(), axis=0))

    data_result = np.concatenate(data, axis=0)
    if verbose == 1:
        stdout.write(cr)
        print 'Matrix shape (samples, feature):', data_result.shape
    return data_result.astype(np.float32)

data_centered = extract_data('/Users/benjamindeleener/data/machine_learning/scikit_learn/original/')

