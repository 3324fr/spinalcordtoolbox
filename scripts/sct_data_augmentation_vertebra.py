#!/usr/bin/env python
#########################################################################################
#
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Camille Van Assel
#
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import os
import sct_utils as sct
from msct_image import Image
import numpy as np


# Extract a table with the name of the file and the list of slices with their corresponding labels
# ==========================================================================================
def extract_label_list(folder_path, data_list, dirs_name):
    list_image_labels = []
    remove_list = []

    if len(dirs_name) == 0:
        for iter in range(0, len(data_list)/2):
            path, file, ext = sct.extract_fname(folder_path + '/' + data_list[iter*2])
            fname_label = folder_path + '/' + file + '_labels.nii.gz'
            try:
                sct.run('sct_label_vertebrae -i ' + folder_path + '/' + data_list[iter*2] + ' -s ' + folder_path + '/' + data_list[iter*2 + 1] + ' -c t2 -o ' + fname_label)
                list_image_labels.append(extract_labels_from_image(fname_label))
            except:
                remove_list.append(iter)

    else:
        for iter in range(0, len(dirs_name)):
            labels = folder_path + '/' + dirs_name[iter] + '/t2s/t2s_levels.txt'
            print labels
            path, file, ext = sct.extract_fname(folder_path + '/' + data_list[iter * 2])
            if not os.path.isfile(labels):
                fname_label = folder_path + '/' + dirs_name[iter] + '/t2s/' + file + '_labels.nii.gz'
                try:
                    sct.run('sct_label_vertebrae -i ' + folder_path + '/' + data_list[iter * 2] + ' -s ' + folder_path + '/' + data_list[iter * 2 + 1] + ' -c t2 -o ' + fname_label)
                    list_image_labels.append(extract_labels_from_image(fname_label))
                except:
                    remove_list.append(iter)
            else:
                list_image_labels.append(extract_labels_from_text(labels))

    for i in range(len(remove_list)-1,-1, -1):
        data_list.remove(data_list[remove_list[i]*2])
        data_list.remove(data_list[remove_list[i]*2 +1])
        dirs_name.remove(dirs_name[remove_list[i]])

    return list_image_labels, data_list, dirs_name


# Extract the vertebral labels from the image created with sct_labels_vertebrae
# ==========================================================================================
def extract_labels_from_image(fname_labels):
    im_labels = Image(fname_labels)
    nx, ny, nz, nt, px, py, pz, pt = Image(im_labels).dim
    data_labels = im_labels.data
    list_labels_vertebrae = np.zeros((nz,1))
    for iter in range(0, nz):
        if np.all(data_labels[:,:,nz] == 0):
            list_labels_vertebrae[iter,:] = 0
        else:
            list_labels_vertebrae[iter,:] = max(max(data_labels[:,:,nz]))
    return list_labels_vertebrae


# Extract the vertebral labels from a textual file
# ==========================================================================================
def extract_labels_from_text(fname_labels_text):
    fp = open(fname_labels_text)
    list_labels_vertebrae = []
    for l, line in enumerate(fp):
        if line[0] == '#':
            continue
        if line[1] == ',':
            if line[3] == '-':
                list_labels_vertebrae.append('0')
            else:
                list_labels_vertebrae.append(line[3])
    fp.close()
    return list_labels_vertebrae


# Extract the vertebral level from an image
# ==========================================================================================
def vertebra_number(fname, list_data, list_image_labels):
    import random
    num_vertebrae = []
    nbre_im = []

    for iter in range(0,len(fname)):
        nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
        rand_slice = random.randint(0, nz)
        while list_image_labels[list_data.index(fname)][rand_slice] == 0:
            rand_slice = random.randint(0, nz)
        num_vertebrae.append(list_image_labels[list_data.index(fname)][rand_slice])
        nbre_im.append(rand_slice)

    return num_vertebrae, nbre_im


# Give a set of 3 images from the same vertebra (from random subjects)
# ==========================================================================================
def random_slice_given_vertebra(list_data, list_image_labels, num_vertebra, GM = None, dirs_name = None):
    import random
    fname = []
    fname_seg = []
    dirs_list=[]

    for iter in range(0,len(num_vertebra)):
        slice_vertebra = []
        # We take a random image and find the vertebra with the choosen label
        while len(slice_vertebra == 0) :
            nbre_rand = random.randint(0, len(list_data/2))
            for i, element in list_image_labels[nbre_rand]:
                if element == num_vertebra[iter]:
                    slice_vertebra.append(i)
        fname.append(list_data[iter*2])
        fname_seg.append(list_data[iter*2 +1])
        dirs_list.append(dirs_name[iter])

        # We choose randomly one of these slice
        rand_slice = random.randint(0, len(slice_vertebra))
        nbre_slice = slice_vertebra[rand_slice]

    return fname, fname_seg, dirs_list, nbre_slice










