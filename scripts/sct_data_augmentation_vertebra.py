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

    for iter in range(0, len(dirs_name)):
        labels = folder_path + '/' + dirs_name[iter] + '/t2s/t2s_levels.txt'
        fname_label = "Slices_" + dirs_name[iter] + '/' + dirs_name[iter] + '_labels.nii.gz'

        if not os.path.isfile(labels):
            if not sct.check_folder_exist("Slices_" + dirs_name[iter], verbose=0):
                os.makedirs("Slices_" + dirs_name[iter])
            if not os.path.isfile(fname_label):
                try:
                    sct.run('sct_label_vertebrae -i ' + data_list[iter * 3] + ' -s ' + data_list[iter * 3 + 2] + ' -c t2 -o ' + fname_label + ' -r 1')
                    list_image_labels.append(extract_labels_from_image(fname_label))
                except:
                    remove_list.append(iter)
                    print remove_list
            else:
                list_image_labels.append(extract_labels_from_image(fname_label))
        else:
            if dirs_name[iter].find('vanderbilt') != -1:
                list_image_labels.append(extract_labels_from_text_inverse(labels))
            else:
                list_image_labels.append(extract_labels_from_text(labels))

    for i in range(len(remove_list)-1, -1, -1):
        data_list.remove(data_list[remove_list[i]*3+2])
        data_list.remove(data_list[remove_list[i]*3+1])
        data_list.remove(data_list[remove_list[i]*3])
        dirs_name.remove(dirs_name[remove_list[i]])

    return list_image_labels, data_list, dirs_name


# Extract the vertebral labels from the image created with sct_labels_vertebrae
# ==========================================================================================
def extract_labels_from_image(fname_labels):
    im_labels = Image(fname_labels)
    nx, ny, nz, nt, px, py, pz, pt = Image(im_labels).dim
    data_labels = np.asarray(im_labels.data)
    list_labels_vertebrae = np.zeros((nz,1))
    for iter in range(0, nz):
        if np.all(data_labels[:,:,iter] == 0):
            list_labels_vertebrae[iter,0] = 0
        else:
            list_labels_vertebrae[iter,0] = np.max(np.max(data_labels[:,:,iter]))
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
        if line[2] == ',':
            if line[4] == '-':
                list_labels_vertebrae.append('0')
            else:
                list_labels_vertebrae.append(line[4])

    fp.close()
    return list_labels_vertebrae


# Extract their vertebral labels from a textual file where the slice are given backward
# ==========================================================================================
def extract_labels_from_text_inverse(fname_labels_text):
    fp = open(fname_labels_text)
    list_labels_vertebrae = []
    for l, line in enumerate(fp):
        if line[0] == '#':
            continue
        if line[1] == ',':
            if line[3] == '-':
                list_labels_vertebrae.insert(0,'0')
            else:
                list_labels_vertebrae.insert(0, line[3])
        if line[2] == ',':
            if line[4] == '-':
                list_labels_vertebrae.insert(0,'0')
            else:
                list_labels_vertebrae.insert(0,line[4])

    fp.close()
    return list_labels_vertebrae

# Crop along the segmentation for the zurich images
# ==========================================================================================
def crop_segmentation(list_data, dirs_name):

    for iter in range(0, len(dirs_name)):
        if dirs_name[iter].find('zurich') != -1:
            if not os.path.isfile('Slices_' + dirs_name[iter] + '/' + dirs_name[iter] + '_crop.nii.gz'):
                nx, ny, nz, nt, px, py, pz, pt = Image(list_data[iter*3]).dim
                data_seg = Image(list_data[iter*3 +2]).data
                if np.all(data_seg[:, :, 0] == np.zeros((nx, ny))) or np.all(data_seg[:, :, nz - 1] == np.zeros((nx, ny))):
                    i = 1
                    while np.all(data_seg[:, :, i] == np.zeros((nx, ny))):
                        i += 1
                    j = nz - 1
                    while np.all(data_seg[:, :, j] == np.zeros((nx, ny))):
                        j -= 1
                if not os.path.isdir("Slices_" + dirs_name[iter]):
                    os.makedirs("Slices_" + dirs_name[iter])

                output_name = 'Slices_' + dirs_name[iter] + '/' + dirs_name[iter] + '_crop'
                sct.run("sct_crop_image -i " + list_data[iter*3] + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + output_name + '.nii.gz', verbose=1)
                sct.run("sct_crop_image -i " + list_data[iter*3 +2] + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + output_name + '_seg.nii.gz', verbose=1)
                sct.run("sct_crop_image -i " + list_data[iter*3 + 1] + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + output_name + '_gmseg.nii.gz', verbose=1)
                list_data[iter*3] = output_name + '.nii.gz'
                list_data[iter*3 + 2] = output_name + '_seg.nii.gz'
                list_data[iter*3 + 1] = output_name + '_gmseg.nii.gz'

    return list_data

# Crop along the segmentation for the zurich images
# ==========================================================================================
def change_orientation(list_data, dirs_name):

    for iter in range(0, len(dirs_name)):
        if dirs_name[iter].find('ucl') != -1:
            if not os.path.isfile('Slices_' + dirs_name[iter] + '/' + dirs_name[iter] + '_rpi.nii.gz'):
                if not os.path.isdir("Slices_" + dirs_name[iter]):
                    os.makedirs("Slices_" + dirs_name[iter])

                output_name = 'Slices_' + dirs_name[iter] + '/' + dirs_name[iter] + '_rpi'
                sct.run('sct_image -i ' + list_data[iter*3] + ' -setorient RPI -o ' + output_name + '.nii.gz')
                sct.run('sct_image -i ' + list_data[iter * 3 + 2] + ' -setorient RPI -o ' + output_name + '_seg.nii.gz')
                sct.run('sct_image -i ' + list_data[iter * 3 + 1] + ' -setorient RPI -o ' + output_name + '_gmseg.nii.gz')
                list_data[iter * 3] = output_name + '.nii.gz'
                list_data[iter * 3 + 2] = output_name + '_seg.nii.gz'
                list_data[iter * 3 + 1] = output_name + '_gmseg.nii.gz'

    return list_data

# Determine a score of vertebral similarity between two images
# ==========================================================================================
def vertebral_similarity(fname, list_data, list_image_labels, nbre):
    nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
    score = [0]*len(list_data)

    for iter in range(0,nz):
        vert = list_image_labels[list_data.index(fname)/3][iter]
        for i in range(0, len(list_data)/3):
            if vert in list_image_labels[i]:
                score[i] += 1
    score[list_data.index(fname)/3] = 0
    fname_out = []
    for j in range(0,nbre):
        try:
            index_max = score.index(max(score))
            fname_out.append(list_data[index_max*3])
            score[index_max] = 0
        except:
            print('There is not enough element in your dataset to perform the given number of transformations')
    return fname_out


# Ge the offset of our registration window to replace it in the original image at the end
# ==========================================================================================
def get_start_crop(fname_seg, fname_size_x, fname_size_y):

    from scipy import ndimage
    start_x = []
    start_y = []

    image_slice_seg = Image(fname_seg)
    nx, ny, nz, nt, px, py, pz, pt = image_slice_seg.dim

    # find the mass center of each slice

    for iter in range(0,nz):
        cof = ndimage.measurements.center_of_mass(image_slice_seg.data[:,:,iter])

        # crop the image along the x,y direction
        start_x.append(int(round(cof[0] - fname_size_x / 2) + 1))
        start_y.append(int(round(cof[1] - fname_size_y / 2) + 1))

    return start_x, start_y


# Given two different images, perform the 2D registration between same vertebral levels (or similar ones)
# ==========================================================================================
def registration_vertebral_levels(fname, fname_out, list_data, dirs_name, list_image_labels):
    nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim

    size_patch = 61
    index_fname = list_data.index(fname)
    hdr_fname = Image(fname).hdr.copy()

    shift = np.zeros((nz, 2))
    data_src = np.zeros((size_patch, size_patch, nz))
    data_src_seg = np.zeros((size_patch, size_patch, nz))
    data_src_gmseg = np.zeros((size_patch, size_patch, nz))

    if not os.path.isdir("Slices_" + dirs_name[index_fname/3]):
        os.makedirs("Slices_" + dirs_name[index_fname/3])

    #if not os.path.isfile('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index_fname/3] + '_cs_crop.nii.gz'):
    if 1 == 1:
        data = Image(fname).data
        data_seg = Image(list_data[index_fname + 2]).data
        data_gm = Image(list_data[index_fname + 1]).data
        shift[:, 0], shift[:, 1] = get_start_crop(list_data[index_fname + 2],size_patch, size_patch)

        for i in range(0,nz):
            data_src[:,:,i] = data[shift[i,0]:shift[i,0] + size_patch, shift[i,1]:shift[i,1] + size_patch, i]
            data_src_seg[:, :, i] = data_seg[shift[i,0]:shift[i,0] + size_patch, shift[i,1]:shift[i,1] + size_patch, i]
            data_src_gmseg[:, :, i] = data_gm[shift[i,0]:shift[i,0] + size_patch, shift[i,1]:shift[i,1] + size_patch, i]

        im_src = Image(data_src)
        im_src.setFileName('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index_fname/3] + '_cs_crop.nii.gz')
        im_src.save()
        im_src = Image(data_src_seg)
        im_src.setFileName('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index_fname / 3] + '_cs_crop_seg.nii.gz')
        im_src.save()
        im_src = Image(data_src_gmseg)
        im_src.setFileName('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index_fname / 3] + '_cs_crop_gmseg.nii.gz')
        im_src.save()

    for iter in range(0, len(fname_out)):

        index = list_data.index(fname_out[iter])

        #if not os.path.exists('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index / 3] + '_cs_crop.nii.gz'):
        if 1 == 1:
            nx_d, ny_d, nz_d, nt_d, px_d, py_d, pz_d, pt_d = Image(fname_out[iter]).dim
            data_dest = np.zeros((size_patch, size_patch, nz))
            data_dest_seg = np.zeros((size_patch, size_patch, nz))
            data_dest_gmseg = np.zeros((size_patch, size_patch, nz))
            shift = np.zeros((nz_d,2))
            shift[:, 0], shift[:, 1] = get_start_crop(list_data[index + 2], size_patch, size_patch)

            data = Image(list_data[index]).data
            data_seg = Image(list_data[index + 2]).data
            data_gm = Image(list_data[index +1]).data

            for i in range(0,nz):
                vert = list_image_labels[list_data.index(fname)/3][i]
                diff = []
                for j in range(0,nz_d):
                    diff.append(abs(int(list_image_labels[index/3][j]) - int(vert)))
                num_slice = diff.index(np.min(diff))

                data_dest[:, :, i] = data[shift[num_slice, 0]:shift[num_slice, 0] + size_patch, shift[num_slice, 1]:shift[num_slice, 1] + size_patch, num_slice]
                data_dest_seg[:, :, i] = data_seg[shift[num_slice, 0]:shift[num_slice, 0] + size_patch, shift[num_slice, 1]:shift[num_slice, 1] + size_patch, num_slice]
                data_dest_gmseg[:, :, i] = data_gm[shift[num_slice, 0]:shift[num_slice, 0] + size_patch, shift[num_slice, 1]:shift[num_slice, 1] + size_patch, num_slice]

            im_dest = Image(data_dest)
            im_dest.setFileName('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index / 3] + '_cs_crop.nii.gz')
            im_dest.save()
            im_dest = Image(data_dest_seg)
            im_dest.setFileName('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index / 3] + '_cs_crop_seg.nii.gz')
            im_dest.save()
            im_dest = Image(data_dest_gmseg)
            im_dest.setFileName('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index / 3] + '_cs_crop_gmseg.nii.gz')
            im_dest.save()

        # We perform the registration between these two images
        out_transfo = str(iter) + '_t_'
        sct.run('isct_antsRegistration ' +
                    '--dimensionality 3 ' +
                    '--transform BSplineSyN[0.5,1,3] ' +
                    '--metric MeanSquares['+ 'Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index_fname/ 3] + '_cs_crop_seg.nii.gz' + ',' + 'Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index/3] + '_cs_crop_seg.nii.gz' + ',1] ' +
                    '--convergence 5x3 ' +
                    '--shrink-factors 2x1 ' +
                    '--restrict-deformation 1x1x0 ' +
                    '--smoothing-sigmas 1x0mm ' +
                    '--output [' + out_transfo + '] ' +
                    '--interpolation BSpline[3] ' +
                    '--verbose 1', verbose=1)
        warp = out_transfo + '0Warp.nii.gz'

        warp_multi_generator('Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index_fname/3] + '_cs_crop.nii.gz', 'Slices_' + dirs_name[index_fname/3] + '/'
                             + dirs_name[index_fname/3] + '_cs_crop_seg.nii.gz', 'Slices_' + dirs_name[index_fname/3] + '/' + dirs_name[index_fname/3] +
                             '_cs_crop_gmseg.nii.gz', warp, 5, dirs_name[index_fname/3] + '_' + dirs_name[index/3])

        shift = np.zeros((nz,2))
        shift[:, 0], shift[:, 1] = get_start_crop(list_data[index_fname + 2], size_patch, size_patch)
        for iter in range(0,5):
            image_matching(fname, dirs_name[index_fname/3] + '_' + dirs_name[index/3] + '_' + str(iter) + '.nii.gz', shift, size_patch, hdr_fname)
            image_matching(list_data[index_fname + 2], dirs_name[index_fname/3] + '_' + dirs_name[index/3] + '_' + str(iter) + '_seg.nii.gz', shift, size_patch, hdr_fname)
            image_matching(list_data[index_fname + 1], dirs_name[index_fname/3] + '_' + dirs_name[index/3] + '_' + str(iter) + '_gmseg.nii.gz', shift, size_patch, hdr_fname)

    return dirs_name[index_fname/3] + '_' + dirs_name[index/3]


# Generate multiple images from a warping field
# ==========================================================================================
def warp_multi_generator(src, src_seg, src_gm, warp, num_of_frames, output_name):

    result = Image(warp)

    for iteration in range(0,num_of_frames):
        print "Iteration #" + str(iteration)
        result_c = result.copy()
        result_c.data *= float(iteration)+1 / 10
        result_c.file_name = "tmp." + result.file_name + "_" + str(iteration)
        result_c.save()

        # Apply the transfo on the source images
        sct.run('sct_apply_transfo -i ' + src + ' -d ' + src + ' -w ' + "tmp." + result.file_name + "_" + str(
            iteration) + '.nii.gz -o ' + output_name + '_' + str(iteration) + '.nii.gz -x linear', verbose=1)
        sct.run('sct_apply_transfo -i ' + src_seg + ' -d ' + src + ' -w ' + "tmp." + result.file_name + "_" + str(
            iteration) + '.nii.gz -o ' + output_name + '_' + str(iteration) + '_seg.nii.gz -x linear', verbose=1)
        sct.run('sct_apply_transfo -i ' + src_gm + ' -d ' + src + ' -w ' + "tmp." + result.file_name + "_" + str(
            iteration) + '.nii.gz -o ' + output_name + '_' + str(iteration) + '_gmseg.nii.gz -x linear', verbose=1)


# Matching of the registered image on the initial image
# ==========================================================================================
def image_matching(image, crop, shift, size, hdr):
    nx, ny, nz, nt, px, py, pz, pt = Image(image).dim
    data = Image(image).data
    data_crop = Image(crop).data

    for iter in range(0,nz):
        data[shift[iter,0]:shift[iter,0]+size, shift[iter,1]:shift[iter,1]+size,iter] = data_crop[:,:,iter]

    im = Image(data)
    im.hdr = hdr
    im.setFileName(Image(crop).file_name + '.nii.gz')
    print str(Image(crop).file_name)
    im.save()


# MAIN
# ==========================================================================================
