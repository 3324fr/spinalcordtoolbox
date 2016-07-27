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


from msct_parser import Parser
import os
import sys
import time
import numpy as np
import sct_utils as sct
from msct_image import Image
import random
from scipy import ndimage
from sct_data_augmentation import extract_name_list
from multiprocessing import Pool


def straighten(fname, fname_seg):
    nx, ny, nz, nt, px, py, pz, pt = Image(fname_seg).dim
    center = [int(nx/2), int(ny/2)]
    cof = np.zeros((2,nz))
    im_seg = Image(fname_seg)
    data_seg = im_seg.data

    for iter in range(0, nz):
        cof[0,iter] = ndimage.measurements.center_of_mass(data_seg[:,:,iter])[0]
        cof[1,iter] = ndimage.measurements.center_of_mass(data_seg[:,:,iter])[1]
    delta_cof_x = np.asarray(cof[0,:] - center[0]*np.ones((nz)), dtype='int')
    delta_cof_y = np.asarray(cof[1,:] - center[1]*np.ones((nz)), dtype='int')

    im = Image(fname)
    data = np.asarray(im.data)

    for iter in range(0,nz):
        if delta_cof_x[iter] > 0:
            data[0:nx - delta_cof_x[iter],:,iter] = data[delta_cof_x[iter]:,:,iter]
            data[nx-delta_cof_x[iter]:nx,:,iter] = 0
            data_seg[0:nx - delta_cof_x[iter], :, iter] = data_seg[delta_cof_x[iter]:, :, iter]
            data_seg[nx - delta_cof_x[iter]:nx, :, iter] = 0
        else:
            data[abs(delta_cof_x[iter]):nx, :, iter] = data[0:nx - abs(delta_cof_x[iter]),:,iter]
            data[0:abs(delta_cof_x[iter]), :, iter] = 0
            data_seg[abs(delta_cof_x[iter]):nx, :, iter] = data_seg[0:nx - abs(delta_cof_x[iter]), :, iter]
            data_seg[0:abs(delta_cof_x[iter]), :, iter] = 0

        if delta_cof_y[iter] > 0:
            data[:, 0:ny - delta_cof_y[iter], iter] = data[:, delta_cof_y[iter]:ny,iter]
            data[:,ny - delta_cof_y[iter]:ny,iter] = 0
            data_seg[:,0:ny - delta_cof_y[iter], iter] = data_seg[:,delta_cof_y[iter]:ny,iter]
            data_seg[:, ny - delta_cof_y[iter]:ny, iter] = 0
        else:
            data[:, abs(delta_cof_y[iter]):ny, iter] = data[:,0:ny - abs(delta_cof_y[iter]),iter]
            data[:,0:abs(delta_cof_y[iter]), iter] = 0
            data_seg[:,abs(delta_cof_y[iter]):ny,iter] = data_seg[:,0:ny - abs(delta_cof_y[iter]),iter]
            data_seg[:,0:abs(delta_cof_y[iter]),iter] = 0

    im.data = data
    im_seg.data = data_seg
    im.setFileName = fname
    im_seg.setFileName = fname_seg
    im.hdr.structarr['qoffset_x'] = im.hdr.structarr['qoffset_y'] = im.hdr.structarr['qoffset_z'] = im.hdr.structarr['srow_x'][-1] = im.hdr.structarr['srow_y'][-1] = im.hdr.structarr['srow_z'][-1] = 0
    im_seg.hdr.structarr['qoffset_x'] = im_seg.hdr.structarr['qoffset_y'] = im_seg.hdr.structarr['qoffset_z'] = im_seg.hdr.structarr['srow_x'][-1] = im_seg.hdr.structarr['srow_y'][-1] = im_seg.hdr.structarr['srow_z'][-1] = 0
    im.save()
    im_seg.save()

    return delta_cof_x, delta_cof_y


# Given two images, crop them to have the same dimension (Dimension of the smallest one)
# ==========================================================================================
def crop_image_to_same_dimension(fname_src, fname_dest, fname_src_seg, fname_dest_seg, dirs_name_src, dirs_name_dest, tmp_path):
    for iter in range(0,len(fname_src)):
        nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = Image(fname_src[iter]).dim
        nx_d, ny_d, nz_d, nt_d, px_d, py_d, pz_d, pt_d = Image(fname_dest[iter]).dim

        if nx_s < nx_d :
            start = int((nx_d - nx_s)/2)
            fname_dest[iter],fname_dest_seg[iter] = crop_image(fname_dest[iter], fname_dest_seg[iter], tmp_path, dirs_name_dest[iter], start, nx_s, 0)
        if nx_d < nx_s:
            start = int((nx_s - nx_d) / 2)
            fname_src[iter], fname_src_seg[iter] = crop_image(fname_src[iter], fname_src_seg[iter], tmp_path, dirs_name_src[iter], start, nx_d, 0)

        if ny_s < ny_d :
            start = int((ny_d - ny_s) / 2)
            fname_dest[iter], fname_dest_seg[iter] = crop_image(fname_dest[iter], fname_dest_seg[iter], tmp_path, dirs_name_dest[iter], start, ny_s, 1)
        if ny_d < ny_s:
            start = int((ny_s - ny_d) / 2)
            fname_src[iter], fname_src_seg[iter] = crop_image(fname_src[iter], fname_src_seg[iter], tmp_path, dirs_name_src[iter], start, ny_d, 1)

        if nz_s < nz_d:
            fname_dest[iter], fname_dest_seg[iter] = resample_image(fname_dest[iter], fname_dest_seg[iter], tmp_path, dirs_name_dest[iter], nx_d, ny_d, nz_s)
        if nz_d < nz_s:
            fname_src[iter], fname_src_seg[iter] = resample_image(fname_src[iter], fname_src_seg[iter], tmp_path, dirs_name_src[iter], nx_s, ny_s, nz_d)

    return fname_src, fname_dest, fname_src_seg, fname_dest_seg


def crop_image(fname, fname_seg, tmp_path, dirs_name, start, nx, dim):
    path, file, ext = sct.extract_fname(fname)
    fname_out = tmp_path + dirs_name + '_' + file + ext
    fname_seg_out = tmp_path + dirs_name + '_' + file + '_seg' + ext
    sct.run('sct_crop_image -i ' + fname + ' -dim ' + str(dim) + ' -start ' + str(start) + ' -end ' + str(start + nx - 1) + ' -o ' + fname_out)
    sct.run('sct_crop_image -i ' + fname_seg + ' -dim ' + str(dim) + ' -start ' + str(start) + ' -end ' + str(start + nx - 1) + ' -o ' + fname_seg_out)
    return fname_out, fname_seg_out


def resample_image(fname, fname_seg, tmp_path, dirs_name, nx, ny, nz):
    path, file, ext = sct.extract_fname(fname)
    fname_out = tmp_path + dirs_name + '_' + file + ext
    fname_seg_out = tmp_path + dirs_name + '_' + file + '_seg' + ext
    sct.run('sct_resample -i ' + fname + ' -vox ' + str(nx) + 'x' + str(ny) + 'x' + str(nz) + ' -o ' + fname_out)
    sct.run('sct_resample -i ' + fname_seg + ' -vox ' + str(nx) + 'x' + str(ny) + 'x' + str(nz) + ' -o ' + fname_seg_out)

    return fname_out, fname_seg_out

# ==========================================================================================
def crop_image_to_same_dimension_2(fname_src, fname_im, fname_im_seg, dirs_name_im, tmp_path):

    for iter in range(0,len(fname_src)):
        nx_s, ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = Image(fname_src[iter]).dim
        for i in range(0, len(fname_im)):
            nx_i, ny_i, nz_i, nt_i, px_i, py_i, pz_i, pt_i = Image(fname_im[i*iter + i]).dim
            fname_im[i*iter + i], fname_im_seg[i*iter + i] = resample_image(fname_im[i*iter + i], fname_im_seg[i*iter + i], tmp_path, dirs_name_im[i*iter + i], nx_s, ny_s, nz_s)

    return fname_im, fname_im_seg

# Create warping field (multiprocess)
# ==========================================================================================
def worker_warping_field(argument):
    src_seg, dest_seg, v, iter = argument

    out = "t2_output_image_transformed.nii.gz"
    out_transfo = str(iter)

    sct.run('isct_antsRegistration ' +
            '--dimensionality 3 ' +
            '--transform BSplineSyN[0.5,1,3] ' +
            '--metric MeanSquares[' + dest_seg + ',' + src_seg + ', 1] ' +
            '--convergence 5x3 ' +
            '--shrink-factors 20x10 ' +
            '--smoothing-sigmas 1x0mm ' +
            '--restrict-deformation 1x1x0 ' +
            '--output [' + out_transfo + ',' + out + '] ' +
            '--interpolation BSpline[3] ' +
            '--verbose 0', verbose=v)

    warp = out_transfo + '0Warp.nii.gz'

    return warp

# Create warping field
# ==========================================================================================
def warping_field(src_seg, dest_seg, nw, v):
    warp = []

    for iter in range(0, nw):
        out = "t2_output_image_transformed.nii.gz"
        out_transfo = str(iter)

        sct.run('isct_antsRegistration ' +
                '--dimensionality 3 ' +
                '--transform BSplineSyN[0.5,1,3] ' +
                '--metric MeanSquares[' + dest_seg[iter] + ',' + src_seg[iter] + ', 1] ' +
                '--convergence 5x3 ' +
                '--shrink-factors 2x1 ' +
                '--smoothing-sigmas 1x0mm ' +
                '--restrict-deformation 1x1x0 ' +
                '--output [' + out_transfo + ',' + out + '] ' +
                '--interpolation BSpline[3] ' +
                '--verbose 0', verbose = v)

        warp.append(out_transfo + '0Warp.nii.gz')
    return warp


# Apply warping field
# ==========================================================================================
def worker_apply_warping_field(argument):
    im, im_seg, dest, nbre_slice, output_folder_path, wrap, j, v, im_GM, dirs_name_src, dirs_name_dest, dirs_name_im = argument

    for iter in range(0, nbre_slice):

        fname_out_im = dirs_name_src + '_' + dirs_name_dest + '_' + dirs_name_im[j+iter] + '.nii.gz'
        fname_out_seg = dirs_name_src + '_' + dirs_name_dest + '_' + dirs_name_im[j+iter] + '_seg.nii.gz'
        fname_out_GM = dirs_name_src + '_' + dirs_name_dest + '_' + dirs_name_im[j+iter] + '_gmseg.nii.gz'

        # Apply warping field to src data
        if not os.path.isfile(fname_out_im):
            sct.run('isct_antsApplyTransforms -i ' + im[j+iter] + ' -r ' + dest + ' -n Linear -t ' + wrap + ' --output ' + output_folder_path + fname_out_im, verbose=1)
            sct.run('isct_antsApplyTransforms -i ' + im_seg[j+iter] + ' -r ' + dest + ' -n Linear -t ' + wrap + ' --output ' + output_folder_path + fname_out_seg, verbose=1)
            if im_GM:
                sct.run('isct_antsApplyTransforms -i ' + im_GM[j + iter] + ' -r ' + dest + ' -n Linear -t ' + wrap + ' --output ' + output_folder_path + fname_out_GM, verbose=1)


# Apply warping field
# ==========================================================================================
def apply_warping_field(im, im_seg, dest, nbre_slice, nbre_wrap, output_folder_path, wrap, v, im_GM, dirs_name_src, dirs_name_dest, dirs_name_im):
    for i in range(0, nbre_wrap):
        for iter in range(0, nbre_slice):
            im_path, im_file, im_ext = sct.extract_fname(im[i * iter + iter])
            fname_out_im = dirs_name_src[i] + '_' + dirs_name_dest[i] + '_' + dirs_name_im[i * iter + iter] + im_file +'.nii.gz'
            fname_out_seg = dirs_name_src[i] + '_' + dirs_name_dest[i] + '_' + dirs_name_im[i * iter + iter] + im_file +'_seg.nii.gz'
            fname_out_GM = dirs_name_src[i] + '_' + dirs_name_dest[i] + '_' + dirs_name_im[i * iter + iter] + im_file +'_gmseg.nii.gz'
            # Apply warping field to src data
            sct.run('isct_antsApplyTransforms -d 3 -i ' + im[i * iter + iter] + ' -r ' + dest[i] + ' -n Linear -t ' + wrap[i] + ' --output ' + output_folder_path + fname_out_im,verbose=v)
            sct.run('isct_antsApplyTransforms -d 3 -i ' + im_seg[i * iter + iter] + ' -r ' + dest[i] + ' -n Linear -t ' + wrap[i] + ' --output ' + output_folder_path + fname_out_seg,verbose=v)
            if im_GM:
                sct.run('isct_antsApplyTransforms -d 3 -i ' + im_GM[i * iter + iter] + ' -r ' + dest[i] + ' -n Linear -t ' + wrap[i] + ' --output ' + output_folder_path + fname_out_GM,verbose=v)


# MAIN
# ==========================================================================================
def main():
    # Initialisation
    start_time_total = time.time()

    # Initialise the parser
    parser = Parser(__file__)
    parser.add_option(name="-f",
                      type_value="folder",
                      description="folder of input image and segmentation",
                      mandatory=False,
                      example="data_patients")
    parser.add_option(name="-dim",
                      type_value=[[','], 'float'],
                      description="size of the image at x and y and resolution of the pixel at x and y",
                      mandatory=True,
                      example="164, 164, 0.5, 0.5")
    parser.add_option(name="-nb",
                      type_value="int",
                      description="number of random slice we apply the wrap on",
                      mandatory=True,
                      example="31")
    parser.add_option(name="-nw",
                      type_value="int",
                      description="number of random wrap we generate",
                      mandatory=True,
                      example='15')
    parser.add_option(name="-o",
                      type_value='str',
                      description="name of the output folder",
                      mandatory=False)
    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="""Verbose.""",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1', '2'])
    parser.add_option(name="-cpu-nb",
                      type_value="int",
                      description="Number of CPU used for straightening. 0: no multiprocessing. If not provided, "
                                  "it uses all the available cores.",
                      mandatory=False,
                      example="8")
    parser.add_option(name="-GM",
                      type_value="multiple_choice",
                      description=" ",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-preprocess",
                      type_value="multiple_choice",
                      description=" ",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])

    # get argument
    arguments = parser.parse(sys.argv[1:])
    dim_list = arguments['-dim']
    nbre_slice = arguments['-nb']
    nbre_wrap = arguments['-nw']
    verbose = int(arguments['-v'])

    v = 0
    if verbose == 2:
        v = 1
    if "-cpu-nb" in arguments:
        cpu_number = int(arguments["-cpu-nb"])
    if '-f' in arguments:
        input_folder = arguments['-f']
        sct.check_folder_exist(str(input_folder), verbose=0)
        folder_path = sct.get_absolute_path(input_folder)
    if '-o' in arguments:
        output_folder = arguments['-o']
        if not sct.check_folder_exist(output_folder, verbose=0):
            os.makedirs(str(output_folder))
        output_folder_path = str(output_folder) + "/"
    else:
        output_folder_path = ''

    if '-GM' in arguments:
        GM = int(arguments['-GM'])
    if '-preprocess' in arguments:
        preprocess = int(arguments['-preprocess'])


# create a folder with all the images
    list_data, dirs_name = extract_name_list(folder_path, GM, preprocess)
    if preprocess:
        from sct_data_augmentation import preprocess
        preprocess(list_data, folder_path, cpu_number, GM, dim_list, verbose)

    delta_cof_x = {}
    delta_cof_y = {}
    #for iter in range(0,len(dirs_name)):
    #    fname = list_data[iter*2]
    #    fname_seg = list_data[iter*2+1]
    #    delta_cof_x[iter], delta_cof_y[iter] = straighten(fname, fname_seg)


    from sct_data_augmentation import random_list
    start_time = time.time()
    if GM:
        fname_src, fname_src_seg, fname_src_GM, dirs_name_src = random_list(list_data, nbre_wrap, GM, dirs_name)
    else:
        fname_src, fname_src_seg, dirs_name_src = random_list(list_data, nbre_wrap, 0, dirs_name)
    if verbose:
        elapsed_time = time.time() - start_time
        sct.printv('\nElapsed time to create a random list of source image: ' + str(int(round(elapsed_time))) + 's')

    start_time = time.time()
    if GM:
        fname_dest, fname_dest_seg, fname_dest_GM, dirs_name_dest = random_list(list_data, nbre_wrap, GM, dirs_name)
    else:
        fname_dest, fname_dest_seg, dirs_name_dest = random_list(list_data, nbre_wrap, 0, dirs_name)
    if verbose:
        elapsed_time = time.time() - start_time
        sct.printv(
            '\nElapsed time to create a random list of destination image: ' + str(int(round(elapsed_time))) + 's')

    start_time = time.time()
    if GM:
        fname_im, fname_im_seg, fname_im_GM, dirs_name_im = random_list(list_data, nbre_slice * nbre_wrap, GM,
                                                                            dirs_name)
    else:
        fname_im, fname_im_seg, dirs_name_im = random_list(list_data, nbre_slice * nbre_wrap, 0, dirs_name)
    if verbose:
        elapsed_time = time.time() - start_time
        sct.printv(
            '\nElapsed time to create a random list of image to tranform: ' + str(int(round(elapsed_time))) + 's')

    # crop images to have the same dimension
    tmp_path = sct.tmp_create()
    fname_src, fname_dest, fname_src_seg, fname_dest_seg = crop_image_to_same_dimension(fname_src, fname_dest, fname_src_seg, fname_dest_seg, dirs_name_src, dirs_name_dest, tmp_path)
    fname_im, fname_im_seg = crop_image_to_same_dimension_2(fname_src, fname_im, fname_im_seg, dirs_name_im, tmp_path)

    # create warping fields beetween source and destination images
    start_time = time.time()

    from sct_data_augmentation import worker_warping_result, warping_field
    if cpu_number != 0:
        pool = Pool(cpu_number)
        arguments = [(fname_src_seg[iter], fname_dest_seg[iter], v, iter) for iter in range(0, nbre_wrap)]
        r = pool.map_async(worker_warping_field, arguments)
        try:
            pool.close()
            pool.join()
            warp = worker_warping_result(r)
        except KeyboardInterrupt:
            print "\nWarning: Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            sys.exit(2)
        except Exception as e:
            print "Error during straightening on line {}".format(sys.exc_info()[-1].tb_lineno)
            print e
            sys.exit(2)
    else:
        print nbre_wrap
        warp = warping_field(fname_src_seg, fname_dest_seg, nbre_wrap, v)
    if verbose:
        elapsed_time = time.time() - start_time
        sct.printv('\nElapsed time to create a list of warping field: ' + str(int(round(elapsed_time))) + 's')

    # Apply warping fields to random images
    start_time = time.time()
    if GM:
        arguments = [(fname_im, fname_im_seg, fname_dest[iter],  nbre_slice, output_folder_path, warp[iter], iter * nbre_slice, v, fname_im_GM, dirs_name_src[iter], dirs_name_dest[iter], dirs_name_im) for iter in range(0, nbre_wrap)]
    else:
        arguments = [(fname_im, fname_im_seg, fname_dest[iter], nbre_slice, output_folder_path, warp[iter], iter * nbre_slice, v, None, dirs_name_src[iter], dirs_name_dest[iter], dirs_name_im) for iter in range(0, nbre_wrap)]
    if cpu_number != 0:
        pool = Pool(cpu_number)
        pool.map(worker_apply_warping_field, arguments)
        try:
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print "\nWarning: Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            sys.exit(2)
        except Exception as e:
            print "Error during straightening on line {}".format(sys.exc_info()[-1].tb_lineno)
            print e
            sys.exit(2)
    else:
        if GM:
            apply_warping_field(fname_im, fname_im_seg, fname_src, fname_dest, nbre_slice, nbre_wrap, output_folder_path, warp, v, fname_im_GM, dirs_name_src, dirs_name_dest, dirs_name_im)

        else:
            apply_warping_field(fname_im, fname_im_seg, fname_src, fname_dest, nbre_slice, nbre_wrap, output_folder_path, warp, v, None, dirs_name_src, dirs_name_dest, dirs_name_im)

    if verbose:
        elapsed_time = time.time() - start_time
        sct.printv(
            '\nElapsed time to apply the warping field to random images: ' + str(int(round(elapsed_time))) + 's')

    # Delete warping files
    # for iter in range(0, nbre_wrap):
     #   sct.run('rm -rf ' + warp[iter], verbose=v)
      #  sct.run('rm -rf ' + str(iter) + '0InverseWarp.nii.gz', verbose=v)

    # Total time
    if verbose:
        elapsed_time = time.time() - start_time_total
        sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's')


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()