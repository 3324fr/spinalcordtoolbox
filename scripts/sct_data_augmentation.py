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


import time
from msct_parser import Parser
import sys
import os
import numpy as np
import sct_utils as sct
from msct_image import Image
import random
from multiprocessing import Pool

#MAIN

def main():
    # Initialisation
    start_time_total = time.time()
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

    # Initialise the parser
    parser = Parser(__file__)
    parser.add_option(name="-f",
                      type_value="folder",
                      description="folder of input image and segmentation",
                      mandatory=False,
                      example="data_patients")
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image source.",
                      mandatory=False,
                      example="src.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="Image destination.",
                      mandatory=False,
                      example="dest.nii.gz")
    parser.add_option(name="-iseg",
                      type_value="file",
                      description="Segmentation source.",
                      mandatory=False,
                      example="src_seg.nii.gz")
    parser.add_option(name="-dseg",
                      type_value="file",
                      description="Segmentation destination.",
                      mandatory=False,
                      example="dest_seg.nii.gz")
    parser.add_option(name="-t",
                      type_value="file",
                      description="image which will be modified",
                      mandatory=False,
                      example="im.nii.gz")
    parser.add_option(name="-tseg",
                      type_value="file",
                      description="Segmentation of the image wich will be modified.",
                      mandatory=False,
                      example="dest_seg.nii.gz")
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
                      description="nameof the output folder",
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
    parser.add_option(name="-all-slices",
                      type_value="multiple_choice",
                      description=" ",
                      mandatory=False,
                      default_value='0',
                      example=['0','1'])
    parser.add_option(name="-center-seg",
                      type_value="multiple_choice",
                      description=" ",
                      mandatory=False,
                      default_value='0',
                      example=['0', '1'])
    parser.add_option(name="-GM",
                      type_value="multiple_choice",
                      description=" ",
                      mandatory=False,
                      default_value='0',
                      example=['0','1'])
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
    if '-all-slices' in arguments:
        all_slices = int(arguments['-all-slices'])
    if '-center-seg' in arguments:
        center_seg = int(arguments['-center-seg'])
    if '-preprocess' in arguments:
        preprocess = int(arguments['-preprocess'])

    # create a folder with all the images
    list_data, dirs_name = extract_name_list(folder_path, GM, preprocess)

    from sct_data_augmentation_vertebra import extract_label_list

    list_image_labels, data_list, dirs_name = extract_label_list(folder_path, list_data, dirs_name)

    print list_image_labels
    print data_list
    print dirs_name

    # resample data to 1x1x1 mm^3 and crop them along the segmentation
    preprocess(list_data, folder_path, cpu_number, GM, dim_list, verbose)

    # crop and slice all these images
    if all_slices == 1:

        if not GM :
            for iter in range(0,len(list_data) / 2 ):
                path, file, ext = sct.extract_fname(str(list_data[iter * 2]))

                # get the number of the slice in the source image and crop it
                if not sct.check_folder_exist("Slices_" + file, verbose = 0):
                    os.makedirs("Slices_" + file)

                nx, ny, nz, nt, px, py, pz, pt = Image(str(list_data[iter * 2])).dim
                arguments = [(str(list_data[iter * 2]), str(list_data[iter * 2 + 1]), dim_list[0], dim_list[1], v, i, center_seg, None, None) for i in range(1,nz)]

                if cpu_number != 0:
                    try:
                        pool = Pool(cpu_number)
                        pool.map(worker_slice_and_crop, arguments)
                        print "OK"
                        pool.close()
                    except KeyboardInterrupt:
                        print "\nWarning: Caught KeyboardInterrupt, terminating workers"
                        pool.terminate()
                        sys.exit(2)
                    except Exception as e:
                        print "Error during straightening on line {}".format(sys.exc_info()[-1].tb_lineno)
                        print e
                        sys.exit(2)
                else:
                    for i in range(1, nz):
                        worker_slice_and_crop(str(list_data[iter * 2]), str(list_data[iter * 2 + 1]), dim_list[0], dim_list[1], v, i, center_seg, None, dirs_name[iter])
        if GM:
            for iter in range(0, len(list_data) / 3):

                # get the number of the slice in the source image and crop it
                if not sct.check_folder_exist("Slices_" + dirs_name[iter], verbose=0):
                    os.makedirs("Slices_" + dirs_name[iter])

                nx, ny, nz, nt, px, py, pz, pt = Image(folder_path + '/' + str(list_data[iter * 3])).dim
                arguments = [(str(list_data[iter * 3]),str(list_data[iter * 3 + 1]), dim_list[0], dim_list[1], v, i, center_seg, str(list_data[iter * 3 + 2]), dirs_name[iter]) for i in range(1, nz)]

                if cpu_number != 0:
                    try:
                        pool = Pool(cpu_number)
                        pool.map(worker_slice_and_crop, arguments)
                        pool.close()
                    except KeyboardInterrupt:
                        print "\nWarning: Caught KeyboardInterrupt, terminating workers"
                        pool.terminate()
                        sys.exit(2)
                    except Exception as e:
                        print "Error during straightening on line {}".format(sys.exc_info()[-1].tb_lineno)
                        print e
                        sys.exit(2)
                else:
                    for i in range(1, nz):
                        worker_slice_and_crop(str(list_data[iter * 2]), str(list_data[iter * 2 + 1]), dim_list[0], dim_list[1], v, i, center_seg)

    if nbre_slice != 0 and nbre_wrap != 0 :
        # choose randomly a source and a destination
        fname_src = []
        fname_src_seg =[]
        fname_dest = []
        fname_dest_seg = []
        fname_im = []
        fname_im_seg = []
        fname_dest_GM = []
        fname_src_GM = []
        fname_im_GM = []


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
            sct.printv('\nElapsed time to create a random list of destination image: ' + str(int(round(elapsed_time))) + 's')

        start_time = time.time()
        if GM:
            fname_im, fname_im_seg, fname_im_GM, dirs_name_im = random_list(list_data, nbre_slice * nbre_wrap, GM, dirs_name)
        else:
            fname_im, fname_im_seg, dirs_name_im = random_list( list_data, nbre_slice * nbre_wrap, 0, dirs_name)
        if verbose:
            elapsed_time = time.time() - start_time
            sct.printv('\nElapsed time to create a random list of image to tranform: ' + str(int(round(elapsed_time))) + 's')

        # choose a random slice in each source image
        start_time = time.time()
        if GM :
            arguments_slice_src = [(fname_src[iter], fname_src_seg[iter], dim_list[0], dim_list[1], folder_path, v, center_seg, fname_src_GM[iter], dirs_name_src[iter]) for iter in range(0, nbre_wrap)]
            arguments_slice_dest = [(fname_dest[iter], fname_dest_seg[iter], dim_list[0], dim_list[1], folder_path, v, center_seg, fname_dest_GM[iter], dirs_name_dest[iter]) for iter in range(0, nbre_wrap)]
            arguments_slice_im = [(fname_im[iter], fname_im_seg[iter], dim_list[0], dim_list[1], folder_path, v, center_seg, fname_im_GM[iter], dirs_name_im[iter]) for iter in range(0, (nbre_wrap * nbre_slice))]
        else:
            arguments_slice_src = [(fname_src[iter],fname_src_seg[iter],dim_list[0], dim_list[1], folder_path, v, center_seg, None, dirs_name_src[iter]) for iter in range(0, nbre_wrap)]
            arguments_slice_dest = [(fname_dest[iter], fname_dest_seg[iter], dim_list[0], dim_list[1], folder_path, v, center_seg, None, dirs_name_dest[iter]) for iter in range(0, nbre_wrap)]
            arguments_slice_im = [(fname_im[iter], fname_im_seg[iter], dim_list[0], dim_list[1], folder_path, v, center_seg, None, dirs_name_im[iter]) for iter in range(0, (nbre_wrap*nbre_slice))]
        if cpu_number != 0:
            pool = Pool(cpu_number)
            r1 = pool.map_async(worker_random_slice, arguments_slice_src)
            r2 = pool.map_async(worker_random_slice, arguments_slice_dest)
            r3 = pool.map_async(worker_random_slice, arguments_slice_im)
            try:
                pool.close()
                pool.join()  # waiting for all the jobs to be done
                if GM:
                    src, src_seg, nbre_src, src_GM = worker_result(r1)
                    dest, dest_seg, nbre_dest, dest_GM = worker_result(r2)
                    im, im_seg, nbre_im, im_GM = worker_result(r3)
                else:
                    src, src_seg, nbre_src = worker_result(r1)
                    dest, dest_seg, nbre_dest = worker_result(r2)
                    im, im_seg, nbre_im = worker_result(r3)
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
                src, src_seg, nbre_src, src_GM = random_slice(fname_src, fname_src_seg, dim_list[0], dim_list[1], folder_path, nbre_wrap, v, center_seg, fname_src_GM, dirs_name_src)
                dest, dest_seg, nbre_dest, dest_GM = random_slice(fname_dest, fname_dest_seg, dim_list[0], dim_list[1], folder_path, nbre_wrap, v, center_seg, fname_dest_GM, dirs_name_dest)
                im, im_seg, nbre_im, im_GM = random_slice(fname_im, fname_im_seg, dim_list[0], dim_list[1], folder_path, nbre_wrap * nbre_slice, v, center_seg, fname_im_GM, dirs_name_im)
            else:
                src, src_seg, nbre_src = random_slice(fname_src,fname_src_seg, dim_list[0], dim_list[1], folder_path, nbre_wrap, v, center_seg, None, None)
                dest, dest_seg, nbre_dest = random_slice(fname_dest, fname_dest_seg, dim_list[0], dim_list[1], folder_path, nbre_wrap, v, center_seg, None, None)
                im, im_seg, nbre_im = random_slice(fname_im, fname_im_seg, dim_list[0], dim_list[1], folder_path, nbre_wrap*nbre_slice, v, center_seg, None, None)

        if verbose :
            elapsed_time = time.time() - start_time
            sct.printv('\nElapsed time to create a random lists of images: ' + str(int(round(elapsed_time))) + 's')

        # create warping fields beetween source and destination images
        start_time = time.time()

        if cpu_number != 0:
            pool = Pool(cpu_number)
            arguments = [(src_seg[iter], dest_seg[iter], v, iter) for iter in range(0,nbre_wrap)]
            r=pool.map_async(worker_warping_field, arguments)
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
            warp = warping_field(src_seg, dest_seg, nbre_wrap, v)
        if verbose :
            elapsed_time = time.time() - start_time
            sct.printv('\nElapsed time to create a list of warping field: ' + str(int(round(elapsed_time))) + 's')

        # Apply warping fields to random images
        start_time = time.time()
        if GM:
            arguments = [(im, im_seg, src[iter], nbre_src[iter], dest[iter], nbre_dest[iter], nbre_slice, output_folder_path, warp[iter], nbre_im, iter * nbre_slice, v, im_GM, dirs_name_src[iter], dirs_name_dest[iter], dirs_name_im) for iter in range(0, nbre_wrap)]
        else:
            arguments = [(im, im_seg, src[iter], nbre_src[iter], dest[iter], nbre_dest[iter], nbre_slice, output_folder_path, warp[iter], nbre_im, iter*nbre_slice, v, None, dirs_name_src[iter], dirs_name_dest[iter], dirs_name_im) for iter in range(0, nbre_wrap)]
        if cpu_number != 0:
            pool = Pool(cpu_number)
            pool.map(worker_apply_warping_field, arguments)
            pool.close()
            pool.join()
        else:
            if GM:
                apply_warping_field(im, im_seg, src, dest, nbre_slice, nbre_wrap, output_folder_path, warp, nbre_im, v, im_GM, dirs_name_src, dirs_name_dest, dirs_name_im)

            else:
                apply_warping_field(im, im_seg, src, dest, nbre_slice, nbre_wrap, output_folder_path, warp, nbre_im, v, None, dirs_name_src, dirs_name_dest, dirs_name_im)


        if verbose :
            elapsed_time = time.time() - start_time
            sct.printv('\nElapsed time to apply the warping field to random images: ' + str(int(round(elapsed_time))) + 's')

        # Delete warping files
        for iter in range(0, nbre_wrap):
            sct.run('rm -rf ' + warp[iter] , verbose=v)
            sct.run('rm -rf ' + str(iter) + '0InverseWarp.nii.gz', verbose=v)

        # Total time
        if verbose :
            elapsed_time = time.time() - start_time_total
            sct.printv('\nFinished! Elapsed time: ' + str(int(round(elapsed_time))) + 's')

# Extract a list of image, seg, and GM from the input directory
# ==========================================================================================
def extract_name_list(folder_path, GM, preprocess):
    list_data = []
    dirs_name = []

    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir.find("t2s") == -1 and dir.find("t2") == -1 and dir.find("mt") == -1 and dir.find("dmri") == -1:
                dirs_name.append(dir)
    if preprocess:
        get_gmseg_from_multilabel(folder_path)

    if len(dirs_name) == 0:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.nii.gz') and file.find('_seg') == -1 and file.find('_gmseg') == -1:
                    pos_t2 = file.find('_t2')
                    subject_name, end_name = file[0:pos_t2 + 3], file[pos_t2 + 3:]
                    file_seg = subject_name + '_seg' + end_name
                    list_data.append(folder_path + '/' + file)
                    list_data.append(folder_path + '/' +file_seg)
                    dirs_name.append(dirs)
    else:
        for iter, dir in enumerate(dirs_name):
            for root, dirs, files in os.walk(folder_path + '/' + dir):
                for file in files:
                    if file.endswith('t2s.nii.gz') and file.find('_gmseg') == -1 and file.find('_seg') == -1:
                        pos_t2 = file.find('t2s')
                        subject_name, end_name = file[0:pos_t2 + 3], file[pos_t2 + 3:]
                        file_seg = dir + '/t2s/' + subject_name + '_seg_manual' + end_name
                        if GM:
                            file_gmseg = dir + '/t2s/' + subject_name + '_gmseg_manual' + end_name
                        file = dir + '/t2s/' + file
                        list_data.append(folder_path + '/' + file)
                         # Get the number of total slices
                        #nx, ny,nz, nt ,px, py, pz, pt = Image(folder_path + '/' + file).dim
                        #sum += nz
                        list_data.append(folder_path + '/' + file_seg)
                        if GM:
                            list_data.append(folder_path + '/' + file_gmseg)
    return list_data, dirs_name


# resample the data, crop the image along the segmentation
# ==========================================================================================
def preprocess(list_data, folder_path, cpu_number,GM, dim_list, verbose ):

    # resample data to 1x1x1 mm^3 and crop them along the segmentation
    if preprocess:
        start_time = time.time()

        if GM:
            arguments = [(list_data[iter * 3], list_data[iter * 3 + 1], v, list_data[iter * 3 + 2], folder_path,
                          dim_list[2], dim_list[3]) for iter in range(0, len(list_data) / 3)]
        else:
            arguments = [(list_data[iter * 2], list_data[iter * 2 + 1], v, list_data[iter * 2 + 1], folder_path,
                          dim_list[2], dim_list[3]) for iter in range(0, len(list_data) / 2)]

        if cpu_number != 0:
            pool = Pool(cpu_number)
            pool.map_async(worker_resized_and_crop, arguments)
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
                for iter in range(0, len(list_data) / 3):
                    arguments_src = (
                    list_data[iter * 3], list_data[iter * 3 + 1], v, list_data[iter * 3 + 2], folder_path, dim_list[2],
                    dim_list[3])
                    worker_resized_and_crop(arguments_src)
            else:
                for iter in range(0, len(list_data) / 2):
                    arguments_src = (
                    list_data[iter * 2], list_data[iter * 2 + 1], v, list_data[iter * 2 + 2], folder_path, dim_list[2],
                    dim_list[3])
                    worker_resized_and_crop(arguments_src)

        from sct_image import get_orientation_3d
        for i in range(0, len(list_data)):
            print str(get_orientation_3d(Image(folder_path + '/' + list_data[i])))
            if not str(get_orientation_3d(Image(folder_path + '/' + list_data[i]))) == "RPI":
                sct.run("sct_image -i " + folder_path + '/' + list_data[i] + " -setorient RPI -o " + folder_path + '/' +
                        list_data[i], verbose=v)
                print "change orient"

        if verbose:
            elapsed_time = time.time() - start_time
            sct.printv('\nElapsed time to crop these images and resized them: ' + str(int(round(elapsed_time))) + 's')


# Extract the GM segmentation from the multilabeled one
# ==========================================================================================
def get_gmseg_from_multilabel(path):
    import os
    from msct_image import Image

    path_data = path
    contrast = 't2s'
    fname_multilabel = 't2s_gmseg_manual_rater_unf.nii.gz'

    fname_gm = 't2s_gmseg_manual.nii.gz'
    fname_sc = 't2s_seg_manual.nii.gz'

    for sub in os.listdir(path_data):
        if os.path.isdir(path_data + '/'+ sub): # and sub.find(".DS_Store") == -1:
            if sub.find('challenge') != -1:
                sub_id = sub.split('_')[1]
                if sub.find('pain') != -1:
                    sub_id = sub.split('_')[2]
                # get multi-label image
                im_ml = Image(path_data + '/' + sub + '/' + contrast + '/' + fname_multilabel)
                # GM = 2, WM= 1
                if int(sub_id) <= 10:
                    # get GM:
                    im_gm = im_ml.copy()
                    im_gm.data[im_gm.data == 2] = 0
                    im_gm.setFileName(path_data + '/' + sub + '/' + contrast + '/' + fname_gm)
                    im_gm.save()
                    # get SC
                    im_sc = im_ml.copy()
                    im_sc.data[im_sc.data > 0] = 1
                    im_sc.setFileName(path_data + '/' + sub + '/' + contrast + '/' + fname_sc)
                    im_sc.save()
                # GM = 1, WM = 2
                elif int(sub_id) > 10:
                    # get GM:
                    im_gm = im_ml.copy()
                    im_gm.data[im_gm.data == 1] = 0
                    im_gm.data[im_gm.data == 2] = 1
                    im_gm.setFileName(path_data + '/' + sub + '/' + contrast + '/' + fname_gm)
                    im_gm.save()
                    # get SC
                    im_sc = im_ml.copy()
                    im_sc.data[im_sc.data > 0] = 1
                    im_sc.setFileName(path_data + '/' + sub + '/' + contrast + '/' + fname_sc)
                    im_sc.save()


# Return a list of random images and segmentation for a given folder
# ==========================================================================================
def random_list(list_data, nw, GM = None, dirs_name = None):
    list=[]
    list_seg=[]
    list_GM = []
    list_dirs_name = []
    if not GM:
        for iter in range(0,nw):
            random_index = random.randrange(0,len(list_data) / 2 )
            fname_src = str(list_data[random_index * 2])
            fname_src_seg = str(list_data[random_index * 2 + 1])
            list.append(fname_src)
            list_seg.append(fname_src_seg)
            dirs = dirs_name[random_index]
            list_dirs_name.append(dirs)
        return list, list_seg, list_dirs_name

    if GM :
        for iter in range(0, nw):
            random_index = random.randrange(0, len(list_data) / 3)
            fname_src =str(list_data[random_index * 3])
            fname_src_seg = str(list_data[random_index * 3 + 1])
            fname_src_GM = str(list_data[random_index * 3 + 2])
            list.append(fname_src)
            list_seg.append(fname_src_seg)
            list_GM.append(fname_src_GM)
            dirs = dirs_name[random_index]
            list_dirs_name.append(dirs)
        return list, list_seg, list_GM, list_dirs_name


# Crop the image to stay in the boundaries of the segmentation
# ==========================================================================================
def crop_segmentation(fname, fname_seg, v, fname_GM):

    if fname_GM == None:
        nx, ny, nz, nt, px, py, pz, pt = Image(fname_seg).dim
        data_seg = Image(fname_seg).data
        if np.all(data_seg[:,:,0] == np.zeros((nx,ny))) or np.all(data_seg[:,:,nz-1] == np.zeros((nx,ny))):
            i = 1
            while np.all(data_seg[:, :, i] == np.zeros((nx,ny))):
                i += 1
            j = nz -1
            while np.all(data_seg[:, :, j] == np.zeros((nx,ny))):
                j -= 1

            sct.run("sct_crop_image -i " + fname + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname, verbose= v )
            sct.run("sct_crop_image -i " + fname_seg + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname_seg, verbose= v )

    if fname_GM:
        nx, ny, nz, nt, px, py, pz, pt = Image(fname_GM).dim
        data_seg = Image(fname_GM).data
        if np.all(data_seg[:,:,0] == np.zeros((nx,ny))) or np.all(data_seg[:,:,nz-1] == np.zeros((nx,ny))):
            i = 1
            while np.all(data_seg[:, :, i] == np.zeros((nx, ny))):
                i += 1
            j = nz - 1
            while np.all(data_seg[:, :, j] == np.zeros((nx, ny))):
                j -= 1

            sct.run("sct_crop_image -i " + fname + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname,verbose=v)
            sct.run("sct_crop_image -i " + fname_seg + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname_seg,verbose=v)
            sct.run("sct_crop_image -i " + fname_GM + " -dim 2 -start " + str(i) + " -end " + str(j) + " -o " + fname_GM, verbose=v)

# Resized and crop the image
# ==========================================================================================
def worker_resized_and_crop(argument):

    fname, fname_seg, v, fname_GM, folder_path, size_px, size_py = argument
    fname = folder_path + '/' + fname
    fname_seg = folder_path + '/' + fname_seg
    fname_GM = folder_path + '/' + fname_GM

    # resample data to 1x1x1 mm^3
    try:

        nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
        if px != size_px or py != size_py:
            sct.run('sct_resample -i ' + fname + ' -mm ' + str(size_px) + 'x' + str(size_py) + 'x' + str(pz) + ' -x nn -o ' + fname, verbose=v)
        nx_s ,ny_s, nz_s, nt_s, px_s, py_s, pz_s, pt_s = Image(fname_seg).dim
        if px_s != size_px or py_s != size_py:
            sct.run('sct_resample -i ' + fname_seg + ' -mm ' + str(size_px) + 'x' + str(size_py) + 'x' + str(pz_s) + ' -x nn -o ' + fname_seg, verbose=v)
        if fname_GM :
            nx_g, ny_g, nz_g, nt_g, px_g, py_g, pz_g, pt_g = Image(fname_GM).dim
            if px_g != size_px or py_g != size_py:
                sct.run('sct_resample -i ' + fname_GM + ' -mm ' + str(size_px) + 'x' + str(size_py) + 'x' + str(pz_g) + ' -x nn -o ' + fname_GM, verbose=v)

        sct.run('sct_image  -i ' + fname + ' -copy-header ' + fname_seg)
        if fname_GM:
            sct.run('sct_image  -i ' + fname + ' -copy-header ' + fname_GM)

        if fname_GM:
            crop_segmentation(fname, fname_seg, v, fname_GM)
        else:
            crop_segmentation(fname, fname_seg, v, fname_GM = None)
    except KeyboardInterrupt:
        return
    except Exception as e:
        raise e


# Crop the image along z, centered  it and crop it along the two others dimensions with a given size (for each slide)
# ==========================================================================================
def worker_slice_and_crop(argument):
    fname, fname_seg, fname_size_x, fname_size_y, v, nbre, center_seg, fname_GM, dirs_name = argument
    try:
        from msct_image import crop_x_y
        # extract path file and extension
        path, file, ext = sct.extract_fname(fname)
        if len(dirs_name) == 0:
            dirs_name = file
        current_path = os.getcwd()
        # crop the image along the z dimension
        sct.run("sct_crop_image -i " + fname + " -dim 2 -start " + str(nbre) + " -end " + str(nbre) + " -o " + current_path + '/' + "Slices_" + dirs_name + '/' +file + "_slice_" + str(nbre) + ".nii.gz", verbose=v)
        sct.run("sct_crop_image -i " + fname_seg + " -dim 2 -start " + str(nbre) + " -end " + str(nbre) + " -o " + current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(nbre) + "_seg.nii.gz", verbose=v)
        if fname_GM:
            sct.run("sct_crop_image -i " + fname_GM + " -dim 2 -start " + str(nbre) + " -end " + str(nbre) + " -o " + current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(nbre) + "_gmseg.nii.gz", verbose=v)

        if center_seg:
            nbre_rep = 1
        else:
            nbre_rep = 10
        for iter in range(0,nbre_rep):
            sub = "_" + str(iter)
            if center_seg:
                sub = ""
            fname_slice = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(nbre) + ".nii.gz"
            fname_slice_seg = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(nbre) + "_seg.nii.gz"

            output_slice = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_crop_" + str(nbre) + sub  + "_o0.nii.gz"
            output_seg = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_crop_" + str(nbre) + sub  + "_seg_o0.nii.gz"

            if fname_GM :
                fname_slice_GM = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(nbre) + "_gmseg.nii.gz"
                output_GM = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_crop_" + str(nbre) + sub  + "_gmseg_o0.nii.gz"
                crop_x_y(fname_slice, fname_slice_seg, fname_size_x, fname_size_y, output_slice, output_seg, center_seg, fname_slice_GM, output_GM)

            else:
                crop_x_y(fname_slice,fname_slice_seg, fname_size_x, fname_size_y, output_slice, output_seg, center_seg)

        if fname_GM:
            sct.run('rm -f ' + fname_slice_GM, verbose=v)
        sct.run('rm -rf ' + fname_slice, verbose=v)
        sct.run('rm -rf ' + fname_slice_seg, verbose=v)

    except KeyboardInterrupt:
        return
    except Exception as e:
        raise e


# Crop the image along z, centered it ans crop it along the two others dimensions with a given size (for a given number of random slides)
# ==========================================================================================
def worker_random_slice(arguments_worker):
    from msct_image import crop_x_y
    fname, fname_seg, fname_size_x, fname_size_y, folder_path, v, center_seg, fname_GM, dirs_name = arguments_worker
    try:
        path, file, ext = sct.extract_fname(fname)
        if len(dirs_name) == 0 :
            dirs_name = file
        # get the number of the slice in the source image and crop it
        if not sct.check_folder_exist("Slices_" + str(dirs_name), verbose=0):
            os.makedirs("Slices_" + str(dirs_name))
        nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
        nbre_im = random.randrange(1, nz )
        im = "Slices_" + dirs_name + '/' + file + "_crop_" + str(nbre_im) + "_o0.nii.gz"

        if not os.path.exists(im):
            current_path = os.getcwd()
            slice_z = nbre_im
            # crop the image along the z dimension
            sct.run("sct_crop_image -i " + fname + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(slice_z) + ".nii.gz", verbose=v)
            sct.run("sct_crop_image -i " + fname_seg + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(slice_z) + "_seg.nii.gz",verbose=v)
            if fname_GM:
                sct.run("sct_crop_image -i " + fname_GM + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(slice_z) + "_gmseg.nii.gz", verbose=v)

            fname_slice = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(slice_z) + ".nii.gz"
            fname_slice_seg = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(slice_z) + "_seg.nii.gz"
            output_slice = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_crop_" + str(slice_z) + "_o0" + ".nii.gz"
            output_seg = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_crop_" + str(slice_z) + "_seg_o0" + ".nii.gz"

            if fname_GM :
                fname_slice_GM = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_slice_" + str(slice_z) + "_gmseg.nii.gz"
                output_GM = current_path + '/' + "Slices_" + dirs_name + '/' + file + "_crop_" + str(slice_z) + "_gmseg_o0.nii.gz"
                crop_x_y(fname_slice, fname_slice_seg, fname_size_x, fname_size_y, output_slice, output_seg, center_seg, fname_slice_GM, output_GM)
                sct.run('rm -rf ' + fname_slice_GM, verbose=v)
            else:
                crop_x_y(fname_slice, fname_slice_seg, fname_size_x, fname_size_y, output_slice, output_seg, center_seg)
            sct.run('rm -rf ' + fname_slice, verbose=v)
            sct.run('rm -rf ' + fname_slice_seg, verbose=v)

        im = "Slices_" + dirs_name + '/'+ file + "_crop_" + str(nbre_im) + "_o0.nii.gz"
        im_seg = "Slices_" + dirs_name + '/'+ file + "_crop_" + str(nbre_im) + "_seg_o0.nii.gz"
        if fname_GM :
            im_GM = "Slices_" + dirs_name + '/'+ file + "_crop_" + str(nbre_im) + "_gmseg_o0.nii.gz"

        else :
            im_GM = None

        return im, im_seg, nbre_im, im_GM

    except KeyboardInterrupt:
        return

    except Exception as e:
        raise e


# Get random slices and their place in a given image
# ==========================================================================================
def random_slice(fname, fname_seg, fname_size_x, fname_size_y, folder_path,nbre_wrap, v, center_seg, fname_GM, dirs_name):

    from msct_image import crop_x_y
    im_list = [] ; im_seg_list = [] ; nbre_im_list = [] ; im_list_GM = []
    for iter in range(0, nbre_wrap):
        path, file, ext = sct.extract_fname(fname[iter])

        if not fname_GM:
            dirs_name[iter] = file

        # change the orientation to RPI
        from sct_image import get_orientation_3d
        if not get_orientation_3d(Image(fname[iter])) == 'RPI':
            sct.run("sct_image -i " + fname[iter] + " -setorient RPI -o " + fname[iter], verbose=v)
            sct.run("sct_image -i " + fname_seg[iter] + " -setorient RPI -o " + fname_seg[iter], verbose=v)
            if fname_GM:
                sct.run("sct_image -i " + fname_GM[iter] + " -setorient RPI -o " + fname_GM[iter], verbose=v)

        # get the number of the slice in the source image and crop it
        if not sct.check_folder_exist("Slices_" + dirs_name[iter], verbose = 0):
            os.makedirs("Slices_" + dirs_name[iter])
        nx, ny, nz, nt, px, py, pz, pt = Image(fname[iter]).dim
        nbre = random.randrange(1, nz )
        im = "Slices_" + dirs_name[iter] + '/' + file + "_crop_" + str(nbre) + "_o0.nii.gz"
        if not os.path.isfile(im):
            current_path = os.getcwd()
            slice_z = nbre
            # crop the image along the z dimension
            sct.run("sct_crop_image -i " + fname[iter] + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_slice_" + str(slice_z) + ".nii.gz", verbose=v)
            sct.run("sct_crop_image -i " + fname_seg[iter] + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_slice_" + str(slice_z) + "_seg.nii.gz", verbose=v)
            if fname_GM:
                sct.run("sct_crop_image -i " + fname_GM[iter] + " -dim 2 -start " + str(slice_z) + " -end " + str(slice_z) + " -o " + current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_slice_" + str(slice_z) + "_gmseg.nii.gz", verbose=v)

            from numpy import asarray
            fname_slice = current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_slice_" + str(slice_z) + ".nii.gz"
            fname_slice_seg = current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_slice_" + str(slice_z) + "_seg.nii.gz"
            output_slice = current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_crop_" + str(slice_z) + "_o0" + ".nii.gz"
            output_seg = current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_crop_" + str(slice_z) + "_seg_o0" + ".nii.gz"
            if fname_seg:
                fname_slice_GM = current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_slice_" + str(slice_z) + "_gmseg.nii.gz"
                output_GM = current_path + '/' + "Slices_" + dirs_name[iter] + '/' + file + "_crop_" + str(slice_z) + "_gmseg_o0" + ".nii.gz"
                crop_x_y(fname_slice, fname_slice_seg, fname_size_x, fname_size_y, output_slice, output_seg, center_seg, fname_slice_GM, output_GM)
            else:
                crop_x_y(fname_slice, fname_slice_seg, fname_size_x, fname_size_y, output_slice, output_seg, center_seg)

            sct.run('rm -rf ' + fname_slice)
            sct.run('rm -rf ' + fname_slice_seg)
            im = "Slices_" + dirs_name[iter] + '/' + file + "_crop_" + str(nbre) + "_o0.nii.gz"
            im_seg = "Slices_" + dirs_name[iter] + '/' + file + "_crop_" + str(nbre) + "_seg_o0.nii.gz"

            im_list.append(im)
            im_seg_list.append(im_seg)
            nbre_im_list.append(nbre)

            if fname_GM:
                sct.run('rm -rf ' + fname_slice_GM)
                im_GM = "Slices_" + dirs_name[iter] + '/' + file + "_crop_" + str(nbre) + "_gmseg_o0.nii.gz"
                im_list_GM.append(im_GM)

    if fname_GM:
        return im_list, im_seg_list, nbre_im_list, im_list_GM

    return im_list, im_seg_list, nbre_im_list, None


# callback function of the random slice worker
# ==========================================================================================
def worker_result(results):
    im_list = [] ; im_seg_list = [] ; nbre_im_list = [] ;  im_GM_list = []
    for im, im_seg, nbre_im, im_GM in results.get():
        if im_GM :
            im_GM_list.append(im_GM)
        im_list.append(im)
        im_seg_list.append(im_seg)
        nbre_im_list.append(nbre_im)
    if im_GM :
        return im_list, im_seg_list, nbre_im_list, im_GM_list
    else:
        return im_list, im_seg_list, nbre_im_list


# Create warping field
# ==========================================================================================
def warping_field(src_seg, dest_seg, nw, v):
    warp = []

    for iter in range(0, nw):
        out = "t2_output_image_transformed.nii.gz"
        out_transfo = str(iter)

        sct.run('isct_antsRegistration ' +
                '--dimensionality 2 ' +
                '--transform BSplineSyN[0.5,1,3] ' +
                '--metric MeanSquares[' + dest_seg[iter] + ',' + src_seg[iter] + ', 1] ' +
                '--convergence 5x3 ' +
                '--shrink-factors 2x1 ' +
                '--smoothing-sigmas 1x0mm ' +
                '--output [' + out_transfo + ',' + out + '] ' +
                '--interpolation BSpline[3] ' +
                '--verbose 0', verbose = v)

        warp.append(out_transfo + '0Warp.nii.gz')
    return warp


# Create warping field (multiprocess)
# ==========================================================================================
def worker_warping_field(argument):
    src_seg, dest_seg, v, iter = argument

    out = "t2_output_image_transformed.nii.gz"
    out_transfo = str(iter)

    sct.run('isct_antsRegistration ' +
            '--dimensionality 2 ' +
            '--transform BSplineSyN[0.5,1,3] ' +
            '--metric MeanSquares[' + dest_seg + ',' + src_seg + ', 1] ' +
            '--convergence 5x3 ' +
            '--shrink-factors 2x1 ' +
            '--smoothing-sigmas 1x0mm ' +
            '--output [' + out_transfo + ',' + out + '] ' +
            '--interpolation BSpline[3] ' +
            '--verbose 0', verbose=v)

    warp = out_transfo + '0Warp.nii.gz'

    return warp


# callback function of the random slice worker
# ==========================================================================================
def worker_warping_result(results):
    warp = []
    for r in results.get():
        warp.append(r)
    return warp


# Apply warping field
# ==========================================================================================
def apply_warping_field(im, im_seg, src, dest, nbre_slice, nbre_wrap,  output_folder_path, wrap, nbre_im, v, im_GM, dirs_name_src, dirs_name_dest, dirs_name_im):
    for i in range(0, nbre_wrap):
        src_path, src_file, src_ext = sct.extract_fname(src[i])
        dest_path, dest_file, dest_ext = sct.extract_fname(dest[i])
        for iter in range(0, nbre_slice):
            im_path, im_file, im_ext = sct.extract_fname(im[i*iter +iter])
            fname_out_im = dirs_name_src[i] + '_' + src_file + '_' + dirs_name_dest[i] + '_' + dest_file + '_' + dirs_name_im[i*iter +iter] + im_file + str(nbre_im[i*iter + iter]) + '.nii.gz'
            fname_out_seg = dirs_name_src[i] + '_' + src_file + '_' + dirs_name_dest[i] + '_' + dest_file + '_' + dirs_name_im[i*iter +iter] + im_file + str(nbre_im[i*iter + iter]) + '_seg.nii.gz'
            fname_out_GM = dirs_name_src[i] + '_' + src_file + '_' + dirs_name_dest[i] + '_' + dest_file + '_' + dirs_name_im[i*iter +iter] + im_file + str(nbre_im[i*iter + iter]) + '_gmseg.nii.gz'
            # Apply warping field to src data
            sct.run('isct_antsApplyTransforms -d 2 -i ' + im[i*iter + iter] + ' -r ' + dest[i] + ' -n Linear -t ' + wrap[i] + ' --output ' + output_folder_path + fname_out_im, verbose=v)
            sct.run('isct_antsApplyTransforms -d 2 -i ' + im_seg[i*iter + iter] + ' -r ' + dest[i] + ' -n Linear -t ' + wrap[i] + ' --output ' + output_folder_path + fname_out_seg, verbose=v)
            if im_GM:
                sct.run('isct_antsApplyTransforms -d 2 -i ' + im_GM[i*iter + iter] + ' -r ' + dest[i] + ' -n Linear -t ' + wrap[i] + ' --output ' + output_folder_path + fname_out_GM,
                        verbose=v)


# Apply warping field
# ==========================================================================================
def worker_apply_warping_field(argument):
    im, im_seg, src, nbre_src, dest, nbre_dest, nbre_slice, output_folder_path, wrap, nbre_im, j, v, im_GM, dirs_name_src, dirs_name_dest, dirs_name_im = argument

    src_path, src_file, src_ext = sct.extract_fname(src)
    pos_crop_src = src_file.find('_crop')
    dest_path, dest_file, dest_ext = sct.extract_fname(dest)
    pos_crop_dest = dest_file.find('_crop')

    for iter in range(0, nbre_slice):
        im_path, im_file, im_ext = sct.extract_fname(im[j+iter])
        pos_crop_im = im_file.find('_crop')
        fname_out_im = dirs_name_src + '_' + src_file[:pos_crop_src] + str(nbre_src) + '_' + dirs_name_dest + '_' +  dest_file[:pos_crop_dest] + str(nbre_dest) + '_' + dirs_name_im[j+iter] + '_' + im_file[:pos_crop_im] + str(nbre_im[j+iter]) + '.nii.gz'
        fname_out_seg = dirs_name_src + '_' + src_file[:pos_crop_src] + str(nbre_src) + '_' + dirs_name_dest + '_' +  dest_file[:pos_crop_dest] + str(nbre_dest) + '_' + dirs_name_im[j+iter] + '_' + im_file[:pos_crop_im] + str(nbre_im[j+iter]) + '_seg.nii.gz'
        fname_out_GM = dirs_name_src + '_' + src_file[:pos_crop_src] + str(nbre_src) + '_' + dirs_name_dest + '_' +  dest_file[:pos_crop_dest] + str(nbre_dest) + '_' + dirs_name_im[j+iter] + '_' + im_file[:pos_crop_im] + str(nbre_im[j+iter]) + '_gmseg.nii.gz'
        # Apply warping field to src data
        if not os.path.isfile(fname_out_im):
            sct.run('isct_antsApplyTransforms -d 2 -i ' + im[j+iter] + ' -r ' + dest + ' -n Linear -t ' + wrap + ' --output ' + output_folder_path + fname_out_im, verbose=1)
            sct.run('isct_antsApplyTransforms -d 2 -i ' + im_seg[j+iter] + ' -r ' + dest + ' -n Linear -t ' + wrap + ' --output ' + output_folder_path + fname_out_seg, verbose=1)
            if im_GM:
                sct.run('isct_antsApplyTransforms -d 2 -i ' + im_GM[j + iter] + ' -r ' + dest + ' -n Linear -t ' + wrap + ' --output ' + output_folder_path + fname_out_GM, verbose=1)

# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()