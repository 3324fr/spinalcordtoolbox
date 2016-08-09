
import random
import os
import numpy as np
from msct_image import Image
import sct_utils as sct
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates, rotate

# Some random transformations
# ==========================================================================================
def random_rotation(data, data_seg, data_gmseg,nz):
    rand_angle = random.uniform(-5,5)
    for iter in range(0, nz):
        data[:,:,iter] = rotate(data[:,:,iter], rand_angle, axes = (0,1), reshape = False)
        data_seg[:,:,iter] = rotate(data_seg[:,:,iter], rand_angle, axes=(0,1), reshape=False)
        data_gmseg[:,:,iter] = rotate(data_gmseg[:,:,iter], rand_angle, axes=(0,1), reshape=False)
    return data, data_seg, data_gmseg


def flipped_lr(data, data_seg, data_gmseg, nz):
    s = np.random.binomial(1, 0.7, 1)
    if s == 1:
        for iter in range(0,nz):
            data[:,:,iter] = np.flipud(data[:,:,iter])
            data_seg[:,:,iter] = np.flipud(data_seg[:,:,iter])
            data_gmseg[:,:,iter] = np.flipud(data_gmseg[:,:,iter])

    return data, data_seg, data_gmseg


def elastic_transform(data, data_seg, data_gmseg, alpha, sigma, nz, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = data[:,:,0].shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)*alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)*alpha

    x,y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1,1)), np.reshape(x+dx,(-1,1))

    for iter in range(0,nz):
        data[:,:,iter] = map_coordinates(data[:,:,iter], indices, order=1).reshape((shape[0],shape[1]))
        data_seg[:,:,iter] = map_coordinates(data_seg[:,:,iter], indices, order=1).reshape((shape[0],shape[1]))
        data_gmseg[:,:,iter] = map_coordinates(data_gmseg[:,:,iter], indices, order=1).reshape((shape[0],shape[1]))

    return data, data_seg, data_gmseg

# Extract the list of image and segmentation from a text file which contains the image we want
# ==========================================================================================
def extract_fname_list(folder_path):
    list_data = []
    dirs_name = []
    fp = open('/Users/cavan/data/list_image_denoised.txt')
    path_tmp = sct.tmp_create()
    for l, line in enumerate(fp):
        for dir in os.listdir(folder_path):
            if dir == str(line[0:len(line)-1]):
                for root, dirs, files in os.walk(folder_path + '/' + dir + '/t2s'):
                    for file in files:
                        if file.endswith('nii.gz') and file.find('_seg') == -1 and file.find('_gmseg') == -1:
                            list_data.append(folder_path + '/' + dir + '/t2s/' + file)
                        if file.find('_gmseg') != -1 and file.find('manual_rater') == -1:
                            list_data.append(folder_path + '/' + dir + '/t2s/' + file)
                        if file.find('_seg') != -1 and file.find('manual_rater') == -1:
                            list_data.append(folder_path + '/' + dir + '/t2s/' + file)
                        if file.find('manual_rater_unf') != -1:
                            fname_seg, fname_gmseg = get_gmseg_from_multilabel(folder_path + '/' + dir + '/t2s', path_tmp, dir)
                            list_data.append(fname_gmseg)
                            list_data.append(fname_seg)
                dirs_name.append(dir)
    fp.close()
    return list_data, dirs_name


# Extract the GM segmentation from the multilabeled one
# ==========================================================================================
def get_gmseg_from_multilabel(path, path_tmp, dir):
    from msct_image import Image

    path_data = path
    fname_multilabel = 't2s_gmseg_manual_rater_unf.nii.gz'

    lim = 10
    fname_gm = 't2s_gmseg_manual.nii.gz'
    fname_sc = 't2s_seg_manual.nii.gz'

    if dir.find('challenge') != -1:
        dir_id = dir.split('_')[1]
        if dir.find('pain') != -1:
            dir_id = dir.split('_')[2]
            lim  = 13
        # get multi-label image
        im_ml = Image(path_data + '/' + fname_multilabel)
        # GM = 2, WM= 1
        if int(dir_id) <= lim:
            # get GM:
            im_gm = im_ml.copy()
            im_gm.data[im_gm.data == 2] = 0
            im_gm.setFileName(path_tmp + dir + '_' + fname_gm)
            im_gm.save()
            # get SC
            im_sc = im_ml.copy()
            im_sc.data[im_sc.data > 0] = 1
            im_sc.setFileName(path_tmp + dir + '_' + fname_sc)
            im_sc.save()
        # GM = 1, WM = 2
        elif int(dir_id) > lim:
            # get GM:
            im_gm = im_ml.copy()
            im_gm.data[im_gm.data == 1] = 0
            im_gm.data[im_gm.data == 2] = 1
            im_gm.setFileName(path_tmp + dir + '_'+ fname_gm)
            im_gm.save()
            # get SC
            im_sc = im_ml.copy()
            im_sc.data[im_sc.data > 0] = 1
            im_sc.setFileName(path_tmp + dir + '_' + fname_sc)
            im_sc.save()
    return path_tmp + dir + '_' + fname_sc, path_tmp + dir + '_' + fname_gm


# ==========================================================================================
def denoised(data_list):
    for iter in range(0, len(data_list)/3):
        sct.run('sct_maths -i ' + data_list[iter*3] + ' -denoise 1 -o ' + data_list[iter*3])


# ==========================================================================================
def main_denoise():
    folder_path = '/Users/cavan/data/essais_t2s_copy'
    data_list, dirs_name = extract_fname_list(folder_path)

    denoised(data_list)


# MAIN
# ==========================================================================================
def main():
    folder_path = '/Volumes/folder_shared/data_machine_learning/dataset_t2s_denoised'
    data_list, dirs_name = extract_fname_list(folder_path)

    from sct_data_augmentation_vertebra import extract_label_list, vertebral_similarity, registration_vertebral_levels, crop_segmentation, change_orientation

    new_name = []

    list_image_labels, data_list, dirs_name = extract_label_list(folder_path, data_list, dirs_name)

    data_list = crop_segmentation(data_list, dirs_name)

    #for iter in range(0,len(dirs_name)):
    #    fname = data_list[iter*3]
    #    fname_out = vertebral_similarity(fname, data_list, list_image_labels, 1)
    #    new_name.append(registration_vertebral_levels(fname, fname_out, data_list, dirs_name, list_image_labels))

    for iter in range(0,len(dirs_name)):
        fname_im = data_list[iter*3]
        fname_seg = data_list[iter*3 +2]
        fname_gmseg = data_list[iter*3+ 1]
        path, file,ext = sct.extract_fname(fname_im)
        file_name = path.split('/')[5]

        im = Image(data_list[iter*3])
        hdr_im = im.hdr.copy()
        nx, ny, nz, nt, px, py, pz, pt = im.dim

        #for j in range(0,5):
        #    path, file_name, ext = sct.extract_fname(new_name[iter] + '_' + str(j) + '.nii.gz')

            #data = Image(new_name[iter] + '_' + str(j) + '.nii.gz')
            # data_seg = Image(new_name[iter] + '_' + str(j) + '_seg.nii.gz')
            # data_gmseg = Image(new_name[iter] + '_' + str(j) + '_gmseg.nii.gz')

        data = Image(fname_im).data
        data_seg = Image(fname_seg).data
        data_gmseg = Image(fname_gmseg).data

        data_t, data_seg_t, data_gmseg_t = random_rotation(data, data_seg, data_gmseg, nz)
        data_t, data_seg_t, data_gmseg_t = flipped_lr(data_t, data_seg_t, data_gmseg_t, nz)

        im_t = Image(data_t)
        im_seg_t = Image(data_seg_t)
        im_gm_t = Image(data_gmseg_t)
        im_t.setFileName(file_name + '_m.nii.gz')
        im_seg_t.setFileName(file_name + '_seg_m.nii.gz')
        im_gm_t.setFileName(file_name + '_gmseg_m.nii.gz')
        im_t.hdr = hdr_im
        im_seg_t.hdr = hdr_im
        im_gm_t.hdr = hdr_im
        im_t.save()
        im_seg_t.save()
        im_gm_t.save()


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()