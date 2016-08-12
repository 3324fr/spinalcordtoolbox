
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
    return data, data_seg, data_gmseg, rand_angle

def flip_lr(data):
    result = np.zeros(data.shape)
    for it_slice in range(0, data.shape[2]):
        result[:, :, it_slice] = np.flipud(data[:, :, it_slice])
    return result

def rotate_data(data, rand_angle, order=0):
    result = np.zeros(data.shape)
    for it_slice in range(0, data.shape[2]):
        result[:, :, it_slice] = rotate(data[:, :, it_slice], rand_angle, axes=(0,1), reshape=False, order=order)
    return result


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
                        if file.endswith('_N4.nii.gz') and file.find('_seg') == -1 and file.find('_gmseg') == -1:
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

def extract_fname_list_multi(folder_path):
    list_data = []
    dirs_name = []
    path_tmp = sct.tmp_create()
    for directory in os.listdir(folder_path):
        if os.path.exists(folder_path + '/' + directory + '/t2s/'):
            result_folder = {}
            result_folder['data'] = None
            result_folder['seg'] = []
            result_folder['gmseg'] = []
            for root, dirs, files in os.walk(folder_path + '/' + directory + '/t2s/'):
                for f in files:
                    if f.endswith('_N4.nii.gz') and f.find('_seg') == -1 and f.find('_gmseg') == -1:
                        result_folder['data'] = folder_path + '/' + directory + '/t2s/' + f
                    if f.find('_gmseg') != -1 and f.find('manual_rater') == -1:
                        result_folder['gmseg'].append(folder_path + '/' + directory + '/t2s/' + f)
                    if f.find('_seg') != -1 and f.find('manual_rater') == -1:
                        result_folder['seg'].append(folder_path + '/' + directory + '/t2s/' + f)
                    if f.find('manual_rater') != -1:
                        fname_seg, fname_gmseg = get_gmseg_from_multilabel(f, folder_path + '/' + directory + '/t2s', path_tmp, directory)
                        result_folder['gmseg'].append(fname_gmseg)
                        result_folder['seg'].append(fname_seg)
                    if f.find('level') != -1 and f.find('_reg') == -1:
                        result_folder['level'] = folder_path + '/' + directory + '/t2s/' + f
            list_data.append(result_folder)
            dirs_name.append(directory)
    return list_data, dirs_name


# Extract the GM segmentation from the multilabeled one
# ==========================================================================================
def get_gmseg_from_multilabel(fname_multilabel, path, path_tmp, dir):
    from msct_image import Image

    path_data = path
    #fname_multilabel = 't2s_gmseg_manual_rater_unf.nii.gz'

    lim = 10
    path, filename, ext = sct.extract_fname(fname_multilabel)
    fname_gm = filename + '_gmseg_manual.nii.gz'
    fname_sc = filename + '_seg_manual.nii.gz'

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
    folder_path = '/Volumes/folder_shared/data_machine_learning/dataset_original/wT2s/training'
    output_folder = '/Volumes/folder_shared/data_machine_learning/dataset_original/wT2s/training_augmented'
    data_list, dirs_name = extract_fname_list_multi(folder_path)

    for i, d in enumerate(data_list):
        print i, '/', len(data_list)
        fname_im = d['data']
        path, filename, ext = sct.extract_fname(fname_im)
        file_name = path.split('/')[-3]

        list_seg = d['seg']
        list_gmseg = d['gmseg']
        level_file = d['level']

        # move level file
        _, _, ext_levels = sct.extract_fname(level_file)
        sct.run('cp ' + level_file + ' ' + output_folder + '/' + file_name + '_levels' + ext_levels, verbose=0)

        # augment data
        im = Image(fname_im)
        data = im.data
        hdr = im.hdr

        nb_rotation = 3
        rand_angles_1 = np.random.uniform(low=-5.0, high=5.0, size=nb_rotation)
        rand_angles_2 = np.random.uniform(low=-5.0, high=5.0, size=nb_rotation)

        data_to_save = []
        data_to_save.append([data, ''])
        data_f = flip_lr(data)
        data_to_save.append([data_f, '_fl'])
        for angle in rand_angles_1:
            data_to_save.append([rotate_data(data, angle, 1), '_r_'+str(round(angle, 3))])
        for angle in rand_angles_2:
            data_to_save.append([rotate_data(data_f, angle, 1), '_fl_r_'+str(round(angle, 3))])

        for j, f in enumerate(list_seg):
            data_seg = Image(f).data
            data_to_save.append([data_seg, '_seg'+str(j+1)])
            for angle in rand_angles_1:
                data_to_save.append([rotate_data(data_seg, angle), '_r_'+str(round(angle, 3))+'_seg'+str(j+1)])
            data_seg_f = flip_lr(data_seg)
            data_to_save.append([data_seg_f, '_fl'+'_seg'+str(j+1)])
            for angle in rand_angles_2:
                data_to_save.append([rotate_data(data_seg_f, angle), '_fl_r_'+str(round(angle, 3))+'_seg'+str(j+1)])

        for j, f in enumerate(list_gmseg):
            data_gmseg = Image(f).data
            data_to_save.append([data_gmseg, '_gmseg'+str(j+1)])
            for angle in rand_angles_1:
                data_to_save.append([rotate_data(data_gmseg, angle), '_r_'+str(round(angle, 3))+'_gmseg'+str(j+1)])
            data_gmseg_f = flip_lr(data_gmseg)
            data_to_save.append([data_gmseg_f, '_fl'+'_gmseg'+str(j+1)])
            for angle in rand_angles_2:
                data_to_save.append([rotate_data(data_gmseg_f, angle), '_fl_r_'+str(round(angle, 3))+'_gmseg'+str(j+1)])

        # save data
        for data2save in data_to_save:
            im_t = Image(data2save[0])
            im_t.hdr = hdr
            im_t.setFileName(output_folder + '/' + file_name + data2save[1] + '.nii.gz')
            im_t.save()



# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    # call main function
    main()