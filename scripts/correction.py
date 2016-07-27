import os
import sct_utils as sct
from msct_image import Image
import numpy as np

#path = os.getcwd()
#for dir in os.listdir(path):
#    if dir.find('Slices')!= -1:
#        for file in os.listdir(dir):
#            if file.find('slice_out')!= -1 or file.find('slice_seg_out')!= -1 or file.find('slice_GM_out')!= -1:
#                sct.run('rm -rf ' + path+ '/' +  dir + '/' +file)

change_name = False
if change_name:
    for dirs in os.listdir('/Users/cavan/data/essais_t2s'):
        if dirs.find('.DS_Store') == -1:
            for sub in os.listdir('/Users/cavan/data/essais_t2s/' + dirs):
                if sub.find('.DS_Store') == -1 :
                    for file in os.listdir('/Users/cavan/data/essais_t2s/' + dirs +'/'+ sub):
                        if file.find('_r_gmseg') !=-1 :
                            print '/Users/cavan/data/essais_t2s/' + dirs +'/'+ sub + '/' + file
                            im = Image('/Users/cavan/data/essais_t2s/' + dirs +'/'+ sub + '/' + file)
                            im.setFileName('/Users/cavan/data/essais_t2s/' + dirs +'/'+ sub + '/' + 't2s_gmseg_manual.nii.gz')
                            im.save()
                        if file.find('_r_seg') != -1 :
                            im = Image('/Users/cavan/data/essais_t2s/' + dirs + '/' + sub + '/' + file)
                            im.setFileName('/Users/cavan/data/essais_t2s/' + dirs +'/'+ sub + '/' +'t2s_seg_manual.nii.gz')
                            im.save()
                        if file.find('_r.nii') != -1 :
                            im = Image('/Users/cavan/data/essais_t2s/' + dirs + '/' + sub + '/' + file)
                            im.setFileName('/Users/cavan/data/essais_t2s/' + dirs +'/'+ sub + '/' + 't2s.nii.gz')
                            im.save()

folder_path = '/Users/cavan/data/essais_t2s_2'
dirs_name = []
get_orientation = False
if get_orientation:
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir.find("t2s") == -1 and dir.find("t2") == -1 and dir.find("mt") == -1 and dir.find("dmri") == -1:
                dirs_name.append(dir)

    list_data = []
    for iter, dir in enumerate(dirs_name):
        for root, dirs, files in os.walk(folder_path + '/' + dir):
            for file in files:
                if file.endswith('t2s.nii.gz') and file.find('_gmseg') == -1 and file.find('_seg') == -1:
                    pos_t2 = file.find('t2s')
                    subject_name, end_name = file[0:pos_t2 + 3], file[pos_t2 + 3:]
                    file_seg = dir + '/t2s/' + subject_name + '_seg_manual' + end_name
                    file_gmseg = dir + '/t2s/' + subject_name + '_gmseg_manual' + end_name
                    file = dir + '/t2s/' + file
                    list_data.append(file)
                    list_data.append(file_seg)
                    list_data.append(file_gmseg)

    from sct_image import get_orientation_3d
    for i in range(0,len(list_data)):
        print list_data[i]
        print(get_orientation_3d(Image(folder_path + '/' + list_data[i])))

delete_false_seg = False
path = '/Volumes/folder_shared/data_machine_learning/T2s/Essai_t2s_2'
if delete_false_seg:
    for files in os.listdir(path):
        print files
        if files.find('vanderbilt')!=-1 :
            sct.run('rm -rf ' + path + '/' + files)

thresholding = False
if thresholding:
    folder_path = '/Users/cavan/data/machine_learning/train/test_patch/'
    fname = 't2s_crop_5_prob.nii.gz'
    im = Image(folder_path + fname)
    data = np.asarray(im.data)
    from scipy import stats
    data_new = stats.threshold(data, threshmin=0.9, newval=0)
    im.data = data_new
    im.save()

folder_path = '/Users/cavan/data/machine_learning/train/test_patch/'
fname = 't2s_crop_5.nii.gz'

im_file = Image(folder_path + fname)
im_file.hdr.structarr['qoffset_x'] = im_file.hdr.structarr['qoffset_y'] = im_file.hdr.structarr['qoffset_z'] = im_file.hdr.structarr['srow_x'][-1] = im_file.hdr.structarr['srow_y'][-1] = im_file.hdr.structarr['srow_z'][-1] = 0
im_file.setFileName(folder_path + fname)
im_file.save()
