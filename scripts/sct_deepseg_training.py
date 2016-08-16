#!/usr/bin/env python
#########################################################################################
#
# Preprocessing for the training the Deepseg model.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 University of British Columbia and Polytechnique Montreal
# Authors: Emil Ljungberg
# Modified: 2016-08-12
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# SCT Libraries
from msct_parser import Parser
from msct_image import Image
from msct_gmseg_utils import resample_image, load_level
from msct_deepseg_utils import *
import sct_utils as sct
#from get_input_data_json import *
from medpy.filter.IntensityRangeStandardization import IntensityRangeStandardization

# Scientific libraries
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

# Python standard libraries
import sys
import pwd
import os
import datetime
import json
import subprocess as sp
import pickle
import PIL
import timeit


### Parser ###
def get_parser():
	# Initialize the parser
	parser = Parser(__file__)
	parser.usage.set_description(
		'''Tool for pre-processing Deepseg training data''')

	parser.add_option(name="-f",
					  type_value="folder",
					  description="Destination folder for training data",
					  mandatory=True,
					  example="./training_data")

	return parser

################


# def add_training_data(jf, path):
		
# 	data = open_json(jf)

# 	i = 1
# 	for site in ['site3', 'site4']:
# 		comment = 'Data from %s' % site
# 		for sub in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
# 			subject = {}
			
# 			subject['path'] = path

# 			# Set image
# 			img_fname = site + '-sc' + sub + '-image.nii.gz'
# 			subject['img'] = img_fname
			
# 			# Add all 4 masks
# 			subject['seg'] = {}
# 			for j in range(1,5):
# 				seg_fname = site + '-sc' + sub + '-mask-r%s.nii.gz' % str(j)
# 				subject['seg'][j] = seg_fname
			
# 			# Set levels
# 			levels_fname = site + '-sc' + sub + '-levels.txt'
# 			subject['levels'] = levels_fname
# 			subject['comment'] = comment

# 			data['data']['subjects'][i] = subject
# 			i += 1

# 	comment = 'Data from site2 in the ISMRM training set.'
# 	site = 'site2'
# 	for sub in ['02', '03', '04', '05', '06', '07', '08', '09', '10']:
# 		# There is something wrong with subject 01 thus no import
# 		subject = {}
		
# 		subject['path'] = path

# 		# Set image
# 		img_fname = site + '-sc' + sub + '-image.nii.gz'
# 		subject['img'] = img_fname
		
# 		# Add all 4 masks
# 		subject['seg'] = {}
# 		for j in range(1,5):
# 			seg_fname = site + '-sc' + sub + '-mask-r%s.nii.gz' % str(j)
# 			subject['seg'][j] = seg_fname
		
# 		# Set levels
# 		levels_fname = site + '-sc' + sub + '-levels.txt'
# 		subject['levels'] = levels_fname
# 		subject['comment'] = comment

# 		data['data']['subjects'][i] = subject
		# i += 1

	# Spacing is wrong in site 1 data. Don't know how to fix it. Cannot run with the 
	# N4 biasfield correction

	# Add in Camilles data here as well

# def apply_N4_correction(jf):
# 	img_list = get_image_list(jf)
# 	jdata = open_json(jf)
# 	subjdata = jdata['data']['subjects']

# 	sct.printv('Performing N4 bias field correction', 1, 'info')
# 	for sub in subjdata.keys():
# 		path = subjdata[sub]['path']

# 		img = subjdata[sub]['img']
# 		in_img_path = os.path.join(path, img)
# 		out_img = img.split('.nii.gz')[0] + '_N4.nii.gz'
# 		out_img_path = os.path.join(path, out_img)

# 		antsN4BiasFieldCorrection(input_img=in_img_path, output_img=out_img_path)
# 		subjdata[sub]['img'] = out_img

# 	add_event(jf, 'Applied N4 bias field correction to training image')

# 	save_json(jdata, jf)

####################

def parse_files(fl):
	cord_segs = []
	gm_segs = []
	img = None
	levels = None
	
	for f in fl:
		if 'level' in f:
			levels = f
		elif '_seg' in f:
			cord_segs.append(f)
		elif '_gmseg' in f:
			gm_segs.append(f)
		else:
			img = f
	
	if len(gm_segs) == 0:
		print 'Cannot find GM segs for: %s' % img
	elif len(cord_segs) == 0:
		print 'Cannot find any segs for: %s' % img
	
	d = {'seg':cord_segs, 'gmseg':gm_segs, 'img':img, 'levels':levels}
	return d

def add_aug_training_data(outpath):
	# First Split up the data in Poly files and Augmented data
	sct.printv('Parsing the data into json structure')
	all_files = os.listdir(outpath)
	subjects = []
	for f in all_files:
		if ('_r' not in f) and ('_fl' not in f) and ('seg' not in f) and ('_levels' not in f):
			sub_name = f.split('.nii')[0]
			subjects.append(sub_name)

	data = {}
	data['challenge'] = {}
	data['poly'] = {}

	# Now we split up first challenge data and then augmented data
	# Loop over all the challenge base names

	for sub in subjects:
		if 'challenge' in sub:
			key = 'challenge'
		else:
			key = 'poly'

		data[key][sub] = {}
		
		# Pick up all the files related to the subject
		sub_files = []
		
		# Loop over all the files again
		for nii in all_files:
			if sub in nii:
				sub_files.append(nii)

		# Now we have all the images from the same subject.
		# Split up in original and augmented data
		org_data = []
		aug_data = []
		for nii in sub_files:
			if ('_fl' not in nii) and ('_r' not in nii):
				org_data.append(nii)
			else:
				aug_data.append(nii)
			
		# Split the augmented data into the subject files
		aug_nii_files = {}
		for nii in aug_data:
			if ('seg' not in nii):
				aug_name = nii.split('.nii')[0]
				aug_nii_files[aug_name] = []
		
		for nii in aug_data:
			for f in aug_nii_files.keys():
				if f in nii and ('_r_' not in nii.split(f)[-1]):
					aug_nii_files[f].append(nii)
		
		data[key][sub]['augdata'] = aug_nii_files
		data[key][sub]['org'] = org_data

	data_sorted = {}
	for k in data['challenge'].keys():
		sub_data = data['challenge'][k]
		d = {}
		d['org'] = parse_files(sub_data['org'])
		
		for i, aug_k in enumerate(sub_data['augdata'].keys()):
			dtemp = parse_files(sub_data['augdata'][aug_k])
			dtemp['levels'] = d['org']['levels']
			d['aug%s' % (i+1)] = dtemp
			
		data_sorted[k] = d
		
	for k in data['poly'].keys():
		sub_data = data['poly'][k]
		d = {}
		d['org'] = parse_files(sub_data['org'])
		
		
		for i, aug_k in enumerate(sub_data['augdata'].keys()):
			dtemp = parse_files(sub_data['augdata'][aug_k])
			dtemp['levels'] = d['org']['levels']
			d['aug%s' % (i+1)] = dtemp

		data_sorted[k] = d        
	 
	json_path = os.path.join(os.path.join(outpath,'db_sorted.json'))   
	with open(json_path, 'w') as jf:
		json.dump(data_sorted, jf)  
	
	return json_path

def move_data(jf, path):

	# This will move all the data speficied in the json to a new folder
	# We here assume that each subject has 3 files:
	# 	1. image 	['img']
	#	2. seg   	['seg']
	# 	3. levels 	['levels']
	
	data = open_json(jf)

	subj_data = data['data']['subjects']
	new_path = os.path.abspath(path)
	for s in subj_data.keys():
		subj = subj_data[s]
		
		subject_path = os.path.join(new_path, str(s).zfill(3))
		os.mkdir(subject_path)
		
		levels = os.path.join(subj['path'], subj['levels'])
		if os.path.isfile(levels):
			new_lvl_path = os.path.join(subject_path, 'levels.txt')
			cmd = ['cp', levels , new_lvl_path]
			cmd = ' '.join(cmd)
			sp.call(cmd, shell=True)
			subj['levels'] = 'levels.txt'

		img = os.path.join(subj['path'], subj['img'])
		if os.path.isfile(img):
			org_img_path = os.path.join(subject_path, 't2s_org.nii.gz')
			cmd = ['cp', img , org_img_path]
			cmd = ' '.join(cmd)
			sp.call(cmd, shell=True)

			training_img_path = os.path.join(subject_path, 't2s_training.nii.gz')
			cmd = ['cp', img , training_img_path]
			cmd = ' '.join(cmd)
			sp.call(cmd, shell=True)

			subj['org_img'] = 't2s_org.nii.gz'
			subj['img'] = 't2s_training.nii.gz'

		i = 1
		for s in subj['seg'].keys():
			s = str(s)
			seg = os.path.join(subj['path'], subj['seg'][s])
			new_seg_name = 't2s_seg%s.nii.gz' % str(s)
			new_seg_path = os.path.join(subject_path, new_seg_name)
			cmd = ['cp', seg , new_seg_path]
			cmd = ' '.join(cmd)
			sp.call(cmd, shell=True)
			subj['seg'][s] = new_seg_name
			i += 1

		subj['path'] = subject_path
	
	save_json(data, jf)
	add_event(jf, 'Moved files to nice folders')

	return data

def get_image_list(jfile):

	jdata = open_json(jfile)
	subj_data = jdata['data']['subjects']
	img_list = []
	for s in subj_data.keys():
		img = os.path.join(subj_data[s]['path'], subj_data[s]['img'])
		img_list.append(img)

	return img_list

def get_seg_list(jfile):
	jdata = open_json(jfile)
	subj_data = jdata['data']['subjects']
	seg_list = []
	for sub in subj_data.keys():
		for i in subj_data[sub]['seg'].keys():
			seg = os.path.join(subj_data[sub]['path'], subj_data[sub]['seg'][i])
			seg_list.append(seg)

	return seg_list

def open_json(jfile):
	with open(jfile) as jf:
		jdata = json.load(jf)

	return jdata

def get_irs_model(jfile):
	
	jdata = open_json(jfile)
	return jdata['models']['irs']

def save_irs_model(irs, jfile):
	jdata = open_json(jfile)
	jdata['models']['irs'] = irs
	save_json(jdata, './data.json')

def save_json(jdata, fname):

	with open(fname, 'w') as jf:
		json.dump(jdata, jf)
	return True

def add_event(json_file, mes):
	
	json_data = open_json(json_file)
	tstamp = '{:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
	json_data['events'][tstamp] = mes

	save_json(json_data, json_file)

def init_json(path):
	# Initialize the json structure.
	# Add boolean fields for:
	# 	N4 correction w. parameters used
	# 	IRS filter w. parameters used
	# 	Mask + crop w. parameters used
	# 	Resampling factor
	# 	Denoising w. parameters
	# 	
	# Empty subject structure that allow us to add data as we want to
	# Make the input data structure just append to the json structure
	sct.printv('Creating new folder structure')

	fullpath = os.path.abspath(path)
	if os.path.exists(fullpath):
		sct.printv('Folder already exists. Appending date and time', 1, 'warning')
		fullpath = fullpath + '_' + datetime.datetime.now().strftime("%y%m%d_%H%M")
	
	os.mkdir(fullpath)

	sct.printv('Initializing json data structure')
	data = {}
	data['data'] = {}
	data['data']['subjects'] = {}

	data['Processing'] = {}
	data['Processing']['N4FieldCorrection'] = {}
	data['Processing']['Resampling'] = {}
	data['Processing']['IRS'] = {}
	data['Processing']['Mask'] = {}
	data['Processing']['Denoising'] = {}

	data['Info'] = {}
	data['Info']['User'] = pwd.getpwuid(os.getuid())[0]
	data['Info']['Created'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
	data['Info']['Analysis started'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
	data['Info']['Path'] = path

	data['events'] = {}

	output_json = os.path.join(fullpath, 'db.json')

	with open(output_json, 'w') as fp:
		json.dump(data, fp)

	sct.printv('json strcuture saved as ' + output_json)

	return output_json


# Processing functions

def train_IRS_model(jf, path):
	
	sct.printv('Training Intensity Range Standardization Model', 1, 'info')
	subdata = open_json(jf)

	all_img_data_list = []
	for s in subdata.keys():
		# We only need to train to model on the original data
		img = subdata[s]['org']['img']
		print img
		img_path = os.path.join(path, img)
		seg = subdata[s]['org']['dilated_seg']
		seg_path = os.path.join(path, seg)

		img_nii = nb.load(str(img_path))
		imdata = img_nii.get_data()
		seg_nii = nb.load(str(seg_path))
		segdata = seg_nii.get_data()
		
		#imdata = cstretch(imdata, 0.8, 0, 100)
		imdata = imdata[segdata > 0]
		all_img_data_list.append(imdata)
	

	# challenge_img_data_list = []
	# for s in subdata.keys():
	# 	if 'challenge' in s:
	# 		# We only need to train to model on the original data
	# 		img = subdata[s]['org']['img']
	# 		img_path = os.path.join(path, img)
	# 		seg = subdata[s]['org']['seg'][0]
	# 		seg_path = os.path.join(path, seg)

	# 		img_nii = nb.load(str(img_path))
	# 		imdata = img_nii.get_data()
	# 		seg_nii = nb.load(str(seg_path))
	# 		segdata = seg_nii.get_data()
			
	# 		#imdata = cstretch(imdata, 0.8, 0, 100)
			
	# 		imdata = imdata[segdata > 0]
	# 		challenge_img_data_list.append(imdata)

	cp = (0,99)
	lp = [10, 20, 30, 40, 50, 60, 70, 80, 90]
	sr = (0,0.8)

	irs = IntensityRangeStandardization(cutoffp=cp, landmarkp=lp, stdrange=sr)
	irs_model = irs.train(all_img_data_list)

	# irs_challenge = IntensityRangeStandardization(cutoffp=cp, landmarkp=lp, stdrange=sr)
	# irs_model_challenge = irs_challenge.train(challenge_img_data_list)

	irs_path = os.path.join(path, 'irs_model.pkl')
	# challenge_irs_path = os.path.join(path, 'challenge_irs_model.pkl')
	
	# Save the irs model as pickle
	with open(irs_path, 'w') as pf:
		pickle.dump(irs_model, pf)

	# with open(challenge_irs_path, 'w') as pf:
	# 	pickle.dump(irs_model_challenge, pf)
	
	save_json(subdata,jf)

def apply_all_IRS_model(jf, path):
	
	subdata = open_json(jf)
	irs_model_file = os.path.join(path, 'irs_model.pkl')

	# -------- Load IRS pickle model ---------- #
	sct.printv('Loading pickle file with IRS model', 1, 'info')
	with open(irs_model_file, 'r') as pf:
		irs_model = pickle.load(pf)

	# --------- Transform images ------------- #
	sct.printv('Transforming images with IRS model', 1, 'info')
	for sub in subdata.keys():
		for s in subdata[sub].keys():
			in_img = subdata[sub][s]['img']
			in_seg = subdata[sub][s]['dilated_seg']

			in_img_path = os.path.join(path, in_img)
			img_nii = nb.load(str(in_img_path))
			imdata = img_nii.get_data()

			in_seg_path = os.path.join(path, in_seg)
			seg_nii = nb.load(str(in_seg_path))
			segdata = seg_nii.get_data()

			irs_data = IRS_transformation(irs_model, imdata, segdata)

			# Save image with histogram normalization
			irs_name = 'IRS_' + in_img
			out_nii = nb.Nifti1Image(irs_data, img_nii.get_affine())
			nb.nifti1.save(out_nii, os.path.join(path, irs_name))
			subdata[sub][s]['img'] = irs_name

	# irs_obj = challenge_irs_model
	# output_path = os.path.join(path, 'IRS_challenge')
	# for sub in subdata.keys():
	# 	for s in sub.keys():
	# 		if 'challenge' in s:
	# 			in_img = subdata[sub][s]['img']
	# 			in_seg = subdata[sub][s]['seg']

	# 			in_img_path = os.path.join(path, in_img)
	# 			img_nii = nb.load(str(in_img_path))
	# 			imdata = img_nii.get_data()

	# 			in_seg_path = os.path.join(path, in_seg)
	# 			seg_nii = nb.load(str(in_seg_path))
	# 			segdata = seg_nii.get_data()

	# 			irs_data = IRS_transformation(irs_obj, imdata, segdata)

	# 			# Save image with histogram normalization
	# 			irs_name = sct.add_suffix(in_img, '_IRS_CH')
	# 			out_nii = nb.Nifti1Image(irs_data, nii.get_affine())
	# 			nb.nifti1.save(out_nii, os.path.join(path, irs_name))


	save_json(subdata, jf)

def crop_around_cord(jf, path):
	
	subdata = open_json(jf)
	sct.printv('Isolating the spinal cord', 1, 'info')

	for ss in subdata.keys():
		for s in subdata[ss].keys():
			sub = subdata[ss][s]

			# Since there might be more than 1 segmentation. We dialate the mask
			# before we crop the image
			seg = sub['seg'][0]
			dilated_seg = 'dilate_' + seg
			
			df = 1 # <<<<<<<< Dialation factor. May need to be adjusted
			cmd = ['fslmaths', os.path.join(path, seg), '-kernel 2D -dilD', os.path.join(path, dilated_seg)]
			# cmd = ['sct_maths', '-i', os.path.join(path, seg), '-dilate', str(df), '-o', os.path.join(path, dilated_seg)]
			sct.run(' '.join(cmd))

			# Now we multiply the image with the dialated seg
			img = sub['img']
			out_img = 'cm_' + img
			cmd = ['sct_maths', '-i', os.path.join(path, img) , '-mul', os.path.join(path, dilated_seg), '-o', os.path.join(path, out_img)]
			sct.run(' '.join(cmd))
			img = out_img

			# Now we crop the image and all the segmentations using the dialated segmentation
			out_img = 'cropped_' + img
			cmd = ['sct_crop_image', '-i', os.path.join(path, img), 
				'-m', os.path.join(path, dilated_seg), '-o', os.path.join(path, out_img)]
			sct.run(' '.join(cmd))
			sub['img'] = out_img

			out_dseg = 'cropped_' + dilated_seg
			cmd = ['sct_crop_image', '-i', os.path.join(path, dilated_seg), 
				'-m', os.path.join(path, dilated_seg), '-o', os.path.join(path, out_dseg)]
			sct.run(' '.join(cmd))
			sub['dilated_seg'] = out_dseg

			new_segs = []
			for seg in sub['seg']:
				out_seg = 'cropped_' + seg
				cmd = ['sct_crop_image', '-i', os.path.join(path, seg), 
					'-m', os.path.join(path, dilated_seg), '-o', os.path.join(path, out_seg)]
				sct.run(' '.join(cmd))
				new_segs.append(out_seg)
			
			sub['seg'] = new_segs

			new_gmsegs = []
			for gmseg in sub['gmseg']:
				out_seg = 'cropped_' + gmseg
				cmd = ['sct_crop_image', '-i', os.path.join(path, gmseg), 
					'-m', os.path.join(path, dilated_seg), '-o', os.path.join(path, out_seg)]
				sct.run(' '.join(cmd))
				new_gmsegs.append(out_seg)

			sub['gmseg'] = new_gmsegs

	save_json(subdata, jf)

def resample(jf, path):
	# Resample data to 0.3x0.3 in plane resolution
	sct.printv('Resampling data to common resolution', 1, 'info')
	subdata = open_json(jf)

	for s in subdata.keys():
		for sub in subdata[s].keys():
			subject = subdata[s][sub]

			# Need to get the current slice thickness
			img_path = os.path.join(subpath, subject['img'])
			input_im = Image(str(in_img_path))
			nx, ny, nz, nt, px, py, pz, pt = input_im.dim
			output_dim = '0.3x0.3x%s' % (str(pz))
			
			# This is a list of all the images we want to resample
			nii_to_resample = img + subject['seg'] + subject['gmseg']
			
			for nii in nii_to_resample:
				img_path = os.path.join(path, img)
				out_nii = sct.add_suffix(img_path, '_rs')

				cmd = ['sct_resample', '-i', in_img_path, '-mm', output_dim, '-o', out_img_path, '-v 0']
				sct.run(' '.join(cmd))

			# Update the file names
			subject['img'] = sct.add_suffix(subject['img'], '_rs')
			subject['seg'] = [sct.add_suffix(s, '_rs') for s in subject['seg']]
			subject['gmseg'] = [sct.add_suffix(s, '_rs') for s in subject['gmseg']]

	save_json(jdata, jf)

def add_tmp_fname(jf, path):

	subdata = open_json(jf)

	for ss in subdata.keys():
		for s in subdata[ss].keys():
			sub = subdata[ss][s]

			# Rename all the files
			new_name  = 'tmp__' + sub['img']
			# new_name = sct.add_suffix(new_name, '_img')
			os.rename(os.path.join(path, sub['img']), os.path.join(path, new_name))
			sub['img'] = new_name

			# Segmentations
			new_segs = []
			for s in sub['seg']:
				new_name = 'tmp__' + s
				os.rename(os.path.join(path, s), os.path.join(path, new_name))
				new_segs.append(new_name)

			sub['seg'] = new_segs

			# GM segs
			new_gmsegs = []
			for s in sub['gmseg']:
				new_name = 'tmp__' + s
				os.rename(os.path.join(path, s), os.path.join(path, new_name))
				new_gmsegs.append(new_name)

			sub['gmseg'] = new_gmsegs

	save_json(subdata, jf)

# def quality_check(jf):
# 	jdata = open_json(jf)['data']

# 	for s in jdata['subjects'].keys():
# 		print s
# 		sub = jdata['subjects'][s]
# 		p = sub['path']

# 		img1 = os.path.join(p, sub['img'])
# 		img2 = os.path.join(p, sub['img'].split('.')[0] + '_N4.nii.gz')

# 		out1 = os.path.join(p, 'original.png')
# 		out2 = os.path.join(p, 'N4corr.png')

# 		m = sp.check_output('fslstats ' + img1 + ' -p 99', shell=True)
# 		cmd1 = ['slicer', img1, '-n -a -i 0', m, out1]
# 		cmd2 = ['slicer', img2, '-n -a -i 0', m, out2]

# 		sp.call(' '.join(cmd1), shell=True)
# 		sp.call(' '.join(cmd2), shell=True)

# 		# list_im = [out1, out2]
# 		# imgs    = [ PIL.Image.open(i) for i in list_im ]
# 		# min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
# 		# imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
# 		# imgs_comb = PIL.Image.fromarray( imgs_comb)
# 		# imgs_comb.save( os.path.join(p, 'overview.png' ))

# def create_mask(jf):

# 	jdata = open_json(jf)
# 	subjdata = jdata['data']['subjects']

# 	for sub in subjdata.keys():
# 		path = subjdata[sub]['path']
# 		in_img = subjdata[sub]['img']
# 		in_img_path = os.path.join(path, in_img)

# 		mask = in_img.split('.')[0] + '_cordmask.nii.gz'
# 		mask_path = os.path.join(path, mask)
		
# 		subjdata[sub]['mask'] = mask

# 		cmd = ['sct_create_mask', '-i', in_img_path, '-p', 'center', 
# 			'-size', '40mm', '-f', 'box','-o', mask_path]

# 		sp.call(' '.join(cmd), shell=True)

# 	save_json(jdata, jf)

# def crop_segmentations(jf):

# 	jdata = open_json(jf)
# 	subdata = jdata['data']['subjects']

# 	sct.printv('Crop segmentations to same space as images',1, 'info')
	
# 	for s in subdata.keys():
# 		seg_list = []
# 		sub = subdata[s]
# 		path = subdata[s]['path']
# 		mask = os.path.join(path, sub['mask'])
# 		segmentations = sub['seg']
# 		for i in segmentations.keys():
# 			in_seg = segmentations[i]
# 			in_seg_path = os.path.join(path, in_seg)

# 			out_seg = in_seg.split('.nii.gz')[0] + '_crop.nii.gz'
# 			out_seg_path = os.path.join(path, out_seg)
# 			cmd = ['sct_crop_image', '-i', in_seg_path, '-m', mask, '-o', out_seg_path]
# 			cmd = ' '.join(cmd)
# 			sp.call(cmd, shell=True)
# 			segmentations[i] = out_seg

# 		sub['seg'] = segmentations

# 	save_json(jdata, jf) 

def make_vert_nifti(jf, path):

	subdata = open_json(jf)
	
	sct.printv('Creating nifti files from vertebrae level text files', 1, 'info')
	for s in subdata.keys():
		for sub in subdata[s].keys():
			img = subdata[s][sub]['img']
			img_path = os.path.join(path, img) 
			level_file = subdata[s][sub]['levels']
			level_path = str(os.path.join(path, level_file))

			out_level_file = sct.add_suffix(img, '_levels')
			out_level_path = os.path.join(path, out_level_file)
			
			# This function can handle both nifti and 
			vert_txt2nii(img_path, level_path, out_level_path)
			subdata[s][sub]['levels'] = out_level_file

	save_json(subdata, jf)

def denoise(jf, v=3, f=1, h=0.05):

	sct.printv('Running denoising', 1, 'info')
	jdata = open_json(jf)
	subdata = jdata['data']['subjects']
	
	for sub in subdata.keys():
		path = subdata[sub]['path']
		in_img = subdata[sub]['img']
		in_img_path = os.path.join(path, in_img)

		out_img = in_img.split('.nii.gz')[0] + '_dn.nii.gz'
		out_img_path = os.path.join(path, out_img)

		# Denoising parameters
		denoise_param = 'v=%s,f=%s,h=%s' % (str(v), str(f), str(h))
		cmd = ['sct_maths', '-i', in_img_path, '-denoise', denoise_param, '-o', out_img_path]
		cmd = ' '.join(cmd)
		sp.call(cmd, shell=True)

		subdata[sub]['img'] = out_img

	save_json(jdata, jf)

# def contrast_stretch(jf):
# 	jdata = open_json(jf)
# 	subdata = jdata['data']['subjects']

# 	for s in subdata.keys():
# 		path = subdata[s]['path']
# 		img = subdata[s]['img']

# 		in_img_path = os.path.join(path, img)
# 		# Make sure we read the N4 image!
# 		nii = nb.load(str(in_img_path))
# 		imdata = nii.get_data()
# 		imdata = cstretch(imdata, 1, 0, 99.9)

# 		# Save data with only contrast stretch from 0 to 1
# 		cstretch_name = sct.add_suffix(img,'_cs')
# 		out_nii = nb.Nifti1Image(imdata, nii.get_affine())
# 		nb.nifti1.save(out_nii, os.path.join(path, cstretch_name))
# 		subdata[s]['img'] = cstretch_name

# 	save_json(jdata, jf)

def plot_histograms(jf, path, qcpath):
	sct.printv('Making some nice histograms!', 1, 'info')
	subdata = open_json(jf)
	b = 50
	cs_hist = np.zeros([b-1,1])
	plt.figure(figsize=(10,10))
	for sub in subdata.keys():
		for s in subdata[sub].keys():
			fname = os.path.join(path, subdata[sub][s]['img'])
			data = nb.load(fname).get_data().flatten()
			y, binEdges=np.histogram(data,bins=b)
			bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
			y = y[1:]
			y = y * 1.0 / np.max(y)
			y = np.reshape(y, [b-1, 1])
			cs_hist = np.append(cs_hist,y, axis=1)

	bincenters = bincenters[1:]
	plt.plot(bincenters, np.mean(cs_hist,axis=1), linewidth=3)
	plt.title('Contrast Stretch average')
	low_sd = np.mean(cs_hist, axis=1) - np.std(cs_hist, axis=1)
	high_sd = np.mean(cs_hist, axis=1) + np.std(cs_hist, axis=1)
	plt.plot(bincenters, low_sd, '--b')
	plt.plot(bincenters, high_sd, '--b')
	plt.axis([0,1,0,1.2])
	plt.legend(['Average histogram', 'Std'])

	plt.savefig(os.path.join(qc_path, 'average_IRS_histograms.png'))

def move_final_files(jf, orgpath, newpath):

	sct.printv('Making a final move of data from the temp folder to the correct directory. Hang tight!')
	print newpath
	sct.run('mkdir %s' % newpath)
	subdata = open_json(jf)

	print subdata
	for ss in subdata.keys():
		for s in subdata[ss].keys():
			sub = subdata[ss][s]

			nii_to_move = [sub['img'], sub['levels']] + sub['seg'] + sub['gmseg']
			
			for nii in nii_to_move: 
				old_img_path = os.path.join(orgpath, nii)
				new_name = nii.split('tmp__')[-1]
				new_img_path = os.path.join(newpath, new_name)
				sct.run('cp %s %s' %(old_img_path, new_img_path))

	# Move pickel files as well
	old_pickle = os.path.join(orgpath, 'irs_model.pkl')
	new_pickle = os.path.join(newpath, 'irs_model.pkl')
	sct.run('cp %s %s' % (old_pickle, new_pickle))

	# Move in original database file as well
	tmp_org_json = os.path.join(orgpath, 'db_sorted_org.json')
	final_json = os.path.join(newpath, 'db.json')
	sct.run('cp %s %s' % (tmp_org_json, final_json))

def export_org_qc(jf, tmp_path, qcpath):
	subdata = open_json(jf)

	for s in subdata.keys():
		sub = subdata[s]['org']

		img = os.path.join(tmp_path, sub['img'])
		gmseg = os.path.join(tmp_path, sub['gmseg'][0])
		print str(img)
		I = Image(str(img))
		new_img_name = img.split('tmp__')[-1]
		I.setFileName(new_img_name)

		S = Image(str(gmseg))
		I.save_quality_control(plane='axial', n_slices=1, seg=S, thr=0, cmap_col='red', format='.png', path_output=qcpath, verbose=1)

def main(arguments):
	
	# Set timer for starting pre-processing
	tic = timeit.default_timer()

	# MASTER_TRAINING_FOLDER = '/Volumes/Monster/Deepseg/data/original/wT2s/training_augmented'
	MASTER_TRAINING_FOLDER = '/Volumes/Monster/Deepseg/data/original/wT2s/training_aug_subset'
	TMP_OUTPUT = os.path.join(arguments['-f'], 'tmp')
	QC_DIR = os.path.join(arguments['-f'], 'quality_control')

	# Start by copying all the data to the new directory so we don't overwrite data in the original directory
	# sct.printv('Copying all data to new directory before pre-processing', 1, 'info')
	# sct.run('mkdir %s' % arguments['-f'])
	# cmd = 'cp -rv ' + MASTER_TRAINING_FOLDER + ' ' + TMP_OUTPUT
	# sct.run(cmd)
	# jf = add_aug_training_data(TMP_OUTPUT)
	# tmp_json = jf.split('.json')[0] + '_org.json'

	# sct.run('cp %s %s' % (jf, tmp_json))

	# add_tmp_fname(jf, TMP_OUTPUT)

	# ------------  1. Resample data to 0.3x0.3 in-plane ------------ 
	#resample(jf, path)

	# ------------  2. Crop image and segmentation around the cord ------------ 	
	# crop_around_cord(jf, TMP_OUTPUT)

	# ------------ 5. Train IRS model ------------
	jf = os.path.join(TMP_OUTPUT, 'db_sorted.json')
	# train_IRS_model(jf, TMP_OUTPUT)

	# ------------ 6. Apply IRS model ------------
	# apply_all_IRS_model(jf, TMP_OUTPUT)

	# ------------ 7. Make vertebrae levels nifti file ------------
	# make_vert_nifti(jf, TMP_OUTPUT)

	# ------------ 10. QA of data. ------------
	# plot_histograms(jf, TMP_OUTPUT, QC_DIR) 	# <<< Will save as .png image

	# Move files to final destination
	FINAL_OUT = os.path.join(arguments['-f'], 'pre_processed')
	# move_final_files(jf, TMP_OUTPUT, FINAL_OUT)

	# Make QC Directory
	try:
		sct.run('mkdir %s' % QC_DIR)
	except:
		print 'Dir exists. Overwriting'

	export_org_qc(jf, TMP_OUTPUT, QC_DIR)

	# sct.run('rm -r %s' % TMP_OUTPUT)

	toc = timeit.default_timer()
	print 'Finished!'
	print 'Total time elapsed for analysis: %s' % str((toc-tic))

if __name__ == "__main__":
	parser = get_parser()
	arguments = parser.parse(sys.argv[1:])

	main(arguments)