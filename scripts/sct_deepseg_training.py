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
from get_input_data_json import *
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

# Helper function #

def add_training_data(jf, path):
		
	data = open_json(jf)

	i = 1
	for site in ['site3', 'site4']:
		comment = 'Data from %s' % site
		for sub in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
			subject = {}
			
			subject['path'] = path

			# Set image
			img_fname = site + '-sc' + sub + '-image.nii.gz'
			subject['img'] = img_fname
			
			# Add all 4 masks
			subject['seg'] = {}
			for j in range(1,5):
				seg_fname = site + '-sc' + sub + '-mask-r%s.nii.gz' % str(j)
				subject['seg'][j] = seg_fname
			
			# Set levels
			levels_fname = site + '-sc' + sub + '-levels.txt'
			subject['levels'] = levels_fname
			subject['comment'] = comment

			data['data']['subjects'][i] = subject
			i += 1

	comment = 'Data from site2 in the ISMRM training set.'
	site = 'site2'
	for sub in ['02', '03', '04', '05', '06', '07', '08', '09', '10']:
		# There is something wrong with subject 01 thus no import
		subject = {}
		
		subject['path'] = path

		# Set image
		img_fname = site + '-sc' + sub + '-image.nii.gz'
		subject['img'] = img_fname
		
		# Add all 4 masks
		subject['seg'] = {}
		for j in range(1,5):
			seg_fname = site + '-sc' + sub + '-mask-r%s.nii.gz' % str(j)
			subject['seg'][j] = seg_fname
		
		# Set levels
		levels_fname = site + '-sc' + sub + '-levels.txt'
		subject['levels'] = levels_fname
		subject['comment'] = comment

		data['data']['subjects'][i] = subject
		i += 1

	# Spacing is wrong in site 1 data. Don't know how to fix it. Cannot run with the 
	# N4 biasfield correction

	# Add in Camilles data here as well

	save_json(data, jf)
	
	return jf

def add_aug_training_data(jf, path):

	jdata = open_json(jf)
	i = 1

	# First parse all the subject IDs
	subjects = {}
	i = 1
	for f in files:
		if '_seg.nii.gz' in f:
			s = {}
			s['original'] = {}
			sname = s.split('_seg.nii.gz')[0]
			s['original']['subj'] = sname
			subjects[i] = s
			i += 1

	# Go through the data again and pick up the original data and augmented data
	for s in subjects.keys():
		sub = subjects[s]
		sub['aug'] = {}
		basename = sub['original']['subj']

		# Original data
		sub['original']['img'] = 
		sub['original']['seg'] = 
		sub['original']['levels'] =

		gmseg = {}
		# See if there is more than one gm seg. Populate the dictionary

		sub['original']['gmseg'] = gmseg

		for f in files:
			if basename in f:
				if f == (basename + '.nii.gz'):
					sub['img'] = f
				elif f == (basename + '_gmseg.nii.gz'):
					sub['gmseg'] = f

	# For each subject
	subject = {}
	subject['path'] = path
	subject['img'] = img
	subject['cordseg'] = cordseg
	subject['gmseg'] = gmseg
	subject['levels'] = level_file

	jdata['data']['subjects'][i] = subject
	
	# Hopefully we can parse the augmentation about each file so we know what we are working with
	subject['comment'] = comment

	save_json(jdata, jf)


def check_files(data):
	subj_data = data['data']['subjects']

	for s in subj_data.keys():
		subj = subj_data[s]
		img = os.path.join(subj['path'], subj['img'])

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

def set_finish_time(jf):
	data = open_json(jf)
	data['Info']['Analysis finished'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
	save_json(data, jf)

# Processing functions

def train_IRS_model(jf):
	
	sct.printv('Training Intensity Range Standardization Model', 1, 'info')
	jdata = open_json(jf)
	subdata = jdata['data']['subjects']

	img_data_list = []

	for sub in subdata.keys():
		path = subdata[sub]['path']
		img = subdata[sub]['img']
		img_path = os.path.join(path, img)

		nii = nb.load(str(img_path))
		imdata = nii.get_data()		
		imdata = cstretch(imdata, 0.8, 0, 100)

		# New part to remove 0 around the cord
		cordseg_data = nb.load(os.path.join(path, 't2s_seg1_cord.nii.gz')).get_data()
		
		imdata = imdata[cordseg_data > 0]
		img_data_list.append(imdata)
	
	cp = (0,99)
	lp = [10, 20, 30, 40, 50, 60, 70, 80, 90]
	sr = (0,0.8)
	irs = IntensityRangeStandardization(
    	cutoffp=cp, landmarkp=lp, stdrange=sr)
	irs_model = irs.train(img_data_list)

	irs_path = os.path.join(jdata['Info']['Path'], 'irs_model.pkl')
	jdata['Processing']['IRS']['model'] = irs_path
	jdata['Processing']['IRS']['cutoffp'] = cp
	jdata['Processing']['IRS']['landmarkp'] = lp
	jdata['Processing']['IRS']['stdrange'] = sr 

	# Save the irs model as pickle
	with open(irs_path, 'w') as pf:
		pickle.dump(irs, pf)

	add_event(jf, 'Created IRS model')

	# Make a note in the json events

	save_json(jdata,jf)

def apply_all_IRS_model(jf):
	
	# Read json file
	# load IRS model
	jdata = open_json(jf)
	subdata = jdata['data']['subjects']
	irs_model = jdata['Processing']['IRS']['model']

	sct.printv('Loading pickle file with IRS model', 1, 'info')
	with open(irs_model, 'r') as pf:
		irs_obj = pickle.load(pf)

	sct.printv('Transforming images with IRS model', 1, 'info')
	for sub in subdata.keys():
		path = subdata[sub]['path']
		in_img = subdata[sub]['img']
		print 'Transforming: ' + in_img

		in_img_path = os.path.join(path, in_img)
		nii = nb.load(str(in_img_path))
		imdata = nii.get_data()
		
		irs_data = IRS_transformation(irs_obj, imdata)

		# Save image with histogram normalization
		irs_name = sct.add_suffic(in_img, '_irs')
 		out_nii = nb.Nifti1Image(irs_data, nii.get_affine())
		nb.nifti1.save(out_nii, os.path.join(path, irs_name))

		subdata[sub]['img'] = irs_name

	# Add event to json
	add_event(jf, 'Transformed all data using model')

	save_json(jdata, jf)

def apply_N4_correction(jf):
	img_list = get_image_list(jf)
	jdata = open_json(jf)
	subjdata = jdata['data']['subjects']

	sct.printv('Performing N4 bias field correction', 1, 'info')
	for sub in subjdata.keys():
		path = subjdata[sub]['path']

		img = subjdata[sub]['img']
		in_img_path = os.path.join(path, img)
		out_img = img.split('.nii.gz')[0] + '_N4.nii.gz'
		out_img_path = os.path.join(path, out_img)

		antsN4BiasFieldCorrection(input_img=in_img_path, output_img=out_img_path)
		subjdata[sub]['img'] = out_img

	add_event(jf, 'Applied N4 bias field correction to training image')

	save_json(jdata, jf)

def quality_check(jf):
	jdata = open_json(jf)['data']

	for s in jdata['subjects'].keys():
		print s
		sub = jdata['subjects'][s]
		p = sub['path']

		img1 = os.path.join(p, sub['img'])
		img2 = os.path.join(p, sub['img'].split('.')[0] + '_N4.nii.gz')

		out1 = os.path.join(p, 'original.png')
		out2 = os.path.join(p, 'N4corr.png')

		m = sp.check_output('fslstats ' + img1 + ' -p 99', shell=True)
		cmd1 = ['slicer', img1, '-n -a -i 0', m, out1]
		cmd2 = ['slicer', img2, '-n -a -i 0', m, out2]

		sp.call(' '.join(cmd1), shell=True)
		sp.call(' '.join(cmd2), shell=True)

		# list_im = [out1, out2]
		# imgs    = [ PIL.Image.open(i) for i in list_im ]
		# min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
		# imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
		# imgs_comb = PIL.Image.fromarray( imgs_comb)
		# imgs_comb.save( os.path.join(p, 'overview.png' ))

def create_mask(jf):

	jdata = open_json(jf)
	subjdata = jdata['data']['subjects']

	for sub in subjdata.keys():
		path = subjdata[sub]['path']
		in_img = subjdata[sub]['img']
		in_img_path = os.path.join(path, in_img)

		mask = in_img.split('.')[0] + '_cordmask.nii.gz'
		mask_path = os.path.join(path, mask)
		
		subjdata[sub]['mask'] = mask

		cmd = ['sct_create_mask', '-i', in_img_path, '-p', 'center', 
			'-size', '40mm', '-f', 'box','-o', mask_path]

		sp.call(' '.join(cmd), shell=True)

	save_json(jdata, jf)

def crop_segmentations(jf):

	jdata = open_json(jf)
	subdata = jdata['data']['subjects']

	sct.printv('Crop segmentations to same space as images',1, 'info')
	
	for s in subdata.keys():
		seg_list = []
		sub = subdata[s]
		path = subdata[s]['path']
		mask = os.path.join(path, sub['mask'])
		segmentations = sub['seg']
		for i in segmentations.keys():
			in_seg = segmentations[i]
			in_seg_path = os.path.join(path, in_seg)

			out_seg = in_seg.split('.nii.gz')[0] + '_crop.nii.gz'
			out_seg_path = os.path.join(path, out_seg)
			cmd = ['sct_crop_image', '-i', in_seg_path, '-m', mask, '-o', out_seg_path]
			cmd = ' '.join(cmd)
			sp.call(cmd, shell=True)
			segmentations[i] = out_seg

		sub['seg'] = segmentations

	save_json(jdata, jf) 

def make_vert_nifti(jf):

	jdata = open_json(jf)
	subdata = jdata['data']['subjects']

	sct.printv('Creating nifti files from vertebrae level text files', 1, 'info')
	for sub in subdata.keys():
		path = subdata[sub]['path']
		img_path = os.path.join(path, subdata[sub]['img']) 
		level_file = subdata[sub]['levels']
		level_path = str(os.path.join(path, level_file))

		output_nii = os.path.join(path, 'levels.nii.gz')
		vert_txt2nii(img_path, level_path, output_nii)

		subdata[sub]['level_nii'] = 'levels.nii.gz'

	save_json(jdata, jf)

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

def resample(jf):
	# Resample data to 0.3x0.3 in plane resolution

	jdata = open_json(jf)
	subjdata = jdata['data']['subjects']

	# Resample N4 image
	for sub in subjdata.keys():
		subpath = subjdata[sub]['path']
		img = subjdata[sub]['img']
		in_img_path = os.path.join(subpath, img)
		input_im = Image(str(in_img_path))
		nx, ny, nz, nt, px, py, pz, pt = input_im.dim

		out_img = img.split('.nii.gz')[0] + '_rs.nii.gz'
		out_img_path = os.path.join(subpath, out_img)
		output_dim = '0.3x0.3x%s' % (str(pz))
		
		cmd = ['sct_resample', '-i', in_img_path, '-mm', output_dim, '-o', out_img_path, '-v 0']
		cmd = ' '.join(cmd)
		sp.call(cmd, shell=True)

		subjdata[sub]['img'] = out_img

	for sub in subjdata.keys():
		segs = subjdata[sub]['seg']
		path = subjdata[sub]['path']

		for i in segs.keys():
			input_seg = segs[i]
			input_seg_path = os.path.join(path, input_seg)
			input_seg_I = Image(str(input_seg_path))

			nx, ny, nz, nt, px, py, pz, pt = input_seg_I.dim
			out_seg = input_seg.split('.nii.gz')[0] + '_rs.nii.gz'
			out_seg_path = os.path.join(path, out_seg)
			output_dim = '0.3x0.3x%s' % (str(pz))
			
			cmd = ['sct_resample', '-i', input_seg_path, '-mm', output_dim, '-o', out_seg_path, '-v 0']
			cmd = ' '.join(cmd)
			sp.call(cmd, shell=True)

			segs[i] = out_seg

	save_json(jdata, jf)

def crop_images(jf):
	jdata = open_json(jf)
	subdata = jdata['data']['subjects']

	sct.printv('Applying mask to image', 1, 'info')
	img_data_list = []

	for sub in subdata.keys():
		path = subdata[sub]['path']
		in_mask = subdata[sub]['mask']
		in_img = subdata[sub]['img']
		cropped_img = in_img.split('.')[0] + '_crop.nii.gz'

		in_img_path = os.path.join(path, in_img)
		in_mask_path = os.path.join(path, in_mask)
		out_img_path = os.path.join(path, cropped_img)

		cmd = ['sct_crop_image', '-i', in_img_path, '-m', in_mask_path, '-o', out_img_path, '-v', '0']
		sp.call(' '.join(cmd), shell=True)

		subdata[sub]['img'] = cropped_img

	save_json(jdata, jf)

def crop_around_cord(jf):
	jdata = open_json(jf)
	subdata = jdata['data']['subjects']

	for s in subdata.keys():
		sub = subdata[s]
		seg = sub['seg']['1']
		path = sub['path']
		img = sub['img']
		cordseg = sct.add_suffix(seg, '_cord')
		cmd = ['sct_maths', '-i', os.path.join(path, seg), '-bin', '-o', os.path.join(path, cordseg)]
		sct.run(' '.join(cmd))

		cord_img = sct.add_suffix(img, '_cord')
		cmd = ['sct_maths', '-i', os.path.join(path, img), '-mul', os.path.join(path, cordseg),
		'-o', os.path.join(path, cord_img)]
		sct.run(' '.join(cmd))
		img = cord_img

		cord_img = sct.add_suffix(img, '_crop')
		cmd = ['sct_crop_image', '-i', os.path.join(path, img), '-m', os.path.join(path, cordseg), '-o', os.path.join(path, cord_img)]
		sct.run(' '.join(cmd))
		sub['img'] = cord_img
		img = cord_img

		for i in sub['seg'].keys():
			
			crop_seg = sct.add_suffix(cordseg, '_crop')
			cmd = ['sct_crop_image', '-i', os.path.join(path, cordseg), '-m', os.path.join(path, cordseg), '-o', os.path.join(path, crop_seg)]
			sct.run(' '.join(cmd))
			sub['seg']['']

	save_json(jdata, jf)

def contrast_stretch(jf):
	jdata = open_json(jf)
	subdata = jdata['data']['subjects']

	for s in subdata.keys():
		path = subdata[s]['path']
		img = subdata[s]['img']

		in_img_path = os.path.join(path, img)
		# Make sure we read the N4 image!
		nii = nb.load(str(in_img_path))
		imdata = nii.get_data()
		imdata = cstretch(imdata, 1, 0, 99.9)

		# Save data with only contrast stretch from 0 to 1
		cstretch_name = sct.add_suffix(img,'_cs')
		out_nii = nb.Nifti1Image(imdata, nii.get_affine())
		nb.nifti1.save(out_nii, os.path.join(path, cstretch_name))
		subdata[s]['img'] = cstretch_name

	save_json(jdata, jf)

def plot_histograms(jf):
	sct.printv('Making some nice histograms!', 1, 'info')
	jdata = open_json(jf)
	subdata = jdata['data']['subjects']
	b = 50
	cs_hist = np.zeros([b-1,1])
	plt.figure(figsize=(10,10))
	for s in subdata.keys():
		fname = os.path.join(subdata[s]['path'], subdata[s]['img'])
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

	plt.savefif('average_IRS_histograms.png')

def main(arguments):
	
	data_folder = arguments['-f']

	# ------------ 0. Add in all training data to the json structure ------------
	# TODO: Separate GM and cord segmentations. They will be used for different things!

	if not os.path.exists(data_folder):
		jf = init_json(data_folder)
		jf = add_training_data(jf, '/Users/emil/Desktop/Deepseg/trainingdata_GM_challenge_with_level_nii/')
	else:
		jf = os.path.join(data_folder, 'db.json')

	
	# ------------ 1. Move the data to a appropriate folder structure ------------ 
	move_data(jf, path=data_folder)

	# ------------ 2. Bias field correction ------------ 
	apply_N4_correction(jf)

	# ------------  3. Resample data to 0.3x0.3 in-plane ------------ 
	#resample(jf)

	# (Creat mask around the cord)
	#create_mask(jf)
	# Crop image
	#crop_images(jf)

	# ------------  4. Crop image and segmentation around the cord ------------ 
	# TODO: Make the function crop the cord and gm segmentations
	crop_around_cord(jf)

	# (5. Denoise data - Will take too long time...)
	#denoise(jf)

	# ------------ 5. Train IRS model ------------
	train_IRS_model(jf)

	# ------------ 6. Apply IRS model ------------
	apply_all_IRS_model(jf)

	# 8. Transform segmentations to the same space as well
	#crop_segmentations(jf)

	# ------------ 7. Make vertebrae levels nifti file ------------
	make_vert_nifti(jf)

	# ------------ 8. Conclude Pre-processing ------------
	set_finish_time(jf)

	# ------------ 10. QA of data. ------------
	plot_histograms(jf) 	# <<< Will save as .png image

	print 'Finished!'

if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    main(arguments)

    











