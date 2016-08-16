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
		'''Tool for pre-processing Deepseg training data. Input data should be organized in a single folder with the image having no suffix. GM segmentations having _gmseg(1..N) and same for cord segmentations.''')

	parser.add_option(name="-i",
					  type_value="folder",
					  description="Training data",
					  mandatory=True,
					  example="./input_training_data")
	parser.add_option(name="-o",
					  type_value="folder",
					  description="Output for processed training data",
					  mandatory=True,
					  example="./processed_training_data")

	return parser


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
	sct.printv('Parsing the data into json structure', 1, 'info')
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

def open_json(jfile):
	with open(jfile) as jf:
		jdata = json.load(jf)

	return jdata

def save_json(jdata, fname):

	with open(fname, 'w') as jf:
		json.dump(jdata, jf)
	return True

# Processing functions
def train_IRS_model(jf, path):
	
	sct.printv('Training Intensity Range Standardization Model on non-augmented data', 1, 'info')
	subdata = open_json(jf)

	all_img_data_list = []
	for s in subdata.keys():
		# We only need to train to model on the original data
		img = subdata[s]['org']['img']
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

	cp = (0,99)
	lp = [10, 20, 30, 40, 50, 60, 70, 80, 90]
	sr = (0,0.8)

	irs = IntensityRangeStandardization(cutoffp=cp, landmarkp=lp, stdrange=sr)
	irs_model = irs.train(all_img_data_list)

	irs_path = os.path.join(path, 'irs_model.pkl')
	
	# Save the irs model as pickle
	with open(irs_path, 'w') as pf:
		pickle.dump(irs_model, pf)


	
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

	save_json(subdata, jf)

def crop_around_cord(jf, path):
	
	subdata = open_json(jf)
	sct.printv('Cropping image around the spinal cord', 1, 'info')

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

def plot_histograms(jf, path, qcpath):
	sct.printv('Generating histogram for all images', 1, 'info')
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

	plt.savefig(os.path.join(qcpath, 'average_IRS_histograms.png'))

def move_final_files(jf, orgpath, newpath):

	sct.printv('Moving files from temporary directory to output')
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
	sct.printv('Exporting QC images from ')
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

def input_data_summary(jf):
	subdata = open_json(jf)

	n_sub = 0
	n_img = 0

	for k in subdata.keys():
		n_sub += 1
		n_img += len(subdata[k].keys())

	sct.printv('Summary of input training data:', 1, 'info')
	sct.printv('\t Number of subjects: %s' % n_sub)
	sct.printv('\t Number of iamges: %s' % n_img)

def main(arguments):
	
	# Set timer for starting pre-processing
	tic = timeit.default_timer()

	MASTER_TRAINING_FOLDER = arguments['-i']
	TMP_OUTPUT = os.path.join(arguments['-o'], 'tmp')
	QC_DIR = os.path.join(arguments['-o'], 'quality_control')

	# Start by copying all the data to the new directory so we don't overwrite data in the original directory
	sct.printv('Copying all data to new directory before pre-processing', 1, 'info')
	sct.run('mkdir %s' % arguments['-o'])
	cmd = 'cp -rv ' + MASTER_TRAINING_FOLDER + ' ' + TMP_OUTPUT
	sct.run(cmd)

	jf = add_aug_training_data(TMP_OUTPUT)
	tmp_json = jf.split('.json')[0] + '_org.json'
	sct.run('cp %s %s' % (jf, tmp_json))

	input_data_summary(jf)

	add_tmp_fname(jf, TMP_OUTPUT)

	# ------------  1. Resample data to 0.3x0.3 in-plane ------------ 
	#resample(jf, path)

	# ------------  2. Crop image and segmentation around the cord ------------ 	
	crop_around_cord(jf, TMP_OUTPUT)

	# ------------ 5. Train IRS model ------------
	train_IRS_model(jf, TMP_OUTPUT)

	# ------------ 6. Apply IRS model ------------
	apply_all_IRS_model(jf, TMP_OUTPUT)

	# ------------ 7. Make vertebrae levels nifti file ------------
	make_vert_nifti(jf, TMP_OUTPUT)

	# ------------ 10. QA of data. ------------
	try:
		sct.run('mkdir %s' % QC_DIR)
	except:
		print 'Dir exists. Overwriting'

	plot_histograms(jf, TMP_OUTPUT, QC_DIR) 	# <<< Will save as .png image

	# Move files to final destination
	FINAL_OUT = os.path.join(arguments['-o'], 'pre_processed')
	move_final_files(jf, TMP_OUTPUT, FINAL_OUT)
	export_org_qc(jf, TMP_OUTPUT, QC_DIR)

	# sct.run('rm -r %s' % TMP_OUTPUT)

	toc = timeit.default_timer()
	print 'Finished!'
	print 'Total time elapsed for analysis: %s' % str((toc-tic))

if __name__ == "__main__":
	parser = get_parser()
	arguments = parser.parse(sys.argv[1:])

	main(arguments)