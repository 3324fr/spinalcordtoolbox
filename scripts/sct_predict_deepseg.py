#!/usr/bin/env python
#########################################################################################
#
# Precition module for the for the Deepseg binary.
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
import sct_utils as sct
from msct_deepseg_utils import *

# Scientific libraries
import numpy as np
import pickle

# Python standard libraries
import sys
import os
import shutil


def get_parser():
    # Initialize the parser
	parser = Parser(__file__)
	parser.usage.set_description(
		'''Deepseg is a deep learning tool using Convolutional Neural networks for gray matter segmentation.''')
	
	parser.usage.addSection("Mandatory arguments")
	parser.add_option(name="-i",
					type_value="file",
					description="input image.",
					mandatory=True,
					example="t2s.nii.gz")
	parser.add_option(name="-s",
					type_value="file",
					description="spinal cord segmentation",
					mandatory=True,
					example='t2s_seg.nii.gz')

	parser.usage.addSection("Optional arguments")
	parser.add_option(name="-t",
					type_value="file",
					description="Textfile with vertebra levels for to help prediction",
					mandatory=False,
					example='levels.txt')
	parser.add_option(name="-denoise",
					type_value=None,
					description="Denoise the input data",
					mandatory=False)
	parser.add_option(name="-bias-correction",
					type_value=None,
					description="Bias field correction",
					mandatory=False)
	parser.add_option(name="-qc",
					type_value=None,
					description="Output quality control of segmentation",
					mandatory=False)

	parser.add_option(name="-v",
					type_value="multiple_choice",
					description="Verbose. 1: display on, 0: display off (default=1)",
					mandatory=False,
					example=["0", "1"],
					default_value="1")

	parser.add_option(name="-rm",
					type_value="multiple_choice",
					description="1: Remove temp files, 0: Keep temp files (default=1)",
					mandatory=False,
					example=["0", "1"],
					default_value="1")



	return parser

def normalize(a, r0=0, r1=1):
	# Linear intensity normalization using
	# Contrast stretching
	# Input:
	# 	a - Data array
	# 	r0 - Lower bound of new range
	#	r1 - Upper bound of new range

	lmin = float(np.amin(a))
	# lmax = float(np.amax(a))
	lmax = float(np.percentile(a, 99.9))
	anorm = (a - lmin)*(r1 - r0)/(lmax - lmin)

	return anorm

def pad_image(cord, seg, nii_dim, fname, ext):

	# nii_dim = I.dim
	nx = nii_dim[0]
	ny = nii_dim[1]
	nz = nii_dim[2]

	dx = nii_dim[4]
	dy = nii_dim[5]
	dz = nii_dim[6]

	xfactor = 1 + int(nx > 0.4)
	yfactor = 1 + int(nx > 0.4)

	# Padding values
	px = (max(512, nx*xfactor) - nx) / 2
	py = (max(512, ny*yfactor) - ny) / 2
	pz = dz

	padding = '%d,%d,%d' % (px, py, pz)

	# Pad Cord
	cmd = ['sct_image', '-i', cord, '-pad']
	cmd.append(padding)	
	cmd.append('-o')
	padded_cord = fname + '_padded' + ext
	cmd.append(padded_cord)
	cmd = ' '.join(cmd)
	sct.run(cmd)

	# Pad segmentation
	cmd = ['sct_image', '-i', seg, '-pad']
	cmd.append(padding)	
	cmd.append('-o')
	padded_seg = fname + '_seg_padded' + ext
	cmd.append(padded_seg)
	cmd = ' '.join(cmd)
	sct.run(cmd)

	return padded_cord, padded_seg

def crop_center_cord(cord, seg, fname, ext):

	fname_mask = fname + '_mask' + ext

	cmd = ['sct_create_mask', '-i', cord, '-p', 'centerline,'+ seg, 
		'-f', 'box', '-size', str(256) ,'-o', fname_mask]

	cmd = ' '.join(cmd)
	sct.run(cmd)

	cord_cropped = fname + '_cropped' + ext
	cmd = ['sct_crop_image', '-i', cord, '-m', fname_mask, '-o', cord_cropped]
	cmd = ' '.join(cmd)
	sct.run()

	return cord_cropped





	# 2. Crop data around cord
	# Image size= 100x100 in plane with 10 slices. Padded if less
	# sct_create_mask

	return fname_istretch

def run_deepseg(prepared_data, vertfile):

	if vertfile:
		# Run through model with vertfile
		model = deepseg_vertebare_model
	else:
		# Run through model witout vertfile
		model = deepseg_novertebare_model
		
	# 1. Run through the binary

	# cmd = ["isct_deepseg", "-i", prepared_data, "-s"]
	# sct.run(cmd, verbose)

	return True

def make_vert_nii(seg, vlevel, vert_nii=None, path=None):

	I = Image(seg)

	cord_levels = load_level(vlevel)

	for i in range(I.dim[2]):
		I.data[:,:,i] = I.data[:,:,i] * cord_levels[i]
	if vert_nii:
		I.setFileName(vert_nii)
	else:
		I.setFileName(I.file_name + '_levels')
	
	if path:
		I.path = path
	else:
		I.path = './'

	I.save()

def main(arguments):
	verbose = 1
	img = arguments["-i"]
	seg = arguments["-s"]

	rm_temp = True				# Add option w. descriotion
	verbose = True 				# Add option w. descriotion

	# ------------ Step 0. Make temporary file ------------
	tmp_path = sct.tmp_create(verbose)
	sct.tmp_copy_nifti(img, tmp_path, 'data.nii')
	sct.tmp_copy_nifti(seg, tmp_path, 'data_seg.nii')
	org_img = img
	img = os.path.join(tmp_path, 'data.nii')
	seg = os.path.join(tmp_path, 'data_seg.nii')

	# ------------ Step 1. Apply biasfield correction ------------
	# Return temp.t2s_N4.nii.gz
	if "-bias-correction" in arguments:
		sct.printv('Applying N4 bias field correction', 1, 'info')
		img_out = sct.add_suffix(img, '_N4')
		img = antsN4BiasFieldCorrection(img, output_img=img_out, dim=3, mask=None, scale=0, weight=None, shrink=4, conv=None, bspline=None, histsharp=None)
		img = img_out

	# ------------ Step 2. Denoise input data ------------
	if "-denoise" in arguments:
		sct.printv('Denoising data', 1, 'info')
		v=3
		f=1
		h=0.01
		img_out = sct.add_suffix(img, '_dn')

		denoise_param = 'v=%s,f=%s,h=%s' % (str(v), str(f), str(h))
		cmd = ['sct_maths', '-i', img, '-denoise', denoise_param, '-o', img_out]
		status, output = sct.run(' '.join(cmd), verbose)
		img = img_out

	# ------------ Step 3. Extract cord through multiplication with seg ------------
	sct.printv('Creating cropping mask around cord based on segmentation', 1, 'info')
	out_img = sct.add_suffix(img, '_cord')
	cmd = ['sct_maths', '-i', img, '-mul', seg, '-o', out_img]
	sct.run(' '.join(cmd), verbose)
	img = out_img

	# ------------ Step 4. Crop image and segmentation around cord ------------
	sct.printv('Cropping image based on mask', 1, 'info')
	out_img = sct.add_suffix(img, '_crop')
	cmd = ['sct_crop_image', '-i', img, '-m', seg, '-o', out_img]
	sct.run(' '.join(cmd), verbose)
	img = out_img

	sct.printv('Cropping segmentation based on segmentation', 1, 'info')
	out_seg = sct.add_suffix(seg, '_crop')
	cmd = ['sct_crop_image', '-i', seg, '-m', seg, '-o', out_seg]
	sct.run(' '.join(cmd), verbose)
	seg = out_seg

	# ------------ Step 5. Apply IRS model ----------------
	# sct.printv('Applying intensity normalization model', 1, 'info')
	# irs_model = './irs_model.pkl'
	# with open(irs_model, 'r') as pf:
	# 	irs_obj = pickle.load(pf)

	# I = Image(img)
	# imdata = I.data
	# S = Image(seg)
	# segdata = S.data

	# new_data = IRS_transformation(irs_obj, imdata, segdata)
	# I.data = new_data
	# out_img = sct.add_suffix(img, '_irs')
	# I.setFileName(out_img)
	# I.save()
	# img = out_img

	# ------------ Step 6. Make vert nii ------------
	if "-t" in arguments:
		vlevel = arguments['-t']
		make_vert_nii(seg, vlevel, 'vert_level.nii', tmp_path)
		levels_provided = True
	else:
		sct.printv('No vertfile given. Will perform prediction without vertebrae level information', 1, 'warning')
		levels_provided = False
		
	# ------------ Step 7. Run data through Deepseg binary ------------
	deepseg_prob = os.path.join(tmp_path, 'deepseg_prob.nii.gz')
	deepseg_bin = os.path.join(tmp_path, 'deepseg_bin.nii.gz')

	if levels_provided:
		sct.printv('6. Performing gray matter prediction using Deepseg w. levels', 1, 'info')
	else:
		sct.printv('6. Performing gray matter prediction using Deepseg wo. levels', 1, 'info')

	# ------------ Step 8. Reformat data to original size and resolution ------------
	sct.printv('7. Reformatting data to input space', 1, 'info')

	path_fname, file_fname, ext_fname = sct.extract_fname(org_img)
	tmp_timestamp = os.path.dirname(tmp_path).split('.')[-1]
	tmp_dest_img = 'reg_dest_' + tmp_timestamp + ext_fname
	tmp_dest_path = os.path.join(tmp_path, tmp_dest_img)
	shutil.copy(org_img, tmp_dest_path)

	source = img 	# <<<< For testing purpose only
	destination = tmp_dest_path
	cmd = ['sct_register_multimodal', '-i', source, '-d', destination, '-identity 1 -x nn', '-o', 'deepseg_prob.nii.gz']
	sct.run(' '.join(cmd), verbose) 

	# Files will now end up in the main folder and we need to remove them
	path_source, file_source, ext_source = sct.extract_fname(source)
	path_dest, file_dest, ext_dest = sct.extract_fname(destination)
	os.remove('warp_' + file_source + '2' + file_dest + ext_dest)
	os.remove('warp_' + file_dest + '2' + file_source + ext_dest)
	os.remove(file_dest + '_reg' + ext_dest)

	#cmd = ['sct_register_multimodal', '-i', deepseg_bin, '-d', org_img, '-identity 1', '-o', 'deepseg_bin.nii.gz']
	#sct.run(' '.join(cmd), verbose)

	# Registration we want to remove

	# ------------ Step 9. Run QC on output data ------------
	if "-qc" in arguments:
		sct.printv('Creating images for QC', 1, 'info')
		im = Image(org_img)
		im_gmseg = Image('deepseg_prob.nii.gz')
		im.save_quality_control(plane='axial', n_slices=5, seg=im_gmseg, thr=float(0.01), cmap_col='red-yellow', path_output='./')

	sct.printv('Segmentation finished. To view results:')
	sct.printv('fslview ' + org_img + 'deepseg_prob.nii.gz -l "Blue-Lightblue" -t 0.7 deepseg_bin.nii.gz -l Red -t 0.7 &')


	# ------------ Step 10. Remove temporary files ------------
	if arguments['-rm'] == "1":
		sct.printv('Removing temp folder', 1, 'info')
		print tmp_path
		shutil.rmtree(tmp_path)

if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    main(arguments)