#!/usr/bin/env python
#########################################################################################
#
# Utilities for pre-training and prediction using Deepseg
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 University of British Columbia and Polytechnique Montreal
# Authors: Emil Ljungberg
# Modified: 2016-08-12
#
# About the license: see the file LICENSE.TXT
#########################################################################################

import numpy as np
import subprocess as sp
import nibabel as nb
from msct_gmseg_utils import load_level

def antsN4BiasFieldCorrection(input_img, output_img=None, dim=3, mask=None, scale=None, weight=None, shrink=4, conv=None, bspline=None, histsharp=None):
	# If no option is given the script is run with standard parameters

	cmd = ['~/scripts/ANTS_YOSEMITE/bin/N4BiasFieldCorrection']

	# Input Image
	cmd.append('--input-image')
	cmd.append(input_img)

	# Output
	cmd.append('--output')
	if output_img:
		cmd.append(output_img)
	else:
		fname, fext = os.path.splitext(input_img)
		cmd.append(fname + '_N4corrected' + fext)
	
	# Dimensions to calculate the bias field
	cmd.append('--image-dimensionality')
	cmd.append(str(dim))

	# Mask to contstrain the bias field calculation
	if mask:
		cmd.append('--mask-image')
		cmd.append(mask)

	# Rescale image at each iteration
	if scale:
		cmd.append('--rescale-intensities')

	# Image weight during bspline fitting
	if weight:
		cmd.append('--weight-image')
		cmd.append(weight) # Image file

	# Shrink factor
	if shrink:
		cmd.append('--shrink-factor')
		cmd.append(str(shrink))

	# Convergence criteria
	if conv:
		cmd.append('--convergence')
		cmd.append(conv)

	# Bspline fitting options  --> Still requires fix on the input arguments!
	if bspline:
		cmd.append('--bspline-fitting')
		cmd.append(bspline)

	# Histogram sharpening
	if histsharp:
		cmd.append('--histogram-sharpening')
		cmd.append(hist)

	# Run command
	cmd = ' '.join(cmd)
	print cmd
	sp.call(cmd, shell=True)

	return output_img

def vert_txt2nii(img, vlevel, output_nii):

	img_nii = nb.load(img)

	if 'nii' in vlevel:
		level_nii = nb.load(vlevel)
		level_data = level_nii.get_data()
		cord_levels = []
		for i in range(level_nii.shape[2]):
			cord_levels.append(np.max(level_data[:,:,i]))

	elif 'txt' in vlevel:
		cord_levels = load_level(vlevel)

	else:
		# Throw exception
		print 'ERROR: Invalid level file'

	level_data = np.ones(img_nii.shape)
	
	for i in range(img_nii.shape[2]):
		level_data[:,:,i] = level_data[:,:,i] * cord_levels[i]

	level_nii = nb.Nifti1Image(level_data, img_nii.get_affine())
	nb.save(level_nii, output_nii)

	return

def cstretch(a, r1, r0, p):
    lmin = float(np.amin(a))
    lmax = float(np.percentile(a, p))
    anorm = (a - lmin)*(r1 - r0)/(lmax - lmin)
    return anorm

def IRS_transformation(irs, imdata, segdata=None):

	imdata = cstretch(imdata, irs.stdrange[1], irs.stdrange[0], 99.9)

	if segdata != None:
		# If we have the segmentation we can pick out the cord coordinates
		# from the non-zero indicies in the seg
		nz_idx = np.nonzero(segdata)
	else:
		nz_idx = np.nonzero(imdata)
		
	imval = []
	for i in range(len(nz_idx[0])):
		imval.append(imdata[nz_idx[0][i], nz_idx[1][i], nz_idx[2][i]])

	trans_imval = irs.transform(imval)
	new_data = np.zeros(np.shape(imdata))

	for i in range(len(imval)):
		new_data[nz_idx[0][i], nz_idx[1][i], nz_idx[2][i]] = trans_imval[i]

	return new_data
