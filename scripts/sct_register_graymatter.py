#!/usr/bin/env python
#
# This program returns the grey matter segmentation given anatomical, landmarks and t2star images
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener, Augustin Roux
# Created: 2014-10-18
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sct_utils as sct
from msct_parser import Parser
from msct_image import Image
from msct_multiatlas_seg import Param
import os
import time
import sys
import getopt


class RegistrationParam:
    def __init__(self):
        self.debug = False
        self.fname_fixed = ''
        self.fname_moving = ''
        self.transformation = 'BSplineSyN'  # 'SyN'
        self.metric = 'CC'  # 'MeanSquares'
        self.gradient_step = '0.5'
        self.radius = '2'  # '4'
        self.interpolation = 'BSpline'
        self.iteration = '10x5' # '20x15'
        self.fname_seg_fixed = ''
        self.fname_seg_moving = ''
        self.fname_output = ''
        self.padding = '10'

        self.verbose = 1
        self.remove_temp = 1


def wm_registration(param, path_tmp):

    moving_name = 'moving'
    fixed_name = 'fixed'

    moving_seg_name = 'moving_seg'
    fixed_seg_name = 'fixed_seg'

    # Extract path/file/extension
    path_output, file_output, ext_output = sct.extract_fname(param.fname_output)

    # copy files to temporary folder
    sct.printv('\nCopy files...', reg_param.verbose, 'normal')
    sct.run("c3d "+param.fname_moving+" -o "+path_tmp+"/"+moving_name+".nii")
    sct.run("c3d "+param.fname_ref+" -o "+path_tmp+"/"+fixed_name+".nii")
    if param.fname_seg_moving != '':
        sct.run("c3d "+param.fname_seg_moving+" -o "+path_tmp+"/"+moving_seg_name+".nii")
    if param.fname_seg_fixed != '':
        sct.run("c3d "+param.fname_seg_fixed+" -o "+path_tmp+"/"+fixed_seg_name+".nii")

    # go to tmp folder
    os.chdir(path_tmp)

    if param.fname_seg_fixed != '':
        # cropping in x & y directions
        fixed_name_temp = fixed_name + "_crop"
        cmd = "sct_crop_image -i " + fixed_name + ".nii -o " + fixed_name_temp + ".nii -m " + fixed_seg_name + ".nii -shift 10,10 -dim 0,1"
        sct.run(cmd)
        fixed_name = fixed_name_temp

        fixed_seg_name_temp = fixed_seg_name+"_crop"
        sct.run("sct_crop_image -i " + fixed_seg_name + ".nii -o " + fixed_seg_name_temp + ".nii -m " + fixed_seg_name + ".nii -shift 10,10 -dim 0,1")
        fixed_seg_name = fixed_seg_name_temp

    # padding the images
    moving_name_pad = moving_name+"_pad"
    fixed_name_pad = fixed_name+"_pad"
    sct.run("c3d "+moving_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+moving_name_pad+".nii")
    sct.run("c3d "+fixed_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+fixed_name_pad+".nii")
    moving_name = moving_name_pad
    fixed_name = fixed_name_pad

    if param.fname_seg_moving != '':
        moving_seg_name_pad = moving_seg_name+"_pad"
        sct.run("c3d "+moving_seg_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+moving_seg_name_pad+".nii")
        moving_seg_name = moving_seg_name_pad
    if param.fname_seg_fixed != '':
        fixed_seg_name_pad = fixed_seg_name+"_pad"
        sct.run("c3d "+fixed_seg_name+".nii -pad 0x0x"+param.padding+"vox 0x0x"+param.padding+"vox 0 -o "+fixed_seg_name_pad+".nii")
        fixed_seg_name = fixed_seg_name_pad

    # offset
    old_min = 0
    old_max = 1
    new_min = 100
    new_max = 200

    fixed_im = Image(fixed_name+".nii")
    fixed_im.data = (fixed_im.data - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
    fixed_im.save()

    moving_im = Image(moving_name+".nii")
    moving_im.data = (moving_im.data - old_min)*(new_max - new_min)/(old_max - old_min) + new_min
    moving_im.save()

    # registration of the gray matter
    sct.printv('\nDeforming the image...', reg_param.verbose, 'normal')
    moving_name_reg = moving_name+"_deformed"

    if param.transformation == 'BSplineSyN':
        transfo_params = ',3,0'
    elif param.transforlation == 'SyN':     # SyN gives bad results...
        transfo_params = ',1,1'
    else:
        transfo_params = ''

    cmd = 'isct_antsRegistration --dimensionality 3 --interpolation '+param.interpolation+' --transform '+param.transformation+'['+param.gradient_step+transfo_params+'] --metric '+param.metric+'['+fixed_name+'.nii,'+moving_name+'.nii,1,4] --output ['+moving_name_reg+','+moving_name_reg+'.nii]  --convergence '+param.iteration+' --shrink-factors 2x1 --smoothing-sigmas 0x0 '

    if param.fname_seg_moving != '':
        cmd += " --masks ["+fixed_seg_name+".nii,"+moving_seg_name+".nii]"
        # cmd += " -m ["+fixed_seg_name+".nii,"+moving_seg_name+".nii]"
    sct.run(cmd)
    moving_name = moving_name_reg

    # removing offset
    fixed_im = Image(fixed_name+".nii")
    fixed_im.data = (fixed_im.data - new_min)*(old_max - old_min)/(new_max - new_min) + old_min
    fixed_im.save()

    moving_im = Image(moving_name+".nii")
    moving_im.data = (moving_im.data - new_min)*(old_max - old_min)/(new_max - new_min) + old_min
    moving_im.save()



    # un-padding the images
    moving_name_unpad = moving_name+"_unpadded"
    sct.run("sct_crop_image -i "+moving_name+".nii -dim 2 -start "+str(int(param.padding)-1)+" -end -"+param.padding+" -o "+moving_name_unpad+".nii")
    sct.run("mv "+moving_name+"0Warp.nii.gz "+file_output+"0Warp"+ext_output)
    sct.run("mv "+moving_name+"0InverseWarp.nii.gz "+file_output+"0InverseWarp"+ext_output)
    moving_name = moving_name_unpad

    moving_name_out = file_output+ext_output
    sct.run("c3d fixed.nii "+moving_name+".nii -reslice-identity -o "+moving_name_out)

    # move output files to initial folder
    sct.run("cp "+file_output+"* ../")
    os.chdir('..')


def segment_gm(target_fname='', sc_seg_fname='', path_to_label='', param=None):
    from sct_segment_graymatter import FullGmSegmentation
    level_fname = path_to_label + '/template/MNI-Poly-AMU_level.nii.gz'
    gmsegfull = FullGmSegmentation(target_fname, sc_seg_fname, None, level_fname, param=param)

    return gmsegfull.res_names['corrected_wm_seg'], gmsegfull.res_names['gm_seg']



def main(seg_params, reg_param, target_fname='', sc_seg_fname='', path_to_label=''):
    # TODO: make tmp folder
    # create temporary folder
    sct.printv('\nCreate temporary folder...', verbose, 'normal' )
    path_tmp = 'tmp.'+time.strftime("%y%m%d%H%M%S")
    sct.run('mkdir '+path_tmp)

    os.chdir(path_tmp)
    sct.run('cp '+target_fname+' '+path_tmp+'/'+''.join(sct.extract_fname(target_fname)[0:1]))
    wm_fname, gm_fname = segment_gm(target_fname=target_fname, sc_seg_fname=sc_seg_fname, path_to_label=path_to_label, param=seg_params)
    os.chdir('..')

    reg_param.fname_fixed = wm_fname
    reg_param.fname_moving = path_to_label + '/template/MNI-Poly-AMU_WM.nii.gz'
    reg_param.fname_seg_fixed = sc_seg_fname
    reg_param.fname_seg_moving = path_to_label + '/template/MNI-Poly-AMU_cord.nii.gz'

    wm_registration(reg_param, path_tmp)

    # remove temporary file
    if reg_param.remove_temp == 1:
        sct.printv('\nRemove temporary files...', verbose, 'normal')
        sct.run("rm -rf "+path_tmp)


    # TODO: get output file for the warping field and the inverse
    # warp the T2star = output
    # sct.run('sct_apply_transfo -i ' + target_fname + ' -d ' + target_fname + ' -w ' + output_inverse_warp + ' -o ' + sct.extract_fname(target_fname)[1] + '_moved.nii.gz')
    # sct.run('sct_apply_transfo -i ' + sc_seg_fname + ' -d ' + sc_seg_fname + ' -w ' + output_inverse_warp + ' -o ' + sct.extract_fname(sc_seg_fname)[1] + '_moved.nii.gz  -x nn ')


if __name__ == "__main__":
    reg_param = RegistrationParam()
    gm_seg_param = Param()
    input_target_fname = ''
    input_sc_seg_fname = ''
    path_to_label = ''

    if reg_param.debug:
        print '\n*** WARNING: DEBUG MODE ON ***\n'
    else:
        param_default = RegistrationParam()

        # Initialize the parser
        parser = Parser(__file__)
        parser.usage.set_description('Register the template on a gray matter segmentation')  # TODO: change description
        parser.usage.addSection('Segmentation parameters')
        parser.add_option(name="-i",
                          type_value="file",
                          description="T2star image (or image with a white/gray matter contrast)",
                          mandatory=True,
                          example='t2star.nii.gz')
        parser.add_option(name="-s",
                          type_value="file",
                          description="Spinal cord segmentation of the T2star target",
                          mandatory=True,
                          example='sc_seg.nii.gz')
        parser.add_option(name="-seg-o",
                          type_value="str",
                          description="output name for the results",
                          mandatory=False,
                          example='t2star_res.nii.gz')
        parser.add_option(name="-dic",
                          type_value="folder",
                          description="Path to the model data",
                          mandatory=True,
                          example='/home/jdoe/gm_seg_model_data/')
        parser.add_option(name="-label",
                          type_value="folder",
                          description="Path to the label directory from the template registration",
                          mandatory=True,
                          example='./label/')
        '''
        parser.add_option(name="-i",
                          type_value="file",
                          description="Fixed image : the white matter automatic segmentation (should be probabilistic)",
                          mandatory=True,
                          example='wm_seg.nii.gz')
        parser.add_option(name="-d",
                          type_value="file",
                          description="Moving image: the white matter probabilistic segmentation from the template",
                          mandatory=True,
                          example='MNI-Poly-AMU_WM.nii.gz')
        parser.add_option(name="-o",
                          type_value="str",
                          description="Output image name",
                          mandatory=False,
                          example='moving_to_fixed.nii.gz')
        parser.add_option(name="-iseg",
                          type_value="file",
                          description="Spinal cord segmentation of the fixed image",
                          mandatory=False,
                          example='sc_seg.nii.gz')
        parser.add_option(name="-dseg",
                          type_value="file",
                          description="Spinal cord segmentation of the moving image (should be the same)",
                          mandatory=False,
                          example='sc_seg.nii.gz')
        '''
        parser.usage.addSection('Registration parameters')
        parser.add_option(name="-t",
                          type_value='multiple_choice',
                          description="type of transformation",
                          mandatory=False,
                          default_value='BSplineSyN',
                          example=['SyN', 'BSplineSyN'])
        parser.add_option(name="-m",
                          type_value='multiple_choice',
                          description="Metric used for the registration",
                          mandatory=False,
                          default_value='CC',
                          example=['CC', 'MeanSquares'])
        parser.add_option(name="-reg-o",
                          type_value="str",
                          description="Output image name",
                          mandatory=False,
                          example='moving_to_fixed.nii.gz')
        parser.usage.addSection("Misc")
        parser.add_option(name="-r",
                          type_value='multiple_choice',
                          description="Remove temporary files",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1'])
        parser.add_option(name="-v",
                          type_value='multiple_choice',
                          description="Verbose",
                          mandatory=False,
                          default_value=1,
                          example=['0', '1', '2'])

        arguments = parser.parse(sys.argv[1:])

        input_target_fname = arguments["-i"]
        input_sc_seg_fname = arguments["-s"]
        gm_seg_param.path_model = arguments["-dic"]
        gm_seg_param.todo_model = 'load'
        path_to_label = arguments["-label"]
        verbose = 1

        if "-seg-o" in arguments:
            gm_seg_param.output_name = arguments["-seg-o"]
        '''
        reg_param.fname_ref = arguments["-i"]
        if "-iseg" in arguments:
            reg_param.fname_seg_fixed = arguments["-iseg"]
        reg_param.fname_moving = arguments["-d"]
        if "-dseg" in arguments:
            reg_param.fname_seg_moving = arguments["-dseg"]

        if "-o" in arguments:
            reg_param.fname_output = arguments["-o"]
        else:
            reg_param.fname_output = sct.extract_fname(reg_param.fname_moving)[1] + '_to_' + sct.extract_fname(reg_param.fname_ref)[1] +  sct.extract_fname(reg_param.fname_ref)[2]
        '''
        if "-t" in arguments:
            reg_param.transformation = arguments["-t"]
        if "-m" in arguments:
            reg_param.metric = arguments["-m"]
        if "-r" in arguments:
            reg_param.remove_temp = int(arguments["-r"])
        if "-v" in arguments:
            verbose = int(arguments["-v"])

        gm_seg_param.verbose = verbose
        reg_param.verbose = verbose

    main(gm_seg_param, reg_param, target_fname=input_target_fname, sc_seg_fname=input_sc_seg_fname, path_to_label=path_to_label)


