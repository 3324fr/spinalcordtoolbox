#!/usr/bin/env python
#########################################################################################
#
# Qc class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Frederic Cloutier
# Modified: 2016-10-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as col
from msct_image import Image
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
import abc


class Qc(object):
    """
    Create a .png file from a 2d image.
    """

    def __init__(self, alpha, color_map,  dpi=300, interpolation='none'):
        self.interpolation = interpolation
        self.alpha = alpha
        self.dpi = dpi
        self.color_map = color_map

    def __call__(self, f):
        def wrapped_f(slice, *args, **kargs):
            name = slice.name
            self.mkdir()
            img, mask = f(slice,*args, **kargs)
            assert isinstance(img, np.ndarray)
            assert isinstance(mask, np.ndarray)
            fig = plt.imshow(img, cmap='gray', interpolation=self.interpolation)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            self.save('{}_gray'.format(name))
            mask = np.ma.masked_where(mask < 1, mask)
            plt.imshow(img, cmap='gray', interpolation=self.interpolation)
            plt.imshow(mask, cmap= self.color_map, interpolation=self.interpolation, alpha=self.alpha, )
            self.save(name)

            plt.close()

        return wrapped_f

    def save(self, name, format='png', bbox_inches='tight', pad_inches=0):
        plt.savefig('{}.png'.format(name), format=format, bbox_inches=bbox_inches,
                    pad_inches=pad_inches, dpi=self.dpi)

    def mkdir(self):
        # TODO : implement function
        # make a new.or update Qc directory
        return  0


class slices(object):
    _discrete_color = col.ListedColormap([ '#f00000', '#ff0000', '#fff000','#ffff00', '#fffff0', '#ffffff',
                                           '#0f0000', '#0ff000', '#0fff00','#0ffff0', '#0fffff', '#f0f0f0',
                                           '#00f000', '#00ff00', '#00fff0','#00ffff', '#f0ffff', '#0f0f0f',
                                           '#000f00', '#000ff0', '#000fff','#f00fff', '#ff0fff', '#123456',
                                           '#0000f0', '#0000ff', '#f000ff','#ff00ff', '#fff0ff', '#654321',
                                           '#00000f', '#f0000f', '#ff000f','#fff00f', '#ffff0f', '#999999'
                                            ], 'indexed')
    def __init__(self, name, imageName, segImageName ):
        self.name = name
        self.image = Image(imageName)
        self.image_seg = Image(segImageName)
        self.image.change_orientation('SAL')  # reorient to SAL
        self.image_seg.change_orientation('SAL')  # reorient to SAL
        self.dim = self.getDim(self.image)
        self.abels_regions = {'PONS': 50, 'MO': 51,
                 'C1': 1, 'C2': 2, 'C3': 3, 'C4': 4, 'C5': 5, 'C6': 6, 'C7': 7,
                 'T1': 8, 'T2': 9, 'T3': 10, 'T4': 11, 'T5': 12, 'T6': 13, 'T7': 14, 'T8': 15, 'T9': 16, 'T10': 17, 'T11': 18, 'T12': 19,
                 'L1': 20, 'L2': 21, 'L3': 22, 'L4': 23, 'L5': 24,
                 'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28, 'S5': 29,
                 'Co': 30}


    __metaclass__ = abc.ABCMeta

    @staticmethod
    def axial_slice(data, i):
        return data[ i, :, : ]

    @staticmethod
    def axial_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nx

    @staticmethod
    def sagital_slice(data, i):
        return data[ :, :, i ]

    @staticmethod
    def sagital_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return nz

    @staticmethod
    def coronal_slice(data, i):
        return data[ :, i, : ]

    @staticmethod
    def coronal_dim(image):
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        return ny

    @staticmethod
    def crop(matrix, center_x, center_y, ray_x, ray_y):
        start_row = center_x - ray_x
        end_row = center_x + ray_x
        start_col = center_y - ray_y
        end_col = center_y + ray_y

        if matrix.shape[ 0 ] < end_row:
            if matrix.shape[ 0 ] < (end_row - start_row):# TODO: throw/raise an exception
                raise OverflowError
            return slices.crop(matrix, center_x - 1, center_y, ray_x, ray_y)
        if matrix.shape[ 1 ] < end_col:
            if matrix.shape[ 1 ] < (end_col - start_col):# TODO: throw/raise an exception
                raise OverflowError
            return slices.crop(matrix, center_x, center_y - 1, ray_x, ray_y)
        if start_row < 0:
            return slices.crop(matrix, center_x + 1 , center_y, ray_x , ray_y)
        if start_col < 0:
            return slices.crop(matrix, center_x, center_y + 1, ray_x, ray_y )

        return matrix[ start_row:end_row, start_col:end_col ]

    @staticmethod
    def add_slice(matrix, i, column, size, patch):
        startCol = (i % column) * size * 2
        endCol = startCol + patch.shape[ 1 ]
        startRow = int(math.ceil(i / column)) * size * 2
        endRow = startRow + patch.shape[ 0 ]
        matrix[ startRow:endRow, startCol:endCol ] = patch
        return matrix

    @staticmethod
    def nan_fill(array):
        array[ np.isnan(array) ] = np.interp(np.isnan(array).ravel().nonzero()[0]
                                             , (-np.isnan(array)).ravel().nonzero()[0]
                                             , array[ -np.isnan(array) ])
        return array


    @abc.abstractmethod
    def getSlice(self, data, i):
        """
        Abstract method to obtain a slice of a 3d matrix.
        :param data: 3d numpy.ndarray
        :param i: int
        :return: 2d numpy.ndarray
        """
        return

    @abc.abstractmethod
    def getDim(self, image):
        """
        Abstract method to obtain the depth of the 3d matrix.
        :param image: 3d numpy.ndarray
        :return: int
        """
        return

    def _center(self):
        axial_dim = self.axial_dim(self.image_seg)
        centers_x = np.zeros(axial_dim)
        centers_y = np.zeros(axial_dim)
        for i in xrange(axial_dim):
            centers_x[ i ], centers_y[ i ] \
                = ndimage.measurements.center_of_mass(self.axial_slice(self.image_seg.data, i))
        try:
            slices.nan_fill(centers_x)
            slices.nan_fill(centers_y)
        except ValueError:
            print "Oops! There are no trace of that spinal cord."  # TODO : raise error
            raise
        return centers_x, centers_y

    @Qc(1, cm.hsv)
    def mosaic(self, nb_column, size):
        matrix0 = np.ones((size * 2 * int((self.dim / nb_column) + 1),size * 2 * nb_column))
        matrix1 = np.empty((size * 2 * int((self.dim / nb_column) + 1), size * 2 * nb_column))
        centers_x, centers_y = self.center()
        for i in range(self.dim):
            x = int(round(centers_x[ i ]))
            y = int(round(centers_y[ i ]))
            matrix0 = slices.add_slice(matrix0, i, nb_column, size,
                                       slices.crop(self.getSlice(self.image.data, i), x, y, size, size))
            matrix1 = slices.add_slice(matrix1, i, nb_column, size,
                                       slices.crop(self.getSlice(self.image_seg.data, i), x, y, size, size))

        return matrix0, matrix1
    @Qc(1, _discrete_color)
    def single(self):
        matrix0 = self.getSlice(self.image.data, self.dim/2)
        matrix1 = self.getSlice(self.image_seg.data,self.dim/2 )
        david = matrix0.shape[0]
        index = self.get_center_spit(self.image_seg)
        for j in range(david):
            matrix0[j] = self.getSlice(self.image.data, int(round(index[j])))[j]
            matrix1[j] = self.getSlice(self.image_seg.data, int(round(index[j])))[j]

        return matrix0, matrix1

    def save(self, nb_column=0, size=10):
        if nb_column > 0:
            return self.mosaic(nb_column, size)
        else:
            return self.single()


class axial(slices):
    def getSlice(self, data, i):
        return self.axial_slice(data, i)

    def getDim(self, image):
        return self.axial_dim(image)

    def get_center_spit(self, image):
        return self.axial_dim(image)/2


class sagital(slices):
    def getSlice(self, data, i):
        return self.sagital_slice(data, i)

    def getDim(self, image):
        return self.sagital_dim(image)

    def get_center_spit(self, image):
        x , y = self._center()
        return y


class coronal(slices):
    def getSlice(self, data, i):
        return  self.coronal_slice(data, i)

    def getDim(self, image):
        return self.coronal_dim(image)

    def get_center_spit(self, image):
        x, y = self._center()
        return x