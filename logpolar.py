#! /usr/bin/env python
#coding=utf-8

r'''Facilities for testing artificial vision agents.
'''

__license__ = r'''
Copyright (c) Helio Perroni Filho <xperroni@gmail.com>

This file is part of  Python Log Polar Transform Redux (PLPTR).

PLPTR is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PLPTR is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PLPTR. If not, see <http://www.gnu.org/licenses/>.
'''

__version__ = '1'

from math import ceil, exp, pi
from math import log, cos, sin

from numpy import array, zeros


def logpolar_naive(image, i_0, j_0, p_n=None, t_n=None):
    r'''"Naive" implementation of the log-polar transform in Python.
    
        Arguments:
            
        image
            The input image.
        
        i_0, j0
            The center of the transform.
        
        p_n, t_n
            Optional. Dimensions of the output transform. If any are None,
            suitable defaults are used.
        
        Returns:
        
        The log-polar transform for the input image.
    '''
    # Shape of the input image.
    (i_n, j_n) = image.shape[:2]
    
    # The distance d_c from the transform's focus (i_0, j_0) to the image's
    # farthest corner (i_c, j_c). This is used below as the default value for
    # p_n, and also to calculate the iteration step across the transform's p
    # dimension.
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5

    if p_n == None:
        # The default value to p_n is defined as the distance d_c.
        p_n = int(ceil(d_c))
    
    if t_n == None:
        # The default value to t_n is defined as the width of the image.
        t_n = j_n
    
    # The scale factors determine the size of each "step" along the transform.
    p_s = log(d_c) / p_n
    t_s = 2.0 * pi / t_n
    
    # The transform's pixels have the same type and depth as the input's.
    transformed = zeros((p_n, t_n) + image.shape[2:], dtype=image.dtype)

    # Scans the transform across its coordinate axes. At each step calculates
    # the reverse transform back into the cartesian coordinate system, and if
    # the coordinates fall within the boundaries of the input image, takes that
    # cell's value into the transform.
    for p in range(0, p_n):
        p_exp = exp(p * p_s)
        for t in range(0, t_n):
            t_rad = t * t_s

            i = int(i_0 + p_exp * sin(t_rad))
            j = int(j_0 + p_exp * cos(t_rad))

            if 0 <= i < i_n and 0 <= j < j_n:
                transformed[p, t] = image[i, j]

    return transformed


_transforms = {}

def _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s):
    # Checks if this transform has been requested before.
    transform = _transforms.get((i_0, j_0, i_n, j_n, p_n, t_n))

    # If the transform is not found...
    if transform == None:
        i_k = []
        j_k = []
        p_k = []
        t_k = []

        # Scans the transform across its coordinate axes. At each step
        # calculates the reverse transform back into the cartesian coordinate
        # system, and if the coordinates fall within the boundaries of the
        # input image, records both coordinate sets.
        for p in range(0, p_n):
            p_exp = exp(p * p_s)
            for t in range(0, t_n):
                t_rad = t * t_s

                i = int(i_0 + p_exp * sin(t_rad))
                j = int(j_0 + p_exp * cos(t_rad))

                if 0 <= i < i_n and 0 <= j < j_n:
                    i_k.append(i)
                    j_k.append(j)
                    p_k.append(p)
                    t_k.append(t)

        # Creates a set of two "fancy-indices", one for retrieving pixels from
        # the input image, and other for assigning them to the transform.
        transform = ((array(p_k), array(t_k)), (array(i_k), array(j_k)))
        _transforms[i_0, j_0, i_n, j_n, p_n, t_n] = transform

    return transform


def logpolar_fancy(image, i_0, j_0, p_n=None, t_n=None):
    r'''Implementation of the log-polar transform based on numpy's fancy
        indexing.
    
        Arguments:
            
        image
            The input image.
        
        i_0, j0
            The center of the transform.
        
        p_n, t_n
            Optional. Dimensions of the output transform. If any are None,
            suitable defaults are used.
        
        Returns:
        
        The log-polar transform for the input image.
    '''
    # Shape of the input image.
    (i_n, j_n) = image.shape[:2]
    
    # The distance d_c from the transform's focus (i_0, j_0) to the image's
    # farthest corner (i_c, j_c). This is used below as the default value for
    # p_n, and also to calculate the iteration step across the transform's p
    # dimension.
    i_c = max(i_0, i_n - i_0)
    j_c = max(j_0, j_n - j_0)
    d_c = (i_c ** 2 + j_c ** 2) ** 0.5
    
    if p_n == None:
        # The default value to p_n is defined as the distance d_c.
        p_n = int(ceil(d_c))
    
    if t_n == None:
        # The default value to t_n is defined as the width of the image.
        t_n = j_n
    
    # The scale factors determine the size of each "step" along the transform.
    p_s = log(d_c) / p_n
    t_s = 2.0 * pi / t_n
    
    
    # Recover the transform fancy index from the cache, creating it if not
    # found.
    (pt, ij) = _get_transform(i_0, j_0, i_n, j_n, p_n, t_n, p_s, t_s)

    # The transform's pixels have the same type and depth as the input's.
    transformed = zeros((p_n, t_n) + image.shape[2:], dtype=image.dtype)

    # Applies the transform to the image via numpy fancy-indexing.
    transformed[pt] = image[ij]
    return transformed


from time import clock
from sys import maxsize
from scipy.misc import imread, imsave


def profile(f):
    image = imread('cartesian.png')
    transformed = None
    t_max = 0
    t_min = maxsize
    for i in range(0, 10):
        t_0 = clock()
        transformed = f(image, 127, 127)
        t_n = clock()
        
        t = t_n - t_0
        if t > t_max:
            t_max = t
        if t < t_min:
            t_min = t

    name = f.__name__
    imsave('%s.png' % name, transformed)
    print('Best and worst time for %s() across 10 runs: (%f, %f)' % (name, t_min, t_max))


if __name__ == '__main__':
    profile(logpolar_naive)
    profile(logpolar_fancy)
