from skimage import io
from skimage.feature import register_translation
from skimage.transform import rotate,EuclideanTransform,warp
from skimage.measure import compare_ssim
from skimage.filters import gaussian

from scipy.ndimage import fourier_shift

from skimage.util import pad
import numpy as np
from logpolar import logpolar_fancy
import math
sig=0.95
orig = gaussian(io.imread('BrainProtonDensitySlice.png'),sigma=sig)
moved = gaussian(io.imread('BrainProtonDensitySliceR10X13Y17.png'),sigma=sig)
#crop translated image to same size
cropped = moved[-217:,-181:]
io.imsave('cropped.png',cropped)

orig_tr = logpolar_fancy(np.fft.fftn(orig),91,108)#91,108
cropped_tr = logpolar_fancy(np.fft.fftn(cropped),82,106)#91,108
shifted_lpt,error_lpt,phased_lpt = register_translation(orig_tr,cropped_tr,10,space='fourier')
rot_rads = math.atan2(shifted_lpt[0],shifted_lpt[1])
rot_degs = math.degrees(rot_rads)
rotated = rotate(cropped,rot_degs)
io.imsave('rotated.png',rotated)
shifts,error,phasediff = register_translation(orig,rotated,1000)
shifted = warp(rotated,EuclideanTransform(translation=shifts))
shifted = (shifted*255).astype('uint8')
mssim,diff = compare_ssim(orig,shifted,full=True)
print(f'Rotation (degs):{rot_degs}\nShift:{shifts}\nMSSIM:{mssim}')
io.imsave('shifted.png',shifted)
io.imsave('diff.png',diff)