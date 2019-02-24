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
#crop translated image to same size
#cropped = moved[-217:,-181:]
#io.imsave('cropped.png',cropped)
xs = [90.5]#np.arange(90,91,0.5)
ys = [107.5]#np.arange(107,108,0.5)
sigmas = [4.735]#np.arange(4.65,4.8,0.005)
best_mssim = 0
best_x = None
best_y = None
for sig in sigmas:
	for x in xs:
		for y in ys:
			orig = gaussian(io.imread('BrainProtonDensitySlice.png'),sigma=sig)
			moved = gaussian(io.imread('BrainProtonDensitySliceR10X13Y17.png'),sigma=sig)
			sft,idc,idk = register_translation(pad(orig,20,'edge'),moved)
			unmoved = warp(moved,EuclideanTransform(translation=[-sft[1],-sft[0]]),output_shape=(257,221,),mode='constant',cval=0)
			cropped = unmoved[20:-20,20:-20]
			io.imsave('unmoved_cropped.png',cropped)
			orig_tr = logpolar_fancy(np.fft.fftn(orig),x,y)#91,108
			cropped_tr = logpolar_fancy(np.fft.fftn(cropped),x,y)#91,108
			shifted_lpt,error_lpt,phased_lpt = register_translation(orig_tr,cropped_tr,space='fourier')
			rot_rads = math.atan2(shifted_lpt[0],shifted_lpt[1])
			rot_degs = math.degrees(rot_rads)
			rotated = rotate(cropped,rot_degs)
			io.imsave('rotated.png',rotated)
			#shifts,error,phasediff = register_translation(orig,rotated,1000)
			#shifted = warp(rotated,EuclideanTransform(translation=shifts))
			#shifted = (shifted*255).astype('uint8')
			mssim,diff = compare_ssim(orig,rotated,full=True)
			if mssim > best_mssim:
				best_mssim=mssim
				best_x = x
				best_y = y
				best_sig = sig
			print(f'Gaussian sigma:{sig}\nRotation (degs):{rot_degs}\nShift:{sft}\nMSSIM:{mssim}')
			#io.imsave('shifted.png',shifted)
			io.imsave('diff.png',diff)
print(f'best mssim:{best_mssim}\nbest_x{best_x} best_y{best_y}\nbest_sig{best_sig}')
'''Gaussian sigma:4.735
Rotation (degs):0.0
Shift:[-16. -12.]
MSSIM:0.9060310863982878
best mssim:0.9060310863982878
best_x90.5 best_y107.5
best_sig4.735'''