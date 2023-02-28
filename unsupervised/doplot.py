import numpy as np
import matplotlib.pyplot as pl
import h5py
import zarr
import az_average as az
from tqdm import tqdm
from astropy.io import fits
from images import read_images

import scipy.stats as stats

def power(image):
    """
    https://bertvandenbroucke.netlify.app/2019/05/24/computing-a-power-spectrum-in-python/
    
    """

    fourier_image = np.fft.fft2(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    npix = image.shape[0]
    
    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                        statistic = "mean",
                                        bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

    return kvals, Abins

npix_apodization = 25
n_pixel = 96
win = np.hanning(npix_apodization)
winOut = np.ones(n_pixel)
winOut[0:npix_apodization//2] = win[0:npix_apodization//2]
winOut[-npix_apodization//2:] = win[-npix_apodization//2:]
window = np.outer(winOut, winOut)

def plot_fov(infile, pix, x, dx, y, dy):    
    f = h5py.File(infile, 'r')

    frame = f['frame0'][:]
    res_nn = f['reconstruction_nn'][:]

    f.close()

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 10), sharex=True, sharey=True, constrained_layout=True)

    ax[0, 0].imshow(frame[0, x:x+dx,y:y+dy], extent=(0, dx*pix, 0, dy*pix))
    ax[0, 1].imshow(res_nn[0, x:x+dx,y:y+dy], extent=(0, dx*pix, 0, dy*pix))


    ax[1, 0].imshow(frame[1, x:x+dx,y:y+dy], extent=(0, dx*pix, 0, dy*pix))
    ax[1, 1].imshow(res_nn[1, x:x+dx,y:y+dy], extent=(0, dx*pix, 0, dy*pix))

    ax[0, 0].set_title('Frame')
    ax[0, 1].set_title('Neural')

    fig.supxlabel('Distance [arcsec]')
    fig.supylabel('Distance [arcsec]')
            
    k_wb, pow_wb = power(frame[0, x:x+dx,y:y+dy])
    k_nb, pow_nb = power(frame[1, x:x+dx,y:y+dy])
    k_nn_wb, pow_nn_wb = power(res_nn[0, x:x+dx,y:y+dy])
    k_nn_nb, pow_nn_nb = power(res_nn[1, x:x+dx,y:y+dy])

    # fig, ax = pl.subplots(constrained_layout=True)
    # ax.loglog(k_wb, pow_wb, '.', color='C0')
    # ax.loglog(k_nb, pow_nb, '.', color='C1')
    # ax.loglog(k_nn_wb, pow_nn_wb, color='C0')
    # ax.loglog(k_nn_nb, pow_nn_nb, color='C1')

########################################
# 3934
########################################
# plot_fov(f'reconstructed/validation_spot_3934_300.h5', pix=0.038, x=300, dx=500, y=500, dy=500)

# pl.savefig('reconstructed/spot_3934_fov.png')



########################################
# 8542
########################################
plot_fov(f'reconstructed/validation_spot_8542_20.h5', pix=0.059, x=300, dx=400, y=500, dy=400)

pl.savefig('reconstructed/spot_8542_fov.png')