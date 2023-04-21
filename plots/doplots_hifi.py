import numpy as np
import matplotlib.pyplot as pl
import h5py
from astropy.io import fits
import az_average

def qs_hifi(label_unroll='', label_unsup=''):

    root_unrolled = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/unrolled'
    root_unsup = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/unsup'
    root_mfbd = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/momfbd'
    root_raw = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/raw'

    origx = 400
    origy = 400
    delta = 400
    pix = 0.059

    f_raw = h5py.File(f'{root_raw}/scan_b000.h5', 'r')
    f_mfbd_wb = fits.open(f'{root_mfbd}/outbb.fz.fits')[0].data
    f_mfbd_nb = fits.open(f'{root_mfbd}/outnb_pos0001.fz.fits')[0].data
    f_unroll = h5py.File(f'{root_unrolled}/scan_b000.{label_unroll}.h5')['reconstruction_nn'][:, 20:, 14:]
    f_unsup = h5py.File(f'{root_unsup}/scan_b000.{label_unsup}.h5')['reconstruction_nn'][:, 20:, 14:]

    fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(7,13.3), sharex=True, sharey=True, constrained_layout=True)
    ax[0,0].imshow(f_raw['wb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,1].imshow(f_raw['nb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,0].set_title('Frame')
    ax[0,0].text(1, 1, 'WB', color='w', weight='bold', fontsize='large')
    ax[0,1].text(1, 1, 'NB', color='w', weight='bold', fontsize='large')

    shx = 3
    shy = 0
    ax[1,0].imshow(f_mfbd_wb[origx+shx:origx+delta+shx, origy+shy:origy+delta+shy], extent=(0,delta*pix,0,delta*pix))
    ax[1,1].imshow(f_mfbd_nb[origx+shx:origx+delta+shx, origy+shy:origy+delta+shy], extent=(0,delta*pix,0,delta*pix))
    ax[1,0].set_title('MOMFBD')

    ax[2,0].imshow(f_unroll[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,1].imshow(f_unroll[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,0].set_title('Unrolling')

    ax[3,0].imshow(f_unsup[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,1].imshow(f_unsup[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,0].set_title('Convolutional')
    
    fig.supxlabel('Distance [arcsec]')
    fig.supylabel('Distance [arcsec]')

    pl.savefig('qs_comparison_hifi.pdf')

    fig, ax = pl.subplots(constrained_layout=True)

    k, power = az_average.power_spectrum(f_raw['wb'][0, 20:520, 14:514])
    k_mfbd, power_mfbd = az_average.power_spectrum(f_mfbd_wb[0:500, 0:500])
    k_unroll, power_unroll = az_average.power_spectrum(f_unroll[0, 0:500, 0:500])
    k_unsup, power_unsup = az_average.power_spectrum(f_unsup[0, 0:500, 0:500])

    ax.semilogy(k, power / power[0], label='Frame')
    ax.semilogy(k_mfbd, power_mfbd / power_mfbd[0], label='MOMFBD')
    ax.semilogy(k_unroll, power_unroll / power_unroll[0], label='Unrolling')
    ax.semilogy(k_unsup, power_unsup / power_unsup[0], label='Unsupervised')

    ax.set_xlim([0, 0.5])
    ax.set_ylim([1e-12, 1])

    ax.set_xlabel(r'Wavenumber [px$^{-1}$]')
    ax.set_ylabel(r'Azimuthal PSD')

    ax.legend()
    pl.savefig('qs_power_hifi.pdf')

def all():
    root_unrolled = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/unrolled'
    root_unsup = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/unsup'
    root_mfbd = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/momfbd'
    root_raw = '/net/drogon/scratch1/aasensio/sst_unroll/hifi/raw'
    label_unroll = '2023-03-07-14:31.All'
    label_unsup = '2023-03-07-14:27.All'

    f_unroll = h5py.File(f'{root_unrolled}/scan_b000.{label_unroll}.h5')
    f_unsup = h5py.File(f'{root_unsup}/scan_b000.{label_unsup}.h5')

    pl.plot(f_unroll['modes_nn'][0, :, 2])
    pl.plot(f_unsup['modes_nn'][0, :, 2])

if __name__ == '__main__':
    label_unroll = '2023-03-11-08:03.All' #'2023-03-07-14:31.All'
    label_unsup = '2023-03-10-12:44.All'#'2023-03-07-14:27.All'

    qs_hifi(label_unroll=label_unroll, label_unsup=label_unsup)
