import numpy as np
import matplotlib.pyplot as pl
import h5py
from astropy.io import fits
import az_average
import torch

def qs(label_unroll='', label_unsup=''):

    root_unrolled = '/net/drogon/scratch1/aasensio/sst_unroll/qs_20190801_081547/unrolled'
    root_unsup = '/net/drogon/scratch1/aasensio/sst_unroll/qs_20190801_081547/unsup'
    root_mfbd = '/net/drogon/scratch1/aasensio/sst_unroll/qs_20190801_081547/momfbd'
    root_raw = '/net/drogon/scratch1/aasensio/sst_unroll/qs_20190801_081547/raw'

    index = 15

    origx = 0
    origy = 0
    delta = 400
    pix = 0.059

    f_raw = h5py.File(f'{root_raw}/wav7_mod0_cam0_{index:05d}.h5', 'r')
    f_mfbd_wb = fits.open(f'{root_mfbd}/camXX_2019-08-01T08:15:47_{index:05d}_8542_8542_+65_lc0.fits')[0].data[:, ::-1]
    f_mfbd_nb = fits.open(f'{root_mfbd}/camXIX_2019-08-01T08:15:47_{index:05d}_8542_8542_+65_lc0.fits')[0].data[:, ::-1]
    f_unroll = h5py.File(f'{root_unrolled}/val_qs_20190801_081547.8542.{index:02d}.{label_unroll}.h5')['reconstruction_nn'][:, 20:, 14:]
    f_unsup = h5py.File(f'{root_unsup}/val_qs_20190801_081547.8542.{index:02d}.{label_unsup}.h5')['reconstruction_nn'][:, 20:, 14:]

    fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(7,13.3), sharex=True, sharey=True, constrained_layout=True)
    ax[0,0].imshow(f_raw['wb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,1].imshow(f_raw['nb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,0].set_title('Frame')
    ax[0,0].text(1, 1, 'WB', color='w', weight='bold', fontsize='large')
    ax[0,1].text(1, 1, 'NB', color='w', weight='bold', fontsize='large')

    
    ax[1,0].imshow(f_mfbd_wb[origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[1,1].imshow(f_mfbd_nb[origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[1,0].set_title('MOMFBD')

    ax[2,0].imshow(f_unroll[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,1].imshow(f_unroll[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,0].set_title('Unrolling')

    ax[3,0].imshow(f_unsup[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,1].imshow(f_unsup[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,0].set_title('Convolutional')
    
    fig.supxlabel('Distance [arcsec]')
    fig.supylabel('Distance [arcsec]')

    pl.savefig('qs_comparison_8542.pdf')

    fig, ax = pl.subplots(constrained_layout=True)

    k, power = az_average.power_spectrum(f_raw['wb'][0, 20:520, 14:514])
    k_mfbd, power_mfbd = az_average.power_spectrum(f_mfbd_wb[0:500, 0:500])
    k_unroll, power_unroll = az_average.power_spectrum(f_unroll[0, 0:500, 0:500])
    k_unsup, power_unsup = az_average.power_spectrum(f_unsup[0, 0:500, 0:500])

    ax.semilogy(k, power / power[0], label='Frame')
    ax.semilogy(k_mfbd, power_mfbd / power_mfbd[0], label='MOMFBD')
    ax.semilogy(k_unroll, power_unroll / power_unroll[0], label='Unrolling')
    ax.semilogy(k_unsup, power_unsup / power_unsup[0], label='Convolutional')

    ax.set_xlim([0, 0.5])
    ax.set_ylim([1e-12, 1])

    ax.set_xlabel(r'Wavenumber [px$^{-1}$]')
    ax.set_ylabel(r'Azimuthal PSD')

    ax.legend()
    pl.savefig('qs_power_8542.pdf')

def spot_8542(label_unroll='', label_unsup=''):

    root_unrolled = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542/unrolled'
    root_unsup = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542/unsup'
    root_mfbd = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542/momfbd'
    root_raw = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542/raw'

    index = 10

    origx = 0
    origy = 0
    delta = 400
    pix = 0.059

    f_raw = h5py.File(f'{root_raw}/wav7_mod0_cam0_{index:05d}.h5', 'r')
    f_mfbd_wb = fits.open(f'{root_mfbd}/camXX_2020-07-27T08:35:09_{index:05d}_8542_8542_+65_lc0.fits')[0].data[:, ::-1]
    f_mfbd_nb = fits.open(f'{root_mfbd}/camXIX_2020-07-27T08:35:09_{index:05d}_8542_8542_+65_lc0.fits')[0].data[:, ::-1]
    f_unroll = h5py.File(f'{root_unrolled}/val_spot_20200727_083509.8542.{index:02d}.{label_unroll}.h5')['reconstruction_nn'][:, 20:, 14:]
    f_unsup = h5py.File(f'{root_unsup}/val_spot_20200727_083509.8542.{index:02d}.{label_unsup}.h5')['reconstruction_nn'][:, 20:, 14:]

    fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(7,13.3), sharex=True, sharey=True, constrained_layout=True)
    ax[0,0].imshow(f_raw['wb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,1].imshow(f_raw['nb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,0].set_title('Frame')
    ax[0,0].text(1, 1, 'WB', color='w', weight='bold', fontsize='large')
    ax[0,1].text(1, 1, 'NB', color='w', weight='bold', fontsize='large')

    ax[1,0].imshow(f_mfbd_wb[origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[1,1].imshow(f_mfbd_nb[origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[1,0].set_title('MOMFBD')

    ax[2,0].imshow(f_unroll[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,1].imshow(f_unroll[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,0].set_title('Unrolling')

    ax[3,0].imshow(f_unsup[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,1].imshow(f_unsup[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,0].set_title('Convolutional')

    fig.supxlabel('Distance [arcsec]')
    fig.supylabel('Distance [arcsec]')

    pl.savefig('spot_comparison_8542.pdf')

def spot_3934(label_unroll='', label_unsup=''):

    root_unrolled = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_3934/unrolled'
    root_unsup = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_3934/unsup'
    root_mfbd = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_3934/momfbd'
    root_raw = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_3934/raw'

    index = 70

    origx = 250
    origy = 1000
    delta = 400
    pix = 0.038

    f_raw = h5py.File(f'{root_raw}/wav13_mod0_cam0_{index:05d}.h5', 'r')
    f_mfbd_wb = fits.open(f'{root_mfbd}/camXXVIII_2020-07-27T08:35:09_{index:05d}_12.00ms_G10.00_3934_3934_+65.fits')[0].data
    f_mfbd_nb = fits.open(f'{root_mfbd}/camXXX_2020-07-27T08:35:09_{index:05d}_12.00ms_G10.00_3934_3934_+65.fits')[0].data
    f_unroll = h5py.File(f'{root_unrolled}/val_spot_20200727_083509.3934.{index:02d}.{label_unroll}.h5')['reconstruction_nn'][:, 20:, 14:]
    f_unsup = h5py.File(f'{root_unsup}/val_spot_20200727_083509.3934.{index:02d}.{label_unsup}.h5')['reconstruction_nn'][:, 20:, 14:]

    fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(7,13.3), sharex=True, sharey=True, constrained_layout=True)
    ax[0,0].imshow(f_raw['wb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,1].imshow(f_raw['nb'][0, 20+origx:20+origx+delta, 14+origy:14+origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[0,0].set_title('Frame')
    ax[0,0].text(1, 1, 'WB', color='w', weight='bold', fontsize='large')
    ax[0,1].text(1, 1, 'NB', color='w', weight='bold', fontsize='large')

    dx = -20
    dy = -18
    ax[1,0].imshow(f_mfbd_wb[origx+dx:origx+delta+dx, origy+dy:origy+delta+dy], extent=(0,delta*pix,0,delta*pix))
    ax[1,1].imshow(f_mfbd_nb[origx+dx:origx+delta+dx, origy+dy:origy+delta+dy], extent=(0,delta*pix,0,delta*pix))
    ax[1,0].set_title('MOMFBD')

    ax[2,0].imshow(f_unroll[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,1].imshow(f_unroll[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[2,0].set_title('Unrolling')

    ax[3,0].imshow(f_unsup[0, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,1].imshow(f_unsup[1, origx:origx+delta, origy:origy+delta], extent=(0,delta*pix,0,delta*pix))
    ax[3,0].set_title('Convolutional')

    fig.supxlabel('Distance [arcsec]')
    fig.supylabel('Distance [arcsec]')

    pl.savefig('spot_comparison_3934.pdf')

def moving_average(x, n, type='simple'):
    x = np.asarray(x)
    if type=='simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a =  np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a

def loss(label_unroll='', label_unsup=''):

    n = 1500

    unroll = torch.load(f'../unroll/weights/{label_unroll}.ep_30.pth', map_location=lambda storage, loc: storage)
    unsup = torch.load(f'../unsupervised/weights/{label_unsup}.ep_30.pth', map_location=lambda storage, loc: storage)

    unroll_xt = np.arange(len(unroll['loss'])) / len(unroll['loss']) * 30.0
    unroll_yt = np.array(unroll['loss'])
    unroll_yt = moving_average(unroll_yt, n)
    unroll_xv = np.arange(len(unroll['val_loss'])) / len(unroll['val_loss']) * 30.0
    unroll_yv = np.array(unroll['val_loss'])
    unroll_yv = moving_average(unroll_yv, n)

    unsup_xt = np.arange(len(unsup['loss'])) / len(unsup['loss']) * 30.0
    unsup_yt = np.array(unsup['loss'])
    unsup_yt = moving_average(unsup_yt, 1500)
    unsup_xv = np.arange(len(unsup['val_loss'])) / len(unsup['val_loss']) * 30.0
    unsup_yv = np.array(unsup['val_loss'])
    unsup_yv = moving_average(unsup_yv, 1500)
    
    fig, ax = pl.subplots()
    ax.plot(unroll_xt[::10], unroll_yt[::10], label='Unrolling [train]')
    ax.plot(unroll_xv[::10], unroll_yv[::10], label='Unrolling [validation]')
    ax.plot(unsup_xt[::10], unsup_yt[::10], label='Convolutional [train]')
    ax.plot(unsup_xv[::10], unsup_yv[::10], label='Convolutional [validation]')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    pl.legend()
    pl.gca().spines['top'].set_visible(False)
    pl.gca().spines['right'].set_visible(False)
    pl.grid()

    pl.savefig('losses.pdf')


if __name__ == '__main__':

    # Trained only with WB
    label_unroll = '2023-03-07-14:31.All'
    label_unsup = '2023-03-07-14:27.All'

    # Trained with WB and NB
    label_unroll = '2023-03-11-08:03.All'
    label_unsup = '2023-03-10-12:44.All'

    # qs(label_unroll=label_unroll, label_unsup=label_unsup)
    # spot_8542(label_unroll=label_unroll, label_unsup=label_unsup)
    # spot_3934(label_unroll=label_unroll, label_unsup=label_unsup)
    loss(label_unroll=label_unroll, label_unsup=label_unsup)
