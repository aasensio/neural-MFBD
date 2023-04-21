import numpy as np
from astropy.io import fits
import scipy.io as io
import matplotlib.pyplot as pl
import h5py

class Demodulate(object):
    def __init__(self, root):
        self.root = root

        print("Reading modulation matrices...")
        refl = fits.open(f'{root}/demodulation/camXIX_8542_polcal.fits')[0].data[:]
        tran = fits.open(f'{root}/demodulation/camXXV_8542_polcal.fits')[0].data[:]
        telescope = io.readsav(f'{root}/demodulation/mtel_8542_20200727_084300.sav')['mtel'][:]
        
        nx, ny, nm = refl.shape
        
        print("Inverting matrices...")
        refl = np.linalg.inv(refl.reshape((nx, ny, 4, 4)))
        tran = np.linalg.inv(tran.reshape((nx, ny, 4, 4)))
        telescope = np.linalg.inv(telescope)
        telescope /= telescope[0, 0]
        
        print("Multiplying telescope+calibration matrices...")
        self.M_r = np.einsum('mnki,jk->mnij', refl, telescope)
        self.M_t = np.einsum('mnki,jk->mnij', tran, telescope)
        
    def demodulate(self, tran, refl, border=0):    
        print("Demodulating...")
        nx, ny, mn = refl.shape

        M_r = self.M_r[100:nx+100, 100:ny+100, :, :][:, ::-1, :, :]
        M_t = self.M_t[100:nx+100, 100:ny+100, :, :][:, ::-1, :, :]

        re0 = np.einsum('mnji,mnj->mni', M_t, tran)
        re1 = np.einsum('mnji,mnj->mni', M_r, refl)

        met = np.mean(re0[border:-border,border:-border,0])
        mer = np.mean(re1[border:-border,border:-border,0])

        scl_t = (met + mer) * 0.25 / met
        scl_r = (met + mer) * 0.25 / mer

        out = re0*scl_t + re1*scl_r

        # Substitute the Stokes I with the average of the two polarizations
        out[:, :, 0] = 0.5 * np.sum(refl + tran, axis=-1) / 4.0
        
        return out, M_r, M_t
    
    def clean(self, data):
        
        print("Cleaning fringes...")
        f = np.fft.fft2(data, axes=(0, 1))
        power = np.abs(f)**2

        x0 = 0.01525
        y0 = -0.00785

        nx, ny = power.shape[0:2]
        x = np.fft.fftfreq(ny)
        y = np.fft.fftfreq(nx)
        X, Y = np.meshgrid(x, y)

        sig = 0.005

        filter = np.exp(-((X-x0)**2 + (Y-y0)**2) / sig**2) + np.exp(-((X+x0)**2 + (Y+y0)**2) / sig**2) + np.exp(-((X-x0)**2 + (Y+y0)**2) / sig**2) + np.exp(-((X+x0)**2 + (Y-y0)**2) / sig**2)
        filter = 1.0 - filter

        corrected = np.fft.ifft2(f * filter[:, :, None], axes=(0, 1)).real

        return power, filter, corrected

            

if __name__ == '__main__':
    root = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542'
    demodulation = Demodulate(root)

    #  Raw
    tran_raw = []
    for lc in range(4):
        filename = f'{root}/raw/wav8_mod{lc}_cam0_00000.h5'
        f = h5py.File(filename, 'r')
        tmp = np.concatenate([f['wb'][0:1, 20:, 14:], f['nb'][0:1, 20:, 14:]], axis=0)
        tran_raw.append(tmp[:, 100:-100, 100:-100, None])        

    refl_raw = []
    for lc in range(4):
        filename = f'{root}/raw/wav8_mod{lc}_cam1_00000.h5'
        f = h5py.File(filename, 'r')
        tmp = np.concatenate([f['wb'][0:1, 20:, 14:], f['nb'][0:1, 20:, 14:]], axis=0)
        refl_raw.append(tmp[:, 100:-100, 100:-100, None])        

    #  Convolutional
    tran_unsup = []
    for lc in range(4):
        filename = f'{root}/unsup/val_spot_20200727_083509.8542.00.+260.lc{lc}.cam0.2023-03-10-12:44.All.h5'
        f = h5py.File(filename, 'r')['reconstruction_nn'][:, 20:, 14:]
        tran_unsup.append(f[:, 100:-100, 100:-100, None])        

    refl_unsup = []
    for lc in range(4):
        filename = f'{root}/unsup/val_spot_20200727_083509.8542.00.+260.lc{lc}.cam1.2023-03-10-12:44.All.h5'
        f = h5py.File(filename, 'r')['reconstruction_nn'][:, 20:, 14:]
        refl_unsup.append(f[:, 100:-100, 100:-100, None])        

    #  Unrolled
    tran_unrolled = []
    for lc in range(4):
        filename = f'{root}/unrolled/val_spot_20200727_083509.8542.00.+260.lc{lc}.cam0.2023-03-11-08:03.All.h5'
        f = h5py.File(filename, 'r')['reconstruction_nn'][:, 20:, 14:]
        tran_unrolled.append(np.nan_to_num(f[:, 100:-100, 100:-100, None]))

    refl_unrolled = []
    for lc in range(4):
        filename = f'{root}/unrolled/val_spot_20200727_083509.8542.00.+260.lc{lc}.cam1.2023-03-11-08:03.All.h5'
        f = h5py.File(filename, 'r')['reconstruction_nn'][:, 20:, 14:]
        refl_unrolled.append(np.nan_to_num(f[:, 100:-100, 100:-100, None]))
    
    refl_raw = np.concatenate(refl_raw, axis=-1)
    tran_raw = np.concatenate(tran_raw, axis=-1)
    
    refl_unrolled = np.concatenate(refl_unrolled, axis=-1)
    tran_unrolled = np.concatenate(tran_unrolled, axis=-1)

    refl_unsup = np.concatenate(refl_unsup, axis=-1)
    tran_unsup = np.concatenate(tran_unsup, axis=-1)

    # Demodulate only NB
    demod_raw, M_r, M_t = demodulation.demodulate(tran_raw[1, ...], refl_raw[1, ...], border=50)
    demod_unsup, M_r, M_t = demodulation.demodulate(tran_unsup[1, ...], refl_unsup[1, ...], border=50)
    demod_unrolled, M_r, M_t = demodulation.demodulate(tran_unrolled[1, ...], refl_unrolled[1, ...], border=50)

    power_raw, filter, demod_raw_clean = demodulation.clean(demod_raw)
    power_unsup, filter, demod_unsup_clean = demodulation.clean(demod_unsup)
    power_unrolled, filter, demod_unrolled_clean = demodulation.clean(demod_unrolled)

    iquv = fits.open(f'{root}/demodulation/stokesIQUV_00000_8542_8542_+260.fits')

    fig, ax = pl.subplots(nrows=4, ncols=2, figsize=(7, 13.3), sharex=True, sharey=True, constrained_layout=True)

    ax[0, 0].imshow(demod_raw_clean[:, ::-1, 0][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[0, 1].imshow(demod_raw_clean[:, ::-1, 3][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[0, 0].set_title('Frame')        
    ax[0, 0].text(1, 1, 'Stokes I', color='w', weight='bold', fontsize='large')
    ax[0, 1].text(1, 1, 'Stokes V', color='w', weight='bold', fontsize='large')
    
    ax[1, 0].imshow(iquv[0].data[0, 0, 0, 100:-100, 100:-100][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[1, 1].imshow(iquv[0].data[0, 3, 0, 100:-100, 100:-100][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[1, 0].set_title('MOMFBD')        

    ax[2, 0].imshow(demod_unrolled_clean[:, ::-1, 0][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[2, 1].imshow(demod_unrolled_clean[:, ::-1, 3][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[2, 0].set_title('Unrolling')

    ax[3, 0].imshow(demod_unsup_clean[:, ::-1, 0][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])    
    ax[3, 1].imshow(demod_unsup_clean[:, ::-1, 3][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[3, 0].set_title('Convolutional')

    fig.supxlabel('Distance [arcsec]')
    fig.supylabel('Distance [arcsec]')

    pl.savefig('polarimetry_IV.pdf')


    fig, ax = pl.subplots(nrows=4, ncols=4, figsize=(13.3, 13.3), sharex=True, sharey=True, constrained_layout=True)

    ax[0, 0].imshow(demod_raw_clean[:, ::-1, 0][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[0, 1].imshow(demod_raw_clean[:, ::-1, 1][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[0, 2].imshow(demod_raw_clean[:, ::-1, 2][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[0, 3].imshow(demod_raw_clean[:, ::-1, 3][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[0, 0].set_title('Frame')        
    ax[0, 0].text(1, 1, 'Stokes I', color='w', weight='bold', fontsize='large')
    ax[0, 1].text(1, 1, 'Stokes Q', color='w', weight='bold', fontsize='large')
    ax[0, 2].text(1, 1, 'Stokes U', color='w', weight='bold', fontsize='large')
    ax[0, 3].text(1, 1, 'Stokes V', color='w', weight='bold', fontsize='large')
    
    ax[1, 0].imshow(iquv[0].data[0, 0, 0, 100:-100, 100:-100][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[1, 1].imshow(iquv[0].data[0, 1, 0, 100:-100, 100:-100][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[1, 2].imshow(iquv[0].data[0, 2, 0, 100:-100, 100:-100][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[1, 3].imshow(iquv[0].data[0, 3, 0, 100:-100, 100:-100][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[1, 0].set_title('MOMFBD')        

    ax[2, 0].imshow(demod_unrolled_clean[:, ::-1, 0][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[2, 1].imshow(demod_unrolled_clean[:, ::-1, 1][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[2, 2].imshow(demod_unrolled_clean[:, ::-1, 2][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[2, 3].imshow(demod_unrolled_clean[:, ::-1, 3][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[2, 0].set_title('Unrolling')

    ax[3, 0].imshow(demod_unsup_clean[:, ::-1, 0][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])    
    ax[3, 1].imshow(demod_unsup_clean[:, ::-1, 1][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[3, 2].imshow(demod_unsup_clean[:, ::-1, 2][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[3, 3].imshow(demod_unsup_clean[:, ::-1, 3][100:500, 300:700], extent=[0,400*0.059,0,400*0.059])
    ax[3, 0].set_title('Convolutional')

    fig.supxlabel('Distance [arcsec]')
    fig.supylabel('Distance [arcsec]')

    pl.savefig('polarimetry_IQUV.pdf')

    

#     pl.savefig('test.png')


# if __name__ == '__main__':
#     root = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542'