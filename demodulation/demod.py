import numpy as np
from astropy.io import fits
import scipy.io as io
import matplotlib.pyplot as pl

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
        
    def demodulate(self, border=0):
        refl = []
        tran = []
        print("Reading data...")
        for i in range(4):
            f = fits.open(f'{self.root}/momfbd/camXIX_2020-07-27T08:35:09_00000_8542_8542_+260_lc{i}.fits')
            refl.append(f[0].data[:, :, None])
        for i in range(4):
            f = fits.open(f'{self.root}/momfbd/camXXV_2020-07-27T08:35:09_00000_8542_8542_+260_lc{i}.fits')
            tran.append(f[0].data[:, :, None])

        refl = np.concatenate(refl, axis=-1)[:, :, :]
        tran = np.concatenate(tran, axis=-1)[:, :, :]
    
        nx, ny, mn = refl.shape

        M_r = self.M_r[0:nx+0, 0:ny+0, :, :][:, ::-1, :, :]
        M_t = self.M_t[0:nx+0, 0:ny+0, :, :][:, ::-1, :, :]

        re0 = np.einsum('mnji,mnj->mni', M_t, tran)
        re1 = np.einsum('mnji,mnj->mni', M_r, refl)

        met = np.mean(re0[border:-border,border:-border,0])
        mer = np.mean(re1[border:-border,border:-border,0])

        scl_t = (met + mer) * 0.25 / met
        scl_r = (met + mer) * 0.25 / mer

        out = re0*scl_t + re1*scl_r

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

        filter = np.exp(-((X-x0)**2 + (Y-y0)**2) / 0.011**2) + np.exp(-((X+x0)**2 + (Y+y0)**2) / 0.011**2)
        filter = 1.0 - filter

        corrected = np.fft.ifft2(f * filter[:, :, None], axes=(0, 1)).real

        return power, filter, corrected

            

if __name__ == '__main__':
    root = '/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542'
    demod = Demodulate(root)
    out, M_r, M_t = demod.demodulate(border=50)

    power, filter, corrected = demod.clean(out[100:-100, 100:-100, :])

    # fig, ax = pl.subplots(nrows=1, ncols=3, figsize=(14, 8))
    # ax[0].imshow(np.fft.fftshift(np.log(power[:,:,1])), extent=[-0.5, 0.5, -0.5, 0.5])
    # ax[1].imshow(np.fft.fftshift(filter), extent=[-0.5, 0.5, -0.5, 0.5])
    # ax[2].imshow(np.fft.fftshift(np.log(power[:,:,1] * filter)), extent=[-0.5, 0.5, -0.5, 0.5])

    iquv = fits.open(f'{root}/demodulation/stokesIQUV_00000_8542_8542_+260.fits')

    fig, ax = pl.subplots(nrows=2, ncols=2, figsize=(10, 8))
    ax[0, 0].imshow(corrected[:, :, 0])
    ax[1, 0].imshow(iquv[0].data[0, 0, 0, 100:-100, 100:-100])
    ax[0, 1].imshow(corrected[:, :, 3])
    ax[1, 1].imshow(iquv[0].data[0, 3, 0, 100:-100, 100:-100])

    pl.savefig('test.png')