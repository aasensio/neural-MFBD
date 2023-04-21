import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import matplotlib.pyplot as pl
import math
import kl_modes
import zern
import util
import config
import h5py
import zarr
from astropy.io import fits
from collections import OrderedDict
from tqdm import tqdm
from noise_svd import noise_estimation
from skimage.morphology import flood
import scipy.ndimage as nd
import scipy.interpolate as interp
import napari
import slomo

class Classic(nn.Module):
    def __init__(self, config):
        """

        Parameters
        ----------
        npix_apodization : int
            Total number of pixel for apodization (divisible by 2)
        device : str
            Device where to carry out the computations
        batch_size : int
            Batch size
        """
        super().__init__()

        self.config = config          
        
        self.cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{self.config['gpus'][0]}" if self.cuda else "cpu")        
        self.slomo_interpolator = None
        
        # Generate Hamming window function for WFS correlation
        self.npix_apod = self.config['npix_apodization']
        win = np.hanning(self.npix_apod)
        winOut = np.ones(self.config['n_pixel'])
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        window = np.outer(winOut, winOut)

        # Compute the overfill to properly generate the PSFs from the wavefronts
        self.overfill = util.psf_scale(self.config['wavelength'], 
                                        self.config['diameter'], 
                                        self.config['pix_size'])

        if (self.overfill < 1.0):
            raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

        # Compute telescope aperture
        pupil = util.aperture(npix=self.config['n_pixel'], 
                        cent_obs = self.config['central_obs'] / self.config['diameter'], 
                        spider=0, 
                        overfill=self.overfill)
                
        # Define modes
        # Zernike
        if (self.config['basis_for_wavefront'].lower() == 'zernike'):
            print("Computing Zernike modes...")
            Z_machine = zern.ZernikeNaive(mask=[])
            x = np.linspace(-1, 1, self.config['n_pixel'])
            xx, yy = np.meshgrid(x, x)
            rho = self.overfill * np.sqrt(xx ** 2 + yy ** 2)
            theta = np.arctan2(yy, xx)
            aperture_mask = rho <= 1.0

            basis = np.zeros((self.config['n_modes'], self.config['n_pixel'], self.config['n_pixel']))
            
            # Precompute all Zernike modes except for piston
            for j in range(self.config['n_modes']):
                n, m = zern.zernIndex(j+2)                
                Z = Z_machine.Z_nm(n, m, rho, theta, True, 'Jacobi')
                basis[j,:,:] = Z * aperture_mask
                basis[j,...] /= np.max(np.abs(basis[j,...]))

        # Karhunen-Loeve
        if (self.config['basis_for_wavefront'].lower() == 'kl'):
            print("Computing KL modes...")
            kl = kl_modes.KL()
            basis = kl.precalculate(npix_image = self.config['n_pixel'], 
                                n_modes_max = self.config['n_modes'], 
                                first_noll = 2, 
                                overfill=self.overfill)            
            basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
        
        self.pupil = torch.tensor(pupil.astype('float32')).to(self.device)
        self.basis = torch.tensor(basis[0:self.config['n_modes'], :, :].astype('float32')).to(self.device)
        self.basis_SD = torch.tensor(basis[self.config['n_modes']:, :, :].astype('float32')).to(self.device)
        self.window = torch.tensor(window.astype('float32')).to(self.device)

        self.cutoff = self.config['diameter'] / (self.config['wavelength'] * 1e-8) / 206265.0
        freq = np.fft.fftfreq(self.config['n_pixel'], d=self.config['pix_size']) / self.cutoff
        
        xx, yy = np.meshgrid(freq, freq)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        mask_diff = rho <= 0.90
        self.mask_diffraction_shift = np.fft.fftshift(mask_diff)
        self.mask_diffraction_th = torch.tensor(self.mask_diffraction_shift.astype('float32')).to(self.device)        
                     
    def compute_psfs(self, modes, derivatives=False):
        """Compute the PSFs and their Fourier transform from a set of modes
        
        Args:
            wavefront_focused ([type]): wavefront of the focused image
            illum ([type]): pupil aperture
            diversity ([type]): diversity for this specific images
        
        """

        # --------------
        # Focused PSF
        # --------------
                
        # Compute wavefronts from estimated modes
        wavefront = torch.einsum('ijk,klm->ijlm', modes, self.basis) #+ self.wavefront_SD

        # Compute the complex phase
        phase = self.pupil[None, None, :, :] * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft.fft2(phase)
        psf = (torch.conj(ft) * ft).real
        
        # Normalize PSF
        psf_norm = psf / torch.sum(psf, [2, 3])[:, :, None, None]

        # FFT of the PSF
        psf_ft = torch.fft.fft2(psf_norm)

        if (derivatives):        

            # Now the derivative of the PSF wrt to the modes
            dW_da = 1j * phase[:, :, None, :, :] * self.basis[None, None, :, :, :]
            dft_da = torch.fft.fft2(dW_da)
            dp_da = torch.conj(dft_da) * ft[:, :, None, :, :] + torch.conj(ft[:, :, None, :, :]) * dft_da        
            sump = torch.sum(psf, dim=(-1,-2))
            
            dp_da = (dp_da * sump[:, :, None, None, None] - psf[:, :, None, :, :] * torch.sum(dp_da, dim=(-1,-2))[:, :, :, None, None]) / sump[:, :, None, None, None]**2

            dpsfft_da = torch.fft.fft2(dp_da)
            
            return wavefront, psf_norm, psf_ft, dpsfft_da
        else:
            return wavefront, psf_norm, psf_ft

    def lofdahl_scharmer_filter(self, Sconj_S, Sconj_I):
        den = torch.conj(Sconj_I) * Sconj_I
        H = (Sconj_S / den).real        

        H = torch.fft.fftshift(H).detach().cpu().numpy()

        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,1,3,3], mode='wrap')    
        
        filt = 1.0 - H * self.sigma[:, :, None, None].cpu().numpy()**2 * self.config['n_pixel']**2
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0
        
        nb, no, nx, ny = filt.shape

        mask = np.zeros_like(filt)

        for ib in range(nb):
            for io in range(no):
                
                mask[ib, io, :, :] = flood(1.0 - filt[ib, io, :, :], (nx//2, ny//2), tolerance=0.9) * self.mask_diffraction_shift
                mask[ib, io, :, :] = np.fft.fftshift(mask[ib, io, :, :])

        return torch.tensor(mask).to(Sconj_S.device)

    def compute_image(self, images_ft, psf_ft, type_filter='gaussian'):
        """Compute the reconstructed image
                        
        """        

        Sconj_S = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * psf_ft[:, :, None, :, :], dim=1)
        Sconj_I = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * images_ft, dim=1)
        
        # Use Lofdahl & Scharmer (1994) filter
        if (type_filter == 'lofdahl_scharmer'):

            mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I)
                        
            out = Sconj_I / Sconj_S * mask            
        
        # Use simple Wiener filter with Gaussian prior
        if (type_filter == 'gaussian'):            
            out = Sconj_I / (self.sigma[:, :, None, None] + Sconj_S) * torch.fft.fftshift(self.mask_diffraction_th[None, None, :, :])            
        return out


    def SVDthreshold(self, reconstructed_ft, bad_frames):
        n_b, n_f, _, _ = reconstructed_ft.shape
        reconstructed = torch.fft.ifft2(reconstructed_ft).real

        x = reconstructed.view(n_b, self.config['n_pixel'] * self.config['n_pixel'])

        x_good = torch.cat([reconstructed[i:i+1, :, :, :] for i in range(n_b) if i not in bad_frames], dim=0)
        x_good = x_good.view(x_good.shape[0], self.config['n_pixel'] * self.config['n_pixel'])

        U, S, Vh = torch.linalg.svd(x_good, full_matrices=False)

        x = (x @ Vh.T) @ Vh

        reconstructed = x.view(n_b, 1, self.config['n_pixel'], self.config['n_pixel'])

        # mean_val = torch.mean(reconstructed, dim=(2, 3), keepdims=True)
        # reconstructed = reconstructed - mean_val
        # reconstructed *= self.window[None, None, :, :]
        # reconstructed += mean_val


        reconstructed_ft = torch.fft.fft2(reconstructed)

        return reconstructed, reconstructed_ft

    def frameFilter(self, reconstructed_ft, bad_frames):

        n_b, n_f, _, _ = reconstructed_ft.shape
        reconstructed = torch.fft.ifft2(reconstructed_ft).real.cpu().numpy()

        x = [i for i in range(n_b) if i not in bad_frames]
        reconstructed_good = [reconstructed[i:i+1, :, :, :] for i in range(n_b) if i not in bad_frames]
        x = np.array(x)
        reconstructed_good = np.concatenate(reconstructed_good, axis=0)
        x_full = np.arange(n_b)
        
        out = torch.tensor(interp.interp1d(x, reconstructed_good, axis=0, kind='linear', fill_value='extrapolate')(x_full).astype('float32')).to(reconstructed_ft.device)
        
        out_ft = torch.fft.fft2(out)

        return out, out_ft
    
    def slomo(self, reconstructed_ft, bad_frames):

        if (self.slomo_interpolator is None):
            self.slomo_interpolator = slomo.VideoInterpolation(device=self.device)
                
        reconstructed = torch.fft.ifft2(reconstructed_ft).real

        all_frames = np.arange(len(reconstructed))

        out = self.slomo_interpolator.fix_video(reconstructed, all_frames, bad_frames, npix_apod=self.npix_apod)[:, None, :, :]
        
        out_ft = torch.fft.fft2(out)

        return out, out_ft

    def deconvolve_torch(self, frames, sigma, bad_frames=[], regularize=None):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        frames = frames.to(self.device)
        sigma = sigma.to(self.device)

        n_b, n_f, _, _, _ = frames.shape

        self.sigma = sigma
        self.weight = 1.0 / sigma
        self.regularize = regularize
                            
        modes = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=True)

        opt = torch.optim.SGD([modes], lr=1.0)

        losses = torch.zeros(self.config['gradient_steps'], device=frames.device)

        # Apodize frames and compute FFT
        mean_val = torch.mean(frames, dim=(3, 4), keepdims=True)
        frames_apod = frames - mean_val
        frames_apod *= self.window[None, None, None, :, :]
        frames_apod += mean_val
        frames_ft = torch.fft.fft2(frames_apod) 

        t = tqdm(range(self.config['gradient_steps']))
        
        for loop in t:

            opt.zero_grad()
            
            # Compute PSF from current wavefront coefficients
            modes_centered = modes.clone()
            modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - torch.mean(modes[:, :, 0:2], dim=1, keepdims=True)
            wavefront, psf, psf_ft = self.compute_psfs(modes_centered, derivatives=False)

            # Predict the corrected image
            reconstructed_ft = self.compute_image(frames_ft, psf_ft, type_filter=self.config['image_filter']).detach()

            # Threshold
            if (self.regularize is not None):
                if (self.regularize == 'SVD'):
                    reconstructed, reconstructed_ft = self.SVDthreshold(reconstructed_ft, bad_frames)
                if (self.regularize == 'interpolate'):
                    reconstructed, reconstructed_ft = self.frameFilter(reconstructed_ft, bad_frames)
                if (self.regularize == 'slomo'):                    
                    reconstructed, reconstructed_ft = self.slomo(reconstructed_ft, bad_frames)
            
            degraded_ft = reconstructed_ft[:, None, :, :, :] * psf_ft[:, :, None, :, :]
            degraded = torch.fft.ifft2(degraded_ft).real
            residual = self.weight[:, None, :, None, None] * (degraded - frames_apod)
            
            loss = torch.sum(residual**2, dim=(1,2,3,4))
    
            loss_sum = torch.sum(loss) / (self.config['n_pixel'] * self.config['n_pixel'] * self.config['n_frames'])

            loss_sum.backward()
            
            opt.step()

            tmp = OrderedDict()
            tmp['loss'] = f'{loss_sum.item():7.4f}'
            tmp['grad'] = f'{torch.max(torch.abs(modes.grad)).item():7.4f}'
            t.set_postfix(ordered_dict=tmp)

        
        modes_centered = modes.clone().detach()
        modes_centered[:, :, 0:2] = modes_centered[:, :, 0:2] - torch.mean(modes_centered[:, :, 0:2], dim=1, keepdims=True)
        wavefront, psf, psf_ft = self.compute_psfs(modes_centered, derivatives=False)
        

        # Predict the corrected image        
        reconstructed_ft = self.compute_image(frames_ft, psf_ft, type_filter='lofdahl_scharmer').detach()
        
        if (self.regularize is not None):
            if (self.regularize == 'SVD'):
                reconstructed, reconstructed_ft = self.SVDthreshold(reconstructed_ft, bad_frames)
            if (self.regularize == 'interpolate'):
                reconstructed, reconstructed_ft = self.frameFilter(reconstructed_ft, bad_frames)
            if (self.regularize == 'slomo'):
                reconstructed, reconstructed_ft = self.slomo(reconstructed_ft, bad_frames)
        
        reconstructed = torch.fft.ifft2(reconstructed_ft).real
                            
        return modes, psf, wavefront, degraded, reconstructed, loss            
    

    def deconvolve(self, frames, sigma, n_eig=None, bad_frames=None, regularize=None):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        n_scans = frames.shape[0]
        self.regularize = regularize

        frames = frames.to(self.device)
        sigma = sigma.to(self.device)

        # Finite difference operators
        # tv = np.zeros((n_scans, n_scans))

        # for i in range(n_scans-1):
        #     tv[i, i] = 1.0
        #     tv[i, i+1] = -1.0
        # tv[-1, -1] = 1.0
            
        # AT_A = tv.T @ tv
        # AT_A = np.eye(n_scans) + regularize**2 * AT_A
        # self.AT_A = np.linalg.inv(AT_A)

        n_b, n_f, _, _, _ = frames.shape

        self.sigma = sigma
        self.weight = 1.0 / sigma
                            
        modes = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=False)

        losses = torch.zeros(self.config['gradient_steps'], device=frames.device)

        # Apodize frames and compute FFT        
        mean_val = torch.mean(frames, dim=(3, 4), keepdims=True)
        frames_apod = frames - mean_val
        frames_apod *= self.window[None, None, None, :, :]
        frames_apod += mean_val
        frames_ft = torch.fft.fft2(frames_apod)


        t = tqdm(range(self.config['gradient_steps']))
        
        for loop in t:
            
            # Compute PSF from current wavefront coefficients
            modes[:, :, 0:2] = modes[:, :, 0:2] - torch.mean(modes[:, :, 0:2], dim=1, keepdims=True)
            wavefront, psf, psf_ft, dpsfft_da = self.compute_psfs(modes, derivatives=True)

            # Predict the corrected image
            reconstructed_ft = self.compute_image(frames_ft, psf_ft, type_filter=self.config['image_filter'])

            # Threshold
            if (self.regularize is not None):
                if (self.regularize == 'SVD'):
                    reconstructed, reconstructed_ft = self.SVDthreshold(reconstructed_ft, n_eig)
                if (self.regularize == 'interpolate'):
                    reconstructed, reconstructed_ft = self.frameFilter(reconstructed_ft, bad_frames)
                if (self.regularize == 'slomo'):
                    reconstructed, reconstructed_ft = self.slomo(reconstructed_ft, bad_frames)
                        
            degraded_ft = reconstructed_ft[:, None, :, :, :] * psf_ft[:, :, None, :, :]
            degraded = torch.fft.ifft2(degraded_ft).real            
            residual = self.weight[:, None, :, None, None] * (degraded - frames_apod)

            loss = torch.sum(residual**2, dim=(1,2,3,4)) / (self.config['n_pixel'] * self.config['n_pixel'] * self.config['n_frames'])

            # residual_ft = self.weight[:, None, :, None, None] * (degraded_ft - frames_ft)
            # loss_ft = torch.mean(residual_ft *  torch.conj(residual_ft), dim=(1,2,3,4)) / (self.config['n_pixel'] * self.config['n_pixel'])


            ddegraded_ft = dpsfft_da[:, :, None, :, :, :] * reconstructed_ft[:, None, :, None, :, :]
            ddegraded = torch.fft.ifft2(ddegraded_ft).real

            residual_weight = residual * self.weight[:, None, :, None, None]

            gradient = torch.sum(2.0 * residual_weight[:, :, :, None, :, :] * ddegraded, dim=(2, -1, -2)).real / (self.config['n_pixel'] * self.config['n_pixel'] * self.config['n_frames'])
            
            modes = modes - 1.0 * gradient            

            loss_sum = loss.sum() #/ (self.config['n_pixel'] * self.config['n_pixel'] * self.config['n_frames'])

            tmp = OrderedDict()
            tmp['loss'] = f'{loss_sum.item():7.4f}'
            tmp['grad'] = f'{torch.max(torch.abs(gradient)).item():7.4f}'
            t.set_postfix(ordered_dict=tmp)
            
        loss = torch.sum(losses)
        
        modes[:, :, 0:2] = modes[:, :, 0:2] - torch.mean(modes[:, :, 0:2], dim=1, keepdims=True)
        wavefront, psf, psf_ft = self.compute_psfs(modes, derivatives=False)

        # Predict the corrected image
        reconstructed_ft = self.compute_image(frames_ft, psf_ft, type_filter='lofdahl_scharmer').detach()

        if (self.regularize is not None):
            if (self.regularize == 'SVD'):
                reconstructed, reconstructed_ft = self.SVDthreshold(reconstructed_ft, bad_frames)
            if (self.regularize == 'interpolate'):
                reconstructed, reconstructed_ft = self.frameFilter(reconstructed_ft, bad_frames)
            if (self.regularize == 'slomo'):
                reconstructed, reconstructed_ft = self.slomo(reconstructed_ft, bad_frames)

        reconstructed = torch.fft.ifft2(reconstructed_ft).real
                                    
        return modes, psf, wavefront, degraded, reconstructed, loss

    
if (__name__ == '__main__'):
    config = {
        'gpus': [2],
        'npix_apodization': 24,
        'basis_for_wavefront': 'kl',
        'n_modes': 44,
        'n_frames' : 12,
        'n_pixel' : 64,
        'gradient_steps' : 100,
        'wavelength': 6563.0,
        'diameter': 144.0,
        'pix_size': 0.04979,
        'central_obs' : 0.0,
        'image_filter': 'gaussian' #'lofdahl_scharmer'
    }
    
    f = h5py.File('/scratch1/aasensio/unroll_mfbd/classic_regularized/test.h5', 'r')

    frames = f['images'][:, :, 200:200+config['n_pixel'], 200:200+config['n_pixel']]
    frames /= np.mean(frames)

    # ff, ax = pl.subplots(nrows=8, ncols=7, figsize=(20, 20))
    # for i in range(50):
    #     ax.flat[i].imshow(frames[i, 0, :, :])

        

    n_scans, n_frames, nx, ny = frames.shape

    x = frames.reshape((n_scans, n_frames, nx*ny))
        
    sigma = noise_estimation(x)
        
    classic = Classic(config)
    
    frames = torch.tensor(frames[:, :, None, :, :].astype('float32')).to(classic.device)
    sigma = torch.tensor(sigma[:, None].astype('float32')).to(classic.device)

    # config['gradient_steps'] = 100
    # modes_th, psf_th, wavefront_th, degraded_th, reconstructed_th, loss_th = classic.deconvolve_torch(frames, sigma)

    classic.config['image_filter'] = 'gaussian'
    classic.config['gradient_steps'] = 40
    modes, psf, wavefront, degraded, reconstructed, loss = classic.deconvolve_torch(frames, sigma, regularize=None)

    classic.config['gradient_steps'] = 120
    modes2, psf2, wavefront2, degraded2, reconstructed2, loss2 = classic.deconvolve_torch(frames, sigma, regularize='slomo', bad_frames=[6, 7, 12, 14, 15, 16, 17, 18, 20, 21, 22, 26, 41, 42, 43, 44, 45, 47, 48])

    # np.save('reconstructed.npy', reconstructed.cpu().numpy())
    
    f.close()

    # comparison = np.concatenate((reconstructed[:, 0, ...].cpu().numpy(), reconstructed2[:, 0, ...].cpu().numpy()), axis=2)

    # viewer = napari.view_image(comparison)

    
    # ff, ax = pl.subplots(nrows=2, ncols=8, figsize=(18, 8))
    # for i in range(8):
    #     ax[0, i].imshow(reconstructed[13+i, 0, :, :].cpu())
    #     ax[1, i].imshow(reconstructed2[13+i, 0, :, :].cpu())
    #     ax[0, i].set_title(f'Orig {i}')
    #     ax[1, i].set_title(f'Regu {i}')

    # tmp = torch.cat([reconstructed, reconstructed2], dim=3)
    # viewer = napari.view_image(tmp[:,0,:,:].cpu().numpy())

    
    # mean_val = np.mean(mfbdbb_patch, keepdims=True)        
    # mfbdbb_patch = mfbdbb_patch - mean_val
    # mfbdbb_patch *= classic.window.cpu().numpy()
    # mfbdbb_patch += mean_val

    # mean_val = np.mean(mfbdnb_patch, keepdims=True)        
    # mfbdnb_patch = mfbdnb_patch - mean_val
    # mfbdnb_patch *= classic.window.cpu().numpy()
    # mfbdnb_patch += mean_val

    # fig, ax = pl.subplots(nrows=2, ncols=3)
    # ax[0, 0].imshow(frames[0, 0, 0, :, :].cpu().numpy())
    # ax[0, 1].imshow(np.mean(frames[0, :, 0, :, :].cpu().numpy(), axis=0))    
    # ax[1, 0].imshow(reconstructed[0, 0, :, :].cpu().numpy())    
    # ax[1, 1].imshow(reconstructed_th[0, 0, :, :].cpu().numpy())    
    # # ax[1, 2].imshow(mfbd)    
    # pl.show()     