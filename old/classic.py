import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import matplotlib.pyplot as pl
import math
import blocks
from millify import millify
import kl_modes
import zern
import util
import config
import zarr
from astropy.io import fits
from collections import OrderedDict
from tqdm import tqdm
from noise_svd import noise_estimation
from skimage.morphology import flood
import scipy.ndimage as nd

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
                                n_modes_max = 200,#self.config['n_modes'], 
                                first_noll = 2, 
                                overfill=self.overfill)            
            basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
        
        self.pupil = torch.tensor(pupil.astype('float32')).to(self.device)
        self.basis = torch.tensor(basis[0:self.config['n_modes'], :, :].astype('float32')).to(self.device)
        self.basis_SD = torch.tensor(basis[self.config['n_modes']:, :, :].astype('float32')).to(self.device)
        self.window = torch.tensor(window.astype('float32')).to(self.device)

        self.gamma_obj = torch.tensor(self.config['gamma_obj'].astype('float32')).to(self.device)
        sigma2 = self.config['sigma2']
        gamma = 1.0 / sigma2
        gamma /= gamma[0]
        self.weight = torch.tensor(gamma.astype('float32')).to(self.device)
        print(f'weight={self.weight}')
        print(f'gamma ={self.gamma_obj}')

        n_SD = 200 - self.config['n_modes']
        sigma_KL = np.sqrt(kl.KL_variance[self.config['n_modes']:200]) * (self.config['diameter'] / 10.0)**(5.0/6.0)
        alpha_SD = sigma_KL * np.random.randn(n_SD)
        alpha_SD = torch.tensor(alpha_SD.astype('float32')).to(self.device)

        self.wavefront_SD = torch.sum(alpha_SD[:,None,None] * self.basis_SD, dim=0)

        self.cutoff = self.config['diameter'] / (self.config['wavelength'] * 1e-8) / 206265.0
        freq = np.fft.fftfreq(self.config['n_pixel'], d=self.config['pix_size']) / self.cutoff
        
        xx, yy = np.meshgrid(freq, freq)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        mask_diff = rho <= 0.90
        self.mask_diffraction_shift = np.fft.fftshift(mask_diff)
                     
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

        H = torch.fft.fftshift(H).cpu().numpy()

        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,1,3,3], mode='wrap')        
        
        filt = 1.0 - H * self.gamma_obj[None, :, None, None].cpu().numpy()**2 * self.config['n_pixel']**2
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0
        
        nb, no, nx, ny = filt.shape

        mask = np.zeros_like(filt)

        for ib in range(nb):
            for io in range(no):
                
                mask[ib, io, :, :] = flood(1.0 - filt[ib, io, :, :], (nx//2, ny//2), tolerance=0.9) * self.mask_diffraction_shift
                mask[ib, io, :, :] = np.fft.fftshift(mask[ib, io, :, :])

        return torch.tensor(mask).to(Sconj_S.device)

    def compute_image(self, images_ft, psf_ft):
        """Compute the reconstructed image
                        
        """        

        Sconj_S = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * psf_ft[:, :, None, :, :], dim=1)
        Sconj_I = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * images_ft, dim=1)
        
        # Use Lofdahl & Scharmer (1994) filter
        if (self.config['image_filter'] == 'lofdahl_scharmer'):

            mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I)
                        
            out = Sconj_I / Sconj_S * mask
        
        # Use simple Wiener filter with Gaussian prior
        if (self.config['image_filter'] == 'gaussian'):            
            out = Sconj_I / (self.gamma_obj[None, :, None, None] + Sconj_S)

        return out

    def deconvolve_torch(self, frames):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        n_b, n_f, _, _, _ = frames.shape
                            
        modes = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=True)

        losses = torch.zeros(self.config['gradient_steps'], device=frames.device)

        # Apodize frames and compute FFT
        mean_val = torch.mean(frames, dim=(2, 3), keepdims=True)
        frames_apod = frames - mean_val
        frames_apod *= self.window[None, None, :, :]
        frames_apod += mean_val
        frames_ft = torch.fft.fft2(frames_apod) 
        
        for loop in range(self.config['gradient_steps']):

            modes = modes.detach().requires_grad_(True)
            
            # Compute PSF from current wavefront coefficients
            wavefront, psf, psf_ft = self.compute_psfs(modes, derivatives=False)

            # Predict the corrected image
            reconstructed_ft = self.compute_image(frames_ft, psf_ft).detach()
            reconstructed = torch.fft.ifft2(reconstructed_ft)

            # # Apodize current image and compute FFT
            # mean_val = torch.mean(reconstructed, dim=(2, 3), keepdims=True)        
            # reconstructed_apod = reconstructed - mean_val
            # reconstructed_apod *= self.window[None, None, :, :]
            # reconstructed_apod += mean_val
            # reconstructed_ft = torch.fft.fft2(reconstructed_apod)            
            
            degraded_ft = reconstructed_ft[:, None, :, :, :] * psf_ft[:, :, None, :, :]
            degraded = torch.fft.ifft2(degraded_ft).real
            residual = degraded - frames_apod
            loss = torch.sum(residual**2)

            loss.backward()
            
            modes = modes - 20 * modes.grad
            print(loss.item())
            
        loss = torch.sum(losses)
                            
        return modes, psf, wavefront, degraded, reconstructed, loss            

    def deconvolve(self, frames):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        n_b, n_f, _, _, _ = frames.shape
                            
        modes = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=False)

        losses = torch.zeros(self.config['gradient_steps'], device=frames.device)

        # Apodize frames and compute FFT
        mean_val = torch.mean(frames, dim=(2, 3), keepdims=True)
        frames_apod = frames - mean_val
        frames_apod *= self.window[None, None, :, :]
        frames_apod += mean_val
        frames_ft = torch.fft.fft2(frames_apod)

        t = tqdm(range(self.config['gradient_steps']))
        
        for loop in t:
            
            # Compute PSF from current wavefront coefficients
            wavefront, psf, psf_ft, dpsfft_da = self.compute_psfs(modes, derivatives=True)

            # Predict the corrected image
            reconstructed_ft = self.compute_image(frames_ft, psf_ft)

            # # Apodize current image and compute FFT
            # mean_val = torch.mean(reconstructed, dim=(2, 3), keepdims=True)        
            # reconstructed_apod = reconstructed - mean_val
            # reconstructed_apod *= self.window[None, None, :, :]
            # reconstructed_apod += mean_val
            # reconstructed_ft = torch.fft.fft2(reconstructed_apod)            
            
            degraded_ft = reconstructed_ft[:, None, :, :, :] * psf_ft[:, :, None, :, :]
            degraded = torch.fft.ifft2(degraded_ft).real            
            residual = self.weight[None, None, :, None, None] * (degraded - frames_apod)            
            loss = torch.sum(residual**2, dim=(-1,-2,-3))

            ddegraded_ft = dpsfft_da[:, :, None, :, :, :] * reconstructed_ft[:, None, :, None, :, :]
            ddegraded = torch.fft.ifft2(ddegraded_ft).real

            gradient = torch.sum(2.0 * residual[:, :, :, None, :, :] * ddegraded, dim=(2, -1, -2)).real
            
            modes = modes - 1.3 * gradient

            modes[:, :, 0:2] = modes[:, :, 0:2] - torch.mean(modes[:, :, 0:2], dim=1, keepdims=True)

            tmp = OrderedDict()
            tmp['loss'] = f'{loss.sum().item():7.4f}'
            t.set_postfix(ordered_dict=tmp)
            
        loss = torch.sum(losses)

        reconstructed = torch.fft.ifft2(reconstructed_ft).real
                                    
        return modes, psf, wavefront, degraded, reconstructed, loss

    
if (__name__ == '__main__'):
    configuration = config.Config('config.ini').hyperparameters
    configuration['n_frames'] = 12
    configuration['n_pixel'] = 64
    configuration['gradient_steps'] = 100

    file = zarr.open(configuration['training_file'], 'r')

    
    
    mfbdbb = fits.open('/net/dracarys/scratch/ckuckein/gregor/hifiplus/2/20220607/scan_b001/outbb.fz.fits')[0].data
    mfbdbb_patch = mfbdbb[270:270+configuration['n_pixel'], 270:270+configuration['n_pixel']]
    mfbdnb = fits.open('/net/dracarys/scratch/ckuckein/gregor/hifiplus/2/20220607/scan_b001/outnb_pos0001.fz.fits')[0].data
    mfbdnb_patch = mfbdnb[270:270+configuration['n_pixel'], 270:270+configuration['n_pixel']]
    
    frames = np.float32(file['20220607/001/bb'][0:configuration['n_frames'], 300:300+configuration['n_pixel'], 300:300+configuration['n_pixel']])
    frames /= np.mean(frames)    
    
    noiseEstimation_bb = noise_estimation(frames.reshape((100,128*128)))
    print(f'N/S={noiseEstimation_bb}')
       
    frames2 = np.float32(file['20220607/001/nb'][0:configuration['n_frames'], 300:300+configuration['n_pixel'], 300:300+configuration['n_pixel']])    
    frames2 /= np.mean(frames2)

    noiseEstimation_nb = noise_estimation(frames2.reshape((100,128*128)))
    
    print(f'N/S={noiseEstimation_nb}')

    sigma = np.array([noiseEstimation_bb, 1.0*noiseEstimation_nb])
    configuration['sigma2'] = sigma**2
    configuration['gamma_obj'] = sigma
    configuration['image_filter'] = 'lofdahl_scharmer' #'gaussian'
    configuration['gpus'] = [0]
    
    classic = Classic(configuration)
    
    frames = torch.tensor(frames[None, :, None, :, :].astype('float32')).to(classic.device)
    frames2 = torch.tensor(frames2[None, :, None, :, :].astype('float32')).to(classic.device)
    frames = torch.cat([frames, frames2], dim=2)

    
    # classic.deconvolve_torch(frames)    
    modes, psf, wavefront, degraded, reconstructed, loss = classic.deconvolve(frames)

    mean_val = np.mean(mfbdbb_patch, keepdims=True)        
    mfbdbb_patch = mfbdbb_patch - mean_val
    mfbdbb_patch *= classic.window.cpu().numpy()
    mfbdbb_patch += mean_val

    mean_val = np.mean(mfbdnb_patch, keepdims=True)        
    mfbdnb_patch = mfbdnb_patch - mean_val
    mfbdnb_patch *= classic.window.cpu().numpy()
    mfbdnb_patch += mean_val

    fig, ax = pl.subplots(nrows=2, ncols=4)
    ax[0, 0].imshow(frames[0, 0, 0, :, :].cpu().numpy())
    ax[1, 0].imshow(frames[0, 0, 1, :, :].cpu().numpy())    
    ax[0, 1].imshow(np.mean(frames[0, :, 0, :, :].cpu().numpy(), axis=0))
    ax[1, 1].imshow(np.mean(frames[0, :, 1, :, :].cpu().numpy(), axis=0))
    ax[0, 2].imshow(reconstructed[0, 0, :, :].cpu().numpy())
    ax[1, 2].imshow(reconstructed[0, 1, :, :].cpu().numpy())
    ax[0, 3].imshow(mfbdbb_patch)
    ax[1, 3].imshow(mfbdnb_patch)
    pl.show()     