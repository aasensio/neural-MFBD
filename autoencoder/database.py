import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.nn.init as init
import matplotlib.pyplot as pl
import kl_modes
import zern
import util
import zarr
from astropy.io import fits
from collections import OrderedDict
from tqdm import tqdm


class Database(nn.Module):
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
        self.device = torch.device(f"cuda:{self.config['gpu']}" if self.cuda else "cpu")      
        
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
                        
        self.kl = kl_modes.KL()
        basis = self.kl.precalculate(npix_image = self.config['n_pixel'], 
                            n_modes_max = self.config['n_modes'], 
                            first_noll = 4,
                            overfill=self.overfill)            
        basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
    
        self.pupil = torch.tensor(pupil.astype('float32')).to(self.device)
        self.basis = torch.tensor(basis.astype('float32')).to(self.device)
                     
    def compute_psfs(self, modes):
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
        wavefront = torch.einsum('ik,klm->ilm', modes, self.basis)

        # Compute the complex phase
        phase = self.pupil[None, :, :] * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft.fft2(phase)
        psf = (torch.conj(ft) * ft).real
        
        # Normalize PSF to unit amplitude        
        psf_norm = psf / torch.amax(psf, dim=(1,2))[:, None, None]
        
        return wavefront, torch.fft.fftshift(psf_norm)

    def calculate(self, batchsize=64, n_batches=100, r0_min=4.0, r0_max=20.0):

        n_psfs = batchsize * n_batches

        psf_all = []
        modes_all = []
        r0_all = []

        for i in tqdm(range(n_batches)):
            
            r0 = np.random.uniform(low=r0_min, high=r0_max, size=batchsize)
            coef = (self.config['diameter'] / r0)**(5.0/6.0)

            # We do not consider the first mode
            sigma_KL = np.sqrt(self.kl.varKL)
            sigma_KL = coef[:, None] * sigma_KL

            modes = np.random.normal(loc=0.0, scale=sigma_KL, size=sigma_KL.shape)
            modes_all.append(modes)
            
            modes = torch.tensor(modes.astype('float32')).to(self.device)
                        
            wavefront, psf_norm = self.compute_psfs(modes)

            psf_all.append(psf_norm.squeeze().cpu().numpy())
            r0_all.append(r0)

        psf_all = np.concatenate(psf_all, axis=0)
        modes_all = np.concatenate(modes_all, axis=0)
        r0_all = np.concatenate(r0_all, axis=0)

        return psf_all, modes_all, r0_all
        
    
if (__name__ == '__main__'):
    config = {
        'gpu': 0,        
        'wavelength': 8542.0,
        'diameter': 100.0,
        'pix_size': 0.059,
        'n_pixel': 64,
        'central_obs': 0.0,
        'n_modes': 44   
        }
            
    db = Database(config)
    psf_all, modes_all, r0_all = db.calculate(batchsize=32, n_batches=1000, r0_min=4.0, r0_max=20.0)

    f = zarr.open('training.zarr', 'w')
    psfd = f.create_dataset('psf', shape=psf_all.shape, dtype=np.float32)
    modesd = f.create_dataset('modes', shape=modes_all.shape, dtype=np.float32)
    r0d = f.create_dataset('r0', shape=r0_all.shape, dtype=np.float32)

    psfd[:] = psf_all
    modesd[:] = modes_all
    r0d[:] = r0_all