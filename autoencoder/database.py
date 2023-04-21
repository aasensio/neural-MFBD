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

        n_wavelengths = len(self.config['wavelengths'])
        print(f"Defining pupils for wavelengths : {self.config['wavelengths']}")

        self.cuda = torch.cuda.is_available()
        self.device = torch.device(f"cuda:{self.config['gpu']}" if self.cuda else "cpu")

        self.pupil = []
        self.basis = []
        self.kl = []
        
        # Compute the overfill to properly generate the PSFs from the wavefronts
        for loop in range(n_wavelengths):

            wl = f"{int(self.config['wavelengths'][loop]):4d}"

            # Compute the overfill to properly generate the PSFs from the wavefronts
            overfill = util.psf_scale(self.config['wavelengths'][loop], 
                                            self.config['diameter'][loop], 
                                            self.config['pix_size'][loop])
            
            if (overfill < 1.0):
                raise Exception(f"The pixel size is not small enough to model a telescope with D={self.telescope_diameter} cm")

            # Compute telescope aperture
            pupil = util.aperture(npix=self.config['n_pixel'], 
                            cent_obs = self.config['central_obs'][loop] / self.config['diameter'][loop], 
                            spider=0, 
                            overfill=overfill)
                    
            # Karhunen-Loeve modes            
            print(f"Computing KL modes for {wl}")
            kl = kl_modes.KL()                
            basis = kl.precalculate(npix_image = self.config['n_pixel'], 
                                n_modes_max = self.config['n_modes'], 
                                first_noll = 2, 
                                overfill=overfill)
            basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
            
            self.pupil.append(torch.tensor(pupil.astype('float32')))
            self.basis.append(torch.tensor(basis.astype('float32')))
            self.kl.append(kl)
                     
    def compute_psfs(self, modes, pupil, basis):
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
        wavefront = torch.einsum('ij,ijlm->ilm', modes, basis)

        # Compute the complex phase
        phase = pupil * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft.fft2(phase)
        psf = (torch.conj(ft) * ft).real
        
        # Normalize PSF to unit amplitude        
        psf_norm = psf / torch.amax(psf, dim=(1, 2))[:, None, None]
        
        return wavefront, torch.fft.fftshift(psf_norm)

    def calculate(self, batchsize=64, n_batches=100, r0_min=4.0, r0_max=20.0):

        n_psfs = batchsize * n_batches

        psf_all = []
        modes_all = []
        r0_all = []
        wl_all = []
        D_all = []

        for i in tqdm(range(n_batches)):
            
            r0 = np.random.uniform(low=r0_min, high=r0_max, size=batchsize)
            ind = np.random.randint(low=0, high=len(self.config['diameter']), size=batchsize)
            D = np.array(self.config['diameter'])[ind]
            D_all.append(D)
            wl = np.array(self.config['wavelengths'])[ind]
            wl_all.append(wl)

            coef = (D / r0)**(5.0/6.0)
            
            # Compute standard deviation of the modes
            sigma_KL = np.zeros((batchsize, self.config['n_modes']))
            pupil = []
            basis = []            
            for j in range(batchsize):
                sigma_KL[j, :] = coef[j] * np.sqrt(self.kl[ind[j]].varKL)
                pupil.append(self.pupil[ind[j]][None, :, :])
                basis.append(self.basis[ind[j]][None, :, :])

            pupil = torch.cat(pupil, dim=0).to(self.device)
            basis = torch.cat(basis, dim=0).to(self.device)
            
            modes = np.random.normal(loc=0.0, scale=sigma_KL, size=sigma_KL.shape)
            modes_all.append(modes)
            
            modes = torch.tensor(modes.astype('float32')).to(self.device)
                        
            wavefront, psf_norm = self.compute_psfs(modes, pupil, basis)

            psf_all.append(psf_norm.squeeze().cpu().numpy())
            r0_all.append(r0)
            wl_all.append(wl)
            D_all.append(D)

        psf_all = np.concatenate(psf_all, axis=0)
        modes_all = np.concatenate(modes_all, axis=0)
        r0_all = np.concatenate(r0_all, axis=0)
        wl_all = np.concatenate(wl_all, axis=0)
        D_all = np.concatenate(D_all, axis=0)

        return psf_all, modes_all, r0_all, wl_all, D_all
        
    
if (__name__ == '__main__'):
    config = {
        'gpu': 0,        
        'wavelengths': [3934.0, 6173.0, 8542.0, 6563.0],
        'diameter': [100.0, 100.0, 100.0, 144.0],
        'pix_size': [0.038, 0.059, 0.059, 0.04979],
        'n_pixel': 64,
        'central_obs': [0.0, 0.0, 0.0, 0.0],
        'n_modes': 44   
        }
            
    db = Database(config)
    psf_all, modes_all, r0_all, wl_all, D_all = db.calculate(batchsize=32, n_batches=10, r0_min=4.0, r0_max=20.0)

    # f = zarr.open('training.zarr', 'w')
    # psfd = f.create_dataset('psf', shape=psf_all.shape, dtype=np.float32)
    # modesd = f.create_dataset('modes', shape=modes_all.shape, dtype=np.float32)
    # r0d = f.create_dataset('r0', shape=r0_all.shape, dtype=np.float32)

    # psfd[:] = psf_all
    # modesd[:] = modes_all
    # r0d[:] = r0_all