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

def weight_init(m, mode='kaiming'):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if (mode == 'kaiming'):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if (mode == 'xavier'):
            nn.init.xavier_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.1)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight.data, 1)                
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(scale_factor),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Model(nn.Module):
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

        n_modes = self.config['n_modes']

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
        self.KL_std = torch.tensor(np.sqrt(self.kl.varKL).astype('float32')).to(self.device)

        # Neural networks

        #----------------
        # Encoder : from 64x64 to a vector of size n_modes
        # 64->32->16->8
        #----------------
        encoder = [DoubleConv(1, 32), 
            Down(32, 64, scale_factor=2), 
            Down(64, 128, scale_factor=2), 
            Down(128, 256, scale_factor=2)]
        self.encoder = nn.Sequential(*encoder)
        self.encoder.apply(weight_init)
        
        if (self.config['beta'] == 0):
            encoder_fc = [nn.Linear(256*8*8, 4*n_modes)]
            self.encoder_fc = nn.Sequential(*encoder_fc)
            self.encoder_fc.apply(weight_init)
        else:
            self.encoder_fc_mu = nn.Linear(256*8*8, 4*n_modes)
            self.encoder_fc_logvar = nn.Linear(256*8*8, 4*n_modes)
            self.encoder_fc_mu.apply(weight_init)
            self.encoder_fc_logvar.apply(weight_init)

        #----------------
        # Transformer : from Z to KL modes
        #----------------
        
        transformer = [nn.Linear(n_modes, 256),
                nn.ReLU(), 
                nn.Linear(256, 256), 
                nn.ReLU(), 
                nn.Linear(256, n_modes)]
        self.transformer = nn.Sequential(*transformer)
        self.transformer.apply(weight_init)
                     
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
        
        # Normalize PSF        
        psf_norm = psf / torch.amax(psf, dim=(1,2))[:, None, None]
                                
        return wavefront, torch.fft.fftshift(psf_norm)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        
        # Transform from latent space to KL modes
        kl = 10.0 * self.KL_std[None, :] * self.transformer(z)

        # Compute wavefronts and PSFs from KL modes
        wavefront, psf = self.compute_psfs(kl)

        return kl, wavefront, psf

    def forward(self, psf):
        
        # CNN encoder       
        tmp = self.encoder(psf)
        tmp = tmp.view(-1, 256*8*8)

        if (self.config['beta'] == 0):
            z = self.encoder_fc(tmp)            
            mu = None
            logvar = None
        else:
            mu = self.encoder_fc_mu(tmp)
            logvar = self.encoder_fc_logvar(tmp)
            
            z = self.reparameterize(mu, logvar)
                
        kl, wavefront, psf = self.decode(z)

        return psf, mu, logvar

    def loss(self, psf):
        psf_sqrt = torch.sqrt(psf)
        recons, mu, logvar = self.forward(psf_sqrt)

        # recons_loss = torch.mean( (psf.squeeze() - recons)**2 / 1e-3**2)

        recons_loss = F.mse_loss(torch.sqrt(recons), psf_sqrt.squeeze())

        kld_loss = torch.tensor(0.0)

        if (self.config['beta'] != 0):
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        
        loss = recons_loss + self.config['beta'] * kld_loss

        return loss, recons_loss, self.config['beta'] * kld_loss, recons
    
if (__name__ == '__main__'):
    config = {
        'gpu': 0,
        'n_pixel': 128,
        'wavelength': 6563.0,
        'diameter': 144.0,
        'pix_size': 0.04979,
        'n_pixel': 64,
        'central_obs': 0.0,
        'n_modes': 44   
        }
        
    f = zarr.open('')
    db = Model(config)
    psf_all, modes_all, r0_all = db.calculate(batchsize=32, n_batches=100, r0_min=4.0, r0_max=20.0)