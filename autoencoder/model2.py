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


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, channels_latent=64, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    

        self.inc = DoubleConv(n_channels, channels_latent)
        self.down1 = Down(channels_latent, 2*channels_latent, scale_factor=2)
        self.down2 = Down(2*channels_latent, 4*channels_latent, scale_factor=2)
        self.down3 = Down(4*channels_latent, 8*channels_latent, scale_factor=2)
        factor = 2 if bilinear else 1
        self.down4 = Down(8*channels_latent, 16*channels_latent // factor, scale_factor=2)
        self.up1 = Up(16*channels_latent, 8*channels_latent // factor, bilinear, scale_factor=2)
        self.up2 = Up(8*channels_latent, 4*channels_latent // factor, bilinear, scale_factor=2)
        self.up3 = Up(4*channels_latent, 2*channels_latent // factor, bilinear, scale_factor=2)
        self.up4 = Up(2*channels_latent, channels_latent, bilinear, scale_factor=2)
        self.outc = OutConv(channels_latent, n_classes)

        fc = [nn.Linear(256*4*4, 256),
                nn.SiLU(), 
                nn.Linear(256, 128), 
                nn.SiLU(), 
                nn.Linear(128, 44), 
                nn.SiLU(), 
                nn.Linear(44, 128), 
                nn.SiLU(), 
                nn.Linear(128, 256),
                nn.SiLU(), 
                nn.Linear(256, 256*4*4)]
        self.bottleneck = nn.Sequential(*fc)

    def forward(self, x):        
        x1 = self.inc(x)        
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5p = self.bottleneck(x5.view(-1, 256*4*4)).view(-1, 256, 4, 4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)        
        out = self.outc(x)        
        return out

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
        self.encoder = UNet(n_channels=1, n_classes=1, channels_latent=32, bilinear=True)
                                     
    def forward(self, psf):
        
        # CNN encoder       
        out = self.encoder(psf)

        return out

    def loss(self, psf):
        recons = self.forward(psf).squeeze()

        recons_loss = torch.mean( (psf.squeeze() - recons) **2 / 1e-3**2)
        kld_loss = torch.tensor(0.0)

        if (self.config['beta'] != 0):
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
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