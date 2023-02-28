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

    def forward(self, x):
        out = self.up(x)
        return self.conv(out)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, channels_latent=8, n_bottleneck=128, bilinear=False, beta=0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear    
        self.beta = beta
        factor = 2 if bilinear else 1

        #--------------
        # Encoder
        #--------------
        inc = DoubleConv(n_channels, channels_latent)
        down1 = Down(channels_latent, 2*channels_latent, scale_factor=2)
        down2 = Down(2*channels_latent, 4*channels_latent, scale_factor=2)
        down3 = Down(4*channels_latent, 8*channels_latent, scale_factor=2)        

        encoder = [inc, down1, down2, down3]
        self.encoder = nn.Sequential(*encoder)

        #--------------
        # Decoder
        #--------------                
        up3 = Up(8*channels_latent, 4*channels_latent, bilinear, scale_factor=2)
        up2 = Up(4*channels_latent, 2*channels_latent, bilinear, scale_factor=2)
        up1 = Up(2*channels_latent, channels_latent, bilinear, scale_factor=2)
        outc = OutConv(channels_latent, n_classes)
        
        decoder = [up3, up2, up1, outc]
        self.decoder = nn.Sequential(*decoder)

        #--------------
        # Bottleneck
        #--------------
        if (beta == 0):
            
            self.bottleneck_encoder = nn.Linear(8*8*64, n_bottleneck)            

        else:
            
            self.bottleneck_encoder_mu = nn.Linear(8*8*64, n_bottleneck)
            self.bottleneck_encoder_logvar = nn.Linear(8*8*64, n_bottleneck)
            
        self.bottleneck_decoder = nn.Linear(n_bottleneck, 8*8*64)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, psf):
        
        # Encode the PSF
        enc = self.encoder(psf)
        
        encp = enc.view(-1, 64*8*8)

        # Now go through the bottleneck
        if (self.beta == 0):
            mu = None
            logvar = None
            z = self.bottleneck_encoder(encp)            
        else:
            mu = self.bottleneck_encoder_mu(encp)
            logvar = self.bottleneck_encoder_logvar(encp)
            z = self.reparameterize(mu, logvar)
        
        zp = self.bottleneck_decoder(z)

        zp = zp.view(-1, 64, 8, 8)

        out = self.decoder(zp)
        
        return out, mu, z, logvar

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
        
        # Neural networks

        #----------------
        # Encoder : from 64x64 to a vector of size n_modes
        # 64->32->16->8
        #----------------        
        self.autoencoder = UNet(n_channels=1, 
                    n_classes=1, 
                    channels_latent=8, 
                    n_bottleneck=self.config['n_bottleneck'], 
                    bilinear=True, 
                    beta=self.config['beta'])
                                     
    def forward(self, psf):
        
        # CNN encoder       
        out, mu, z, logvar = self.autoencoder(psf)

        return out, mu, z, logvar

    def loss(self, psf):

        recons, mu, z, logvar = self.forward(psf)

        recons_loss = torch.mean( (psf.squeeze() - recons.squeeze()) **2 / 1e-3**2)
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