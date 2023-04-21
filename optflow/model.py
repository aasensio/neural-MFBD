import numpy as np
import torch.nn as nn
import torch
import torch.functional as F

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

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)        

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

    
class backWarp(nn.Module):
    """
    A class for creating a backwarping object.
    This is used for backwarping to an image:
    Given optical flow from frame I0 to I1 --> F_0_1 and frame I1, 
    it generates I0 <-- backwarp(F_0_1, I1).
    ...
    Methods
    -------
    forward(x)
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
    """


    def __init__(self, W, H):
        """
        Parameters
        ----------
            W : int
                width of the image.
            H : int
                height of the image.
            device : device
                computation device (cpu/cuda). 
        """


        super(backWarp, self).__init__()
        # create a grid
        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        gridX = torch.tensor(gridX, requires_grad=False)
        gridY = torch.tensor(gridY, requires_grad=False)

        self.register_buffer('gridX', gridX)
        self.register_buffer('gridY', gridY)

    def update_grid(self, W, H):

        gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
        self.W = W
        self.H = H
        gridX = torch.tensor(gridX, requires_grad=False)
        gridY = torch.tensor(gridY, requires_grad=False)

        device = self.gridX.device

        self.gridX = gridX.to(device)
        self.gridY = gridY.to(device)
        
    def forward(self, img, flow):
        """
        Returns output tensor after passing input `img` and `flow` to the backwarping
        block.
        I0  = backwarp(I1, F_0_1)
        Parameters
        ----------
            img : tensor
                frame I1.
            flow : tensor
                optical flow from I0 and I1: F_0_1.
        Returns
        -------
            tensor
                frame I0.
        """

        # Extract horizontal and vertical flows.
        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]
        x = self.gridX.unsqueeze(0).expand_as(u).float() + u
        y = self.gridY.unsqueeze(0).expand_as(v).float() + v
        # range -1 to 1
        x = 2*(x/self.W - 0.5)
        y = 2*(y/self.H - 0.5)
        
        # stacking X and Y
        grid = torch.stack((x,y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=False)
        return imgOut, grid

def cross_correlation(im1, im2, eps=1e-8):
    """
    A function for computing the cross-correlation between two images.
    ...
    Parameters
    ----------
    im1 : tensor
    """
    im1_mean = torch.mean(im1, dim=(2,3), keepdim=True)
    im2_mean = torch.mean(im2, dim=(2,3), keepdim=True)
    im1 = im1 - im1_mean
    im2 = im2 - im2_mean

    num = torch.sum(im1 * im2)
    den = torch.sqrt(torch.sum(im1**2) * torch.sum(im2**2) + eps)
    
    return num / den

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.config = config

        self.n_images = 2
        self.n_output = 2
        self.channels_latent = self.config['channels']
        self.n_pixel = self.config['n_pixel']

        #--------------
        # Encoder
        #--------------
        self.inc = DoubleConv(self.n_images, self.channels_latent)                             # N
        self.down1 = Down(self.channels_latent, 2*self.channels_latent, scale_factor=2)        # N/2
        self.down2 = Down(2*self.channels_latent, 4*self.channels_latent, scale_factor=2)      # N/4
        self.down3 = Down(4*self.channels_latent, 8*self.channels_latent, scale_factor=2)      # N/8
        self.down4 = Down(8*self.channels_latent, 8*self.channels_latent, scale_factor=2)      # N/16
        self.down5 = Down(8*self.channels_latent, 8*self.channels_latent, scale_factor=2)      # N/32
        self.down6 = Down(8*self.channels_latent, 8*self.channels_latent, scale_factor=2)      # N/64

        proj4 = []
        proj4.append(DoubleConv(8*self.channels_latent, 8*self.channels_latent))
        proj4.append(nn.Conv2d(8*self.channels_latent, self.n_output, kernel_size=1))
        self.proj4 = nn.Sequential(*proj4)

        proj5 = []
        proj5.append(DoubleConv(8*self.channels_latent, 8*self.channels_latent))
        proj5.append(nn.Conv2d(8*self.channels_latent, self.n_output, kernel_size=1))
        self.proj5 = nn.Sequential(*proj5)

        proj6 = []
        proj6.append(DoubleConv(8*self.channels_latent, 8*self.channels_latent))
        proj6.append(nn.Conv2d(8*self.channels_latent, self.n_output, kernel_size=1))
        self.proj6 = nn.Sequential(*proj6)

        self.bilinear4 = nn.Upsample(scale_factor=2**4, mode='bilinear', align_corners=True)
        self.bilinear5 = nn.Upsample(scale_factor=2**5, mode='bilinear', align_corners=True)
        self.bilinear6 = nn.Upsample(scale_factor=2**6, mode='bilinear', align_corners=True)

        self.backwarp = backWarp(self.n_pixel, self.n_pixel)
        

    def update_grid(self, W, H):

        self.backwarp.update_grid(W, H)

    def forward(self, x, mode='combined'):

        I0 = x[:, 0:1, :, :]
        I1 = x[:, 1:2, :, :]
        
        out_N = self.inc(x)
        out_N2 = self.down1(out_N)
        out_N4 = self.down2(out_N2)
        out_N8 = self.down3(out_N4)
        out_N16 = self.down4(out_N8)
        out_N32 = self.down5(out_N16)
        out_N64 = self.down6(out_N32)

        flow_64 = self.bilinear6(self.proj6(out_N64))  # Low resolution
        flow_32 = self.bilinear5(self.proj5(out_N32))  # Mid resolution
        flow_16 = self.bilinear4(self.proj4(out_N16))  # High resolution

        if (mode == 'combined'):

            flow_64 = torch.clamp(flow_64, -10.0, 10.0)     # Low resolution
            flow_32 = torch.clamp(flow_32, -5.0, 5.0)       # Mid resolution
            flow_16 = torch.clamp(flow_16, -3.0, 3.0)       # High resolution

            flow = flow_16 + flow_32 + flow_64
        
            I1_warped, grid = self.backwarp(I1, flow)

        if (mode == 'sequential'):
            flow_64 = torch.clamp(flow_64, -10.0, 10.0)     # Low resolution
            I1_warped_64, grid = self.backwarp(I1, flow_64)

            flow_32 = torch.clamp(flow_32, -5.0, 5.0)       # Mid resolution
            I1_warped_32, grid = self.backwarp(I1_warped_64, flow_32)

            flow_16 = torch.clamp(flow_16, -3.0, 3.0)       # High resolution
            I1_warped, grid = self.backwarp(I1_warped_32, flow_32)

            flow = flow_16 + flow_32 + flow_64
                
        loss = 1.0 - cross_correlation(I0[:, :, 15:-15, 15:-15], I1_warped[:, :, 15:-15, 15:-15])
                
        return I1_warped, flow_16, flow_32, flow_64, loss
    
if (__name__ == '__main__'):
    config = {        
        'n_pixel': 64,
        'channels': 8          
        }
    
    x = torch.zeros((10, 2, 64, 64))
            
    tmp = Model(config)

    out = tmp(x)


    