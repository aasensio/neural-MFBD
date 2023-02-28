import numpy as np
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
from skimage import measure
from skimage.morphology import flood
import scipy.ndimage as nd

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

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

    # elif isinstance(m, (nn.GRU, nn.LSTM)):
    #     breakpoint()        
    #     nn.init.xavier_uniform_(m.weight_ih.data)
    #             # elif 'weight_hh' in name:
    #             #     nn.init.orthogonal_(p.data)
    #             # elif 'bias_ih' in name:
    #             #     p.data.fill_(0)
    #             #     # Set forget-gate bias to 1
    #             #     n = p.size(0)
    #             #     p.data[(n // 4):(n // 2)].fill_(1)
    #             # elif 'bias_hh' in name:
    #             #     p.data.fill_(0)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class CleanNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_modes = config['n_modes']
                        
        self.net = []
        for i in range(self.config['gradient_steps']):
            nets = []
            if (self.config['type_cleaning'] == 'fc'):
                nets.append(nn.Linear(self.config['n_modes'], 128))
                nets.append(nn.ReLU())
                nets.append(nn.Linear(128, 128))
                nets.append(nn.ReLU())
                nets.append(nn.Linear(128, self.config['n_modes']))
            
            if (self.config['type_cleaning'] == 'conv1d'):
                nets.append(nn.Conv1d(self.config['n_modes'], 2*self.config['n_modes'], kernel_size=7, padding=3))
                nets.append(nn.ReLU())
                nets.append(nn.Conv1d(2*self.config['n_modes'], 2*self.config['n_modes'], kernel_size=5, padding=2))
                nets.append(nn.ReLU())
                nets.append(nn.Conv1d(2*self.config['n_modes'], self.config['n_modes'], kernel_size=3, padding=1))

            net = nn.Sequential(*nets)
            net.apply(weight_init)
            self.net.append(net)

        self.net = nn.ModuleList(self.net)
                            
        print('CleanNet')
        print('--------')
        npar_total = 0
        npar_total_learn = 0
        for i, network in enumerate(self.net):
            npar = sum(x.numel() for x in network.parameters())            
            npar_learn = sum(x.numel() for x in network.parameters() if x.requires_grad)
            npar_total += npar
            npar_total_learn += npar_learn
            print(f" Number of params net_{i}             : {millify(npar, precision=3)} / {millify(npar_learn, precision=3)}")

        print(f" Total                                         : {millify(npar_total_learn, precision=3)} / {millify(npar_total, precision=3)}")

        self.apply(weight_init)

        self.tanh = nn.Tanh()

    def forward(self, x, index):
        
        nb, nt, nm = x.shape

        if (self.config['type_cleaning'] == 'fc'):
            out = self.net[index](x)                        

        if (self.config['type_cleaning'] == 'conv1d'):
            
            xt = torch.transpose(x, 1, 2)
            xt = self.net[index](xt)
            out = torch.transpose(xt, 1, 2)
        
        # Subtract the mean tip-tilt
        mean_tt = torch.mean(out[:, :, 0:2], dim=1, keepdims=True)
        mean_tt = F.pad(mean_tt.expand(nb, nt, 2),(0, nm-2, 0, 0, 0, 0),value=0.0)

        out = 5.0*self.tanh(0.1*(out - mean_tt))

        # for p in self.net[index].parameters():
        #     p.requires_grad = False

        return out

class ObjectNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.channels = config['object_enc_channels']
        self.channels_out = config['object_enc_channels_out']
        self.channels_proj = config['object_channels_proj']
        self.channels_weight = config['object_channels_weight']

        wavelength = config['wavelength']
        D = config['diameter']
        pix = config['pix_size']

        # This factor 2 comes from the fact that the basis for the wavefront in the
        # file is normalized to unit amplitude for the tip-tilt, which is a factor
        # 2 smaller than the amplitude of the Zernike tip-tilt
        self.tt_coeff = 2.0 * wavelength * 1e-8 / (np.pi * pix * D) * 206265 / 2.0

        padding_mode = 'reflect'
        activation = 'lrelu'
                
        #-----------
        # Encoder
        #-----------
        self.encoder = []

        # Conv2d+ReLU
        self.encoder.append(blocks.conv_block(in_planes=2, 
                    out_planes=self.channels, 
                    kernel_size=3,
                    padding = 1,
                    padding_mode=padding_mode,
                    activation=activation))

        # 3 x ResBlock
        for i in range(3):
            self.encoder.append(blocks.ResBlock(in_planes=self.channels, 
                    out_planes=self.channels, 
                    kernel_size=3,
                    padding = 1,
                    padding_mode=padding_mode,
                    activation=activation))

        # Conv2d+ReLU
        self.encoder.append(blocks.conv_block(in_planes=self.channels, 
                    out_planes=2, 
                    kernel_size=3, 
                    padding = 1,
                    padding_mode=padding_mode,
                    activation=activation))

        self.encoder = nn.Sequential(*self.encoder)

        networks = [self.encoder]
        labels = ['Encoder']

        print('ObjectNet')
        print('--------')
        npar_total = 0
        npar_total_learn = 0
        for i, network in enumerate(networks):
            npar = sum(x.numel() for x in network.parameters())            
            npar_learn = sum(x.numel() for x in network.parameters() if x.requires_grad)
            npar_total += npar
            npar_total_learn += npar_learn
            print(f" Number of params {labels[i]:<16}             : {millify(npar, precision=3)} / {millify(npar_learn, precision=3)}")

        print(f" Total                                         : {millify(npar_total_learn, precision=3)} / {millify(npar_total, precision=3)}")


    def forward(self, x):
                
        # Shared encoder for predicting features
        out = x + self.encoder(x)
                                
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

        self.cutoff = self.config['diameter'] / (self.config['wavelength'] * 1e-8) / 206265.0

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
        
        self.register_buffer('pupil', torch.tensor(pupil.astype('float32')))
        self.register_buffer('basis', torch.tensor(basis.astype('float32')))            
        self.register_buffer('window', torch.tensor(window.astype('float32')))

        gamma_obj = np.array(self.config['gamma_obj'])
        self.register_buffer('gamma_obj', torch.tensor(gamma_obj.astype('float32')))

        self.rho = nn.Parameter(torch.ones(self.config['gradient_steps'], requires_grad=True))
        
        # We define the modal network                
        # self.modalnet = ModalNetwork(self.config)
        
        # We define the object network
        # self.denoise_object_net = ObjectNetwork(self.config)

        self.denoise_modes_net = CleanNetwork(self.config)

        self.optimizer = [None] * self.config['gradient_steps']
        for i in range(self.config['gradient_steps']):
            self.optimizer[i] = torch.optim.Adam(self.denoise_modes_net.net[i].parameters(), 
                        lr=self.config['lr'], 
                        weight_decay=self.config['wd'])


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
        wavefront = torch.einsum('ijk,klm->ijlm', modes, self.basis)

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

    def set_gamma_obj(self, gamma):
        print(f'Setting gamma = {np.array(gamma)}')
        self.gamma_obj = torch.from_numpy(np.array(gamma).astype('float32')).to(self.rho.device)

    def lofdahl_scharmer_filter(self, Sconj_S, Sconj_I, sigma):
        den = torch.conj(Sconj_I) * Sconj_I
        H = (Sconj_S / den).real        

        H = torch.fft.fftshift(H).cpu().numpy()

        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,1,3,3], mode='wrap')        
        
        filt = 1.0 - H * sigma[:, :, None, None].cpu().numpy()**2 * self.config['n_pixel']**2
        
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0

        nb, no, nx, ny = filt.shape

        mask = np.zeros_like(filt)

        for ib in range(nb):
            for io in range(no):
                
                mask[ib, io, :, :] = flood(1.0 - filt[ib, io, :, :], (nx//2, ny//2), tolerance=0.9) * self.mask_diffraction_shift
                mask[ib, io, :, :] = np.fft.fftshift(mask[ib, io, :, :])

        return torch.tensor(mask).to(Sconj_S.device)

    def compute_image(self, images_ft, psf_ft, sigma, image_filter='gaussian'):
        """Compute the reconstructed image
                        
        """

        Sconj_S = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * psf_ft[:, :, None, :, :], dim=1)
        Sconj_I = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * images_ft, dim=1)
        
        # Use Lofdahl & Scharmer (1994) filter
        if (image_filter == 'lofdahl_scharmer'):

            mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I, sigma)
                        
            out = Sconj_I / Sconj_S * mask
        
        # Use simple Wiener filter with Gaussian prior
        if (image_filter == 'gaussian'):            
            out = Sconj_I / (sigma[:, :, None, None] + Sconj_S)

        return out


    def image_from_modes(self, modes, frames_ft, frames_apod, sigma, weight, image_filter='gaussian'):
        wavefront, psf, psf_ft, dpsfft_da = self.compute_psfs(modes, derivatives=True)

        # Predict the corrected image
        reconstructed_ft = self.compute_image(frames_ft, psf_ft, sigma, image_filter=image_filter)

        degraded_ft = reconstructed_ft[:, None, :, :, :] * psf_ft[:, :, None, :, :]
        degraded = torch.fft.ifft2(degraded_ft).real
        residual = (degraded - frames_apod) * weight[:, None, :, None, None]
        loss = torch.sum(residual**2, dim=(-1,-2,-3))

        return loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da

    def update_modes(self, modes, residual, reconstructed_ft, dpsfft_da, rho):
        ddegraded_ft = dpsfft_da[:, :, None, :, :, :] * reconstructed_ft[:, None, :, None, :, :]
        ddegraded = torch.fft.ifft2(ddegraded_ft).real

        gradient = torch.sum(2.0 * residual[:, :, :, None, :, :] * ddegraded, dim=(2, -1, -2)).real
        
        modes = modes - 1.2 * rho * gradient

        return modes


    def forward(self, frames, frames_apod, sigma, weight, optimize=True, image_filter='gaussian'):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        # frames : n_batch, n_frames, n_object, nx, ny

        n_b, n_f, _, _, _ = frames.shape
                    
        modes = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=False)

        total_loss = 0.0

        frames_ft = torch.fft.fft2(frames_apod)

        # Compute image from modes
        loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, image_filter='gaussian')

        for loop in range(self.config['gradient_steps']):

            if (optimize):
                self.optimizer[loop].zero_grad()
            
            # Update modes using gradient
            modes = self.update_modes(modes, residual, reconstructed_ft, dpsfft_da, 1.0)#self.rho[loop])

            # Clean modes
            modes = self.denoise_modes_net(modes, loop)
            
            # Compute image from modes
            loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, image_filter='gaussian')
            
            loss = torch.mean(loss) / self.config['gradient_steps']

            if (optimize):
                loss.backward()

            total_loss += loss
                        
            if (optimize):
                self.optimizer[loop].step()

                modes = modes.clone().detach()
                residual = residual.clone().detach()
                reconstructed_ft = reconstructed_ft.clone().detach()
                dpsfft_da = dpsfft_da.clone().detach()            
            
        if (image_filter == 'lofdahl_scharmer'):
            loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, image_filter=image_filter)

        reconstructed = torch.fft.ifft2(reconstructed_ft).real
        mean_val = torch.mean(reconstructed, dim=(2, 3), keepdims=True)
        reconstructed_apod = reconstructed - mean_val
        reconstructed_apod *= self.window[None, None, :, :]
        reconstructed_apod += mean_val            
                                            
        return modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, total_loss

    def forward_modes(self, frames):
                
        # Estimate the modes        
        modes = self.modalnet(frames)

        return modes

    def forward_reconstructed(self, frames, frames_apod, sigma, weight, image_filter='gaussian'):
                        
        # Predict the corrected image        
        modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, loss = self.forward(frames, frames_apod, sigma, weight, optimize=False, image_filter=image_filter)

        return reconstructed, modes

    def forward_pst(self, frames, diversity):
                
        # Estimate the modes        
        modes = self.modalnet(frames)
                                            
        # Compute the PSF and their Fourier transform from the modes        
        wavefront, psf = self.compute_psfs(modes, diversity)
        
        return modes, psf, wavefront