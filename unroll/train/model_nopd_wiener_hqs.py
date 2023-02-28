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
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='leaky_relu')
        if (mode == 'xavier'):
            nn.init.xavier_normal_(m.weight.data, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
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


class CleanNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.n_modes = config['n_modes']
        self.n_channels = config['n_internal_channels']
                        
        self.net = []
        self.rho = []
        for i in range(self.config['gradient_steps']):
            nets = []
            if (self.config['type_cleaning'] == 'fc'):
                nets.append(nn.Linear(self.config['n_modes'], 128))
                nets.append(nn.ReLU())
                nets.append(nn.Linear(128, 128))
                nets.append(nn.ReLU())
                nets.append(nn.Linear(128, self.config['n_modes']))
            
            if (self.config['type_cleaning'] == 'conv1d'):
                nets.append(nn.Conv1d(2*self.config['n_modes'], self.n_channels, kernel_size=3, padding=1))
                nets.append(nn.GELU())
                nets.append(nn.Conv1d(self.n_channels, self.n_channels, kernel_size=3, padding=1))
                nets.append(nn.GELU())
                nets.append(nn.Conv1d(self.n_channels, self.n_channels, kernel_size=3, padding=1))
                nets.append(nn.GELU())
                nets.append(nn.Conv1d(self.n_channels, self.config['n_modes'], kernel_size=3, padding=1))

            net = nn.Sequential(*nets)
            net.apply(weight_init)
            self.net.append(net)

            self.rho.append(nn.Parameter(torch.tensor(1.0)))

        self.net = nn.ModuleList(self.net)
        self.rho = nn.ParameterList(self.rho)
                            
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

    def forward(self, x, v, index):
        
        nb, nt, nm = x.shape

        if (self.config['type_cleaning'] == 'fc'):
            out = x + self.net[index](x)                        

        if (self.config['type_cleaning'] == 'conv1d'):
            xt = torch.transpose(x, 1, 2)
            vt = torch.transpose(v, 1, 2)
            xt = torch.cat([xt, vt], dim=1)
            xt = self.net[index](xt)
            out = v + torch.transpose(xt, 1, 2)
        
        # Subtract the mean tip-tilt
        mean_tt = torch.mean(out[:, :, 0:2], dim=1, keepdims=True)
        mean_tt = F.pad(mean_tt.expand(nb, nt, 2),(0, nm-2, 0, 0, 0, 0), value=0.0)
        
        out = out - mean_tt

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
        self.mu = 2.0
        
        # Generate Hamming window function for WFS correlation
        self.npix_apod = self.config['npix_apodization']
        win = np.hanning(self.npix_apod)
        winOut = np.ones(self.config['n_pixel'])
        winOut[0:self.npix_apod//2] = win[0:self.npix_apod//2]
        winOut[-self.npix_apod//2:] = win[-self.npix_apod//2:]
        window = np.outer(winOut, winOut)

        n_wavelengths = len(self.config['wavelengths'])
        print(f"Defining pupils for wavelengths : {self.config['wavelengths']}")

        self.mask_diffraction_shift = {}
        
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
                    
            # Define modes
            # Zernike
            if (self.config['basis_for_wavefront'].lower() == 'zernike'):
                print("Computing Zernike modes...")
                Z_machine = zern.ZernikeNaive(mask=[])
                x = np.linspace(-1, 1, self.config['n_pixel'])
                xx, yy = np.meshgrid(x, x)
                rho = overfill * np.sqrt(xx ** 2 + yy ** 2)
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
                print(f"Computing KL modes for {wl}")
                kl = kl_modes.KL()
                basis = kl.precalculate(npix_image = self.config['n_pixel'], 
                                    n_modes_max = self.config['n_modes'], 
                                    first_noll = 2, 
                                    overfill=overfill)
                basis /= np.max(np.abs(basis), axis=(1, 2), keepdims=True)
            
            self.register_buffer(f'pupil_{wl}', torch.tensor(pupil.astype('float32')))
            self.register_buffer(f'basis_{wl}', torch.tensor(basis.astype('float32')))

            # Define the diffraction mask to tamper high frequencies for each wavelength
            cutoff = self.config['diameter'][loop] / (self.config['wavelengths'][loop] * 1e-8) / 206265.0
            freq = np.fft.fftfreq(self.config['n_pixel'], d=self.config['pix_size'][loop]) / cutoff
            
            xx, yy = np.meshgrid(freq, freq)
            rho = np.sqrt(xx ** 2 + yy ** 2)
            mask_diff = rho <= 0.80            
            mask_diffraction_shift = np.fft.fftshift(mask_diff)

            self.register_buffer(f'mask_{wl}', torch.tensor(mask_diffraction_shift.astype('float32')))

        self.register_buffer('window', torch.tensor(window.astype('float32')))
        
        # Define the cleaning networks
        self.denoise_modes_net = CleanNetwork(self.config)

        # Define the optimizers for each network
        self.optimizer = [None] * self.config['gradient_steps']
        for i in range(self.config['gradient_steps']):   
            # Add the learning rate to the set of weights of the network
            pars = list(self.denoise_modes_net.net[i].parameters()) + list([self.denoise_modes_net.rho[i]])
            self.optimizer[i] = torch.optim.Adam(pars, 
                        lr=self.config['lr'], 
                        weight_decay=self.config['wd'])

        # # Define the scaler in case of half precision
        # if (self.config['precision'] == 'half'):
        #     print("Using half precision training...")
        #     self.scaler = torch.cuda.amp.GradScaler(enabled=True)        
        
    def compute_psfs(self, modes, pupil, basis, derivatives=False):
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
        wavefront = torch.einsum('ijk,iklm->ijlm', modes, basis)

        # Compute the complex phase
        phase = pupil[:, None, :, :] * torch.exp(1j * wavefront)

        # Compute FFT of the pupil function and compute autocorrelation
        ft = torch.fft.fft2(phase)
        psf = (torch.conj(ft) * ft).real
        
        # Normalize PSF
        psf_norm = psf / torch.sum(psf, [2, 3])[:, :, None, None]

        # FFT of the PSF
        psf_ft = torch.fft.fft2(psf_norm)

        if (derivatives):        

            # Now the derivative of the PSF wrt to the modes
            dW_da = 1j * phase[:, :, None, :, :] * basis[:, None, :, :, :]
            dft_da = torch.fft.fft2(dW_da)
            dp_da = torch.conj(dft_da) * ft[:, :, None, :, :] + torch.conj(ft[:, :, None, :, :]) * dft_da        
            sump = torch.sum(psf, dim=(-1,-2))
            
            dp_da = (dp_da * sump[:, :, None, None, None] - psf[:, :, None, :, :] * torch.sum(dp_da, dim=(-1,-2))[:, :, :, None, None]) / sump[:, :, None, None, None]**2

            dpsfft_da = torch.fft.fft2(dp_da)
            
            return wavefront, psf_norm, psf_ft, dpsfft_da
        else:
            return wavefront, psf_norm, psf_ft


    def lofdahl_scharmer_filter(self, Sconj_S, Sconj_I, sigma, mask):
        den = torch.conj(Sconj_I) * Sconj_I
        H = (Sconj_S / den).real        

        H = torch.fft.fftshift(H).cpu().numpy()

        # noise = 1.35 / np.median(H[:, :, 0:10, 0:10], axis=(2,3))

        H = nd.median_filter(H, [1,1,3,3], mode='wrap')        
        
        filt = 1.0 - H * sigma[:, :, None, None].cpu().numpy()**2 * self.config['n_pixel']**2
        
        filt[filt < 0.2] = 0.0
        filt[filt > 1.0] = 1.0

        nb, no, nx, ny = filt.shape

        mask_out = np.zeros_like(filt)

        for ib in range(nb):
            for io in range(no):
                
                mask_out[ib, io, :, :] = flood(1.0 - filt[ib, io, :, :], (nx//2, ny//2), tolerance=0.9) * mask[ib, :, :].cpu().numpy()
                mask_out[ib, io, :, :] = np.fft.fftshift(mask_out[ib, io, :, :])

        return torch.tensor(mask_out).to(Sconj_S.device)

    def compute_image(self, images_ft, psf_ft, sigma, mask, image_filter='gaussian'):
        """Compute the reconstructed image
                        
        """

        Sconj_S = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * psf_ft[:, :, None, :, :], dim=1)
        Sconj_I = torch.sum(torch.conj(psf_ft[:, :, None, :, :]) * images_ft, dim=1)
        
        # Use Lofdahl & Scharmer (1994) filter
        if (image_filter == 'lofdahl_scharmer'):

            mask = self.lofdahl_scharmer_filter(Sconj_S, Sconj_I, sigma, mask)
                        
            out = Sconj_I / Sconj_S * mask
        
        # Use simple Wiener filter with Gaussian prior
        if (image_filter == 'gaussian'):
            out = Sconj_I / (sigma[:, :, None, None] + Sconj_S) * torch.fft.fftshift(mask[:, None, :, :])

        return out


    def image_from_modes(self, modes, frames_ft, frames_apod, sigma, weight, pupil, basis, mask, image_filter='gaussian'):

        # Compute PSFs from modes
        wavefront, psf, psf_ft, dpsfft_da = self.compute_psfs(modes, pupil, basis, derivatives=True)

        # Predict the corrected image
        reconstructed_ft = self.compute_image(frames_ft, psf_ft, sigma, mask, image_filter=image_filter)

        degraded_ft = reconstructed_ft[:, None, :, :, :] * psf_ft[:, :, None, :, :]
        degraded = torch.fft.ifft2(degraded_ft).real
        residual = (degraded - frames_apod) * weight[:, None, :, None, None]
        loss = torch.sum(residual**2, dim=(-1,-2,-3))

        return loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da

    def update_modes(self, modes, v, residual, reconstructed_ft, weight, dpsfft_da, rho):
        ddegraded_ft = dpsfft_da[:, :, None, :, :, :] * reconstructed_ft[:, None, :, None, :, :]
        ddegraded = torch.fft.ifft2(ddegraded_ft).real

        residual_weight = residual * weight[:, None, :, None, None]
        
        gradient = torch.sum(2.0 * residual_weight[:, :, :, None, :, :] * ddegraded, dim=(2, -1, -2)).real
        
        modes = modes - rho * gradient - rho * self.mu * (modes - v)

        return modes, gradient


    def forward(self, frames, frames_apod, wl, sigma, weight, optimize=True, image_filter='gaussian'):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        # frames : n_batch, n_frames, n_object, nx, ny
        n_b, n_f, _, _, _ = frames.shape
                    
        modes = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=False)
        v = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=False)

        # Get correct pupil and basis for each element of the batch
        # They have different wavelengths
        pupil = []
        basis = []
        mask = []
        for i in range(n_b):
            if (int(wl[i]) == 3934):
                pupil.append(self.pupil_3934[None, :, :])
                basis.append(self.basis_3934[None, :, :, :])
                mask.append(self.mask_3934[None, :, :])
            if (int(wl[i]) == 6173):
                pupil.append(self.pupil_6173[None, :, :])
                basis.append(self.basis_6173[None, :, :, :])
                mask.append(self.mask_6173[None, :, :])
            if (int(wl[i]) == 6563):
                pupil.append(self.pupil_6563[None, :, :])
                basis.append(self.basis_6563[None, :, :, :])
                mask.append(self.mask_6563[None, :, :])
            if (int(wl[i]) == 8542):
                pupil.append(self.pupil_8542[None, :, :])
                basis.append(self.basis_8542[None, :, :, :])
                mask.append(self.mask_8542[None, :, :])

        pupil = torch.cat(pupil, dim=0)
        basis = torch.cat(basis, dim=0)
        mask = torch.cat(mask, dim=0)

        total_loss = 0.0
        self.mu = 2.0

        frames_ft = torch.fft.fft2(frames_apod)

        # Compute image from modes
        loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, pupil, basis, mask, image_filter='gaussian')

        for loop in range(self.config['gradient_steps']):

            if (optimize):
                self.optimizer[loop].zero_grad()

            print(loop, loss, total_loss) 
            
            # Update modes using gradient descent
            modes, gradient = self.update_modes(modes, v, residual, reconstructed_ft, weight, dpsfft_da, 0.1 * self.denoise_modes_net.rho[loop])
            
            # Clean modes
            v = self.denoise_modes_net(modes, v, loop)
            
            # Compute image from modes
            loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, pupil, basis, mask, image_filter='gaussian')
            
            loss = torch.mean(loss) / self.config['gradient_steps']# + 0.1 * torch.mean(torch.abs(v))
            
            if (optimize):
                loss.backward()

            total_loss += loss  

            print(loop, loss, total_loss) 

            self.mu *= 1.2
                        
            if (optimize):
                self.optimizer[loop].step()

                modes = modes.clone().detach()
                v = v.clone().detach()
                residual = residual.clone().detach()
                reconstructed_ft = reconstructed_ft.clone().detach()
                dpsfft_da = dpsfft_da.clone().detach()            
            
        if (image_filter == 'lofdahl_scharmer'):
            loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, pupil, basis, mask, image_filter=image_filter)

        reconstructed = torch.fft.ifft2(reconstructed_ft).real
        mean_val = torch.mean(reconstructed, dim=(2, 3), keepdims=True)
        reconstructed_apod = reconstructed - mean_val
        reconstructed_apod *= self.window[None, None, :, :]
        reconstructed_apod += mean_val            
                                            
        return modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, total_loss

    def classic(self, frames, frames_apod, wl, sigma, weight, optimize=True, image_filter='gaussian'):
                
        # Estimate the modes                
        # modes = self.modalnet(frames)

        # frames : n_batch, n_frames, n_object, nx, ny
        n_b, n_f, _, _, _ = frames.shape
                    
        modes = torch.zeros((n_b, self.config['n_frames'], self.config['n_modes']), device=frames.device, requires_grad=False)
        
        # Get correct pupil and basis for each element of the batch
        # They have different wavelengths
        pupil = []
        basis = []
        mask = []
        for i in range(n_b):
            if (int(wl[i]) == 3934):
                pupil.append(self.pupil_3934[None, :, :])
                basis.append(self.basis_3934[None, :, :, :])
                mask.append(self.mask_3934[None, :, :])
            if (int(wl[i]) == 6173):
                pupil.append(self.pupil_6173[None, :, :])
                basis.append(self.basis_6173[None, :, :, :])
                mask.append(self.mask_6173[None, :, :])
            if (int(wl[i]) == 8542):
                pupil.append(self.pupil_8542[None, :, :])
                basis.append(self.basis_8542[None, :, :, :])
                mask.append(self.mask_8542[None, :, :])

        pupil = torch.cat(pupil, dim=0)
        basis = torch.cat(basis, dim=0)
        mask = torch.cat(mask, dim=0)

        total_loss = 0.0

        frames_ft = torch.fft.fft2(frames_apod)
        
        # Compute image from modes
        loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, pupil, basis, mask, image_filter='gaussian')
        
        for loop in range(self.config['gradient_steps']):

            # Update modes using gradient descent
            modes, gradient = self.update_modes(modes, residual, reconstructed_ft, weight, dpsfft_da, 0.05)

            # Average zero tip-tilt
            modes[:, :, 0:2] = modes[:, :, 0:2] - torch.mean(modes[:, :, 0:2], dim=1, keepdims=True)
                        
            # Compute image from modes
            loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, pupil, basis, mask, image_filter='gaussian')
            
            loss = torch.mean(loss)
                        
        if (image_filter == 'lofdahl_scharmer'):
            loss, wavefront, psf, degraded, reconstructed_ft, residual, dpsfft_da = self.image_from_modes(modes, frames_ft, frames_apod, sigma, weight, pupil, basis, mask, image_filter=image_filter)

        reconstructed = torch.fft.ifft2(reconstructed_ft).real
        mean_val = torch.mean(reconstructed, dim=(2, 3), keepdims=True)
        reconstructed_apod = reconstructed - mean_val
        reconstructed_apod *= self.window[None, None, :, :]
        reconstructed_apod += mean_val            
                                            
        return modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, loss

    def forward_modes(self, frames):
                
        # Estimate the modes        
        modes = self.modalnet(frames)

        return modes

    def forward_reconstructed(self, frames, frames_apod, wl, sigma, weight, image_filter='gaussian'):
                        
        # Predict the corrected image        
        modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, loss = self.forward(frames, frames_apod, wl, sigma, weight, optimize=False, image_filter=image_filter)

        return reconstructed, modes

    def classic_reconstructed(self, frames, frames_apod, wl, sigma, weight, image_filter='gaussian'):
                        
        # Predict the corrected image        
        modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, loss = self.classic(frames, frames_apod, wl, sigma, weight, optimize=False, image_filter=image_filter)

        return reconstructed, modes

    def forward_psf(self, frames, diversity):
                
        # Estimate the modes        
        modes = self.modalnet(frames)
                                            
        # Compute the PSF and their Fourier transform from the modes        
        wavefront, psf = self.compute_psfs(modes, diversity)
        
        return modes, psf, wavefront