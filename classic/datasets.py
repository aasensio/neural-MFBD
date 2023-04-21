import numpy as np
import zarr
import sys
import torch.utils.data
from noise_svd import noise_estimation
import images
import patchify

def augment(out_images):
    """
    Augment images. Assumed dimensions are [Seq, Channel, H, W]
    
    """
    # Augmentations
    rot90 = np.random.randint(4)
    out_images = np.rot90(out_images, k=rot90, axes=(2, 3))

    flipud = np.random.rand()
    if flipud > 0.5:
        out_images = np.flip(out_images, axis=2)

    fliplr = np.random.rand()
    if fliplr > 0.5:
        out_images = np.flip(out_images, axis=3)

    return out_images

class DatasetSST(torch.utils.data.Dataset):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, config):
        super(DatasetSST, self).__init__()
                
        print(f"Reading SST file : {config['training_file']}")

        # Number of pixel for apodization
        self.npix_apodization = config['npix_apodization']

        # Sizes, number of frames and number of modes in the burst are obtained from the training file
        self.n_pixel = config['n_pixel']        
        self.n_modes = config['n_modes']        
        self.bands = config['bands']
                
        # Open file and read all focused/defocused images, together with the modes
        self.file = zarr.open(config['training_file'], 'r')

        n_images, n_frames, nx, ny = self.file['wb'].shape

        self.n_frames = config['n_frames']
        if (self.n_frames > n_frames):
            print(f'The training set has {n_frames} frames and you are asking for {self.n_frames}. Cannot continue.')
            sys.exit()

        if (self.n_pixel != nx):
            print(f'The training set has images of size {nx}x{ny} and you are asking for images of size {self.n_pixel}x{self.n_pixel}. Cannot continue.')
            sys.exit()
                                                       
        # Generate Hamming window function for WFS correlation
        win = np.hanning(self.npix_apodization)
        winOut = np.ones(self.n_pixel)
        winOut[0:self.npix_apodization//2] = win[0:self.npix_apodization//2]
        winOut[-self.npix_apodization//2:] = win[-self.npix_apodization//2:]
        self.window = np.outer(winOut, winOut)

        self.n_training = n_images

    def __getitem__(self, index):
                
        # Select images from the database [nseq, nobj, nx, ny] -> nseq frames of size nxxny, with nobj objects        
        out_wb = self.file['wb'][index, :, :, :].astype('float32')
        out_nb = self.file['nb'][index, :, :, :].astype('float32')
        out_wl = self.file['wl'][index]
        sigma_wb = self.file['sigma_wb'][index]
        sigma_nb = self.file['sigma_nb'][index]
                
        out_images = np.concatenate([out_wb[:, None, :, :], out_nb[:, None, :, :]], axis=1)

        # Augmentations
        out_images = augment(out_images)
                        
        # Apodize the images
        out_images_apodized = np.copy(out_images)

        # Subtract the mean of each image, apply apodization window and add again the mean
        med = np.mean(out_images, axis=(2, 3), keepdims=True)
        out_images_apodized -= med
        out_images_apodized *= self.window[None, None, :, :]
        out_images_apodized += med

        # Normalized images
        maxval = np.max(out_images, axis=(0,2,3), keepdims=True)
        minval = np.min(out_images, axis=(0,2,3), keepdims=True)        
        out_images = (out_images - minval) / (maxval - minval)
        out_images_apodized = (out_images_apodized - minval) / (maxval - minval)

        # Noise estimation
        # Estimate noise of images        
        x = np.transpose(out_images, axes=(1,0,2,3)).reshape((2, self.n_frames, self.n_pixel*self.n_pixel))
        sigma = noise_estimation(x)
        # sigma = np.array([sigma_wb, sigma_nb]) / (maxval - minval)

        # From the noise, estimate the weight in the loss of each object
        # The weight is simply 1/stddev, which will later be squared
        weight = 1.0 / sigma
        weight /= 50.0

        # Compute contrast and change the weight for the objects with low (or very high) contrast
        contrast = 100.0 * np.std(out_wb) / np.mean(out_wb)
        if (contrast > 25.0):
            weight *= 0.0
        if (contrast < 3.0):
            weight *= 0.05
       
        # Outputs are focused+defocused image, modes and diversity    
        return out_images.astype('float32'), out_images_apodized.astype('float32'), out_wl.astype('float32'), sigma.astype('float32'), weight.astype('float32')
        
    def __len__(self):
        return self.n_training


class DatasetHiFi(torch.utils.data.Dataset):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, config):
        super(DatasetHiFi, self).__init__()
        
        
        print(f"Reading {config['training_file']}")

        # Number of pixel for apodization
        self.npix_apodization = config['npix_apodization']
        self.n_frames = config['n_frames']      

        # Sizes, number of frames and number of modes in the burst are obtained from the training file
        self.n_pixel = config['n_pixel']
        self.n_modes = config['n_modes']        
        self.border_pixel = config['border_pixel']
        
        # Open file and read all focused/defocused images, together with the modes
        self.file = zarr.open(config['training_file'], 'r')

        self.groups = [i[0] for i in self.file.groups()]
        self.obs = []
        self.im_sizes = []
        self.indf = []
        self.indx = []
        self.indy = []
        self.indt = []
        n_images = 0

        # Generate the patches for training
        for g in self.groups:
            tmp = [i[0] for i in self.file[g].groups()]
            self.obs.append(tmp)            
            dimt, dimx, dimy = self.file[g][tmp[0]]['bb'].shape
            self.im_sizes.append(self.file[g][tmp[0]]['bb'].shape)
            
            for j in tmp:
                self.indf.extend([f'{g}/{j}' for i in range(config['n_patches_per_image'])])
                self.indx.extend(np.random.randint(low=self.border_pixel, high=dimx-self.n_pixel-self.border_pixel, size=config['n_patches_per_image']))
                self.indy.extend(np.random.randint(low=self.border_pixel, high=dimy-self.n_pixel-self.border_pixel, size=config['n_patches_per_image']))
                self.indt.extend(np.random.randint(low=0, high=dimt-self.n_frames, size=config['n_patches_per_image']))

            n_images += len(tmp)            
                                               
        # Generate Hamming window function for WFS correlation
        win = np.hanning(self.npix_apodization)
        winOut = np.ones(self.n_pixel)
        winOut[0:self.npix_apodization//2] = win[0:self.npix_apodization//2]
        winOut[-self.npix_apodization//2:] = win[-self.npix_apodization//2:]
        self.window = np.outer(winOut, winOut)

        # Random indices for the extraction of the images
        self.n_images = n_images
        self.n_training = len(self.indf)

    def __getitem__(self, index):
                
        # Select images from the database [nseq, nobj, nx, ny] -> nseq frames of size nxxny, with nobj objects
        indf = self.indf[index]
        indx = self.indx[index]
        indy = self.indy[index]
        indt = self.indt[index]

        out_bb = self.file[indf]['bb'][indt:indt+self.n_frames, indx:indx+self.n_pixel, indy:indy+self.n_pixel].astype('float32')
        out_nb = self.file[indf]['nb'][indt:indt+self.n_frames, indx:indx+self.n_pixel, indy:indy+self.n_pixel].astype('float32')
        
        out_images = np.concatenate([out_bb[:, None, :, :], out_nb[:, None, :, :]], axis=1)

        # Augmentations
        out_images = augment(out_images)
                               
        # Apodize the images
        out_images_apodized = np.copy(out_images)

        out_wl = np.array([6563.0])

        # Subtract the mean of each image, apply apodization window and add again the mean
        med = np.mean(out_images, axis=(2, 3), keepdims=True)
        out_images_apodized -= med
        out_images_apodized *= self.window[None, None, :, :]
        out_images_apodized += med

        # Normalized images
        maxval = np.max(out_images, axis=(0,2,3), keepdims=True)
        minval = np.min(out_images, axis=(0,2,3), keepdims=True)        
        out_images = (out_images - minval) / (maxval - minval)
        out_images_apodized = (out_images_apodized - minval) / (maxval - minval)

        # Estimate noise of images        
        x = np.transpose(out_images, axes=(1,0,2,3)).reshape((2, self.n_frames, self.n_pixel*self.n_pixel))
        sigma = noise_estimation(x)

        # From the noise, estimate the weight in the loss of each object normalized to the first one
        weight = 1.0 / sigma
        weight /= 50.0  
       
        # Outputs are focused+defocused image, modes and diversity    
        return out_images.astype('float32'), out_images_apodized.astype('float32'), out_wl.astype('float32'), sigma.astype('float32'), weight.astype('float32')
        
    def __len__(self):
        return self.n_training

class Dataset_validation(torch.utils.data.Dataset):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, config):
        super(Dataset_validation, self).__init__()
        
        # Number of pixel for apodization
        self.npix_apodization = config['npix_apodization']
        
        # Sizes, number of frames and number of modes in the burst are obtained from the training file
        self.n_pixel = config['n_pixel']
        self.n_modes = config['n_modes']            

    def patchify(self, step=40):
        self.n_frames, self.nx, self.ny = self.wb.shape

        self.wb_patch = patchify.patchify(self.wb, (self.n_frames, self.n_pixel, self.n_pixel), step=step)
        self.nb_patch = patchify.patchify(self.nb, (self.n_frames, self.n_pixel, self.n_pixel), step=step)

        x = np.arange(self.wb.shape[1])
        y = np.arange(self.wb.shape[2])
        X, Y = np.meshgrid(y, x)
        XY = np.concatenate([X[None, :, :], Y[None, :, :]], axis=0)

        self.XY_patch = patchify.patchify(XY, (2, self.n_pixel, self.n_pixel), step=step)
    
        nt, nx, ny, dimt, dimx, dimy = self.wb_patch.shape

        self.wb_patch = self.wb_patch.reshape((nt*nx*ny, dimt, dimx, dimy))
        self.nb_patch = self.nb_patch.reshape((nt*nx*ny, dimt, dimx, dimy))
        self.XY_patch = self.XY_patch.reshape((1*nx*ny, 2, dimx, dimy))
                                               
        # Generate Hamming window function for WFS correlation
        win = np.hanning(self.npix_apodization)
        winOut = np.ones(self.n_pixel)
        winOut[0:self.npix_apodization//2] = win[0:self.npix_apodization//2]
        winOut[-self.npix_apodization//2:] = win[-self.npix_apodization//2:]
        self.window = np.outer(winOut, winOut)

        # Random indices for the extraction of the images        
        self.n_training = self.wb_patch.shape[0]
                
    def __getitem__(self, index):
        
        # index += 16
        # Select images from the database [64, 2, 96, 96] -> 64 frames of size 96x96, with 2 channels
        wb = self.wb_patch[index, ...].astype('float32')
        nb = self.nb_patch[index, ...].astype('float32')
        out_wl = float(self.wl)
        xy = self.XY_patch[index, ...]
        
        out_images = np.concatenate([wb[:, None, :, :], nb[:, None, :, :]], axis=1)

        # Apodize the images
        out_images_apodized = np.copy(out_images)

        # Subtract the mean of each image, apply apodization window and add again the mean
        med = np.mean(out_images, axis=(2, 3), keepdims=True)
        out_images_apodized -= med
        out_images_apodized *= self.window[None, None, :, :]
        out_images_apodized += med

        # Normalized images
        maxval = np.max(out_images, axis=(0,2,3), keepdims=True)
        minval = np.min(out_images, axis=(0,2,3), keepdims=True)        
        out_images = (out_images - minval) / (maxval - minval)
        out_images_apodized = (out_images_apodized - minval) / (maxval - minval)

        # Noise estimation
        # Estimate noise of images        
        x = np.transpose(out_images, axes=(1,0,2,3)).reshape((2, self.n_frames, self.n_pixel*self.n_pixel))
        sigma = noise_estimation(x)
        # sigma = np.array([sigma_wb, sigma_nb]) / (maxval - minval)

        # From the noise, estimate the weight in the loss of each object normalized to the first one
        weight = 1.0 / sigma
        weight /= 50.0
        
        # Outputs are focused+defocused image, modes and diversity    
        return out_images.astype('float32'), out_images_apodized.astype('float32'), out_wl, xy[:, 0, 0].astype('float32'), minval, maxval, sigma.astype('float32'), weight.astype('float32')
        
    def __len__(self):
        return self.n_training

class DatasetSST_validation(Dataset_validation):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, config, iregion, index, wav=0, mod=0, cam=0, step=40):
        super(DatasetSST_validation, self).__init__(config)
                
        self.wb, self.nb, self.wl = images.read_images_sst(iregion, index, wav, mod, cam)
        self.patchify(step=step)

class DatasetHiFi_validation(Dataset_validation):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, config, root, nac=12, step=40):
        super(DatasetHiFi_validation, self).__init__(config)
                
        self.wb, self.nb, self.wl = images.read_images_hifi(root, nac)
        self.patchify(step=step)