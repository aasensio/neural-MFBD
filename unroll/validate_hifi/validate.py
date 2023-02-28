import numpy as np
import torch
import torch.nn as nn
import h5py
from tqdm import tqdm
import model_nopd_wiener as model
import nvidia_smi
import matplotlib.pyplot as pl
import patchify
from noise_svd import noise_estimation
from astropy.io import fits
from images import read_images


def align(a, b):        

    if(a.shape[0] != b.shape[0] or a.shape[1] != b.shape[1]):
        print("align: ERROR, both images must have the same size")
        return(0.0,0.0)
    
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)

    corr = np.fft.ifft2(np.conj(fa) * fb)
    cc = np.fft.fftshift(corr.real)
        
    mm = np.argmax(cc)
    xy = ( mm // fa.shape[1], mm % fa.shape[1])

    cc = cc[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]

    y = 2.0*cc[1,1]
    x = (cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-y)*0.5
    y = (cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-y)*0.5

    x += xy[1] - fa.shape[1]//2
    y += xy[0] - fa.shape[0]//2

    return (-y,-x)
    

class Dataset(torch.utils.data.Dataset):
    """
    Dataset

      Scripts to produce the training sets : db.py
    
    """
    def __init__(self, config, root, nac):
        super(Dataset, self).__init__()
        
        # Number of pixel for apodization
        self.npix_apodization = config['npix_apodization']
        
        # Sizes, number of frames and number of modes in the burst are obtained from the training file
        self.n_pixel = config['n_pixel']
        self.n_modes = config['n_modes']
        
        self.wb, self.nb = read_images(root, nac)

        self.n_frames, self.nx, self.ny = self.wb.shape

        self.wb_patch = patchify.patchify(self.wb, (self.n_frames, self.n_pixel, self.n_pixel), step=40)
        self.nb_patch = patchify.patchify(self.nb, (self.n_frames, self.n_pixel, self.n_pixel), step=40)

        x = np.arange(self.wb.shape[1])
        y = np.arange(self.wb.shape[2])
        X, Y = np.meshgrid(y, x)
        XY = np.concatenate([X[None, :, :], Y[None, :, :]], axis=0)

        self.XY_patch = patchify.patchify(XY, (2, self.n_pixel, self.n_pixel), step=40)
    
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
        weight = 1.0 / sigma**2
        weight /= weight[0]
        
        # Outputs are focused+defocused image, modes and diversity    
        return out_images.astype('float32'), out_images_apodized.astype('float32'), xy[:, 0, 0].astype('float32'), minval, maxval, sigma.astype('float32'), weight.astype('float32')
        
    def __len__(self):
        return self.n_training
        
class FastMFBD(object):
    def __init__(self, checkpoint, n_frames, gpu=2):
        """
        Train a deep neural network for self-supervised learning of multiframe deconvolution
                
        """        

        self.batch_size = 6

        self.checkpoint = checkpoint
        print("=> loading '{}'".format(self.checkpoint))
        chk = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        print("=> done")

        self.config = chk['hyperparameters']

        self.config['n_frames'] = n_frames
                        
        # Is CUDA available?
        self.cuda = torch.cuda.is_available()
                        
        print("Using GPU : {0}".format(gpu))
        self.device = torch.device(f"cuda:{gpu}" if self.cuda else "cpu")      
        torch.cuda.set_device(gpu)

        print(f"Device : {self.device}")      
        
        # Ger handlers to later check memory and usage of GPUs
        nvidia_smi.nvmlInit()
        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu)
        print("Computing in {1} - cuda:{0}".format(gpu, nvidia_smi.nvmlDeviceGetName(self.handle)))
                
        # Define the neural network model
        print("Defining the model...")
        netmodel = model.Model(self.config)
                    
        # Move model to GPU/CPU
        self.model = netmodel.to(self.device)        

        self.model.load_state_dict(chk['state_dict'])        

        print(f'rho={self.model.rho}')

        for param in self.model.parameters():
            param.requires_grad = False
                
    def validate_fov(self, root, outf):
        """
        Validate for one epoch
        """

        # Read training and validation sets
        self.dataset = Dataset(self.config, root, self.config['n_frames'])
        
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
                
        # Data loaders that will inject data during training
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, 
                    batch_size=self.batch_size,
                    shuffle=False, 
                    drop_last=True,
                    **kwargs)        

        # Put the model in evaluation mode
        self.model.eval()

        filter = 'lofdahl_scharmer'
        # filter = 'gaussian'

        t = tqdm(self.validation_loader)

        nx, ny = self.dataset.nx, self.dataset.ny
        out_nn = np.zeros((2, nx, ny))
        orig_nn = np.zeros((2, nx, ny))
        overlap = np.zeros((nx, ny))
        
        # Patches of 64 pixels, with a step of 40 pixels
        n = (64 - 40) // 2
        
        # filename = self.checkpoint.split('/')[-1]
        fout = h5py.File(outf, 'w')
        dset_modes_nn = fout.create_dataset('modes_nn', (self.dataset.n_training, self.dataset.n_frames, 44), dtype='float32')        
        dset_images_nn = fout.create_dataset('reconstruction_nn', (2, nx, ny), dtype='float32')
        dset_orig_nn = fout.create_dataset('frame0', (2, nx, ny), dtype='float32')
        dset_minval_nn = fout.create_dataset('minval', (self.dataset.n_training, 2), dtype='float32')
        dset_maxval_nn = fout.create_dataset('maxval', (self.dataset.n_training, 2), dtype='float32')
        dset_xy_nn = fout.create_dataset('xy', (self.dataset.n_training, 2), dtype='float32')
        
        dset_modes_nn_mem = np.zeros((self.dataset.n_training, self.dataset.n_frames, 44), dtype='float32')
        dset_images_nn_mem = np.zeros((self.dataset.n_training, 2, 64, 64), dtype='float32')
        dset_orig_nn_mem = np.zeros((self.dataset.n_training, 2, 64, 64), dtype='float32')
        dset_xy_nn_mem = np.zeros((self.dataset.n_training, 2), dtype='float32')
        dset_minval_nn_mem = np.zeros((self.dataset.n_training, 2), dtype='float32')
        dset_maxval_nn_mem = np.zeros((self.dataset.n_training, 2), dtype='float32')
                            
        for batch_idx, (frames, frames_apod, xy, minval, maxval, sigma, weight) in enumerate(t):
                            
            # Move all data to GPU/CPU
            frames = frames.to(self.device)
            frames_apod = frames_apod.to(self.device)
            sigma = sigma.to(self.device)
            weight = weight.to(self.device)

                                                
            with torch.no_grad():
                reconstructed, modes = self.model.forward_reconstructed(frames, frames_apod, sigma, weight, image_filter=filter)

            left = batch_idx*self.batch_size
            right = (batch_idx+1)*self.batch_size
            dset_images_nn_mem[left:right, :, :] = reconstructed[:, :, :, :].cpu().numpy()
            dset_orig_nn_mem[left:right, :, :] = frames[:, 0, :, :, :].cpu().numpy()
            dset_modes_nn_mem[left:right, :, :] = modes[:, :, :].cpu().numpy()
            dset_xy_nn_mem[left:right, :] = xy.cpu().numpy()            
            dset_minval_nn_mem[left:right, :] = np.squeeze(minval)
            dset_maxval_nn_mem[left:right, :] = np.squeeze(maxval)

        
        print("Stitching images...")
        for i in tqdm(range(self.dataset.n_training)):
            indy, indx = dset_xy_nn_mem[i, :].astype('int')    
            minv = dset_minval_nn_mem[i, :][:, None, None]
            maxv = dset_maxval_nn_mem[i, :][:, None, None]
                
            out_nn[:, indx+n:indx+64-n, indy+n:indy+64-n] += dset_images_nn_mem[i, :, n:-n, n:-n] * (maxv - minv) + minv
            orig_nn[:, indx+n:indx+64-n, indy+n:indy+64-n] += dset_orig_nn_mem[i, :, n:-n, n:-n] * (maxv - minv) + minv
            overlap[indx+n:indx+64-n, indy+n:indy+64-n] += 1

        dset_images_nn_mem = out_nn / overlap
        dset_orig_nn_mem = orig_nn / overlap
            
        print("Saving file...")
        dset_images_nn[:] = dset_images_nn_mem
        dset_orig_nn[:] = dset_orig_nn_mem
        dset_xy_nn[:] = dset_xy_nn_mem
        dset_modes_nn[:] = dset_modes_nn_mem
        dset_minval_nn[:] = dset_minval_nn_mem
        dset_maxval_nn[:] = dset_maxval_nn_mem

        fout.close()            

if (__name__ == '__main__'):

    checkpoint = 'weights/2022-10-26-12:44_ep_30.pth'

    root = '/net/delfin/scratch/ckuckein/gregor/hifi2/momfbd/20221128/'

    deep_mfbd_network = FastMFBD(checkpoint=checkpoint, n_frames=50, gpu=0)

    for i in range(0, 257):
        deep_mfbd_network.validate_fov(root=f'{root}/scan_b{i:03d}', outf=f'reconstructed/scan_b{i:03d}.h5')
