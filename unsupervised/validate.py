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
import datasets


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
    

        
class FastMFBD(object):
    def __init__(self, checkpoint, gpu=2, batch_size=12):
        """
        Train a deep neural network for self-supervised learning of multiframe deconvolution
                
        """        

        self.batch_size = batch_size

        self.checkpoint = checkpoint
        print("=> loading '{}'".format(self.checkpoint))
        chk = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        print("=> done")

        self.config = chk['hyperparameters']
                        
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

        print("Setting weights of the model")
        self.model.load_state_dict(chk['state_dict'])        

        for param in self.model.parameters():
            param.requires_grad = False

    def set_observations_SST(self, iregion, index, wav=0, mod=0, cam=0):
        # Read training and validation sets
        self.dataset = datasets.DatasetSST_validation(self.config, iregion, index, wav=wav, mod=mod, cam=cam)

    def set_observations_HiFi(self, root, nac=12):
        # Read training and validation sets
        self.dataset = datasets.DatasetHiFi_validation(self.config, root, nac=nac)
                
    def validate_fov(self, outf):
        """
        Validate for one epoch
        """        
        
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
                            
        for batch_idx, (frames, frames_apod, wl, xy, minval, maxval, sigma, weight) in enumerate(t):
                            
            # Move all data to GPU/CPU
            frames = frames.to(self.device)
            frames_apod = frames_apod.to(self.device)
            sigma = sigma.to(self.device)
            weight = weight.to(self.device)
            wl = wl.to(self.device)
                                                
            with torch.no_grad():
                reconstructed, modes = self.model.forward_reconstructed(frames, frames_apod, wl, sigma, weight, image_filter=filter)

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

    instrument = 'SST'    

    # HiFi
    if (instrument == 'HiFi'):
    
    
        checkpoint = 'weights/2023-02-06-12:46.HiFi.ep_30.pth'
        deep_mfbd_network = FastMFBD(checkpoint=checkpoint, gpu=3, batch_size=24)

        # Hifi
        root = '/net/delfin/scratch/ckuckein/gregor/hifi2/momfbd/20221128/scan_b000'
        deep_mfbd_network.set_observations_HiFi(root, nac=12)
        deep_mfbd_network.validate_fov(f'reconstructed/test.HiFi.h5')

    
    # SST
    if (instrument == 'SST'):
        checkpoint = 'weights/2023-02-27-20:34.SST.ep_30.pth'
        # checkpoint = 'weights/2023-02-05-19:11.SST.ep_30.pth'
        # checkpoint = 'weights/2023-02-07-21:27.SST.ep_30.pth'

        deep_mfbd_network = FastMFBD(checkpoint=checkpoint, gpu=2, batch_size=24)

        time = checkpoint.split('/')[-1].split('.')[0]

        for i in range(70, 130):
            deep_mfbd_network.set_observations_SST(iregion=8, index=i, wav=13)
            deep_mfbd_network.validate_fov(f'reconstructed/val_spot_20200727_083509.3934.{i:02d}.{time}.SST.h5')
        
        for i in range(0, 50):
            deep_mfbd_network.set_observations_SST(iregion=9, index=i, wav=7)
            deep_mfbd_network.validate_fov(f'reconstructed/val_spot_20200727_083509.8542.{i:02d}.{time}.SST.h5')

        for i in range(10, 50):
            deep_mfbd_network.set_observations_SST(iregion=10, index=i, wav=7)
            deep_mfbd_network.validate_fov(f'reconstructed/val_qs_20190801_081547.8542.{i:02d}.{time}.SST.h5')
            
        # for i in range(20, 40):
        #     deep_mfbd_network.set_observations_SST(iregion=7, index=i, wav=7)
        #     deep_mfbd_network.validate_fov(f'reconstructed/validation_spot_8542_{i:02d}.{time}.SST.h5')
        
        # for i in range(11):
        #     deep_mfbd_network.set_observations_SST(iregion=4, index=i, wav=7)
        #     deep_mfbd_network.validate_fov(f'reconstructed/validation_plage_8542_{i:02d}.{time}.SST.h5')

        # for i in range(300, 320):
        #     deep_mfbd_network.set_observations_SST(iregion=5, index=i, wav=13)
        #     deep_mfbd_network.validate_fov(f'reconstructed/validation_spot_3934_{i:02d}.{time}.SST.h5')
