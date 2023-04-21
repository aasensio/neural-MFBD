import sys
sys.path.append('../modules')
import numpy as np
import torch
import h5py
from tqdm import tqdm
import model_nopd_wiener as model
import nvidia_smi
import datasets

        
class FastMFBD(object):
    def __init__(self, checkpoint, gpu=2, batch_size=12, diffraction_limit=0.9, npix=None, npix_apodization=None):
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

        self.model.load_state_dict(chk['state_dict'])

        if (npix is not None):            
            self.config['n_pixel'] = npix
            self.config['npix_apodization'] = npix_apodization
        else:
            npix = self.config['n_pixel']
            npix_apodization = self.config['npix_apodization']
        
        self.model.change_pixel_size(npix=npix, npix_apodization=npix_apodization, diffraction_limit=diffraction_limit)            

        for param in self.model.parameters():
            param.requires_grad = False

    def set_observations_SST(self, iregion, index, wav=0, mod=0, cam=0, npix=None, step=40):
        # Read training and validation sets
        self.dataset = datasets.DatasetSST_validation(self.config, iregion, index, wav=wav, mod=mod, cam=cam, npix=npix, step=step)
        self.step = step

    def set_observations_HiFi(self, root, nac=12, npix=None, step=40):
        # Read training and validation sets
        self.dataset = datasets.DatasetHiFi_validation(self.config, root, nac=nac, npix=npix, step=step)
        self.step = step
                
    def validate_fov(self, outf):
        """
        Validate for one epoch
        """        
        
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
                
        # Data loaders that will inject data during training
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, 
                    batch_size=self.batch_size,
                    shuffle=False, 
                    drop_last=False,
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
        npix = self.config['n_pixel']
        n = (npix - self.step) // 2
        
        # filename = self.checkpoint.split('/')[-1]
        fout = h5py.File(outf, 'w')
        dset_modes_nn = fout.create_dataset('modes_nn', (self.dataset.n_training, self.dataset.n_frames, 44), dtype='float32')        
        dset_images_nn = fout.create_dataset('reconstruction_nn', (2, nx, ny), dtype='float32')
        dset_minval_nn = fout.create_dataset('minval', (self.dataset.n_training, 2), dtype='float32')
        dset_maxval_nn = fout.create_dataset('maxval', (self.dataset.n_training, 2), dtype='float32')
        dset_xy_nn = fout.create_dataset('xy', (self.dataset.n_training, 2), dtype='float32')
        
        dset_modes_nn_mem = np.zeros((self.dataset.n_training, self.dataset.n_frames, 44), dtype='float32')
        dset_images_nn_mem = np.zeros((self.dataset.n_training, 2, npix, npix), dtype='float32')        
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
                                                
            # with torch.no_grad():
            reconstructed, modes = self.model.forward_reconstructed(frames, frames_apod, wl, sigma, weight, image_filter=filter)

            left = batch_idx*self.batch_size
            right = (batch_idx+1)*self.batch_size
            dset_images_nn_mem[left:right, :, :] = reconstructed[:, :, :, :].detach().cpu().numpy()
            dset_modes_nn_mem[left:right, :, :] = modes[:, :, :].detach().cpu().numpy()
            dset_xy_nn_mem[left:right, :] = xy.detach().cpu().numpy()            
            dset_minval_nn_mem[left:right, :] = np.squeeze(minval)
            dset_maxval_nn_mem[left:right, :] = np.squeeze(maxval)

        
        print("Stitching images...")
        for i in tqdm(range(self.dataset.n_training)):
            indy, indx = dset_xy_nn_mem[i, :].astype('int')    
            minv = dset_minval_nn_mem[i, :][:, None, None]
            maxv = dset_maxval_nn_mem[i, :][:, None, None]
                
            out_nn[:, indx+n:indx+64-n, indy+n:indy+64-n] += dset_images_nn_mem[i, :, n:-n, n:-n] * (maxv - minv) + minv
            overlap[indx+n:indx+64-n, indy+n:indy+64-n] += 1

        dset_images_nn_mem = out_nn / overlap
            
        print("Saving file...")
        dset_images_nn[:] = dset_images_nn_mem        
        dset_xy_nn[:] = dset_xy_nn_mem
        dset_modes_nn[:] = dset_modes_nn_mem
        dset_minval_nn[:] = dset_minval_nn_mem
        dset_maxval_nn[:] = dset_maxval_nn_mem

        fout.close()            

if (__name__ == '__main__'):

    npix = 64
    npix_apodization = 24
    step = npix - npix_apodization - 4
    gpu = 3
    batch_size = 4


    checkpoint = 'weights/2023-03-07-14:31.All.ep_30.pth'
    checkpoint = 'weights/2023-03-11-08:03.All.ep_30.pth'
    
    diffraction_limit = [0.75, 0.75, 0.75, 0.75]

    deep_mfbd_network = FastMFBD(checkpoint=checkpoint, gpu=3, batch_size=batch_size, diffraction_limit=diffraction_limit, npix=npix, npix_apodization=npix_apodization)

    time = checkpoint.split('/')[-1].split('.')[0]
    

    # # Hifi
    # deep_mfbd_network.config['n_frames'] = 90
    # root = '/net/delfin/scratch/ckuckein/gregor/hifi2/momfbd/20221128/scan_b000'
    # deep_mfbd_network.set_observations_HiFi(root, nac=90, npix=npix, step=step)
    # deep_mfbd_network.validate_fov(f'/scratch1/aasensio/sst_unroll/hifi/unrolled/scan_b000.{time}.All.h5')

    # SST
    deep_mfbd_network.config['n_frames'] = 12
    
    # CRSIP Spot : [-1.755, -0.780, -0.260, -0.130, -0.065, 0., 0.065, 0.130, 0.260, 0.780, 1.755]  -> wave=7 -> 130 mA
    # for i in range(0, 50):
    #     deep_mfbd_network.set_observations_SST(iregion=9, index=i, wav=7, npix=npix, step=step)
    #     deep_mfbd_network.validate_fov(f'/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542/unrolled/val_spot_20200727_083509.8542.{i:02d}.{time}.All.h5')
    
    # CRISP QS : [-1.755, -0.845, -0.390, -0.195, -0.130, -0.065, 0., 0.065, 0.130, 0.195, 0.390, 0.845, 1.755]   -> wave=7 -> 65 mA
    # for i in range(10, 50):
    #     deep_mfbd_network.set_observations_SST(iregion=10, index=i, wav=7, step=38)
    #     deep_mfbd_network.validate_fov(f'/scratch1/aasensio/sst_unroll/qs_20190801_081547/unrolled/val_qs_20190801_081547.8542.{i:02d}.{time}.All.h5')
    
    # CHROMIS : [-1236, -845, -651, -585, -520, -454, -391, -325, -260, -195, -131, -65, +0, +65, +131, +195, +260, +325, +391, +454, +520, +585, +651, +845, +1234]    -> wave=7 -> 65 mA
    # for i in range(70, 130):
    #     deep_mfbd_network.set_observations_SST(iregion=8, index=i, wav=13, step=38)
    #     deep_mfbd_network.validate_fov(f'/scratch1/aasensio/sst_unroll/spot_20200727_083509_3934/unrolled/val_spot_20200727_083509.3934.{i:02d}.{time}.All.h5')

    for mod in range(4):
        for cam in range(2):            
            deep_mfbd_network.set_observations_SST(iregion=9, index=0, wav=8, npix=npix, step=step, mod=mod, cam=cam)  # 260 mA
            deep_mfbd_network.validate_fov(f'/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542/unrolled/val_spot_20200727_083509.8542.00.+260.lc{mod:01d}.cam{cam:01d}.{time}.All.h5')