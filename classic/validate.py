import numpy as np
import torch
import h5py
from tqdm import tqdm
import matplotlib.pyplot as pl
import classic
import datasets

        
class MFBD(object):
    def __init__(self, config, batch_size=12):
        """
        Train a deep neural network for self-supervised learning of multiframe deconvolution
                
        """        
        self.config = config
        self.batch_size = batch_size
        self.classic = classic.Classic(config)


    def set_observations_SST(self, iregion, index, wav=0, mod=0, cam=0, step=40):
        # Read training and validation sets
        self.dataset = datasets.DatasetSST_validation(self.config, iregion, index, wav=wav, mod=mod, cam=cam, step=step)
        self.step = step

    def set_observations_HiFi(self, root, nac=12, step=40):
        # Read training and validation sets
        self.dataset = datasets.DatasetHiFi_validation(self.config, root, nac=nac, step=step)
        self.step = step
                
    def validate_fov(self, outf):
        """
        Validate for one epoch
        """        
        
        kwargs = {'num_workers': 4, 'pin_memory': True}        
                
        # Data loaders that will inject data during training
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, 
                    batch_size=self.batch_size,
                    shuffle=False, 
                    drop_last=False,
                    **kwargs)        

        t = tqdm(self.validation_loader)

        npix = self.config['n_pixel']

        nx, ny = self.dataset.nx, self.dataset.ny
        out_nn = np.zeros((2, nx, ny))
        orig_nn = np.zeros((2, nx, ny))
        overlap = np.zeros((nx, ny))
        
        # Patches of 64 pixels, with a step of 40 pixels
        n = (npix - self.step) // 2
        
        # filename = self.checkpoint.split('/')[-1]
        fout = h5py.File(outf, 'w')
        dset_modes_nn = fout.create_dataset('modes_nn', (self.dataset.n_training, self.dataset.n_frames, 44), dtype='float32')        
        dset_images_nn = fout.create_dataset('reconstruction_nn', (2, nx, ny), dtype='float32')
        dset_orig_nn = fout.create_dataset('frame0', (2, nx, ny), dtype='float32')
        dset_minval_nn = fout.create_dataset('minval', (self.dataset.n_training, 2), dtype='float32')
        dset_maxval_nn = fout.create_dataset('maxval', (self.dataset.n_training, 2), dtype='float32')
        dset_xy_nn = fout.create_dataset('xy', (self.dataset.n_training, 2), dtype='float32')
        
        dset_modes_nn_mem = np.zeros((self.dataset.n_training, self.dataset.n_frames, 44), dtype='float32')
        dset_images_nn_mem = np.zeros((self.dataset.n_training, 2, npix, npix), dtype='float32')
        dset_orig_nn_mem = np.zeros((self.dataset.n_training, 2, npix, npix), dtype='float32')
        dset_xy_nn_mem = np.zeros((self.dataset.n_training, 2), dtype='float32')
        dset_minval_nn_mem = np.zeros((self.dataset.n_training, 2), dtype='float32')
        dset_maxval_nn_mem = np.zeros((self.dataset.n_training, 2), dtype='float32')
                            
        for batch_idx, (frames, frames_apod, wl, xy, minval, maxval, sigma, weight) in enumerate(t):
                            
            modes, psf, wavefront, degraded, reconstructed, loss = self.classic.deconvolve(frames, sigma)
                                                            
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

    config = {
        'gpus': [2],
        'npix_apodization': 24,
        'basis_for_wavefront': 'kl',
        'n_modes': 44,
        'n_frames' : 12,
        'n_pixel' : 64,
        'gradient_steps' : 100,
        'wavelength': 8542.0,
        'diameter': 100.0,
        'pix_size': 0.04979,
        'central_obs' : 0.0,
        'image_filter': 'lofdahl_scharmer'
    }
        
    instrument = 'SST'    

    # HiFi
    if (instrument == 'HiFi'):
    
    
        checkpoint = 'weights/2023-02-06-12:46.HiFi.ep_30.pth'
        deep_mfbd_network = FastMFBD(checkpoint=checkpoint, gpu=3, batch_size=24)

        # Hifi
        root = '/net/delfin/scratch/ckuckein/gregor/hifi2/momfbd/20221128/scan_b000'
        deep_mfbd_network.set_observations_HiFi(root, nac=12, step=38)
        deep_mfbd_network.validate_fov(f'reconstructed/test.HiFi.h5')

    
    # SST
    if (instrument == 'SST'):
        config['wavelength'] = 3934.0
        config['diameter'] = 100.0
        config['pix_size'] = 0.038
        mfbd = MFBD(config=config, batch_size=24)
        for i in range(70, 130):
            mfbd.set_observations_SST(iregion=8, index=i, wav=13, step=38)
            mfbd.validate_fov(f'reconstructed/val_spot_20200727_083509.3934.{i:02d}.SST.h5')
        
        config['wavelength'] = 8542.0
        config['diameter'] = 100.0
        config['pix_size'] = 0.059
        mfbd = MFBD(config=config, batch_size=24)
        for i in range(0, 50):
            mfbd.set_observations_SST(iregion=9, index=i, wav=7, step=38)
            mfbd.validate_fov(f'reconstructed/val_spot_20200727_083509.8542.{i:02d}.SST.h5')

        
        for i in range(10, 50):
            mfbd.set_observations_SST(iregion=10, index=i, wav=7, step=38)
            mfbd.validate_fov(f'reconstructed/val_qs_20190801_081547.8542.{i:02d}.SST.h5')
            
