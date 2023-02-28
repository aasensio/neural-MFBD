import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import model_autoenc as model
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
from collections import OrderedDict
import pathlib
import matplotlib.pyplot as pl
import database
import zarr
try:
    import telegram
    TELEGRAM_BOT = True
except:
    TELEGRAM_BOT = False

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, n_training):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        try:
            f = zarr.open('/scratch1/aasensio/unroll_mfbd/autoencoder/training.zarr', 'r')
        except:
            raise Exception('training.zarr does not exists')

        print("Reading PSFs...")
        self.psf = f['psf']
        self.modes = f['modes']
        self.r0 = f['r0']

        self.n_pixel = f['psf'].attrs['n_pixel']
        
        self.n_training = len(self.r0)
        
    def __getitem__(self, index):

        psf = self.psf[index, :].reshape((self.n_pixel, self.n_pixel))[None, :, :]
        
        psf += np.random.normal(loc=0, scale=1e-3, size=psf.shape)

        min_psf = np.min(psf)
        max_psf = np.max(psf)
        psf = (psf - min_psf) / (max_psf - min_psf)
        
        modes = self.modes[index, :]

        return psf.astype('float32'), modes.astype('float32')

    def __len__(self):
        return self.n_training
        

class Validation(object):
    def __init__(self, checkpoint, gpu=1):
        self.checkpoint = checkpoint

        print("=> loading '{}'".format(self.checkpoint))
        chk = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)
        print("=> done")

        self.hyperparameters = chk['hyperparameters']
        
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = 128
                
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}        
        
        print('Instantiating model...')
        self.model = model.Model(config=self.hyperparameters).to(self.device)

        self.model.load_state_dict(chk['state_dict'])
                
        self.dataset = Dataset(n_training=None)
        
        self.validation_split = self.hyperparameters['validation_split']
        idx = np.arange(self.dataset.n_training)
        
        self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        
    def validate(self):
        self.model.eval()
        t = tqdm(self.validation_loader)
        loss_avg = 0.0

        z_all = []

        with torch.no_grad():
            for batch_idx, (psf, modes) in enumerate(t):
                psf = psf.to(self.device)
                modes = modes.to(self.device)
                        
                recons, mu, z, logvar = self.model.forward(psf)

                z_all.append(z)

        breakpoint()
            
        return loss_avg

if (__name__ == '__main__'):

    deepnet = Validation(checkpoint='weights//2023-01-25-15:02:37.best.pth')    
    deepnet.validate()