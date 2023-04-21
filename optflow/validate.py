import sys
sys.path.append('../modules')
import numpy as np
import torch
from tqdm import tqdm
import nvidia_smi
from astropy.io import fits
import model
import matplotlib.pyplot as pl
import napari
import h5py
        
class FastOptFlow(object):
    def __init__(self, checkpoint, gpu=2, batch_size=12, npix=128):
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

        self.model.update_grid(npix, npix)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
                
    def validate_fov(self):
        """
        Validate for one epoch
        """        
        
        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres'
        region = 'spot_20200727_083509_8542'
        tmp = region.split('_')[1:]
        tmp = '_'.join(tmp)
        index = 10
        wav = 7
        mod = 0
        cam = 0
                
        filename = f'{root}/{region}/wb_{tmp}_nwav_al_{index:05d}.fits'
        f_wb = fits.open(filename)

        nac, nwav, nmod, ny, nx = f_wb[0].data.shape

        wb = f_wb[0].data[:]

        original = wb[:, wav, mod, 0:512, 0:512]

        warped_all = [original[0:1, 0:512, 0:512]]
        
        for loop in tqdm(range(11)):

            im_wb = np.copy(original[loop:loop+2, 0:512, 0:512])
            im_wb[0, ...] = warped_all[0][0, ...]

            mn = np.mean(im_wb, axis=(1, 2), keepdims=True)
            std = np.std(im_wb, axis=(1, 2), keepdims=True)

            images_norm = (im_wb - mn) / std
            images_norm = torch.tensor(images_norm[None, :, :, :].astype('float32')).to(self.device)
            with torch.no_grad():
                warped, flow_16, flow_32, flow_64, loss = self.model(images_norm, mode=self.config['mode'])
            breakpoint()

            warped = warped.cpu().numpy()
            warped *= std[0]
            warped += mn[0]            
            
            warped_all.append(warped[0:1, 0, ...])
        
        warped_all = np.concatenate(warped_all, axis=0)

        return original, warped_all
    
                

if (__name__ == '__main__'):

        
    checkpoint = 'weights/2023-03-09-10:50:47.best.pth'
    deep_flow = FastOptFlow(checkpoint=checkpoint, gpu=0, batch_size=24, npix=512)

    original, images = deep_flow.validate_fov()

    dest = h5py.File('/net/drogon/scratch1/aasensio/sst_unroll/spot_20200727_083509_8542/raw/wav7_mod0_cam0_00010.h5', 'r')
    tmp = np.concatenate([original, images, dest['wb'][:, 0:512, 0:512]], axis=2)
    viewer = napari.view_image(tmp)
    # viewer = napari.view_image(images-dest['wb'][:, 0:512, 0:512])