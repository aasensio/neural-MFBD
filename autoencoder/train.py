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
    import os
except:
    TELEGRAM_BOT = False

def merge_images(image_batch, size):
    b, h, w = image_batch.shape    
    img = np.zeros((int(h*size[0]), int(w*size[1])))
    for idx in range(b):
        i = idx % size[1]
        j = idx // size[1]
        maxval = np.max(image_batch[idx, :, :])
        minval = np.min(image_batch[idx, :, :])
        img[j*h:j*h+h, i*w:i*w+w] = (image_batch[idx, :, :] - minval) / (maxval - minval)
    return img


class TelegramBot(object):

    def __init__(self) -> None:
        self.token = os.environ['TELEGRAM_TOKEN']
        self.chat_id = os.environ['TELEGRAM_CHATID']

        self.bot = telegram.Bot(token=self.token)

    def send_message(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)

    def send_image(self, image):
        self.bot.send_photo(chat_id=self.chat_id, photo=open(image, 'rb'))

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, training_file=None):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        try:
            f = zarr.open(training_file, 'r')
        except:
            raise Exception('training.zarr does not exists')

        print("Reading PSFs...")
        self.psf = f['psf'][:]
        self.modes = f['modes']
        self.r0 = f['r0']
        self.D = f['D']
        self.wl = f['wl']

        self.n_pixel = f['psf'].attrs['n_pixel']
        
        self.n_training = len(self.r0)
        
    def __getitem__(self, index):

        psf = self.psf[index, :].reshape((self.n_pixel, self.n_pixel))[None, :, :]
        
        psf += np.random.normal(loc=0, scale=1e-3, size=psf.shape)

        min_psf = np.min(psf)
        max_psf = np.max(psf)
        psf = (psf - min_psf) / (max_psf - min_psf)
        
        modes = self.modes[index, :]

        wl = self.wl[index] / 5000.0
        D = self.D[index] / 100.0    

        return psf.astype('float32'), modes.astype('float32'), wl.astype('float32'), D.astype('float32')

    def __len__(self):
        return self.n_training
        

class Training(object):
    def __init__(self, hyperparameters):

        self.hyperparameters = hyperparameters

        self.cuda = torch.cuda.is_available()
        self.gpu = hyperparameters['gpu']
        self.smooth = hyperparameters['smooth']
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")        

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = hyperparameters['batch_size']        
                
        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}        

        # Create new training set if needed
        if (self.hyperparameters['new_trainingset']):
            print("Creating new training set...")
            db = database.Database(self.hyperparameters)
            psf_all, modes_all, r0_all, wl_all, D_all = db.calculate(batchsize=hyperparameters['training']['batch_size'], 
                n_batches=hyperparameters['training']['n_batches'],
                r0_min=hyperparameters['training']['r0_min'], 
                r0_max=hyperparameters['training']['r0_max'])
            
            psf_all = psf_all.reshape((-1, hyperparameters['n_pixel'] * hyperparameters['n_pixel']))
            _, n = psf_all.shape
            
            print('Saving training.zarr...')
            f = zarr.open(hyperparameters['training_file'], 'w')
            psfd = f.create_dataset('psf', shape=psf_all.shape, dtype=np.float32, chunks=(100, n))
            modesd = f.create_dataset('modes', shape=modes_all.shape, dtype=np.float32)
            r0d = f.create_dataset('r0', shape=r0_all.shape, dtype=np.float32)
            Dd = f.create_dataset('D', shape=D_all.shape, dtype=np.float32)
            wld = f.create_dataset('wl', shape=wl_all.shape, dtype=np.float32)

            psfd[:] = psf_all
            modesd[:] = modes_all
            r0d[:] = r0_all
            Dd[:] = D_all
            wld[:] = wl_all

            psfd.attrs['n_pixel'] = hyperparameters['n_pixel']
        
        print('Instantiating model...')
        self.model = model.Model(config=self.hyperparameters).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.dataset = Dataset(training_file=hyperparameters['training_file'])
        
        self.validation_split = hyperparameters['validation_split']
        idx = np.arange(self.dataset.n_training)
        
        self.train_index = idx[0:int((1-self.validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.validation_split)*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.train_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, sampler=self.validation_sampler, batch_size=self.batch_size, shuffle=False, **kwargs)

        if (TELEGRAM_BOT):
            self.bot = TelegramBot()

    def init_optimize(self):

        self.lr = self.hyperparameters['lr']
        self.wd = self.hyperparameters['wd']
        self.n_epochs = self.hyperparameters['n_epochs']
        
        print('Learning rate : {0}'.format(self.lr))        
        
        p = pathlib.Path('weights/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'weights/{0}'.format(current_time)

        # Copy model
        f = open(model.__file__, 'r')
        self.hyperparameters['model_code'] = f.readlines()
        f.close()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler, gamma=0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=0.2*self.lr)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = 1e100
        
        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            loss = self.train(epoch)
            loss_val = self.test()

            self.loss.append(loss)
            self.loss_val.append(loss_val)

            self.scheduler.step()

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'loss': self.loss,
                'loss_val': self.loss_val,
                'optimizer': self.optimizer.state_dict(),
                'hyperparameters': self.hyperparameters
            }

            if (loss_val < best_loss):
                print(f"Saving model {self.out_name}.best.pth")                
                best_loss = loss_val
                torch.save(checkpoint, f'{self.out_name}.best.pth')

            if (self.hyperparameters['save_all_epochs']):
                torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')


    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (psf, modes, wl, D) in enumerate(t):
            psf = psf.to(self.device)
            modes = modes.to(self.device)
            wl = wl.to(self.device)
            D = D.to(self.device)            
            
            self.optimizer.zero_grad()
            
            # Loss
            loss, recons_loss, kld_loss, recons = self.model.loss(psf, wl, D)
                    
            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()
                recons_loss_avg = recons_loss.item()
                kld_loss_avg = kld_loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                recons_loss_avg = self.smooth * recons_loss.item() + (1.0 - self.smooth) * recons_loss_avg
                kld_loss_avg = self.smooth * kld_loss.item() + (1.0 - self.smooth) * kld_loss_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                gpu_usage = f'{tmp.gpu}'
                tmp = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                memory_usage = f' {tmp.used / tmp.total * 100.0:4.1f}'                
            else:
                gpu_usage = 'NA'
                memory_usage = 'NA'

            tmp = OrderedDict()
            tmp['gpu'] = gpu_usage
            tmp['mem'] = memory_usage
            tmp['lr'] = current_lr
            tmp['L'] = loss_avg
            tmp['rec_L'] = recons_loss_avg
            tmp['kld_L'] = kld_loss_avg
            t.set_postfix(ordered_dict = tmp)

            if (batch_idx % self.hyperparameters['frequency_png'] == 0):                
                tmp = torch.cat([psf[0:8, 0, :, :], recons[0:8, 0, :, :]], dim=0)
                loop = 8              
                for j in range(3):
                    tmp = torch.cat([tmp, psf[loop:loop+8, 0, :, :]], dim=0)
                    tmp = torch.cat([tmp, recons[loop:loop+8, 0, :, :]], dim=0)
                    loop += 8                    
                im_merged = merge_images(tmp.detach().cpu().numpy(), [8,8])
                pl.imsave('samples/samples.png', im_merged)
                if (TELEGRAM_BOT):
                    self.bot.send_message(f'Autoencoder - Ep: {epoch} - L={loss_avg:7.4f}')                
                    self.bot.send_image('samples/samples.png')
            
        self.loss.append(loss_avg)
                
        return loss_avg

    def test(self):
        self.model.eval()
        t = tqdm(self.validation_loader)
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (psf, modes, wl, D) in enumerate(t):
                psf = psf.to(self.device)
                modes = modes.to(self.device)
                wl = wl.to(self.device)
                D = D.to(self.device)
                        
                loss, recons_loss, kld_loss, recons = self.model.loss(psf, wl, D)

                if (batch_idx == 0):
                    loss_avg = loss.item()
                    recons_loss_avg = recons_loss.item()
                    kld_loss_avg = kld_loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                    recons_loss_avg = self.smooth * recons_loss.item() + (1.0 - self.smooth) * recons_loss_avg
                    kld_loss_avg = self.smooth * kld_loss.item() + (1.0 - self.smooth) * kld_loss_avg
                
                t.set_postfix(loss=loss_avg)
            
        return loss_avg

if (__name__ == '__main__'):

    training_hyper = {
        'n_batches': 5000,
        'batch_size': 32,
        'r0_min': 3.0,
        'r0_max': 15.0
    }

    hyperparameters = {
        'batch_size': 128,
        'training_file': '/net/drogon/scratch1/aasensio/sst_unroll/psf.zarr',
        'validation_split': 0.1,
        'gpu': 0,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 300,
        'smooth': 0.15,
        'save_all_epochs': True,
        'wavelengths': [3934.0, 6173.0, 8542.0, 6563.0],
        'diameter': [100.0, 100.0, 100.0, 144.0],
        'pix_size': [0.038, 0.059, 0.059, 0.04979],
        'central_obs': [0.0, 0.0, 0.0, 0.0],
        'n_pixel': 64,        
        'n_modes': 44,
        'new_trainingset': False,
        'beta': 1e-3,
        'training': training_hyper,
        'frequency_png': 200,
        'n_bottleneck': 64,
        'channels_latent': 8
    }

    deepnet = Training(hyperparameters)
    deepnet.init_optimize()
    deepnet.optimize()
