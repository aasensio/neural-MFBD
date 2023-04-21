import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import time
from tqdm import tqdm
import model
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
from collections import OrderedDict
import pathlib
import matplotlib.pyplot as pl
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
        
        self.images = f['wb']

        self.n_training, _, self.n_pixel, self.n_pixel = f['wb'].shape
        
    def __getitem__(self, index):

        images = self.images[index, ...]

        rotate = np.random.randint(0, 4)
        images = np.rot90(images, rotate, axes=(1, 2))

        flip = np.random.randint(0, 2)
        if (flip):
            images = np.flip(images, axis=1)

        flip = np.random.randint(0, 2)
        if (flip):
            images = np.flip(images, axis=2)

        mn = np.mean(images, axis=(1, 2), keepdims=True)
        std = np.std(images, axis=(1, 2), keepdims=True)

        images_norm = (images - mn) / std
                
        return images_norm.astype('float32')

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
                
        print('Instantiating model...')
        self.model = model.Model(config=self.hyperparameters).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        print('Reading dataset...')
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

        for batch_idx, (images) in enumerate(t):
            images = images.to(self.device)  
            
            self.optimizer.zero_grad()
                        
            warped, _, _, _, loss = self.model(images, mode=self.hyperparameters['mode'])
                    
            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()                
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg                

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
            t.set_postfix(ordered_dict = tmp)

            # if (batch_idx % self.hyperparameters['frequency_png'] == 0):                
            #     tmp = torch.cat([images[0:8, 0, :, :] - warped[0:8, 0, :, :], flow[0:8, 0, :, :], flow[0:8, 1, :, :]], dim=0)
            #     loop = 8              
            #     for j in range(3):
            #         tmp = torch.cat([tmp, images[loop:loop+8, 0, :, :] - warped[loop:loop+8, 0, :, :]], dim=0)
            #         tmp = torch.cat([tmp,  flow[loop:loop+8, 0, :, :]], dim=0)
            #         tmp = torch.cat([tmp,  flow[loop:loop+8, 1, :, :]], dim=0)
            #         loop += 8                    
            #     im_merged = merge_images(tmp.detach().cpu().numpy(), [12,8])
            #     pl.imsave('samples/samples.png', im_merged)
            #     if (TELEGRAM_BOT):
            #         self.bot.send_message(f'Optflow - Ep: {epoch} - L={loss_avg:7.4f}')                
            #         self.bot.send_image('samples/samples.png')
            
        self.loss.append(loss_avg)
                
        return loss_avg

    def test(self):
        self.model.eval()
        t = tqdm(self.validation_loader)
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (images) in enumerate(t):
                images = images.to(self.device)
                                        
                warped, _, _, _, loss = self.model(images, mode=self.hyperparameters['mode'])

                if (batch_idx == 0):
                    loss_avg = loss.item()                    
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                
                t.set_postfix(loss=loss_avg)
            
        return loss_avg

if (__name__ == '__main__'):

    hyperparameters = {
        'batch_size': 128,
        'training_file': '/scratch1/aasensio/sst_unroll/training_optflow.zarr',
        'validation_split': 0.1,
        'gpu': 0,
        'lr': 3e-4,
        'wd': 0.0,
        'n_epochs': 30,
        'smooth': 0.15,
        'save_all_epochs': True,
        'n_pixel': 128,
        'channels': 16,
        'frequency_png': 25000,
        'mode': 'combined'
    }

    deepnet = Training(hyperparameters)
    deepnet.init_optimize()
    deepnet.optimize()
