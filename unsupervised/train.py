import matplotlib
import numpy as np
import torch
import torch.utils.data
import time
from tqdm import tqdm
import model_nopd_wiener as model
import argparse
import nvidia_smi
import sys
import config
import matplotlib.pyplot as pl
import pathlib
from collections import OrderedDict
from images import merge_images
import datasets
matplotlib.use('Agg')

try:
    import telegram
    TELEGRAM_BOT = True
    import os
except:
    TELEGRAM_BOT = False

class TelegramBot(object):
    """
    Telegram Bot for messaging with information during training

    """

    def __init__(self) -> None:
        self.token = os.environ['TELEGRAM_TOKEN']
        self.chat_id = os.environ['TELEGRAM_CHATID']

        self.bot = telegram.Bot(token=self.token)

    def send_message(self, message):
        self.bot.send_message(chat_id=self.chat_id, text=message)

    def send_image(self, image):
        self.bot.send_photo(chat_id=self.chat_id, photo=open(image, 'rb'))

        
class FastMFBD(object):
    def __init__(self, configuration_file=None, checkpoint=None):
        """
        Train a deep neural network for self-supervised learning of multiframe deconvolution
                
        """        

        # Set checkpoint file in case a previous training is to be resumed
        self.checkpoint = checkpoint

        # Read configuration file
        if (configuration_file is None):
            self.configuration_file = 'config.ini'
            self.config = config.Config('config.ini').hyperparameters
        else:
            self.configuration_file = configuration_file
            self.config = config.Config(configuration_file).hyperparameters

        # Is CUDA available?
        self.cuda = torch.cuda.is_available()        
        self.n_gpus = len(self.config['gpus'])
                
        # Smooth factor for the loss function
        self.smooth = 0.15
        
        # Number of GPUs in use
        n_gpus_available = torch.cuda.device_count()
        if (len(self.config['gpus']) > 1):
            print("Using GPUs : {0}".format(self.config['gpus']))            
            self.device = torch.device(f"cuda:{self.config['gpus'][0]}" if self.cuda else "cpu")
        else:
            print("Using single GPU : {0}".format(self.config['gpus']))
            self.device = torch.device(f"cuda:{self.config['gpus'][0]}" if self.cuda else "cpu")      
            torch.cuda.set_device(self.config['gpus'][0])

        print(f"Device : {self.device}")      
        
        # Ger handlers to later check memory and usage of GPUs
        nvidia_smi.nvmlInit()
        self.handle = [None] * self.n_gpus
        for i, gpu in enumerate(self.config['gpus']):
            self.handle[i] = nvidia_smi.nvmlDeviceGetHandleByIndex(int(gpu))
            print("Computing in {1} - cuda:{0}".format(gpu, nvidia_smi.nvmlDeviceGetName(self.handle[i])))

        # Read training and validation sets
        if (self.config['dataset_instrument'] == 'SST'):
            self.dataset = datasets.DatasetSST(self.config)
        if (self.config['dataset_instrument'] == 'HiFi'):
            self.dataset = datasets.DatasetHiFi(self.config)

        # Shuffle the training set
        idx = np.arange(self.dataset.n_training)
        np.random.shuffle(idx)

        self.train_index = idx[0:int((1-self.config['validation_split'])*self.dataset.n_training)]
        self.validation_index = idx[int((1-self.config['validation_split'])*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)

        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, 
                    batch_size=self.config['batch_size'], 
                    shuffle=False, 
                    drop_last=True, 
                    sampler=self.train_sampler, 
                    **kwargs)

        self.validation_loader = torch.utils.data.DataLoader(self.dataset, 
                    batch_size=self.config['batch_size'],
                    shuffle=False, 
                    drop_last=True, 
                    sampler=self.validation_sampler, 
                    **kwargs)
        
        # Get number of pixels and frames from the dataset                
        self.config['n_pixel'] = self.dataset.n_pixel
        self.config['n_frames'] = self.dataset.n_frames        
                
        # Instantiate the model
        print("Defining the model...")
        netmodel = model.Model(self.config)
        
        # If training in more than one GPU, instantiate the DataParallel class
        n_gpus_available = torch.cuda.device_count()
        if (len(self.config['gpus']) > n_gpus_available):
            raise("You do not have enough GPUs")

        if (len(self.config['gpus']) > 1):
            netmodel = torch.nn.DataParallel(netmodel, device_ids=self.config['gpus'])
            self.multi_gpu = True
        else:
            self.multi_gpu = False
        
        print(f"Training sample size : {len(self.train_loader) * self.config['batch_size']}")
        print(f"Validation sample size : {len(self.validation_loader) * self.config['batch_size']}")
        
        # Move model to GPU/CPU
        self.model = netmodel.to(self.device)

        # Initialize the Telegram bot if present
        if (TELEGRAM_BOT):
            self.bot = TelegramBot()
            print(f'Telegram Bot active: {self.bot}')
                
    def read_config(self):
        """
        Read configuration file
        """
        self.config = config.Config(self.configuration_file)
            
    def init_optimize(self):
        """
        Initialize the training
        """

        # Create directory with trained outputs if it does not exist
        p = pathlib.Path('weights/')
        p.mkdir(parents=True, exist_ok=True)
        
        # Get output file (it uses the time for getting a unique file)
        current_time = time.strftime("%Y-%m-%d-%H:%M")
        self.out_name = f"weights/{current_time}.{self.config['dataset_instrument']}"

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config['wd'])
                
        # Instantiate scheduler
        n_batches = len(self.train_loader)
        print(f'Number of batches : {n_batches}')
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                        self.config['n_epochs'] * n_batches, 
                        eta_min=self.config['scheduler_decay']*self.config['lr'])
        
        # Load checkpoint in case it is present
        if (self.checkpoint is not None):
            print(f"Loading checkpoint {self.checkpoint}")
            chk = torch.load(self.checkpoint)
            out_name = self.checkpoint.split('.')
            out_name = out_name[0].split('_')[0]
            self.out_name = f'{out_name}'
            if (self.multi_gpu):
                self.model.module.load_state_dict(chk['state_dict'])
            else:
                self.model.load_state_dict(chk['state_dict'])
            self.optimizer.load_state_dict(chk['optimizer_state_dict'])
            self.epoch_initial = chk['epoch'] + 1
        else:
            self.epoch_initial = 1
        
    def optimize(self):
        """
        Do the optimization
        """

        self.loss = []
        self.loss_val = []
        best_loss = 1e100       
        
        print('Model : {0}'.format(self.out_name))

        # Loop over epochs
        for epoch in range(self.epoch_initial, self.config['n_epochs'] + 1):
                        
            # Do one epoch for the training set
            loss_avg = self.train(epoch)

            # Do one epoch for the validation set
            loss_avg = self.validate(epoch)

            if (self.multi_gpu):
                model = self.model.module.state_dict()
            else:
                model = self.model.state_dict()
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'state_dict': model,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_loss': best_loss,
                'hyperparameters': self.config,
                'loss': self.loss,
                'val_loss': self.loss_val
            }

            best_loss = loss_avg
                    
            print(f"Saving model {self.out_name}.ep_{epoch}.pth...")
            torch.save(checkpoint, f'{self.out_name}.ep_{epoch}.pth')            
                    

    def train(self, epoch):
        """
        Train for one epoch
        """

        # Set model in training mode
        self.model.train()

        current_time = time.strftime("%Y-%m-%d-%H:%M")
        print(f"Epoch {epoch}/{self.config['n_epochs']} - {current_time}")
        t = tqdm(self.train_loader)
        
        # Get current learning rate
        if (self.multi_gpu):
            for param_group in self.module.optimizer.param_groups:
                current_lr = param_group['lr']
        else:
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']

        for batch_idx, (frames, frames_apod, wl, sigma, weight) in enumerate(t):
                        
            # Move all data to GPU/CPU
            frames = frames.to(self.device)
            frames_apod = frames_apod.to(self.device)
            sigma = sigma.to(self.device)
            weight = weight.to(self.device)
            wl = wl.to(self.device)

            n = self.config['npix_apodization']

            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Evaluate the model
            modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, loss = self.model(frames, 
                frames_apod,
                wl, 
                sigma, 
                weight, 
                image_filter=self.config['image_filter'])

            # Backpropagate
            loss.backward()

            # Update parameters
            self.optimizer.step()

            # Update averaged loss function                                                                        
            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (np.isnan(loss_avg)):
                if (TELEGRAM_BOT):
                    try:
                        self.bot.send_message(f'Exited. Loss became NaN')
                    except:
                        pass
                sys.exit()
                
            # Do some printing
            lo_max = torch.max(modes[:, :, 0:2]).item()
            lo_min = torch.min(modes[:, :, 0:2]).item()
            ho_max = torch.max(modes[:, :, 2:]).item()
            ho_min = torch.min(modes[:, :, 2:]).item()
            i1_min = torch.min(reconstructed_apod[0, 0, n:-n, n:-n]).item()
            i1_max = torch.max(reconstructed_apod[0, 0, n:-n, n:-n]).item()
            i2_min = torch.min(reconstructed_apod[0, 1, n:-n, n:-n]).item()
            i2_max = torch.max(reconstructed_apod[0, 1, n:-n, n:-n]).item()            

            n = self.config['npix_apodization'] // 2
            
            # Do some plots
            if (batch_idx % self.config['frequency_png'] == 0):

                n_images = 8
                
                labels = ['Frames WB', 'Frames NB', 'Deg WB', 'Deg NB', 'Diff WB', 'Diff NB', 'WF', 'sqrtPSF', 'PSF', 'Rec WB', 'Avg WB', 'Rec NB', 'Avg NB']

                tmp = frames_apod[0:n_images, 0, 0, :, :]
                tmp = torch.cat([tmp, frames_apod[0:n_images, 0, 1, :, :]], dim=0)
                tmp = torch.cat([tmp, degraded[0:n_images, 0, 0, :, :]], dim=0)
                tmp = torch.cat([tmp, degraded[0:n_images, 0, 1, :, :]], dim=0)
                tmp = torch.cat([tmp, frames_apod[0:n_images, 0, 0, :, :] - degraded[0:n_images, 0, 0, :, :]], dim=0)
                tmp = torch.cat([tmp, frames_apod[0:n_images, 0, 1, :, :] - degraded[0:n_images, 0, 1, :, :]], dim=0)
                tmp = torch.cat([tmp, wavefront[0:n_images, 3, :, :]], dim=0)
                tmp = torch.cat([tmp, torch.sqrt(torch.fft.fftshift(psf[0:n_images, 3, :, :]))], dim=0)
                tmp = torch.cat([tmp, torch.fft.fftshift(psf[0:n_images, 3, :, :])], dim=0)
                
                _, dimx, dimy = reconstructed[0, 0:1, :, :].shape
                
                t1 = reconstructed_apod[0:n_images, 0, :, :]
                tmp = torch.cat([tmp, t1], dim=0)

                t1 = torch.mean(frames_apod[0:n_images, :, 0, :, :], dim=1)
                tmp = torch.cat([tmp, t1], dim=0)

                t1 = reconstructed_apod[0:n_images, 1, :, :]
                tmp = torch.cat([tmp, t1], dim=0)

                t1 = torch.mean(frames_apod[0:n_images, :, 1, :, :], dim=1)
                tmp = torch.cat([tmp, t1], dim=0)

                im_merged = merge_images(tmp.detach().cpu().numpy(), [13,n_images], labels=labels)
                pl.imsave('samples/images.png', im_merged)

                fig, ax = pl.subplots()
                ax.plot(modes[0,:,0:20].detach().cpu().numpy())
                pl.savefig('samples/modes.png')
                pl.close()
                              
            
                if (TELEGRAM_BOT):
                    try:
                        self.bot.send_message(f'Ep: {epoch} - batch: {batch_idx} - L={loss_avg:7.4f}')
                        self.bot.send_image('samples/modes.png')                    
                        self.bot.send_image('samples/images.png')
                    except:
                        pass


            # Get GPU usage for printing
            gpu_usage = ''
            memory_usage = ''
            for i in range(self.n_gpus):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle[i])
                gpu_usage = gpu_usage+f' {tmp.gpu}'
                tmp = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle[i])
                memory_usage = memory_usage+f' {tmp.used / tmp.total * 100.0:4.1f}'

            # Update the progress bar
            tmp = OrderedDict()
            tmp['gpu'] = gpu_usage
            tmp['mem'] = memory_usage
            tmp['lr'] = f'{current_lr:8.6f}'
            tmp['lo'] = f'{lo_min:6.3f}/{lo_max:6.3f}'
            tmp['ho'] = f'{ho_min:6.3f}/{ho_max:6.3f}'
            tmp['imin'] = f'{i1_min:6.3f}/{i2_min:6.3f}'
            tmp['imax'] = f'{i1_max:6.3f}/{i2_max:6.3f}'
            tmp['loss'] = f'{loss_avg:8.5f}'
            t.set_postfix(ordered_dict = tmp)
                            
            self.loss.append(loss_avg)

            # Update learning rate if needed
            self.scheduler.step()

        return loss_avg

    def validate(self, epoch):
        """
        Validate for one epoch
        """

        # Put the model in evaluation mode
        self.model.eval()

        t = tqdm(self.validation_loader)
        n = 1
        loss_avg = 0
        loss_l1_avg = 0
        
        if (self.multi_gpu):
            for param_group in self.module.optimizer.param_groups:
                current_lr = param_group['lr']
        else:
            for param_group in self.optimizer.param_groups:
                current_lr = param_group['lr']
            
        # with torch.no_grad():
        for batch_idx, (frames, frames_apod, wl, sigma, weight) in enumerate(t):

            # Move all data to GPU/CPU
            frames = frames.to(self.device)
            frames_apod = frames_apod.to(self.device)
            sigma = sigma.to(self.device)
            weight = weight.to(self.device)
            wl = wl.to(self.device)
                                    
            modes, psf, wavefront, degraded, reconstructed, reconstructed_apod, loss = self.model(frames, frames_apod, wl, sigma, weight, optimize=False, image_filter='gaussian')

            loss = torch.sum(loss)

            if (batch_idx == 0):
                loss_avg = loss.item()                
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
            
            gpu_usage = ''
            memory_usage = ''
            for i in range(self.n_gpus):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle[i])
                gpu_usage = gpu_usage+f' {tmp.gpu}'
                memory_usage = memory_usage+f' {tmp.memory}'

            tmp = OrderedDict()
            tmp['gpu'] = gpu_usage
            tmp['mem'] = memory_usage
            tmp['loss'] = loss_avg
            t.set_postfix(ordered_dict = tmp)
                    
            self.loss_val.append(loss_avg)

        return loss_avg

if (__name__ == '__main__'):

    pl.ioff()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.ini', type=str)
    args = parser.parse_args()
    
    deep_mfbd_network = FastMFBD(configuration_file=args.config, checkpoint=None)
    deep_mfbd_network.init_optimize()
    deep_mfbd_network.optimize()