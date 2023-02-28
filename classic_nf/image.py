import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
from siren import SirenNet
from tqdm import tqdm
from collections import OrderedDict
from astropy.io import fits


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self, image, XYT):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()
                
        self.XYT = XYT
        self.image = image

        self.n_training = XYT.shape[0]
        
    def __getitem__(self, index):
        
        XYT = self.XYT[index, :]
        image = self.image[index]
        
        return XYT.astype('float32'), image.astype('float32')

    def __len__(self):
        return self.n_training

class RT(object):

    def __init__(self, configuration):

        self.device = 'cuda:0'

        self.configuration = configuration
    
        # Number of iterations for the optimization
        self.n_epochs = configuration['n_epochs']

        # Number of iterations for the optimization
        self.batch_size = configuration['batch_size']

        # Loss smooth
        self.smooth = configuration['smooth']
    
        # Learning rate
        self.lr = configuration['lr']

        # x_min, x_max
        self.xmin = 0.0
        self.xmax = 1.0
        self.ymin = 0.0
        self.ymax = 1.0

        self.outputfile = f"trained/test.pth"

        # Architecture
        self.dim_hidden = configuration['dim_hidden']
        self.n_hidden = configuration['n_hidden']

        self.n_pix = configuration['n_pix']

        root = '/net/diablos/scratch/sesteban/reduc/reduc_andres'

        images = []
        for i in range(20,30):
            
            region = 'spot_20200726_090257_8542'
            
            tmp = region.split('_')[1:]
            tmp = '_'.join(tmp)

            filename = f'{root}/{region}/wb_{tmp}_nwav_al_{i:05d}.fits'
            
            f_wb = fits.open(filename)
        
            image = f_wb[0].data[0, 0, 0, 500:500+self.n_pix, 500:500+self.n_pix]
            image /= np.mean(image)
            image = torch.tensor(image.astype('float32')).to(self.device)

            images.append(image[:, :, None])

        self.image = torch.cat(images, dim=-1)

        self.n_t = self.image.shape[-1]

        x = np.linspace(0.0, 1.0, self.n_pix)
        y = np.linspace(0.0, 1.0, self.n_pix)
        t = np.linspace(0.0, 1.0, self.n_t)
        
        X, Y, T = np.meshgrid(x, y, t)
        self.X = torch.from_numpy(X).float().to(self.device)
        self.Y = torch.from_numpy(Y).float().to(self.device)
        self.T = torch.from_numpy(T).float().to(self.device)

        self.XYT = torch.stack([self.X, self.Y, self.T], axis=-1).reshape(-1, 3)

        self.dataset = Dataset(self.image.view(-1).cpu().numpy(), self.XYT.cpu().numpy())

        kwargs = {'num_workers': 0, 'pin_memory': False}

        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, **kwargs)

        self.n_batches = self.dataset.n_training // self.batch_size
        
    def init_optimize(self):

        #####################
        # Neural network
        #####################

        n_output = 1

        self.model = SirenNet(dim_in=3, dim_hidden=self.dim_hidden, dim_out=1, num_layers=self.n_hidden, w0_initial=[100.0, 100.0, 10.0]).to(self.device)
        
        # Optimizer
        # optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            
        # Learning rate scheduler
        self.scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_epochs * self.n_batches * 2, eta_min=self.lr / 50.0)

        self.loss_mse = nn.MSELoss().to(self.device)
        
            
    def optimize(self):

        best_loss = 1e10

        loop = 0

        eta = 0.1

        # pbar = tqdm(total=self.n_epochs)        

        for epoch in range(self.n_epochs):

            t = tqdm(self.train_loader)

            for batch_idx, (XYT, image) in enumerate(t):

                XYT = XYT.to(self.device)
                image = image.to(self.device)
                
                # Evaluate the neural network 
                I, coords = self.model(XYT)

                residual = I[:, 0] - image

                loss = torch.mean(torch.abs(residual))

                # Zero the gradients
                self.optimizer.zero_grad()

                # Compute all gradients by backpropagation
                loss.backward()

                # Evolve the optimizer and update the learning rate
                self.optimizer.step()
                self.scheduler.step()

                # Extract the current learning rate
                for param_group in self.optimizer.param_groups:
                    current_lr = param_group['lr']

                # And do some printing
                output = OrderedDict()
                output['epoch'] = f'{epoch:03d}/{self.n_epochs:03d}'
                # output['weight'] = f'{weight_iter:8.4f}'
                output['lr'] = f'{current_lr:6.2e}'
                output['loss'] =f'{loss.item():7.3e}'
            
                t.set_postfix(output) 

            # pbar.update()

            if (loss.item() < best_loss):
                self.best_model = self.model
                best_loss = loss.item()

            loop += 1

        # pbar.close()

        # print("Saving best model...")
        # checkpoint = {'configuration': self.configuration,
        #     'state_dict': self.best_model.state_dict(),
        #     'losses': self.losses
        #     }

        # torch.save(checkpoint, f'{self.outputfile}')
        
        with torch.no_grad():
            I, _ = self.model(self.XYT)

        I = I.cpu().numpy().reshape((self.n_pix, self.n_pix, self.n_t))
        self.image = self.image.cpu().numpy()

        fig, ax = pl.subplots(nrows=2, ncols=10, figsize=(18,7))
        for i in range(10):
            ax[0, i].imshow(self.image[:, :, i])
            ax[1, i].imshow(I[:, :, i])
        pl.show()

        breakpoint()

        

if __name__ == '__main__':

    configuration = {'n_epochs': 20,
                        'lr': 3e-4,
                        'smooth': 0.05,
                        'dim_hidden': 256,
                        'n_hidden': 5,
                        'n_pix': 128,
                        'batch_size': 4096}

    out = RT(configuration)
    
    out.init_optimize()
    # B_pot = nlfff.init_potential(n_epochs=5)
    out.optimize()