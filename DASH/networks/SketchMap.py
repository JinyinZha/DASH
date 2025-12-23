from .modules import MLP
import torch
from torch import nn
from torch.optim import lr_scheduler
import numpy as np


class SketchMap(nn.Module):
    #https://mlcolvar.readthedocs.io/en/latest/notebooks/tutorials/adv_newcv_subclass.html
    def __init__(self,xs,info,outdir,high_dim_params=None,low_dim_params=None,device="cuda"):
        super(SketchMap, self).__init__()
        high_dim = xs.shape[1]
        self.encoder = MLP([high_dim,high_dim,16,2])
        self.xs = xs
        self.info = info
        self.outdir = outdir
        self.high_dim_params = high_dim_params if high_dim_params is not None else dict(sigma=0.75, a=12, b=6)
        self.low_dim_params = low_dim_params if low_dim_params is not None else dict(sigma=1, a=2, b=6)
        self.device = device
        
    def forward(self,x):
        z = self.encoder(x)
        return z

    def get_z(self,x):
        return self.encoder(x)
        
    def sigmoid(self,x,sigma,a,b):
        return 1 - (1+(2**(a/b) -1)*( x/sigma)) **(-b/a)

    def get_loss(self,choice,use_all = False):
        if use_all:
            x = torch.from_numpy(self.xs).to(self.device)
            dm_h = self.dm_h_all
        else:
            x = torch.from_numpy(self.xs[choice]).to(self.device)
            dm_h = self.dm_h_all[choice][:,choice]
        z = self.forward(x)
        dm_l = torch.cdist(z,z)
        n_dist = (x.shape[0]-1)**2
        return 1/n_dist * torch.sum( ( self.sigmoid(dm_h, **self.high_dim_params ) - self.sigmoid(dm_l, **self.low_dim_params ) )**2 )
        
    def get_encoder(self):
        return self.encoder
    
    def train(self,random_seed):
        #1, parameters
        epoch = self.info["epoch"]
        batch_size = self.info["batch_size"]
        lr = self.info["lr"]
        schedular_step_size = self.info["schedular_step_size"]
        schedular_gamma = self.info["schedular_gamma"]
        #random_seed = self.info["random_seed"]
        #2 prepare batches and optimizers
        x_train = np.copy(self.xs)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        np.random.shuffle(x_train)
        n_batch= int(len(x_train) / batch_size)
        xs_split = np.array_split(x_train,n_batch)
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer,step_size=schedular_step_size,gamma=schedular_gamma,last_epoch=-1)
        #3, Train
        x_torch = torch.from_numpy(self.xs).to(self.device)
        self.dm_h_all = torch.cdist(x_torch,x_torch)
        losses = []
        for n in range(epoch):
            #for xnp in xs_split: 
            for i in range(400):
                loss = self.get_loss(np.random.choice(len(self.xs),1000,replace=False))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = self.get_loss(-1,True)
            losses.append(float(loss))
            scheduler.step()
            print("Seed: %d Epoch: %d  Loss:%.3f"%(random_seed,n+1,loss))
        #4, save
        return losses
