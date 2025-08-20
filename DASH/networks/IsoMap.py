from sklearn.manifold import Isomap
from .modules import MLP
import torch
from torch import nn
from torch.optim import lr_scheduler
import numpy as np

class IsoMap(nn.Module):
    def __init__(self,xs,info,outdir,device="cuda"):
        super(IsoMap, self).__init__()
        high_dim = xs.shape[1]
        self.encoder = MLP([high_dim,high_dim,16,2])
        self.xs = xs
        self.info = info
        self.outdir = outdir
        self.device = device
        
    def forward(self,x):
        return self.encoder(x)

    def get_dm_h_all(self,x):
        ism = Isomap().fit(x)
        return ism.dist_matrix_

    def get_z(self,x):
        return self.encoder(x)

    def get_loss(self,choice,use_all = False):
        if use_all:
            x = torch.from_numpy(self.xs).to(self.device)
            dm_h = self.dm_h_all
        else:
            x = torch.from_numpy(self.xs[choice]).to(self.device)
            dm_h = self.dm_h_all[choice][:,choice]
        z = self.forward(x)
        dm_l = torch.cdist(z,z)
        return nn.MSELoss()(dm_l,dm_h)
        
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
        self.dm_h_all = torch.from_numpy(self.get_dm_h_all(self.xs).astype(np.float32)).to(self.device)
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
        return losses
