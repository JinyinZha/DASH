from .modules import MLP
import torch
from torch import nn
from torch.optim import lr_scheduler
import random
import numpy as np
from plot import Plot

class Contrast(nn.Module):
    def __init__(self,xs,rmsds,x1_ids,x2_ids,info,outdir,scorer_dim=[4,8,1],device="cuda"):
        super(Contrast, self).__init__()
        high_dim = xs.shape[1]        
        self.encoder = MLP([high_dim,high_dim,16,2])
        self.scorer = MLP(scorer_dim)
        self.xs = xs
        self.rmsds = rmsds
        self.x1_ids = x1_ids
        self.x2_ids = x2_ids
        self.info =  info
        self.outdir = outdir
        self.device = device
        self.y_ref_all = torch.from_numpy(rmsds.reshape(-1,1)).to(self.device)


    def forward(self,x1,x2):
        out1 = self.encoder(x1)
        out2 = self.encoder(x2)
        x = torch.cat((out1,out2),1)
        return self.scorer(x)

    def get_z(self,x1):
        return self.encoder(x1)
    
    def get_loss(self,this_ids):
        x1 = torch.from_numpy(self.xs[self.x1_ids[this_ids]]).to(self.device)
        x2 = torch.from_numpy(self.xs[self.x2_ids[this_ids]]).to(self.device)
        y_ref = torch.from_numpy(self.rmsds[this_ids].reshape(-1,1)).to(self.device)
        y = self.forward(x1,x2)
        return nn.MSELoss()(y,y_ref),y,y_ref
    
    def get_encoder(self):
        return self.encoder
    
    def train(self,random_seed):
        #1, parameters
        epoch = self.info["epoch"]
        batch_size = self.info["batch_size"]
        max_nbatch = self.info["max_nbatch"]
        #random_seed = self.info["random_seed"]
        lr = self.info["lr"]
        schedular_step_size = self.info["schedular_step_size"]
        schedular_gamma = self.info["schedular_gamma"]
        #2, prepare batches and optimizers
        ids = np.array(list(range(len(self.rmsds))))
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        np.random.shuffle(ids) 
        n_batch = int(len(ids) / batch_size)
        ids_split = np.array_split(ids,n_batch)
        optimizer = torch.optim.Adam(self.parameters(),lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer,step_size=schedular_step_size,gamma=schedular_gamma,last_epoch=-1)
        #3, train
        losses = []
        for n in range(epoch):
            sampled_ids_split = random.sample(ids_split,max_nbatch)
            for this_ids in sampled_ids_split:         
                loss = self.get_loss(this_ids)[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sampled_ids_split = random.sample(ids_split,7)
            for this_ids in sampled_ids_split:         
                loss,y,y_ref = self.get_loss(this_ids)
            losses.append(float(loss))
            scheduler.step()
            print("Epoch: %d  Loss:%.3f"%(n+1,loss))
        #4, plot
        pt = Plot(self.outdir)
        y = y.cpu().detach().numpy().flatten()
        y_ref = y_ref.cpu().detach().numpy().flatten()
        pt.pre_ref_line(y,y_ref,"rmsd_reg_%d"%(random_seed))
        return losses