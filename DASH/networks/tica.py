import torch
from torch import nn
from deeptime.decomposition import TICA as TICA_Base
import numpy as np

class TICA(nn.Module):
    def __init__(self,x,lag=1,dim=2,device="cuda"):
        super(TICA, self).__init__()
        self.dim=dim
        self.x = x
        self.device=device
        self.lag = lag
        
    def forward(self,x):
        return torch.matmul(x-self.x_mean, self.U[:,:self.dim])
    
    def get_z(self,x):
        return self.forward(x)
    
    def get_encoder(self):
        return self
        
    def train(self):
        tica = TICA_Base(lagtime=self.lag)
        tica.fit(self.x)
        model = tica.fetch_model()
        self.U =  torch.from_numpy(model.instantaneous_coefficients.astype(np.float32)).to(self.device)
        self.x_mean = torch.from_numpy(self.x.mean(0).astype(np.float32)).to(self.device)
