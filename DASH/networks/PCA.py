import torch
from torch import nn

class PCA(nn.Module):
    def __init__(self,x,dim=2,device="cuda"):
        super(PCA, self).__init__()
        self.dim=dim
        self.x = x
        self.device=device
        
    def forward(self,x):
	    return torch.matmul(x, self.U[:,:self.dim])
    
    def get_z(self,x):
        return self.forward(x)
    
    def get_encoder(self):
        return self
        
    def train(self):
        x = torch.from_numpy(self.x).to(self.device)
        x_mean = x.mean(dim=0)
        x = x - x_mean.expand_as(x)
        self.U, S, V = torch.svd(torch.t(x)) # U*Diag(S)*V_T