import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from scipy.spatial.distance import pdist, squareform
from egnn_pytorch import EGNN

device = "cuda"

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, traj,cutoff=0.5):
        self.traj = traj
        dm = squareform(pdist(traj[0]))
        src, dst = np.where((dm < cutoff) & (dm > 0))
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        #self.edge_index = np.argwhere((dm>0) & (dm<0.5))

    def __len__(self):
        return len(self.traj)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.traj[idx], dtype=torch.float)
        return Data(x=x, edge_index=self.edge_index)

cset = np.random.rand(100,20,3)
dataset = ProteinDataset(cset)
loader = DataLoader(dataset, batch_size=5, shuffle=True)

class GCNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 64)
        self.conv3 = GCNConv(64, 2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        print(x,x.shape)
        print(edge_index,edge_index.shape)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return global_mean_pool(x, data.batch)
    
net = GCNEncoder().to(device)
for batch in loader:
    batch = batch.to(device)
    print(net(batch))
    break