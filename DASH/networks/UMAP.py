from .modules import MLP
import torch
from torch import nn
from torch.optim import lr_scheduler
import random
import numpy as np

from .parametric_umap.parametric_umap.datasets.covariates_datasets import TorchSparseDataset, VariableDataset
from .parametric_umap.parametric_umap.datasets.edge_dataset import EdgeDataset
from .parametric_umap.parametric_umap.utils.graph import compute_all_p_umap
from .parametric_umap.parametric_umap.core import ParametricUMAP
from .parametric_umap.parametric_umap.utils.losses import compute_correlation_loss

class UMAP(nn.Module):
    #https://kkgithub.com/fcarli/parametric_umap/
    def __init__(self,xs,info,outdir,device="cuda"):
        super(UMAP, self).__init__()
        high_dim = xs.shape[1]
        self.encoder = MLP([high_dim,high_dim,16,2])
        self.xs = xs
        self.info = info
        self.outdir = outdir
        self.umap_net = ParametricUMAP()
        self.device = device

    def forward(self,x):
        return self.encoder(x)

    def get_z(self,x):
        return self.encoder(x)
                
    def get_encoder(self):
        return self.encoder
    
    def train(self,random_seed,low_memory=False):
        #1, parameters
        epoch = self.info["epoch"]
        batch_size = self.info["batch_size"]
        max_nbatch = self.info["max_nbatch"]
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
        dataset = VariableDataset(self.xs).to(self.device)
        P_sym = compute_all_p_umap(self.xs, k=self.umap_net.n_neighbors)
        ed = EdgeDataset(P_sym)
        target_dataset = TorchSparseDataset(P_sym).to(self.device)
        loader = list(ed.get_loader(
            batch_size=batch_size,
            sample_first=True,
            random_state=random_seed,
            n_processes=1,
            verbose=True,
        ))
        losses = []
        for n in range(epoch):
            for edge_batch in random.sample(loader,min(max_nbatch,len(loader))):
                # Get src and dst indexes from edge_batch
                src_indexes = [i for i, j in edge_batch]
                dst_indexes = [j for i, j in edge_batch]
                # Get values from dataset
                src_values = dataset[src_indexes]
                dst_values = dataset[dst_indexes]
                targets = target_dataset[edge_batch]
                # If low memory, the dataset is not on GPU, so we need to move the values to GPU
                if low_memory:
                    src_values = src_values.to(self.device)
                    dst_values = dst_values.to(self.device)
                    targets = targets.to(self.device)
                # Get embeddings from model
                src_embeddings = self.encoder(src_values)
                dst_embeddings = self.encoder(dst_values)
                # Compute distances
                Z_distances = torch.norm(src_embeddings - dst_embeddings, dim=1)
                X_distances = torch.norm(src_values - dst_values, dim=1)
                # Compute losses
                qs = torch.pow(1 + self.umap_net.a * torch.norm(src_embeddings - dst_embeddings, dim=1, p=2 * self.umap_net.b), -1)
                umap_loss = self.umap_net.loss_fn(qs, targets)
                corr_loss = compute_correlation_loss(X_distances, Z_distances)
                loss = umap_loss + self.umap_net.correlation_weight * corr_loss
                loss.backward()
                optimizer.step()
            tmp=  []
            for edge_batch in random.sample(loader,min(max_nbatch,7)):
                # Get src and dst indexes from edge_batch
                src_indexes = [i for i, j in edge_batch]
                dst_indexes = [j for i, j in edge_batch]
                # Get values from dataset
                src_values = dataset[src_indexes]
                dst_values = dataset[dst_indexes]
                targets = target_dataset[edge_batch]
                # If low memory, the dataset is not on GPU, so we need to move the values to GPU
                if low_memory:
                    src_values = src_values.to(self.device)
                    dst_values = dst_values.to(self.device)
                    targets = targets.to(self.device)
                # Get embeddings from model
                src_embeddings = self.encoder(src_values)
                dst_embeddings = self.encoder(dst_values)
                # Compute distances
                Z_distances = torch.norm(src_embeddings - dst_embeddings, dim=1)
                X_distances = torch.norm(src_values - dst_values, dim=1)
                # Compute losses
                qs = torch.pow(1 + self.umap_net.a * torch.norm(src_embeddings - dst_embeddings, dim=1, p=2 * self.umap_net.b), -1)
                umap_loss = self.umap_net.loss_fn(qs, targets)
                corr_loss = compute_correlation_loss(X_distances, Z_distances)
                tmp.append(float(umap_loss + self.umap_net.correlation_weight * corr_loss))
            loss = np.mean(tmp)
            losses.append(loss)
            scheduler.step()
            print("Seed: %d Epoch: %d  Loss:%.3f"%(random_seed,n+1,loss))
        #4, save
        return losses