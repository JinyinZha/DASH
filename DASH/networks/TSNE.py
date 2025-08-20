from .modules import MLP
import torch
from torch import nn
from torch.optim import lr_scheduler
import numpy as np


class TSNE(nn.Module):
    #https://github.com/mxl1990/tsne-pytorch/blob/master/tsne_torch.py

    def __init__(self,xs,info,outdir,perplexity=30.0,device="cuda"):
        super(TSNE, self).__init__()
        self.device = "cuda"
        self.EPS = torch.tensor([1e-12]).to(self.device)
        high_dim = xs.shape[1]
        self.encoder = MLP([high_dim,high_dim,16,2])
        self.xs = xs
        self.info = info
        self.outdir = outdir
        self.perplexity = perplexity
        

    def forward(self,x):
        return self.encoder(x)

    def get_z(self,x):
        return self.encoder(x)
        
    def Hbeta_torch(self,D, beta=1.0):
        P = torch.exp(-D.clone() * beta)
        sumP = torch.sum(P) + self.EPS
        H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
        P = P / sumP
        return H, P
    
    def x2p_torch(self,X, tol=1e-5, perplexity=30.0):
        """
            Performs a binary search to get P-values in such a way that each
            conditional Gaussian has the same perplexity.
        """
        # Initialize some variables
        X = torch.from_numpy(X).to(self.device)
        n, d = X.shape
        sum_X = torch.sum(X*X, 1)
        D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)
        P = torch.zeros(n, n).to(self.device)
        beta = torch.ones(n, 1).to(self.device)
        logU = torch.log(torch.tensor([perplexity]).to(self.device))
        n_list = [i for i in range(n)]
        # Loop over all datapoints
        for i in range(n):
            # Compute the Gaussian kernel and entropy for the current precision
            # there may be something wrong with this setting None
            betamin = None
            betamax = None
            Di = D[i, n_list[0:i]+n_list[i+1:n]]
            H, thisP = self.Hbeta_torch(Di, beta[i])
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU
            tries = 0
            while torch.abs(Hdiff) > tol and tries < 50:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].clone()
                    if betamax is None:
                        beta[i] = beta[i] * 2.
                    else:
                        beta[i] = (beta[i] + betamax) / 2.
                else:
                    betamax = beta[i].clone()
                    if betamin is None:
                        beta[i] = beta[i] / 2.
                    else:
                        beta[i] = (beta[i] + betamin) / 2.
                # Recompute the values
                H, thisP = self.Hbeta_torch(Di, beta[i])
                Hdiff = H - logU
                tries += 1
            # Set the final row of P
            P[i, n_list[0:i]+n_list[i+1:n]] = thisP
        # Return final P-matrix
        return P
    
    def y2q_torch(self,Y):
        n = Y.shape[0]
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        return Q

    def get_loss(self,choice,P,use_all=False):
        if use_all:
            x = torch.from_numpy(self.xs).to(self.device)
            P = self.P
        else:
            x = torch.from_numpy(self.xs[choice]).to(self.device)
            P = self.P[choice][:,choice]
            
        y = self.forward(x)
        Q = self.y2q_torch(y)
        return (P * torch.log((P + self.EPS) / (Q + self.EPS))).sum()

        
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
        P = self.x2p_torch(self.xs, 1e-5, self.perplexity)
        P = P + P.t()
        self.P = P / torch.sum(P)

        losses = []
        for n in range(epoch):
            #for xnp in xs_split: 
            for i in range(400):
                choice = np.random.choice(len(self.xs),1000,replace=False)
                loss = self.get_loss(choice,P)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss = self.get_loss(-1,P,True)
            losses.append(float(loss))
            scheduler.step()
            print("Seed: %d Epoch: %d  Loss:%.3f"%(random_seed,n+1,loss))
        return losses