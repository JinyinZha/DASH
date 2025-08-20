from .modules import MLP,GNN,EGNN
import torch
from torch import nn
from torch.optim import lr_scheduler
import random
import numpy as np
from plot import Plot
import copy
from scipy.spatial.distance import pdist, squareform

class Split_Contrast_MAE(nn.Module):
    def __init__(self,xs,rmsds,x1_ids,x2_ids,info,outdir,c_con=1,c_mae=1,device="cuda",encoder_type="MLP",gnn_cutoff=12):
        super(Split_Contrast_MAE, self).__init__() 
        sel_ids = [] # for selection in xs  
        for res_ids in info["res_id_sels"]: #res_ids is a list of res_id for one CV
            tmp = []
            for i in range(len(info["resids4xi"])):
                if info["resids4xi"][i] in res_ids:
                    tmp.append(i)
            sel_ids.append(copy.deepcopy(tmp))
            #print("sel_ids in SC",sel_ids)

    
        if encoder_type == "MLP":
            self.encoders = nn.ModuleList([MLP([len(sel_id),len(sel_id),8,1]) for sel_id in sel_ids])
        elif encoder_type in ["GCN","GAT","EGNN"]:
            self.encoders = nn.ModuleList()
            n_frame, n_feat = xs.shape
            #cset = xs.reshape(n_frame,-1,3)
            for sel_id in sel_ids:
                cset_sel = xs[:,sel_id].reshape(n_frame,-1,3)               
                dm = squareform(pdist(cset_sel.mean(0)))
                src, dst = np.where((dm < gnn_cutoff) & (dm > 0))
                edge_index = torch.tensor([src, dst], dtype=torch.long)
                if encoder_type in ["EGNN"]:
                    self.encoders.append(EGNN([3,16,1],edge_index,n_layer=3))
                else:
                    self.encoders.append(GNN([3,16,32,64,1],edge_index,gnn_type=encoder_type))

            


                    
        #self.scorer = MLP([2*len(self.encoders),8,1])
        self.scorer = MLP([2,8,1])
        self.decoder = MLP([2,16,xs.shape[1],xs.shape[1]])
        self.xs = xs
        self.rmsds = rmsds
        self.x1_ids = x1_ids
        self.x2_ids = x2_ids
        self.info =  info
        self.sel_ids = sel_ids #x3
        self.sel_resids = info["res_id_sels"] #not x3
        self.outdir = outdir
        self.device = device
        self.y_ref_all = torch.from_numpy(rmsds.reshape(-1,1)).to(self.device)
        self.c_con = c_con
        self.c_mae = c_mae
        print(info)

    def forward(self,x1s,x2s,xs):
        out1 = self.get_z(x1s)
        out2 = self.get_z(x2s)
        out3 = self.get_z(xs)
        #x = torch.cat((out1,out2),1)
        #x = torch.cat(((out1+out2).pow(2),(out1-out2).pow(2)),1)
        d2 = (out1-out2).pow(2)
        return self.scorer(d2),self.decoder(out3)

    def get_z(self,xs):
        return torch.cat([self.encoders[i](xs[:,self.sel_ids[i]]) for i in range(len(self.sel_ids))],1)
    
    def reconst(self,xs):
        return self.decoder(self.get_z(xs))
    
    def get_loss(self,this_ids):
        x1_ids = self.x1_ids[this_ids]
        x2_ids = self.x2_ids[this_ids]
        unq_ids = np.unique(np.hstack((x1_ids,x2_ids)).flatten())
        x1s = torch.from_numpy(self.xs[x1_ids]).to(self.device)
        x2s = torch.from_numpy(self.xs[x2_ids]).to(self.device)
        xs = torch.from_numpy(self.xs[unq_ids]).to(self.device)

        #print("x.shape",x1s.shape,x2s.shape,self.xs.shape)
        y_ref = torch.from_numpy(self.rmsds[this_ids].reshape(-1,1)).to(self.device)
        y,x_reconst = self.forward(x1s,x2s,xs)
        return nn.MSELoss()(y,y_ref), nn.MSELoss()(x_reconst,xs),y,y_ref

    
    def get_encoder(self,i):
        return self.encoders[i]
    
    def train(self,random_seed):
        #1, parameters
        epoch = self.info["epoch"]
        batch_size = self.info["batch_size"]
        max_nbatch = self.info["max_nbatch"]
        #random_seed = self.info["random_seed"]
        print("random_seed",random_seed)
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
        con_losses = []
        mae_losses = []
        for n in range(epoch):
            sampled_ids_split = random.sample(ids_split,max_nbatch)
            for this_ids in sampled_ids_split:         
                con_loss,mae_loss = self.get_loss(this_ids)[0:2]
                loss = self.c_con*con_loss + self.c_mae*mae_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            sampled_ids_split = random.sample(ids_split,7)
            tmp_losses = []
            tmp_con_losses = []
            tmp_mae_losses = []
            for this_ids in sampled_ids_split:         
                con_loss,mae_loss,y,y_ref = self.get_loss(this_ids)
                tmp_con_losses.append(float(con_loss))
                tmp_mae_losses.append(float(mae_loss))
                tmp_losses.append(self.c_con*float(con_loss) + self.c_mae*float(mae_loss))
            losses.append(np.mean(tmp_losses))
            con_losses.append(np.mean(tmp_con_losses))
            mae_losses.append(np.mean(tmp_mae_losses))
            scheduler.step()
            print("Seed: %d Epoch: %d  Con Loss:%.3f  MAE Loss:%.3f  Tot Loss:%.3f"%(random_seed,n+1,
                                                                                     con_losses[-1],mae_losses[-1],losses[-1]))
        #4, plot
        pt = Plot(self.outdir)
        y = y.cpu().detach().numpy().flatten()
        y_ref = y_ref.cpu().detach().numpy().flatten()
        pt.pre_ref_line(y,y_ref,"rmsd_reg_%d"%(random_seed))
        pt.loss([con_losses,mae_losses,losses],["Con","MAE","Tot"],"loss_%d"%(random_seed))
        return losses
