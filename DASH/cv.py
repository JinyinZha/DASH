#!/usr/bin/env python3

import os
import sys
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
import torch
ctx = torch.multiprocessing.get_context("spawn")
from torch import nn,isnan
from torch.optim import lr_scheduler
from torch.cuda import OutOfMemoryError
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import random
import prody
import matplotlib.pyplot as plt
from openmm.app.metadynamics import *

from networks import Contrast,Contrast_AE,Split_Contrast_MAE,Split_Contrast,PCA,AE,IsoMap,EncoderMap,TSNE,UMAP,TICA
from data_loader import Data_Loader
from utils import get_d_matrix,get_most_fluc_regions,cluster_fluc_regions,get_repre_conf,cv_from_knowledge
from plot import Plot
from score_net import Net_Scorer
import config

class Packed_Encoder(nn.Module):
    def __init__(self,encoder,ref,align_ids,sub_ids,out_id,):
        super(Packed_Encoder,self).__init__()
        self.encoder = encoder
        self.align_ids = align_ids
        self.sub_ids = sub_ids# selecting key atoms from coordinates 
        self.out_id = out_id
        ref = ref[align_ids]
        self.ref_center = ref.mean(dim=0)
        self.centered_ref = ref - self.ref_center.expand_as(ref) 

    def forward(self,x):
        if x.device != self.centered_ref.device:
            x = x.to(self.centered_ref.device)
            x = x.to(torch.float32)
        x = x * 10#nm to amgstrong
        x_align = x[self.align_ids]
        x_sub = x[self.sub_ids]
        x_sub = self.align(x_sub,x_align).flatten()
        x = self.encoder(x_sub)
        return x[self.out_id]
        
    def align(self,x,x_align):
        #https://hunterheidenreich.com/posts/kabsch_algorithm/
        x_align_center = x_align.mean(dim=0)
        centered_x_align = x_align - x_align_center.expand_as(x_align)
        centered_x = x - x_align_center.expand_as(x)
        H = torch.matmul(centered_x_align.transpose(0, 1), self.centered_ref)
        U, S, Vt = torch.linalg.svd(H)
        if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
            Vt[:, -1] *= -1.0
        R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1)).detach()
        return torch.matmul(centered_x,R.transpose(0, 1)) + self.ref_center.expand_as(x)

class CV():
    def __init__(self,outdir):
        self.outdir = outdir
        
    def train(self,info,trained=False):
        #1, Parameters
        traj_info = info["traj_info"]
        epoch = info["epoch"]###
        batch_size = info["batch_size"]###
        max_nbatch = info["max_nbatch"]###
        self.random_seed_base = info["random_seed"]
        lr = info["lr"]###
        schedular_step_size = info["schedular_step_size"]###
        schedular_gamma = info["schedular_gamma"]###
        high_dim_type = info["high_dim_type"]
        if "n_train" in info.keys():
            n_train = info["n_train"]
        else:
            n_train = 1
        #1, Load data
        traj_info = info["traj_info"]
        high_dim_type = info["high_dim_type"]
        dl = Data_Loader(traj_info,align=False)
        if high_dim_type == "heavy":
            sel_str = 'not name "H.*"'
            align_str = sel_str
            t = "atom"
        elif high_dim_type == "CA":
            sel_str = "protein and name CA"
            align_str = sel_str
            t = "atom"
        pdb, sub_ids, resids4xi, resids4sub, xs, align_ids, align_coords,align_resids = dl.get_atomic_subset(sel_str,align_str,t)
        info["resids4xi"] = resids4xi
        info["resids4sub"] = resids4sub
        #2, Make RMSD Matrix
        name = traj_info["name"]
        if __name__ == "__main__" or "traj_id_sel" in info.keys():
            print("Training Mode")
            if os.path.exists("%s/%s_dmatrix.npy"%(traj_info["dir"],name)):
                x1_ids,x2_ids,rmsds = np.load("%s/%s_dmatrix.npy"%(traj_info["dir"],name))
                
            else:
                x1_ids,x2_ids,rmsds = get_d_matrix(align_coords)
                np.save("%s/%s_dmatrix.npy"%(traj_info["dir"],name),(x1_ids,x2_ids,rmsds))
        else:
            if os.path.exists("%s/%s_dmatrix.npy"%(self.outdir,name)):
                x1_ids,x2_ids,rmsds = np.load("%s/%s_dmatrix.npy"%(self.outdir,name))
            else:
                print("make RMSD matrix...")
                x1_ids,x2_ids,rmsds = get_d_matrix(align_coords)
                np.save("%s/%s_dmatrix.npy"%(self.outdir,name),(x1_ids,x2_ids,rmsds))
        x1_ids = x1_ids.astype(np.int32)
        x2_ids = x2_ids.astype(np.int32)
        rmsds = rmsds.astype(np.float32)

        if "traj_id_sel" in info.keys():
            tmp1,tmp2,tmp3 = [],[],[]
            traj_id_sel = set(info["traj_id_sel"])
            id_fix_dict = dict()
            n = 0
            for i in range(len(xs)):
                if i in traj_id_sel:
                    id_fix_dict[i] = n
                    n += 1
            for i in range(len(x1_ids)):
                if x1_ids[i] in traj_id_sel and x2_ids[i] in traj_id_sel:
                    tmp1.append(id_fix_dict[x1_ids[i]])
                    tmp2.append(id_fix_dict[x2_ids[i]])
                    tmp3.append(rmsds[i])
            
            x1_ids,x2_ids,rmsds = np.array(tmp1),np.array(tmp2),np.array(tmp3)
            xs_test = np.vstack([xs[i] for i in range(len(xs)) if i not in traj_id_sel])
            xs = xs[info["traj_id_sel"]]
        
        #3, Train
        if (__name__ == "__main__" or "traj_id_sel" in info.keys()) and os.path.exists("%s/model"%(self.outdir)):
            print("train result already found")
            net = torch.load("%s/model"%(self.outdir))
            self.net = net
            f = open("%s/resid_for_cvs.txt"%(self.outdir))
            info["res_id_sels"] = eval(f.read())
            f.close()
            pt = Plot(self.outdir)
        else:
            if info["method_name"] in ["SplitContrast","SplitContrastMAE"]:
                if "cv_knowledge" in traj_info.keys():
                    info["res_id_sels"] = cv_from_knowledge(pdb,traj_info["cv_knowledge"],self.outdir)
                elif info["msr"] == 0:
                    info["res_id_sels"] = cluster_fluc_regions(pdb.getCoords(),set(resids4sub),align_resids,align_coords,self.outdir)
                elif info["msr"] == 1:
                    info["res_id_sels"] = cluster_fluc_regions(pdb.getCoords(),set(resids4sub),align_resids,align_coords,self.outdir,skip_ter=True)
                elif info["msr"] == 2:
                    info["res_id_sels"] = cluster_fluc_regions(pdb.getCoords(),set(resids4sub),align_resids,align_coords,self.outdir,"top")
                elif info["msr"] == 3:
                    info["res_id_sels"] = cluster_fluc_regions(pdb.getCoords(),set(resids4sub),align_resids,align_coords,self.outdir,"top",skip_ter=True)
                elif info["msr"] == 4:
                    info["res_id_sels"] = cluster_fluc_regions(pdb.getCoords(),set(resids4sub),align_resids,align_coords,self.outdir,"local_linear")
                else:
                    info["res_id_sels"] = cluster_fluc_regions(pdb.getCoords(),set(resids4sub),align_resids,align_coords,self.outdir,"local_linear",skip_ter=True)
    
            if info["method_name"] in ["PCA","TICA"]:
                if info["method_name"] == "PCA":
                    net = PCA(xs).to(device)
                elif info["method_name"] == "TICA":
                    if "lag_time" in info.keys():
                        lag = info["lag_time"]
                    else:
                        lag = 1
                    if "split_file" in info.keys():
                        x_dict = dict()
                        for i,x in enumerate(np.load(info["split_file"])):
                            k = "_".join(x.split("-")[0:2])
                            if k[0] == "0":
                                k = "0"
                            if k not in x_dict.keys():
                                x_dict[k] = []
                            x_dict[k].append(xs[i])
                        xs_li = [np.vstack(x_dict[k]) for k in x_dict.keys()]
                        net = TICA(xs_li,lag,device=device).to(device)
                    else:
                        net = TICA(xs,lag,device=device).to(device)
                net.train()
                loss = None
            else:
                train_res = []
                trainers = []
                #if info["method_name"] == "EncoderMap" and xs.shape[1] > 800:
                    #info["n_proc"] = min(max(int(900*4/xs.shape[1]),1),5)
                if info["method_name"] == "IsoMap" and xs.shape[1] > 900:
                    info["n_proc"] = min(max(int(900*4/xs.shape[1]),1),5)
                p = ctx.Pool(info["n_proc"])
                #Split_Contrast_MAE
                if "encoder_type" in info.keys():
                    encoder_type = info["encoder_type"]
                else:
                    encoder_type = "MLP"

                for i in range(n_train):
                    if info["method_name"] == "Contrast":
                        net = Contrast(xs,rmsds,x1_ids,x2_ids,info,self.outdir,device=device).to(device)
                    elif info["method_name"] == "ContrastAE":
                        net = Contrast_AE(xs,rmsds,x1_ids,x2_ids,info,self.outdir,device=device).to(device)
                    elif info["method_name"] == "SplitContrast":
                        net = Split_Contrast(xs,rmsds,x1_ids,x2_ids,info,self.outdir,device=device).to(device)
                    elif info["method_name"] == "SplitContrastMAE":
                        if "c_con" in info.keys() and "c_mae" in info.keys():
                            net = Split_Contrast_MAE(xs,rmsds,x1_ids,x2_ids,info,self.outdir,info["c_con"],info["c_mae"],device=device,encoder_type = encoder_type).to(device) 
                        else:
                            net = Split_Contrast_MAE(xs,rmsds,x1_ids,x2_ids,info,self.outdir,device=device,encoder_type=encoder_type).to(device)             
                    elif info["method_name"] == "AE":
                        net = AE(xs,info,self.outdir,device=device).to(device)
                    elif info["method_name"] == "EncoderMap":
                        net = EncoderMap(xs,info,self.outdir,device=device).to(device)
                    elif info["method_name"] == "TSNE":
                        net = TSNE(xs,info,self.outdir,device=device).to(device)
                    elif info["method_name"] == "UMAP":
                        net = UMAP(xs,info,self.outdir,device=device).to(device)
                    elif info["method_name"] == "IsoMap":
                        net = IsoMap(xs,info,self.outdir,device=device).to(device)
                    if i == 0:
                        print(net)
                        total_params = sum(p.numel() for p in net.parameters())
                        print("Total parameters: %d"%(total_params))
                    trainers.append([p.apply_async(net.train,((i+1) * self.random_seed_base,)),net])
                p.close()
                train_res = [[trainer[1],trainer[0].get()] for trainer in trainers]
                p.join()
                p.terminate()
                train_res.sort(key=lambda x:x[1][-1])
                net,loss = train_res[0]
            #4, Save
            #4.1 Loss
            pt = Plot(self.outdir)
            if loss:
                pt.loss(loss)
            self.net = net
            torch.save(net,"%s/model"%(self.outdir))
        #4.2 Z
        z = self.get_z(xs)
        if "old_traj_info" in info.keys():
            old_traj_info = info["old_traj_info"]
            dl_old = Data_Loader(old_traj_info,align=False)
            if high_dim_type == "heavy":
                sel_str = 'not name "H.*"'
            else:
                sel_str = "protein and name CA"
            if "res_id_sel" in traj_info.keys():
                sel_str = "(%s) and (%s)"%(sel_str,traj_info["res_id_sel"])
            coords_old = dl_old.get_atomic_subset(sel_str)[3]
            n_frame_old, n_atom, n_dim = coords_old.shape
            xs_old = coords_old.reshape(n_frame_old,n_atom*n_dim)
            z_old = net.get_z(torch.from_numpy(xs_old).to(device)).cpu().detach().numpy()
            z,z_train = z_old,z
            self.z0_train = z_train
            np.save("%s/z_train.npy"%(self.outdir),z_train)
        np.save("%s/z.npy"%(self.outdir),z)
        self.z0 = z
        self.coords = align_coords
        np.save("%s/coords.npy"%(self.outdir),self.coords)
        pt.z(z)
        #4.3 CV
        z_minx = np.min(z[:,0])
        z_maxx = np.max(z[:,0])
        z_miny = np.min(z[:,1])
        z_maxy = np.max(z[:,1])
        dx = z_maxx - z_minx
        dy = z_maxy - z_miny
        
        z_minx -= dx
        z_maxx += dx
        z_miny -= dy
        z_maxy += dy
        
        if info["method_name"] in ["SplitContrast","SplitContrastMAE"]:
            self.fs = []
            for i in range(len(info["res_id_sels"])):
                sub_sub_ids = []
                for j in range(len(resids4sub)):
                    if resids4sub[j] in info["res_id_sels"][i]:
                        sub_sub_ids.append(sub_ids[j])
                print(i,"sub_sub_ids",sub_sub_ids,len(sub_sub_ids))
                out = Packed_Encoder(net.get_encoder(i),torch.from_numpy(dl.pdb_all.getCoords().astype(np.float32)).to(device),align_ids,sub_sub_ids,0)
                mod = torch.jit.script(out)
                mod.save("%s/CVnet_%d.pt"%(self.outdir,i))
                #f = TorchForce("%s/CVnet_%d.pt"%(self.outdir,i))
                f = "%s/CVnet_%d.pt"%(self.outdir,i)
                self.fs.append(f)
            np.save("%s/forceid.npy"%(self.outdir),list(range(len(info["res_id_sels"]))))
        else:
            out = Packed_Encoder(net.get_encoder(),torch.from_numpy(dl.pdb_all.getCoords().astype(np.float32)).to(device),align_ids,sub_ids,0)
            mod = torch.jit.script(out)
            mod.save("%s/CVnet_1.pt"%(self.outdir))
            out = Packed_Encoder(net.get_encoder(),torch.from_numpy(dl.pdb_all.getCoords().astype(np.float32)).to(device),align_ids,sub_ids,1)
            mod = torch.jit.script(out)
            mod.save("%s/CVnet_2.pt"%(self.outdir))
            self.fs = ["%s/CVnet_%d.pt"%(self.outdir,i) for i in [1,2]]
        self.drs = [dx,dy]
        #4.4 Seed structures for US
        dcd_total=dl.traj_all
        if "bins_for_margin" in info.keys():
            bins = info["bins_for_margin"]
        else:
            bins = 20
        if "cluster_margin" in info.keys():
            cluster_margin = info["cluster_margin"]
        else:
            cluster_margin = True
        self.margin_frameIDs,self.cluster_margin_centers,self.dxys = self.get_margin_frameIDs(z,bins=bins,cluster=cluster_margin,r0type=info["r0type"],n_seed=info["n_seed"])
        for n,i in enumerate(self.margin_frameIDs):
            prody.writePDB("%s/seed%d.pdb"%(self.outdir,n),dcd_total[i])        
        np.save("%s/margin_frameIDs.npy"%(self.outdir),self.margin_frameIDs)
        np.save("%s/cluster_margin_centers.npy"%(self.outdir),self.cluster_margin_centers)
        np.save("%s/dxys.npy"%(self.outdir),self.dxys)
        #4.5, analyze xs_test, if valid
        if "traj_id_sel" in info.keys():
            z_test = self.get_z(xs_test)
            np.save("%s/z_test.npy"%(self.outdir),z_test)
            plt.figure()
            plt.scatter(self.z0[:,0],self.z0[:,1],s=2)
            plt.scatter(z_test[:,0],z_test[:,1],s=2)
            plt.savefig("%s/train_test_z_scatter.jpg"%(self.outdir))
            plt.close()
        self.margin_dots = z[self.margin_frameIDs]
        pt.cvs_on_oldz(z,self.margin_dots,"margin",s=10)
        self.sel_str = sel_str
        
    def get_z(self,xs,n_split=1):
        if n_split == 1:
            return self.net.get_z(torch.from_numpy(xs).to(device)).cpu().detach().numpy()
        else:
            return np.vstack(([self.net.get_z(torch.from_numpy(x).to(device)).cpu().detach().numpy() for x in (np.array_split(xs,n_split))]))
        
        
    def load_net(self,force_graph=False): 
        print("Loading net of %s"%(self.outdir))
        self.net = torch.load("%s/model"%(self.outdir))
        self.z0 = np.load("%s/z.npy"%(self.outdir))
        z = self.z0
        if os.path.exists("%s/z_train.npy"%(self.outdir)):
            self.z0_train = np.load("%s/z_train.npy"%(self.outdir))
            
        self.coords = np.load("%s/coords.npy"%(self.outdir))
        z_minx = np.min(z[:,0])
        z_maxx = np.max(z[:,0])
        z_miny = np.min(z[:,1])
        z_maxy = np.max(z[:,1])
        dx = z_maxx - z_minx
        dy = z_maxy - z_miny
        
        z_minx -= dx
        z_maxx += dx
        z_miny -= dy
        z_maxy += dy

        if os.path.exists("%s/forceid.npy"%(self.outdir)):
            self.fs = ["%s/CVnet_%d.pt"%(self.outdir,i) for i in np.load("%s/forceid.npy"%(self.outdir))]
        else:
            self.fs = ["%s/CVnet_%d.pt"%(self.outdir,i) for i in [1,2]]
        self.drs = [dx,dy]
        self.cluster_margin_centers = np.load("%s/cluster_margin_centers.npy"%(self.outdir))
        self.margin_frameIDs = np.load("%s/margin_frameIDs.npy"%(self.outdir))
        try:
            self.dxys = np.load("%s/dxys.npy"%(self.outdir))
            self.unit_dxy = np.load("%s/unit_dxy.npy"%(self.outdir))
        except:
            pass
        self.margin_dots = self.z0[self.margin_frameIDs]

        
    def get_cv_values(self,traj_info,high_dim_type,draw_cut = None,plot=True,align=True):
        dl = Data_Loader(traj_info,align=align)
        if high_dim_type == "heavy":
            sel_str = 'not name "H.*"'
            align_str = sel_str
            t = "atom"
        elif high_dim_type == "CA":
            sel_str = "protein and name CA"
            align_str = sel_str
            t = "atom"
        elif high_dim_type == "CASC":
            sel_str = "protein and (%s)"%(config.casc_sel)
            align_str = "protein and name CA"
            t = "atom"
        elif  high_dim_type == "dihe":
            sel_str = "protein and name CA"
            align_str = sel_str
            t = "dihe"
        xs = dl.get_atomic_subset(sel_str,align_str,t)[4]
        z = self.net.get_z(torch.from_numpy(xs).to(device)).cpu().detach().numpy()
        pt = Plot(self.outdir)
        if plot:
            if draw_cut:
                pt.cvs_on_oldz(self.z0,z[-draw_cut:],"cvs_on_oldz")
            else:
                pt.cvs_on_oldz(self.z0,z,"cvs_on_oldz")
        return z
        
    def get_margin_frameIDs(self,zs,bins=20,n_seed=10,cluster=True,r0type="out"):
        #Make normalized hist2d
        zss = (zs-zs.min(axis=0))/(zs.max(axis=0)-zs.min(axis=0))
        zh0,xb,yb = np.histogram2d(zss[:,0],zss[:,1],bins=bins)
        zh = np.zeros((bins+2,bins+2))
        zh[1:-1,1:-1] = zh0
        dx = xb[1] - xb[0]
        dy = yb[1] - yb[0]
        xb = np.hstack((xb[0]-dx,xb,xb[-1]+dx))
        yb = np.hstack((yb[0]-dy,yb,yb[-1]+dy))
        x = (xb[1:]+xb[:-1])/2
        y = (yb[1:]+yb[:-1])/2
        #get margin centers
        margin_centers = []
        for i in range(bins+2):
            for j in range(bins+2):
                around = zh[np.max([0,i-1]):np.min([bins+2,i+2]),np.max([0,j-1]):np.min([bins+2,j+2])]
                if zh[i,j] == 0 and around.max() > 0:
                    margin_centers.append([x[i],y[j]])
        margin_centers = np.array(margin_centers)
        #cluster
        dxys = []
        cluster_margin_centers = []
        if cluster:
            km = KMeans(n_clusters=n_seed).fit(margin_centers)
            for i in range(n_seed):
                pos = np.argwhere(km.labels_==i).flatten()
                centers_in_cluster = margin_centers[pos]
                cc = centers_in_cluster.mean(0)
                cluster_margin_centers.append(margin_centers[pos[np.argmin(np.linalg.norm(centers_in_cluster-cc,axis=1))]]*(zs.max(axis=0)-zs.min(axis=0)) + zs.min(axis=0))
                #cluster_margin_centers.append(margin_centers[pos[np.argmin([np.linalg.norm(c-zss,axis=1).min() for c in centers_in_cluster])]] * (zs.max(axis=0)-zs.min(axis=0)) + zs.min(axis=0))
                dxys.append(np.abs(centers_in_cluster-km.cluster_centers_[i]).max(0) * (zs.max(axis=0)-zs.min(axis=0)))
            cluster_margin_centers = np.array(cluster_margin_centers)
        else:
            cluster_margin_centers = margin_centers * (zs.max(axis=0)-zs.min(axis=0)) + zs.min(axis=0)
            dxys = np.array([[dx,dy]]*len(cluster_margin_centers)) * (zs.max(axis=0)-zs.min(axis=0)) / 2
        self.unit_dxy =  np.array([dx,dy]) * (zs.max(axis=0)-zs.min(axis=0))
        np.save("%s/unit_dxy.npy"%(self.outdir),self.unit_dxy)
        print("unit_dxy",self.unit_dxy)
        print(dxys)
        
        margin_frameIDs = [np.argmin(np.linalg.norm(zs-c,axis=1)) for c in cluster_margin_centers]
        margin_centers = margin_centers * (zs.max(axis=0)-zs.min(axis=0)) + zs.min(axis=0)
        if r0type=="old":
            cluster_margin_centers = zs[margin_frameIDs]
        #plot
        plt.figure()
        plt.plot(zs[:,0],zs[:,1],"o",ms=0.5,color="black")
        if "z0_train" in dir(self):
            plt.plot(self.z0_train[:,0],self.z0_train[:,1],"^",ms=1,color="green")
        if cluster:
            for i in range(n_seed):
                pos = np.argwhere(km.labels_==i).flatten()
                plt.plot(margin_centers[pos,0],margin_centers[pos,1],"o",ms=2)
        plt.plot(cluster_margin_centers[:,0],cluster_margin_centers[:,1],"*",color="red",ms=10)
        plt.savefig("%s/margin_centers.jpg"%(self.outdir))
        plt.close()
        return margin_frameIDs,cluster_margin_centers,dxys
    
    def write_closest_frame_of_r0(self,traj_info,dcd_file,r0,out_name,dxy):
        traj_info["dcd"] = dcd_file
        dl = Data_Loader(traj_info)
        if "res_id_sel" in traj_info.keys():
            sel_str = "(name CA) and (%s)"%(traj_info["res_id_sel"])#############
        if high_dim_type == "heavy":
            sel_str = 'not name "H.*"'
            align_str = sel_str
            t = "atom"
        elif high_dim_type == "CA":
            sel_str = "protein and name CA"
            align_str = sel_str
            t = "atom"
        elif high_dim_type == "CASC":
            sel_str = "protein and (%s)"%(config.casc_sel)
            align_str = "protein and name CA"
            t = "atom"
        elif  high_dim_type == "dihe":
            sel_str = "protein and name CA"
            align_str = sel_str
            t = "dihe"
        xs = dl.get_atomic_subset(sel_str,align_str,t)[4]
        z = self.net.get_z(torch.from_numpy(xs).to(device)).cpu().detach().numpy()
        min_id = np.argmin(np.linalg.norm(z-r0,axis=1))
        dx,dy = dxy
        dr = (dx*dx+dy*dy)**0.5 / 2
        print("dr",dr)
        if np.linalg.norm(z[min_id]-r0) > dr:
            return False
        prody.writePDB(out_name,dl.traj_all[min_id])
        return True
              
def get_traj_info(file):
    d = dict()
    f = open(file)
    fli = f.read().split("\n")
    f.close()
    for line in fli:
        tmp = line.strip().split("#")[0].split()
        if len(tmp) >= 2:
            d[tmp[0]] = " ".join(tmp[1:])
            if tmp[0] in ["step","n_proc","n_train"]:
                d[tmp[0]] = int(tmp[1])
            elif os.path.exists(tmp[1]) and tmp[0] != "name":
                d[tmp[0]] = os.path.abspath(tmp[1])
    return d


def train_single(method_name,traj_info,high_dim_type,workdir,encoder_type=None,b=20,score=True,c_con=1,c_mae=1):
    os.system("mkdir %s"%(workdir))
    net = CV(workdir)
    traj_info["sel_str"] = "not resname WAT"
    if "n_proc" == traj_info.keys():
        n_proc = int(traj_info["n_proc"])
    else:
        n_proc = 5
    if "n_train" == traj_info.keys():
        n_train = int(traj_info["n_train"])
    else:
        n_train = 5
    cv_info = {"traj_info":traj_info,
           "method_name": method_name,
           "epoch":100,
           "batch_size":512,
           "max_nbatch":400,
           "random_seed":111,
           "lr":0.005,
           "schedular_step_size":20,
           "schedular_gamma":0.7,
           "high_dim_type":high_dim_type,
           "msr":0,
           "r0type":"out",
           "n_seed":20,
           "n_proc":n_proc,
           "n_train":n_train,
           "c_con":c_con,
           "c_mae":c_mae}
    if encoder_type:
        cv_info["encoder_type"] = encoder_type
    
    net.train(cv_info)
    get_repre_conf(net.z0,traj_info["dcd"],traj_info["pdb"],workdir,traj_info["name"],pdb_file_sel_str="all",compare_coord_sel_str=traj_info["sel_str"],bx=b,by=b)
    '''if score:
        ns = Net_Scorer()
        ns.total(net.z0,net.coords,workdir,traj_info["dir"],traj_info)
    net.load_net()'''

def get_folder_names(path,d=0,dmax=2):
    folders = []
    for f in os.listdir(path):
        if os.path.isdir("%s/%s"%(path,f)):
            folders.append(f)
            if d <= dmax-1:
                folders += get_folder_names("%s/%s"%(path,f),d+1)
    return set(folders)


if __name__ == "__main__":
    method_name = sys.argv[1]
    traj_info = get_traj_info(sys.argv[2])
    high_dim_type = sys.argv[3]
    workdir = "Train_%s_%s"%(method_name,traj_info["name"])
    if high_dim_type != "CA":
        workdir += "_%s"%(high_dim_type)
    if len(sys.argv) >= 6:
        c_con,c_mae = int(sys.argv[4]),int(sys.argv[5])
        if c_con != 1 or c_mae != 1:
            workdir += "_ccon%d_cmae%d"%(c_con,c_mae)
    else:
        c_con = c_mae = 1
    if len(sys.argv) >= 7:
        encoder_type = sys.argv[6]
        if encoder_type != "MLP":
            workdir += "_%s"%(encoder_type)
    else:
        encoder_type = None
    train_single(method_name,traj_info,high_dim_type,workdir,encoder_type=encoder_type,c_con=c_con,c_mae=c_mae)
            
         
        
        
        
    
