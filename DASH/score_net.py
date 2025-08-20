#!/usr/bin/env python3

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.cluster import KMeans, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from utils import  get_d_matrix, get_dists,get_pwd_matrix,ref_anti_sheet_coords,ref_helix_coords,ref_para_sheet_coords
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
import prody
from multiprocessing import Pool
import sys
import os

class Net_Scorer():

    def __init__(self):
        self.avg_lowd = None
        self.out = dict()
    
    def total(self,z,coords,outdir,root,traj_info,sel_str="protein and name CA",high_dim_type="CA",n_seed=20):
        self.outdir = outdir
        self.sel_str = sel_str
        self.root = root
        if os.path.exists("%s/score.txt"%(outdir)):
            f = open("%s/score.txt"%(outdir))
            lines = f.read().split("\n")
            f.close()
            for line in lines:
                if ":" not in line:
                    continue
                k,v = line.split(":")
                self.out[k] = float(v)
        methods = [
                    [self.intra_grid_info,[i],"Grid(%d) max RMSD"%(i)] for i in [40,10]
                  ] + [
                    [self.rmsd_of_seed,[],"Intra Seed RMSD"],
                    [self.reg_ss_info,[traj_info],"ss_score"],
                  ] 
        for method,para,k in methods:
            if k in self.out.keys():
                continue
            method(z,coords,outdir,*para)
            f = open("%s/score.txt"%(outdir),"w")
            for kk in self.out.keys(): 
                f.write("%s:%.3f\n"%(kk,self.out[kk]))
            f.close()

    def reg_ss_info(self,z,coords,outdir,traj_info):
        if os.path.exists("%s/sscv.npy"%(traj_info["dir"])):
            y = np.load("%s/sscv.npy"%(traj_info["dir"]))
            if len(y.shape) == 2:
                nf = y.shape[1]
                if y.shape[1] == 1:
                    y = y.flatten()
            else:
                nf = 1
            z = (z-z.min(0))/(z.max(0)-z.min(0))
            
            reg = MLPRegressor(hidden_layer_sizes=(4,8,4), activation='relu', max_iter=1000, random_state=111).fit(z,y)

            plt.figure(figsize=(nf*6,6))
            if nf == 1:
                plt.scatter(z[:,0],z[:,1],c=y,cmap="rainbow",s=0.5)
                plt.colorbar()
            else:
                for i in range(nf):
                    plt.subplot(1,nf,i+1)
                    plt.scatter(z[:,0],z[:,1],c=y[:,i],cmap="rainbow",s=0.5)
                    plt.colorbar() 
            plt.savefig("%s/ss_on_cv.jpg"%(outdir))
            plt.close()
            self.out["ss_score"] = reg.score(z,y)

    def rmsd_of_seed(self,z,coords,outdir,n_seed=20,n_close=3,n_far=3):
        z_scale = (z-z.min(0))/(z.max(0)-z.min(0))
        z_seed = z_scale[np.load("%s/margin_frameIDs.npy"%(self.outdir))]
        seed_ca_coords = np.array([prody.parsePDB("%s/seed%d.pdb"%(self.outdir,i)).select("(%s) and protein and name CA"%(self.sel_str)).getCoords() for i in range(n_seed)])
        self.out["Intra Seed RMSD"] = np.mean(get_d_matrix(seed_ca_coords)[2])
        self.out["Seed to All RMSD"] = np.mean(get_pwd_matrix(seed_ca_coords,coords)[2])           
        close_seed_rmsds = []
        far_seed_rmsds = []
        for i,zi in enumerate(z_seed):
            ds = np.linalg.norm(zi-z_seed,axis=1)
            li = [[j,ds[j]] for j in range(len(ds))]
            li.sort(key=lambda x:x[1])
            close_sel = [tmp[0] for tmp in li[1:1+n_close]]
            far_sel = [tmp[0] for tmp in li[len(li)-n_far:len(li)]]
            close_seed_rmsds.append(np.linalg.norm(seed_ca_coords[i] - seed_ca_coords[close_sel],axis=2).mean())
            far_seed_rmsds.append(np.linalg.norm(seed_ca_coords[i] - seed_ca_coords[far_sel],axis=2).mean())
        self.out["Close Seed RMSD"] = np.mean(close_seed_rmsds)
        self.out["Far Seed RMSD"] = np.mean(far_seed_rmsds)

    def intra_grid_info(self,z,coords,outdir,n_grid=50):
        if os.path.exists("%s/ses.txt"%(self.root)):
            f = open("%s/ses.txt"%(self.root))
            pots = np.array([float(line) for line in f.read().split("\n") if line != ""])
            f.close()
        else:
            pots = []
        x_max = np.max(z[:,0])
        x_min = np.min(z[:,0])
        y_max = np.max(z[:,1])
        y_min = np.min(z[:,1])
        dx = (x_max - x_min) / n_grid
        dy = (y_max - y_min) / n_grid
        max_rmsds = []
        rmsds = []
        max_dpots = []
        matrix = [[[] for j in range(n_grid)] for i in range(n_grid)]
        for i in range(len(z)):
             matrix[min(n_grid-1,int((z[i][1]-y_min)/dy))][min(n_grid-1,int((z[i][0]-x_min)/dx))].append(i)
        for line in matrix:
            for grid in line:
                if len(grid) == 0:
                    continue
                elif len(grid) == 1:
                    max_rmsds.append(0)
                    rmsds.append(0)
                    max_dpots.append(0)
                else:
                    dm = get_d_matrix(coords[grid])[2]
                    max_rmsds.append(np.max(dm))
                    rmsds.append(np.average(dm))
                    if len(pots) > 0:
                        max_dpots.append(pots[grid].max()-pots[grid].min())
                    else:
                        max_dpots.append(-1)
        all_link_rmsds = []
        link_not_link_ratio = []
        for i in range(n_grid*n_grid):
            xi = int(i // n_grid)
            yi = int(i % n_grid)
            if len(matrix[xi][yi]) == 0 :
                continue
            link_rmsds,not_link_rmsds = [],[]
            for j in range(n_grid*n_grid):
                if i==j:
                    continue
                xj = int(j // n_grid)
                yj = int(j % n_grid)
                if len(matrix[xj][yj]) == 0 :
                    continue
                r = np.linalg.norm(coords[matrix[xi][yi]].mean(0)-coords[matrix[xj][yj]].mean(0),axis=1).mean()
                if abs(xi-xj) <= 1 and abs(yi-yj) <= 1:
                    link_rmsds.append(r)
                else:
                    not_link_rmsds.append(r)
            if len(link_rmsds) > 0 and len(not_link_rmsds) > 0:
                all_link_rmsds.append(np.min(link_rmsds))
                link_not_link_ratio.append(np.min(link_rmsds)/np.mean(not_link_rmsds))
            else:
                all_link_rmsds.append(1)
                link_not_link_ratio.append(1)
        self.out["Grid(%d) Link/Not Ratio"%(n_grid)] = np.mean(link_not_link_ratio)
        self.out["Grid(%d) Link RMSD"%(n_grid)] = np.mean(all_link_rmsds)  
        plt.figure(dpi=500)
        plt.hist(rmsds,bins=50)
        plt.savefig("%s/intra_grid(%d)_rmsd_hist.jpg"%(outdir,n_grid))
        plt.close()
        np.save("%s/intra_grid(%d)_rmsd.npy"%(outdir,n_grid),rmsds)
        plt.figure(dpi=500)
        plt.hist(max_rmsds,bins=50)
        plt.savefig("%s/intra_grid(%d)_rmsd_max_hist.jpg"%(outdir,n_grid))
        plt.close()
        np.save("%s/intra_grid(%d)_rmsd_max.npy"%(outdir,n_grid),max_rmsds)
        self.out["Grid(%d) max RMSD"%(n_grid)] = np.mean(max_rmsds)
        self.out["Grid(%d) avg RMSD"%(n_grid)] = np.mean(rmsds)
        
def single_score(z,traj_info,system,outdir,root):
    ns = Net_Scorer()
    dl = Data_Loader(traj_info,align=False)
    if system == "diala":
        coords = dl.get_atomic_subset('not name "H.*"')[6]
    else:
        coords = dl.get_atomic_subset("protein and name CA")[6]
    ns.total(z,coords,outdir,root,traj_info)

if __name__ == "__main__":
    import sys
    import os
    from data_loader import Data_Loader
    from cv import get_traj_info

    outdir = sys.argv[1]
    z = np.load("%s/z.npy"%(outdir))
    traj_info = get_traj_info(sys.argv[2])
    dl = Data_Loader(traj_info,False)
    coords = dl.get_atomic_subset("protein and name CA")[6]
    ns = Net_Scorer()
    ns.total(z,coords,outdir,traj_info["dir"],traj_info)
