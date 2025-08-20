from md_engine import Energy_Cal
from utils import get_pwd_matrix

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ctx = torch.multiprocessing.get_context("spawn")

import numpy as np
import prody
import matplotlib.pyplot as plt
import copy
from scipy import stats
import os

import pydssp

def dssp_converge(traj_info,folder,cutoff=0.2):
    i = int(folder)
    p = prody.parsePDB(traj_info["pdb"])
    p0 = p.select("protein").copy()
    sel_ids = []
    res_ids = p0.getResnums()
    atom_names = p0.getNames()
    n_res = 0

    names = ["N","CA","C","O",]
    for res_id in np.unique(p0.getResnums()):
        tmp_sel_ids = []
        pos = np.argwhere(res_ids==res_id).flatten()
        for name in names:
            for j in pos:
                if atom_names[j] == name:
                    tmp_sel_ids.append(j)
                    break
        if len(tmp_sel_ids) == len(names):
            sel_ids += tmp_sel_ids
            n_res += 1

    c0 = p0.getCoordsets()[:,sel_ids,:]
    c0 = c0.reshape(c0.shape[0],n_res,len(names),-1)
    n_loop0 = [li.count("-") for li in pydssp.assign(c0, out_type='c3').tolist()][0]
    seed_pdb_files = ["%s/%s"%(folder,f) for f in os.listdir(folder) if f[0:4]=="seed" and f[-4:]==".pdb"]
    p.setCoords(np.array([prody.parsePDB(f).getCoords() for f in seed_pdb_files]))
    p1 = p.select("protein").copy()
    c1 = p1.getCoordsets()[:int(10000/(i+1)),sel_ids,:]
    c1 = c1.reshape(c1.shape[0],n_res,len(names),-1)

    seed_n_loop_res = [li.count("-") for li in pydssp.assign(c1, out_type='c3').tolist()]
    seed_n_loop_res.sort(reverse=True)
    print((np.mean(seed_n_loop_res[0:10]) - n_loop0) / (n_res - n_loop0) )
    return (np.mean(seed_n_loop_res[0:10]) - n_loop0) / (n_res - n_loop0) > cutoff
    

def energy_converge(traj_info,folder,n_proc=4):
    pro = prody.parsePDB(traj_info["pdb"])
    pro.setCoords(prody.parseDCD("%s/md.dcd"%(folder)).getCoordsets())
    pool = ctx.Pool(n_proc)
    ps = []
    for cs in np.array_split(pro.getCoordsets(),10):
        ps.append(pool.apply_async(get_pot,(cs,traj_info)))
    pool.close()
    es = []
    for p in ps:
        es += p.get()
    pool.join()
    pool.terminate()
    f = open("%s/es.txt"%(folder),"w")
    f.write("\n".join([str(i) for i in es]))
    f.close()
    es = np.array(es)
    labels = ["-".join(i.split("-")[0:2]) for i in np.load("%s/labels_of_combined.npy"%(folder))]
    plt.figure()
    e2plot = [es[0]]
    cl1,cl2 = labels[0].split("-")
    ts = [0]
    e_rounds = []
    e_rounds_means = []
    e_runs = []
    std_reses_means = []
    for i in range(1,len(es)):
        l1,l2 = labels[i].split("-")
        if (l1 == cl1 and l2 == cl2) or l1=="0":
            e2plot.append(es[i])
            ts.append(i)
        else:
            plt.plot(ts,e2plot)
            e_runs.append(copy.deepcopy(e2plot))
            if l1 != cl1:
                plt.vlines(i,es.min(),es.max(),ls="--",color="black")
                if int(cl1) >= 5:
                    y = []
                    x = []
                    for j in range(len(e_rounds_means)):
                        if j < 5:
                            y.append(e_rounds_means[j])
                            x.append(j)
                        elif -12 <= std_reses_means[j-5] <= 6:
                            y.append(e_rounds_means[j])
                            x.append(j)
                    y = np.array(y)
                    x = np.array(x)
                    lin_mod = stats.linregress(x,y)
                    k, b = lin_mod.slope,lin_mod.intercept
                    res_std = np.std(y - (k * x + b))
                    y_news = np.array([np.mean(li) for li in e_runs])
                    x_news = np.array(len(y_news)*[x[-1]+1])
                    std_reses = (y_news - (k * x_news + b)) / res_std
                    std_reses_means.append(std_reses.mean())
                e_rounds.append([i for li in e_runs for i in li])
                e_rounds_means.append(np.mean(e_rounds[-1]))
                e_runs = []
            e2plot = [es[i]]
            ts = [i]
            cl1 = l1
            cl2 = l2
    plt.plot(ts,e2plot)
    plt.savefig("%s/e_t.jpg"%(folder))
    plt.close()
    e_runs.append(copy.deepcopy(e2plot))
    y = []
    x = []
    for j in range(len(e_rounds_means)):
        if j < 5:
            y.append(e_rounds_means[j])
            x.append(j)
        elif -12 <= std_reses_means[j-5] <= 6:
            y.append(e_rounds_means[j])
            x.append(j)
    y = np.array(y)
    x = np.array(x)
    lin_mod = stats.linregress(x,y)
    k, b = lin_mod.slope,lin_mod.intercept
    res_std = np.std(y - (k * x + b))
    y_news = np.array([np.mean(li) for li in e_runs])
    x_news = np.array(len(y_news)*[x[-1]+1])
    std_reses = (y_news - (k * x_news + b)) / res_std
    std_reses_means.append(std_reses.max())

    plt.figure()
    plt.plot(np.linspace(5,4+len(std_reses_means),len(std_reses_means)),std_reses_means,label="mean")
    plt.legend()
    plt.savefig("%s/e_res.jpg"%(folder))
    plt.close()

    return not -12 <= std_reses_means[-1] <= 6


def get_pot(cs,traj_info):
    ec = Energy_Cal(traj_info["prmtop"],traj_info["pdb"],sel_str="all",t="sol")
    return [ec.get_energy(c) for c in cs]

def rmsd_converge(traj_info,i,cv_sel_str="protein and name CA"):
    pro = prody.parsePDB(traj_info["pdb"])
    pro.setCoords(prody.parseDCD(traj_info["dcd"]).getCoordsets())
    ca_coords = pro.select(cv_sel_str).getCoordsets()
    cut_pose = int(10000//(i+1))
    rmsds = get_pwd_matrix(ca_coords[:-cut_pose],ca_coords[-cut_pose:])[2]
    f = open("rmsd_convergte.txt","a")
    f.write("%d   %f\n"%(i,np.mean(rmsds)))
    f.close()
    return np.mean(rmsds) < 2

if __name__ == "__main__":
    import os,sys
    from traj_info import traj_infos

    os.chdir(sys.argv[1])
    traj_info = traj_infos[sys.argv[2]]
    st = int(sys.argv[3])
    ed = int(sys.argv[4])

    for i in range(st,ed+1):
        energy_converge(traj_info,str(i),n_proc=8)
