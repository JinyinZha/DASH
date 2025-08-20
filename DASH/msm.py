import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import sys,os
sys.path.append("../")
import pyemma
import pickle

def pyemma_plot_free_energy(
        xall, yall, weights=None, ax=None, nbins=30, ncontours=15,
        offset=-1, avoid_zero_count=False, minener_zero=True, kT=1.0,
        vmin=None, vmax=None, cmap='rainbow', cbar=True,
        cbar_label='free energy / kT', cax=None, levels=None,
        legacy=True, ncountours=None, cbar_orientation='vertical',
        **kwargs):
    x, y, z = pyemma.plots.plots2d.get_histogram(
        xall, yall, nbins=nbins, weights=weights,
        avoid_zero_count=avoid_zero_count)
    f = pyemma.plots.plots2d._to_free_energy(z, minener_zero=minener_zero) * kT
    fig, ax, misc = pyemma.plots.plots2d.plot_map(
        x, y, f, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels=levels,
        cbar=cbar, cax=cax, cbar_label=cbar_label,
        cbar_orientation=cbar_orientation, norm=None,
        **kwargs)
    return f,x,y,misc

def msm_and_TP(name,ni,n_bins,z_file,n_clusters=50,lag_time=0.5,s=23,upper_bar=False,
               xticks=None,yticks=None,cbar_ticks=None,xlim=None,ylim=None,xlabel="CV1",ylabel="CV2",
               z_st_x=None,z_st_y=None,z_ed_x=None,z_ed_y=None):
    zs = np.load("%d/%s"%(ni,z_file))
    if zs.shape[1] != 2:
        zs = zs.T
    z_dict = dict()
    for i,x in enumerate(np.load("%d/labels_of_combined.npy"%(ni))):
        k = "_".join(x.split("-")[0:2])
        if k[0] == "0":
            k = "0"
        if k not in z_dict.keys():
            z_dict[k] = []
        z_dict[k].append(zs[i])
    zs_li = [np.vstack(z_dict[k]) for k in z_dict.keys()]
    if os.path.exists("%d/%s_msm_%.2fns/%s_msm.pkl"%(ni,name,lag_time,name)):
        with open("%d/%s_msm_%.2fns/%s_msm.pkl"%(ni,name,lag_time,name),"rb") as f:
            msm = pickle.load(f)
    else:

        dt = (ni+1) / 1000
        lag = int(len(zs)*lag_time/(ni+1)/10)
        os.system("mkdir %d/%s_msm_%.2fns"%(ni,name,lag_time))
        cluster = pyemma.coordinates.cluster_kmeans(zs_li, k=n_clusters, max_iter=50, stride=10, fixed_seed=111)
        its = pyemma.msm.its(cluster.dtrajs, lags=4*lag, nits=10, errors='bayes')
        plt.figure(figsize=(8,6))
        pyemma.plots.plot_implied_timescales(its, units='ns', dt=dt)
        s1=22
        s2=s1+2
        plt.xticks(fontsize=s1)
        plt.yticks(fontsize=s1)
        plt.xlabel("Lag Time (ns)",fontdict={"size":s2})
        plt.ylabel("Timescale (ns)",fontdict={"size":s2})
        plt.tight_layout()
        plt.savefig("%d/%s_msm_%.2fns/%s_its.jpg"%(ni,name,lag_time,name))
        plt.close()
        msm = pyemma.msm.bayesian_markov_model(cluster.dtrajs, lag=lag, dt_traj='%f ns'%(dt))
        nc = 5 
        cktest = msm.cktest(nc, mlags=6)
        pyemma.plots.plot_cktest(cktest,dt=dt, units='ns')
        plt.savefig("%d/%s_msm_%.2fns/%s_ck.jpg"%(ni,name,lag_time,name))
        plt.close()
        with open("%d/%s_msm_%.2fns/%s_msm.pkl"%(ni,name,lag_time,name),"wb") as f:
            pickle.dump(msm,f)
        with open("%d/%s_msm_%.2fns/%s_cluster.pkl"%(ni,name,lag_time,name),"wb") as f:
            pickle.dump(cluster,f)
    s1=s
    s2=s1+2
    plt.figure(dpi=300,figsize=(8,8))
    e,x,y,misc =  pyemma_plot_free_energy(
        *zs[:, :2].T,ax=plt.gca(),
        weights=np.concatenate(msm.trajectory_weights()),
        legacy=False,nbins=n_bins)
    if type(xlim) == type((1,2)):
        plt.xlim(xlim)
    if type(ylim) == type((1,2)):
        plt.ylim(ylim)
    if type(xticks) == type([]):
        plt.xticks(xticks,fontsize=s1)
    else:
        plt.xticks(fontsize=s1)
    if type(yticks) == type([]):
        plt.yticks(yticks,fontsize=s1)
    else:
        plt.yticks(fontsize=s1)
    plt.xlabel(xlabel,fontdict={"size":s2})
    plt.ylabel(ylabel,fontdict={"size":s2})
    cbar = misc["cbar"]
    if upper_bar:
        cbar.remove()
        cbar = plt.colorbar(misc["mappable"],orientation='horizontal', location='top',pad=0.02)
    if type(cbar_ticks) == type([]):
        cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=s1)
    if upper_bar:
        cbar.set_label("Free Energy / k$_B$T", fontsize=s2, rotation=0,labelpad=10)
    else:
        cbar.set_label("Free Energy / k$_B$T", fontsize=s2, rotation=270,labelpad=40)
    ax=plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    plt.tight_layout()
    plt.savefig("%d/%s_msm_%.2fns/FES.jpg"%(ni,name,lag_time))
    
    print("FES saved as %d/%s_msm_%.2fns/FES.jpg"%(ni,name,lag_time))
    if z_st_x!=None and z_st_y!=None and z_ed_x !=None and z_ed_y!=None:
        basins = []
        for i in range(len(x)):
            ist = max(0,i-1)
            ied = min(i+2,len(x))
            for j in range(len(y)):
                if e[i,j] == np.inf:
                    continue
                jst = max(0,j-1)
                jed = min(j+2,len(y))
                if e[i,j] <= e[ist:ied,jst:jed].min():
                    basins.append((i,j))
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        x_min = x[0] - 0.5 * dx
        y_min = y[0] - 0.5 * dy
        all_grid_centers = np.array([[[xx,yy] for xx in x] for yy in y])
        frameID_in_bins = [[[] for j in range(n_bins)] for i in range(n_bins)]
        for i,z in enumerate(zs):
            zx,zy = z
            frameID_in_bins[min(n_bins-1,int((zx-x_min)/dx))][min(n_bins-1,int((zy-y_min)/dy))].append(i)
        grids = [[e[i,j],(i,j),all_grid_centers[i,j],[]] for i in range(n_bins) for j in range(n_bins) if e[i,j] != np.inf]

        for j,(cvx,cvy) in enumerate(zs):
            ij = (np.clip(int((cvy-y_min)/dy),0,len(y)-1),np.clip(int((cvx-x_min)/dx),0,len(x)-1))
            for i in range(len(grids)):
                if grids[i][1] == ij:
                    grids[i][3].append(j)
                    break
        d_min = np.inf
        i_grid_st = None
        i_st = np.clip(int((z_st_x-x_min)/dx),0,len(x)-1)
        j_st = np.clip(int((z_st_y-y_min)/dy),0,len(y)-1)
        z_st_ij = np.array([j_st,i_st])
        for i,g in enumerate(grids):
            if g[1] not in basins:
                continue
            d = np.linalg.norm(z_st_ij-np.array(g[1]))
            if d < d_min:
                i_grid_st = i
                d_min = d
        d_min = np.inf
        i_grid_ed = None
        i_ed = np.clip(int((z_ed_x-x_min)/dx),0,len(x)-1)
        j_ed = np.clip(int((z_ed_y-y_min)/dy),0,len(y)-1)
        z_ed_ij = np.array([j_ed,i_ed])
        for i,g in enumerate(grids):
            if g[1] not in basins:
                continue
            d = np.linalg.norm(z_ed_ij-np.array(g[1]))
            if d < d_min:
                i_grid_ed = i
                d_min = d
        if d_min > 3 or i_grid_st == i_grid_ed:
            d_min = np.inf
            i_grid_ed = None
            for i,g in enumerate(grids):
                d = np.linalg.norm(z_ed_ij-np.array(g[1]))
                if d < d_min:
                    i_grid_ed = i
                    d_min = d

        traj,barrier = dijkstra(grids,i_grid_st,i_grid_ed)
        es = [grids[t][0] for t in traj]
        
        traj_x = [grids[t][2][0] for t in traj]
        traj_y = [grids[t][2][1] for t in traj]
        plt.plot(traj_x,traj_y,lw=5,color="black")
        plt.savefig("%d/%s_msm_%.2fns/FES_traj_%.2fkbT.jpg"%(ni,name,lag_time,barrier))
        plt.close()

        for i in range(len(es)):
            print(traj_x[i],traj_y[i],es[i])
        plt.figure(dpi=500,figsize=(7,6))
        plt.plot(es,lw=5,color="black")
        plt.xticks([],fontsize=s1)
        plt.yticks(fontsize=s1)
        plt.ylabel("Free Energy / k$_B$T",fontdict={"size":s2})
        plt.xlabel("Transition Path",fontdict={"size":s2})
        ax=plt.gca()
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_linewidth(3)
        plt.tick_params(width=3)
        plt.tight_layout()
        plt.savefig("%d/%s_msm_%.2fns/%s_path.jpg"%(ni,name,lag_time,name))
        plt.close()
    else:
        plt.close()

def dijkstra(grids,st,ed):
    n_v = len(grids)
    traj = [[i] for i in range(n_v)]
    es = np.array([grids[i][0] for i in range(n_v)])
    visited = [False] * n_v
    barrier = [np.inf] * n_v
    barrier[st] = 0
    while not visited[ed]:
        barrier_min = np.inf
        i_min = None
        tmp_e = np.inf
        for i in range(n_v):
            if (not visited[i]) and barrier[i] < barrier_min:
                barrier_min = barrier[i]
                i_min = i
                tmp_e = es[i]
            elif (not visited[i]) and barrier[i] == barrier_min and tmp_e > es[i]:
                barrier_min = barrier[i]
                i_min = i
                tmp_e = es[i]
        visited[i_min] = True
        for i in range(n_v):
            d1 = np.abs(grids[i][1][0]-grids[i_min][1][0])
            d2 = np.abs(grids[i][1][1]-grids[i_min][1][1])
            if d1 <=1 and d2 <= 1 and i_min != i:
                tmp_traj = traj[i_min] + [i]
                e_traj = es[tmp_traj].flatten()
                new_barrier = max([e_traj[0:j+1].max()-e_traj[0:j+1].min() for j in range(1,len(tmp_traj))])
                if new_barrier < barrier[i]:
                    barrier[i] = new_barrier
                    traj[i] = traj[i_min] + [i]
                elif new_barrier == barrier[i] and len(traj[i]) > 1:
                    if e_traj[-2] < es[traj[i]].flatten()[-2]:                  
                        barrier[i] = new_barrier
                        traj[i] = traj[i_min] + [i]
    return traj[ed],barrier[ed]


