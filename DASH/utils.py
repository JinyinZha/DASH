import prody
import numpy as np
from numba import jit
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import copy
from tqdm import tqdm



ref_helix_coords = np.array(
[[0.733, 0.519, 5.298],
 [1.763, 0.81, 4.301],
 [3.166, 0.543, 4.881],
 [1.527, -0.045, 3.053],
 [1.646, 0.436, 1.928],
 [1.18, -1.312, 3.254],
 [0.924, -2.203, 2.126],
 [0.65, -3.626, 2.626],
 [-0.239, -1.711, 1.261],
 [-0.19, -1.815, 0.032],
 [-1.28, -1.172, 1.891],
 [-2.416, -0.661, 1.127],
 [-3.548, -0.217, 2.056],
 [-1.964, 0.529, 0.276],
 [-2.364, 0.659, -0.88],
 [-1.13, 1.391, 0.856],
 [-0.62, 2.565, 0.148],
 [0.228, 3.439, 1.077],
 [0.231, 2.129, -1.032],
 [0.179, 2.733, -2.099],
 [1.028, 1.084, -0.833],
 [1.872, 0.593, -1.919],
 [2.85, -0.462, -1.397],
 [1.02, 0.02, -3.049],
 [1.317, 0.227, -4.224],
 [-0.051, -0.684, -2.696],
 [-0.927, -1.261, -3.713],
 [-1.933, -2.219, -3.074],
 [-1.663, -0.171, -4.475],
 [-1.916, -0.296, -5.673]])

ref_anti_sheet_coords = np.array(
[[2.263, -3.795, 1.722],
 [2.493, -2.426, 2.263],
 [3.847, -1.838, 1.761],
 [1.301, -1.517, 1.921],
 [0.852, -1.504, 0.739],
 [0.818, -0.738, 2.917],
 [-0.299, 0.243, 2.748],
 [-1.421, -0.076, 3.757],
 [0.273, 1.68, 2.854],
 [0.902, 1.993, 3.888],
 [0.119, 2.532, 1.813],
 [0.683, 3.916, 1.68],
 [1.58, 3.94, 0.395],
 [-0.394, 5.011, 1.63],
 [-1.459, 4.814, 0.982],
 [-2.962, 3.559, -1.359],
 [-2.439, 2.526, -2.287],
 [-1.189, 3.006, -3.087],
 [-2.081, 1.231, -1.52],
 [-1.524, 1.324, -0.409],
 [-2.326, 0.037, -2.095],
 [-1.858, -1.269, -1.554],
 [-3.053, -2.199, -1.291],
 [-0.869, -1.949, -2.512],
 [-1.255, -2.07, -3.71],
 [0.326, -2.363, -2.072],
 [1.405, -2.992, -2.872],
 [2.699, -2.129, -2.917],
 [1.745, -4.399, -2.33],
 [1.899, -4.545, -1.102]]
)

ref_para_sheet_coords = np.array(
[[-1.439, -5.122, -1.144],
 [-0.816, -3.803, -1.013],
 [0.099, -3.509, -2.206],
 [-1.928, -2.77, -0.952],
 [-2.991, -2.97, -1.551],
 [-1.698, -1.687, -0.215],
 [-2.681, -0.613, -0.143],
 [-3.323, -0.477, 1.267],
 [-1.984, 0.681, -0.574],
 [-0.807, 0.921, -0.273],
 [-2.716, 1.492, -1.329],
 [-2.196, 2.731, -1.883],
 [-2.263, 2.692, -3.418],
 [-2.989, 3.949, -1.433],
 [-4.214, 3.989, -1.583],
 [2.464, -4.352, 2.149],
 [3.078, -3.17, 1.541],
 [3.398, -3.415, 0.06],
 [2.08, -2.021, 1.639],
 [0.938, -2.178, 1.225],
 [2.525, -0.886, 2.183],
 [1.692, 0.303, 2.346],
 [1.541, 0.665, 3.842],
 [2.42, 1.41, 1.608],
 [3.567, 1.733, 1.937],
 [1.758, 1.976, 0.6],
 [2.373, 2.987, -0.238],
 [2.367, 2.527, -1.72],
 [1.684, 4.331, -0.148],
 [0.486, 4.43, -0.415]]
)



@jit(nopython=True)
def get_d_matrix(coords):
    x1_ids = []
    x2_ids = []
    rmsds = []
    n_frame = len(coords)
    n_atom = len(coords[0])
    dim = len(coords[0][0])
    for i in range(n_frame):
        for j in range(i+1,n_frame):
            sum_d = 0
            for k in range(n_atom):
                tmp = 0
                for l in range(dim):
                    d = coords[i][k][l] - coords[j][k][l]
                    tmp += d*d
                sum_d += tmp
            rmsds.append((sum_d / n_atom)**0.5)
            x1_ids.append(i)
            x2_ids.append(j)
    x1_ids = np.array(x1_ids)  
    x2_ids = np.array(x2_ids)
    rmsds = np.array(rmsds)
    return (x1_ids,x2_ids,rmsds)



@jit(nopython=True)
def get_pwd_matrix(coords1,coords2):
    x1_ids = []
    x2_ids = []
    rmsds = []
    n_frame1 = len(coords1)
    n_frame2 = len(coords2)
    n_atom = len(coords1[0])
    dim = len(coords1[0][0])
    for i in range(n_frame1):
        for j in range(n_frame2):
            sum_d = 0
            for k in range(n_atom):
                tmp = 0
                for l in range(dim):
                    d = coords1[i][k][l] - coords2[j][k][l]
                    tmp += d*d
                sum_d += tmp
            rmsds.append((sum_d / n_atom)**0.5)
            x1_ids.append(i)
            x2_ids.append(j)
    x1_ids = np.array(x1_ids)  
    x2_ids = np.array(x2_ids)
    rmsds = np.array(rmsds)
    return (x1_ids,x2_ids,rmsds)


@jit(nopython=True)
def get_dists(xs):
    ds = []
    n = len(xs)
    dim = len(xs[0])
    for i in range(n):
        for j in range(i+1,n):
            tmp = 0
            for k in range(dim):
                d = xs[i][k] - xs[j][k]
                tmp += d*d
            ds.append(tmp**0.5)
    return ds



def get_repre_conf(z,dcd_file,pdb_file,outdir,name,bx=20,by=20,pdb_file_sel_str="all",compare_coord_sel_str="protein and name CA"):
    dcd = prody.parseDCD(dcd_file)
    pdb = prody.parsePDB(pdb_file).select(pdb_file_sel_str).copy()
    dcd.setAtoms(pdb)
    pdb.setCoords(dcd.getCoordsets())
    coords = pdb.select(compare_coord_sel_str).getCoordsets()
    zh,xb,yb=np.histogram2d(z[:,0],z[:,1],bins=(bx,by))
    x = np.array([(xb[i+1]+xb[i])/2 for i in range(bx)])
    y = np.array([(yb[i+1]+yb[i])/2 for i in range(by)])

    repre_zs = []
    repre_names = []
    for i in range(bx):
        left = max(0,i-1)
        right = min(bx,i+2)
        for j in range(by):
            up = max(0,j-1)
            down = min(by,j+2)
            if zh[i,j] == zh[left:right,up:down].max() and zh[i,j] != 0:
                ids = np.argwhere((z[:,0]>=xb[i]) & (z[:,0]<=xb[i+1]) & (z[:,1]>=yb[j]) & (z[:,1]<=yb[j+1])).flatten()
                inbox_coords = coords[ids]
                min_id = ids[np.argmin(np.linalg.norm(inbox_coords-inbox_coords.mean(0),axis=2).mean(1))]

                print(x[i],y[j],min_id,z[min_id],zh[i,j],)
                prody.writePDB("%s/%s_x%.3f_y%.3f_%dframes.pdb"%(outdir,name,x[i],y[j],zh[i,j]),dcd[min_id])
                repre_zs.append(z[min_id])
                repre_names.append("%s_x%.3f_y%.3f_%dframes.pdb"%(name,x[i],y[j],zh[i,j]))
    return np.array(repre_zs),repre_names
 
    
def combine_dcd(name,n_round,pdb_file,dcd_files,n0,n1):
    coords0 = prody.parseDCD(dcd_files[0]).getCoordsets()
    if n_round > 1:
        li0 = np.load("%d/labels_of_combined.npy"%(n_round-1))
    else:
        li0 = np.array(["0-%d"%(i) for i in range(len(coords0))])
    li1 = []
    coords1 = []
    for j in range(10):
        if os.path.exists("%d/md_%d.dcd"%(n_round,j)):
            try:
                c = prody.parseDCD("%d/md_%d.dcd"%(n_round,j)).getCoordsets()
            except:
                continue
            li1 += ["%d-%d-%d"%(n_round,j,k) for k in range(len(c))]
            coords1.append(c)
    li1 = np.array(li1)
    coords1 = np.vstack(coords1)
    select0 = np.linspace(0,len(coords0)-1,n0).astype(np.int32)
    select1 = np.linspace(0,len(coords1)-1,n1).astype(np.int32)
    coords = np.vstack([coords0[select0],coords1[select1]])
    labels = np.hstack([li0[select0],li1[select1]])
    np.save("%d/labels_of_combined.npy"%(n_round),labels)
    pdb = prody.parsePDB(pdb_file)
    pdb.setCoords(coords)
    prody.writeDCD(name,pdb)

def combine_dcd_eq(pdb_file,dcd_files,name,sel_str,n_frame=None):
    pdb = prody.parsePDB(pdb_file).select(sel_str).copy()
    coords = np.vstack([prody.parseDCD(dcd_file).getCoordsets() for dcd_file in dcd_files])
    if n_frame and n_frame < len(coords):
        sel = np.linspace(0,len(coords)-1,n_frame).astype(np.int32)
        coords = coords[sel]
    pdb.setCoords(coords)
    prody.writeDCD(name,pdb)
    return len(coords)

def get_most_fluc_regions(coord0,coords,outdir,n_region=2):
    e = prody.Ensemble()
    e.setCoords(coord0)
    e.addCoordset(coords)
    r = e.getRMSFs()
    r_mean = r.mean()
    val_id_li = [[r[i],i] for i in range(len(r))]
    val_id_li.sort(reverse=True,key=lambda x:x[0])
    recorded_ids = []
    out = []
    tmp_out = []
    for ri,i in val_id_li:
        if ri < r_mean:
            break
        if i < 5 or i > len(r)-5:
            continue
        if i in recorded_ids:
            continue
        left = []
        for j in range(i-1,0,-1):
            if r[j] < r_mean:
                break
            left = [j] + left
            recorded_ids.append(j)
        right = []
        for j in range(i+1,len(r),1):
            if r[j] < r_mean:
                break
            right = right + [j]
            recorded_ids.append(j)
        tot = left + [i] + right
        print("tot",tot,len(tot))
        if len(tot) < 10:
            lhgap = rhgap = hgap = int(np.ceil((10 - len(tot))/2))
            for j in range(tot[0]-1,tot[0]-lhgap-1,-1):
                tot = [j] + tot
            for j in range(tot[-1]+1,tot[-1]+rhgap+1,1):
                tot = tot + [j]
            print("tot fixed",tot,len(tot))
        out.append(tot)
        if len(out) == n_region:
            break
        print(out)

    if len(out) < n_region:
        n0 = len(out)
        n_split = int(np.ceil(n_region/(n0)))
        newout = []
        for n,i in enumerate(out):
            for j in np.array_split(i,n_split):
                newout.append(j.tolist())
            if len(newout) + len(out) - (n+1) >= n_region:
                out = newout+out[n+1:]
                out = out[0:n_region]
                break
    f = open("%s/resid_for_cvs.txt"%(outdir),"w")
    f.write(str(out))
    f.close()
    plt.figure()
    plt.plot(r,lw=2)
    for i in out:
        plt.plot(i,r[i],lw=3)
    plt.hlines(r_mean,0,len(r),linestyles="--",color="grey",lw=2)
    plt.savefig("%s/rmsf.jpg"%(outdir))

    return out

def cv_from_knowledge(pdb,sel_strs,outdir):
    info2id_dict = dict()
    for i,a in enumerate(pdb):
        info2id_dict["%s:%s:%s:%s"%(a.getChid(),a.getResnum(),a.getResname(),a.getResname())] = i
    
    out = [[info2id_dict["%s:%s:%s:%s"%(a.getChid(),a.getResnum(),a.getResname(),a.getResname())] for a in pdb.select(sel_str)] for sel_str in sel_strs.split("@@")]
    if len(out) == 1:
        coords = pdb.select(sel_strs).getCoords()
        km = KMeans(2).fit(coords)
        tmp = [[],[]]
        for n,i in enumerate(km.labels_):
            tmp[i].append(out[0][n])
        out = tmp

    print("id from cv_knowledge")
    f = open("%s/resid_for_cvs.txt"%(outdir),"w")
    f.write(str(out))
    f.close()
    print(out)
    print(len(out[0]),len(out[1]))
    return out


def cluster_fluc_regions(coord0,resids4sub,resids4align,coords,outdir,method="cluster",skip_ter=False,min_con=5,merge_gap=5,cutoff4pep=5):
    e = prody.Ensemble()
    e.setCoords(coord0)
    e.addCoordset(coords)
    n_frame = len(coords)
    n_ca = len(coord0)
    if n_ca < 50:
        min_con = int(n_ca/10)
    if n_ca <= 20:
        min_con  = 0
        cutoff4pep = 0
    r = e.getRMSFs()
    r_sel = np.array([r[i] for i in range(len(r)) if resids4align[i] in resids4sub])
    r_mean = r_sel.mean()

    tmp = []
    ids = []
    merged_id_groups = []
    ids2resids_dict = dict()
    for i,ri in enumerate(r):
        if ri < r_mean and not (ri > cutoff4pep and n_ca < 50):
            if len(tmp) >= min_con:
                if not (skip_ter and 0 in tmp):
                    ids += [j for j in tmp]
                    if len(merged_id_groups) > 0:
                        if tmp[0] - merged_id_groups[-1][-1] <= merge_gap:
                            merged_id_groups[-1] += [j for j in tmp]
                        else:
                            merged_id_groups.append([j for j in tmp])
                    else:
                        merged_id_groups.append([j for j in tmp])
            tmp = []
        else:
            if resids4align[i] in resids4sub:
                tmp.append(i)
                ids2resids_dict[i] = resids4align[i]
    if len(tmp) >= min_con:
        if not (skip_ter and len(r)-1 in tmp):
            ids += [j for j in tmp]
            if len(merged_id_groups) > 0:
                if tmp[0] - merged_id_groups[-1][-1] <= merge_gap:
                    merged_id_groups[-1] += [j for j in tmp]
                else:
                    merged_id_groups.append([j for j in tmp])
            else:
                merged_id_groups.append([j for j in tmp])

    if len(ids) == 0:
        ids = [i for i,ri in enumerate(r) if ri >= r_mean]
    all_res = np.array(ids)

    if method == "cluster":
        km = KMeans(2).fit(coord0[all_res])
        out = [[],[]]
        for n,i in enumerate(km.labels_):
            out[i].append(int(all_res[n]))
    elif method == "top":
        if len(merged_id_groups) < 2:
            print("Too few continous regions with large fluctuation. Swithc to msr=0 or msr=1")
            km = KMeans(2).fit(coord0[all_res])
            out = [[],[]]
            for n,i in enumerate(km.labels_):
                out[i].append(int(all_res[n]))
        else:
            idgs_fluc = [[r[idg].sum(),idg] for idg in merged_id_groups]
            idgs_fluc.sort(key=lambda x:x[0],reverse=True)
            out = [idgs_fluc[0][1],idgs_fluc[1][1]]
    elif method == "local_linear":
        if len(merged_id_groups) < 2:
            print("Too few continous regions with large fluctuation. Switch to msr=0 or msr=1")
            km = KMeans(2).fit(coord0[all_res])
            out = [[],[]]
            for n,i in enumerate(km.labels_):
                out[i].append(int(all_res[n]))
        else:
            ys = coords.reshape(n_frame,-1)
            idg_lr_scores = []
            for idg in merged_id_groups:
                xs = coords[:,idg,:].reshape(n_frame,-1)
                lr = LinearRegression().fit(xs,ys)
                idg_lr_scores.append([idg,lr.score(xs,ys)])
            idg_lr_scores.sort(key=lambda x:x[1],reverse=True)
            f = open("%s/idg_lr_scores"%(outdir),"w")
            f.write(str(idg_lr_scores))
            f.close()
            out = [idg_lr_scores[0][0],idg_lr_scores[1][0]]


    f = open("%s/resid_for_cvs.txt"%(outdir),"w")
    f.write(str(out))
    f.close()
    plt.figure()
    plt.plot(r,lw=2)
    for i in out:
        plt.plot(i,r[i],lw=3)
    plt.hlines(r_mean,0,len(r),linestyles="--",color="grey",lw=2)
    plt.savefig("%s/rmsf.jpg"%(outdir))

    out2 = [[ids2resids_dict[i] for i in o] for o in out]
    print(out2)
    print(len(out2[0]),len(out2[1]))
    return out2


def get_resID_of_sel(pdb_file,sel_str):
    pdb = prody.parsePDB(pdb_file)
    #mk_dict
    resInfo2resID_dict = dict()
    res_id = 0
    for a in pdb:
        res_info = "%s:%s:%d"%(a.getChid(),a.getResname(),a.getResnum())
        if res_info not in resInfo2resID_dict.keys():
            resInfo2resID_dict[res_info] = res_id
            res_id += 1
    #get sel ID
    sel_res_ids = []
    for a in pdb.select(sel_str):
        res_info = "%s:%s:%d"%(a.getChid(),a.getResname(),a.getResnum())
        res_id = resInfo2resID_dict[res_info]
        if res_id not in sel_res_ids:
            sel_res_ids.append(res_id)
    return sel_res_ids

def get_ss_score(pdb,sel_str,get_max_score_only=False):
    pdb = pdb.select("protein").copy()
    #1, get the residue selections
    sel_res_set = set()
    for a in pdb.select(sel_str):
        c = a.getChid()
        if c == "":
            c = "_"
        k = "%s:%s:%d"%(c,a.getResname(),a.getResnum())
        if k not in sel_res_set:
            sel_res_set.add(k)
    #2, get coordinates and sel_ids
    all_res_set = set()
    coordset = []
    res_i = 0
    sel_res_is = []
    for a in pdb:
        c = a.getChid()
        if c == "":
            c = "_"
        k = "%s:%s:%d"%(c,a.getResname(),a.getResnum())
        if k not in all_res_set:
            all_res_set.add(k)
            if a.getResname() == "GLY":
                atom_names = ["N","CA","HA2","C","O"]
            else:
                atom_names = ["N","CA","CB","C","O"]
            
            coordset.append(np.vstack([pdb.select("chain %s and resname %s and resnum %d and name %s"%(c,a.getResname(),a.getResnum(),an)).getCoords().flatten() for an in atom_names]))
            if k in sel_res_set:
                sel_res_is.append(res_i)
            res_i += 1
    if get_max_score_only:
        return len(sel_res_is)
    coordset = np.array(coordset)
    sel_coordset = coordset[sel_res_is]
    ss_scores = [[] for i in range(len(sel_res_is))]
    #3, helix score
    helix_coordset = np.array([np.vstack(sel_coordset[i:i+6]) for i in range(len(sel_coordset)-5)])
    e = prody.Ensemble()
    e.setCoords(ref_helix_coords)
    e.addCoordset(helix_coordset)
    e.superpose()
    rmsds = e.getRMSDs()
    helix_scores = ((1-(rmsds/0.8)**8)/(1-(rmsds/0.8)**12))
    for i in range(len(sel_coordset)-5):
        for j in range(6):
            ss_scores[i+j].append(helix_scores[i])
    #4, sheet_score
    for i in range(len(sel_res_is)-2):
        sheet_coordset = []
        for j in range(len(coordset)-2):
            if j - sel_res_is[i+2] > 1:
                sheet_coordset.append(np.vstack([coordset[sel_res_is[i]],coordset[sel_res_is[i+1]],coordset[sel_res_is[i+2]],
                                             coordset[j],coordset[j+1],coordset[j+2]]))
            elif sel_res_is[i] - (j+2) > 1:
                sheet_coordset.append(np.vstack([coordset[j],coordset[j+1],coordset[j+2],
                                                 coordset[sel_res_is[i]],coordset[sel_res_is[i+1]],coordset[sel_res_is[i+2]]]))
        sheet_coordset = np.array(sheet_coordset)
        e1 = prody.Ensemble()
        e1.setCoords(ref_para_sheet_coords)
        e1.addCoordset(sheet_coordset)
        e1.superpose()
        e2 = prody.Ensemble()
        e2.setCoords(ref_anti_sheet_coords)
        e2.addCoordset(sheet_coordset)
        e2.superpose()
        rmsd = min(e1.getRMSDs().min(),e2.getRMSDs().min())
        sheet_score = ((1-(rmsd/0.8)**8)/(1-(rmsd/0.8)**12))
        for j in range(3):
            ss_scores[i+j].append(sheet_score)
    #5, get ss score
    return sum([max(i) for i in ss_scores])

if __name__ == "__main__":
    #get_ss_score(prody.parsePDB("/home/jyzha/project/enhanced_sampling/sample/ssgMD_SplitContrastMAE_CA_kras4b_idr_rt10_nsd20_ecsol_msr0_pct20l10_resnum160to185@@resnum160to185/6/tmp/draged11.pdb"),"resnum 160 to 185",)
    #print("-"*50)
    #get_ss_score(prody.parsePDB("/home/jyzha/project/enhanced_sampling/sample/ssgMD_SplitContrastMAE_CA_kras4b_idr_rt10_nsd20_ecsol_msr0_pct20l10_resnum160to185@@resnum160to185/0/tmp/draged9.pdb"),"resnum 160 to 185",)
    
    from traj_info import traj_infos
    folder = ""
    k = ""
    os.chdir(folder)
    z = np.load("z.npy")
    dcd_file = traj_infos[k]["dcd"]
    pdb_file = traj_infos[k]["pdb"]
    outdir = "./"
    name = k
       
    get_repre_conf(z,dcd_file,pdb_file,outdir,name,bx=20,by=20,pdb_file_sel_str="all",compare_coord_sel_str="protein and name CA")