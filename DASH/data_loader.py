import prody
import numpy as np
import os
import config



class Data_Loader():
    def __init__(self,traj_info,align=True):
        #1, Load data
        pdb_file = traj_info["pdb"]
        dcd_file = traj_info["dcd"]
        sel_str = traj_info["sel_str"]
        if "step" not in traj_info.keys():
            step = 1
        else:
            step = traj_info["step"]
        if "cv_sel_str" in traj_info.keys():
            self.cv_sel_str = traj_info["cv_sel_str"]
        else:
            self.cv_sel_str = ""
        self.pdb_all = prody.parsePDB(pdb_file)
        if "pre_sel" in traj_info.keys():
            self.pdb_all = self.pdb_all.select(traj_info["pre_sel"]).copy()
        self.traj_all = prody.parseDCD(dcd_file,step=step)
        self.traj_all.setAtoms(self.pdb_all)
        #2, Select the main part
        self.pdb = self.pdb_all.select(sel_str).copy()
        pdb_traj = self.pdb_all.copy()
        pdb_traj.setCoords(self.traj_all.getCoordsets())
        self.pdb_traj = pdb_traj.select(sel_str).copy()
        #3 Align and save
        if align or (not os.path.exists("%s_aligned.dcd"%(dcd_file.split(".dcd")[0]))):
            e = prody.Ensemble()
            e.setCoords(self.pdb.getCoords())
            e.addCoordset(self.pdb_traj.getCoordsets())
            e.superpose()
            prody.writeDCD("%s_aligned.dcd"%(dcd_file.split(".dcd")[0]),e)
        self.traj = prody.parseDCD("%s_aligned.dcd"%(dcd_file.split(".dcd")[0]))
        self.pdb_traj.setCoords(self.traj.getCoordsets())

            
    def get_atomic_subset(self,sel_str="protein and name CA",align_sel_str="protein and name CA",t="atom"):
        

        pdb = self.pdb.select(align_sel_str).copy()
        resinfo2resid_dict = dict()
        tmp = set()
        sel_reses = []
        i = 0
        for a in self.pdb:
            k = (a.getChid(),a.getResname(),a.getResnum())
            if k not in tmp:
                resinfo2resid_dict[k] = i
                tmp.add(k)
                sel_reses.append(k)
                i += 1
        
        atominfo2IDofPDB = dict()
        for i,a in enumerate(self.pdb):
            si = "%s:%s:%d:%s"%(a.getChid(),a.getResname(),a.getResnum(),a.getName())
            atominfo2IDofPDB[si] = i
            
        #sub_id and coords
        if self.cv_sel_str != "":
            sel_str = "(%s) and (%s)"%(sel_str,self.cv_sel_str)
        sel_info = ["%s:%s:%d:%s"%(i.getChid(),i.getResname(),i.getResnum(),i.getName()) for i in self.pdb.select(sel_str)]
        sub_ids = []    
        resid4sub = []
        if t == "atom":
            sub_ids4pdb = []                
            for i,a in enumerate(self.pdb_all):
                si = "%s:%s:%d:%s"%(a.getChid(),a.getResname(),a.getResnum(),a.getName())
                ri = (a.getChid(),a.getResname(),a.getResnum())
                if si in sel_info:
                    sub_ids.append(i)
                    resid4sub.append(resinfo2resid_dict[ri])
                    sub_ids4pdb.append(atominfo2IDofPDB[si])
              
            sub_coords = self.pdb_traj.getCoordsets()[:,sub_ids4pdb,:]
            n_frame, n_atom, n_dim = sub_coords.shape
            xs = sub_coords.reshape(n_frame,n_atom*n_dim)
            resid4xi = [resid4sub[i] for i in range(n_atom) for j in range(n_dim)]
                    
        elif t == "dihe":
            self.id_dict = dict()
            for i,a in enumerate(self.pdb_all.select(sel_str)):
                self.id_dict[(a.getChid(),a.getResname(),a.getResnum(),a.getName())] = i
            self.dihe_ids = [[] for i in range(len(sel_reses))]
            
            for i in range(len(sel_reses)):
                chain,resname,resnum = sel_reses[i]
                if resname not in aa_chi_dict.keys():
                    continue
                
                #phi
                if i >= 1:
                    pre_chain,pre_resname,pre_resnum = sel_reses[i-1]
                    self.get_dihedral_ids([( pre_chain,pre_resname,pre_resnum,"C"),(chain,resname,resnum,"N"),(chain,resname,resnum,"CA"),(chain,resname,resnum,"C")],i)
                #psi
                if i <= len(sel_reses)-2:
                    next_chain,next_resname,next_resnum = sel_reses[i+1]
                    self.get_dihedral_ids([(chain,resname,resnum,"N"),(chain,resname,resnum,"CA"),(chain,resname,resnum,"C"),(next_chain,next_resname,next_resnum,"N")],i)
                #chis
                for n1,n2,n3,n4 in aa_chi_dict[resname]:
                    self.get_dihedral_ids([(chain,resname,resnum,n1),(chain,resname,resnum,n2),(chain,resname,resnum,n3),(chain,resname,resnum,n4)],i)
            dihes = []
            resid4xi = []
            coordset = self.traj_all.getCoordsets()
            for i in range(len(self.dihe_ids)):
                for i1,i2,i3,i4 in self.dihe_ids[i]:
                    dihes.append(calc_dihe(coordset,i1,i2,i3,i4))
                    resid4xi.append(i)
                    sub_ids.append([i1,i2,i3,i4])
                    resid4sub.append(i)
            dihes = np.array(dihes).T
            xs = dihes

        #make align info
        align_sel_info = ["%s:%s:%d:%s"%(i.getChid(),i.getResname(),i.getResnum(),i.getName()) for i in self.pdb.select(align_sel_str)]
        align_ids = []
        align_ids4pdb = []
        align_resids = []
        for i,a in enumerate(self.pdb_all):
            si = "%s:%s:%d:%s"%(a.getChid(),a.getResname(),a.getResnum(),a.getName())
            if si in align_sel_info:
                align_ids.append(i)
                align_ids4pdb.append(atominfo2IDofPDB[si])
                align_resids.append(resinfo2resid_dict[(a.getChid(),a.getResname(),a.getResnum())])

        align_coords = self.pdb_traj.getCoordsets()[:,align_ids4pdb,:]

        return pdb, sub_ids, resid4xi, resid4sub, xs,  align_ids, align_coords,align_resids
     



    def get_dihedrals(self,):
        self.id_dict = dict()
        ca_ids = []
        for i,a in enumerate(self.pdb.select("protein")):
            self.id_dict[(a.getChid(),a.getResname(),a.getResnum(),a.getName())] = i
            if a.getName() == "CA":
                ca_ids.append(i)
        sel_reses = []
        tmp_set = set()
        for i in self.pdb.select("protein"):
            k = (i.getChid(),i.getResname(),i.getResnum())
            if k not in tmp_set:
                tmp_set.add(k)
                sel_reses.append(k)
        print(sel_reses)
        self.dihe_ids = [[] for i in range(len(sel_reses))]
        for i in range(len(sel_reses)):
            chain,resname,resnum = sel_reses[i]
            #phi
            if i >= 1:
                pre_chain,pre_resname,pre_resnum = sel_reses[i-1]
                self.get_dihedral_ids([( pre_chain,pre_resname,pre_resnum,"C"),(chain,resname,resnum,"N"),(chain,resname,resnum,"CA"),(chain,resname,resnum,"C")],i)
            #psi
            if i <= len(sel_reses)-2:
                next_chain,next_resname,next_resnum = sel_reses[i+1]
                self.get_dihedral_ids([(chain,resname,resnum,"N"),(chain,resname,resnum,"CA"),(chain,resname,resnum,"C"),(next_chain,next_resname,next_resnum,"N")],i)
            #chis
            for n1,n2,n3,n4 in aa_chi_dict[resname]:
                self.get_dihedral_ids([(chain,resname,resnum,n1),(chain,resname,resnum,n2),(chain,resname,resnum,n3),(chain,resname,resnum,n4)],i)
        dihes = []
        resid4dihes = []
        coordset = self.traj_all.getCoordsets()
        for i in range(len(self.dihe_ids)):
            for i1,i2,i3,i4 in self.dihe_ids[i]:
                dihes.append(calc_dihe(coordset,i1,i2,i3,i4))
                resid4dihes.append(i)
        dihes = np.array(dihes).T

        pdb = self.pdb.select("protein and name CA").copy()
        return  self.pdb.select("protein and name CA").copy(), ca_ids, dihes, resid4dihes, self.pdb_traj.select("protein and name CA").getCoordsets()



    def get_dihedral_ids(self,ks,i):
        ids = []
        for k in ks:
            if k not in self.id_dict.keys():
                return 
            else:
                ids.append(self.id_dict[k])
        self.dihe_ids[i].append(ids)



def calc_dihe(coordset,i1,i2,i3,i4):
    A = coordset[:,i1,:].reshape(-1,3)
    B = coordset[:,i2,:].reshape(-1,3)
    C = coordset[:,i3,:].reshape(-1,3)
    D = coordset[:,i4,:].reshape(-1,3)
    a1 = B - A # n_frame * n_atom * 3
    a2 = C - B # n_frame * n_atom * 3
    a3 = D - C # n_frame * n_atom * 3
    v1 = np.cross(a1,a2) # n_frame * n_atom
    v1 = v1 / np.linalg.norm(v1,axis=-1)[:,np.newaxis] # n_frame 
    v2 =  np.cross(a2,a3) # n_frame * n_atom
    v2 = v2 / np.linalg.norm(v2,axis=-1)[:,np.newaxis] # n_frame 
    porm = np.sign((v1 * a3).sum(-1))
    rad = np.arccos(np.clip((v1*v2).sum(-1) / ((v1**2).sum(-1) * (v2**2).sum(-1))**0.5,-1.0,1.0))
    porm_fixed = []
    for i in porm.flatten():
        if i == 0:
            porm_fixed.append(1)
        else:
            porm_fixed.append(i)
    porm = np.array(porm_fixed)
    rad = rad * porm
    return np.degrees(rad)

if __name__ == "__main__":
    from traj_info import traj_infos
    dl = Data_Loader(traj_infos["sirt6"])
    pdb, sub_ids, sub_names, sub_coords, id_dict = dl.get_atomic_subset("protein and name CA")
    print(pdb, sub_ids, sub_names, sub_coords, id_dict)
                   