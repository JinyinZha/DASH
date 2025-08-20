#!/usr/bin/env python3 

from traj_info import traj_infos
import os,sys
import argparse
import time
import datetime
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
import prody
from utils import combine_dcd,combine_dcd_eq,get_repre_conf,get_ss_score,get_resID_of_sel
import numpy as np
from openmm import *
from openmm.app import *
from simtk.unit import *
from md_engine import MD_Engine,Energy_Cal
import config

adfr_prepare = config.adfr_prepare
adfr_autosite = config.adfr_autosite

def get_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("-i","--input_file", type=str, help="Path to input file")
    base_args, remaining_argv = base_parser.parse_known_args()
    if base_args.input_file:
        f = open(base_args.input_file)
        fli = f.read().split("\n")
        f.close()
        for line in fli:
            tmp = line.strip().split("#")[0].split()
            if len(tmp) == 2:
                sys.argv += ["--"+tmp[0],tmp[1]]
    parser = argparse.ArgumentParser(parents=[base_parser])
    parser.add_argument("-g","--guide_type",help="Method to select seed structures. bp (Binding Potential) or e (Potential Energy) or ebp (Mix)", type=str,choices=["e","bp","ebp",],default="ebp")
    parser.add_argument("-ss","--seed_sample_type",help="Type of seeded sampling. get_sim or MD",type=str,choices=["get_sim","MD"],default="MD")
    parser.add_argument("-m","--method_name",help="Method of Dimention Reduction",type=str)
    parser.add_argument("-t","--traj_info",help="Traj Info",type=str)
    parser.add_argument("-hi","--high_dim_type",help="High Dimension Type",type=str)
    parser.add_argument("-c","--cuda_id",help="CUDA ID",type=str)
    parser.add_argument("-r","--res_id_sel",help="Res ID sel",type=str)
    parser.add_argument("-ntr","--n_train",help="Number of paralle trainings",type=int,default=1)
    parser.add_argument("-np","--n_proc",help="Number of paralle processes",type=int,default=1)
    parser.add_argument("-temp","--T",help="temperature of simulation",type=float,default=300.0)
    parser.add_argument("-rt","--round_time",help="Time of ns of each round. Only intergers are accepted.",type=int,default=10)
    parser.add_argument("-mr","--max_round",help="Maximum iteration rounds",type=int,default=30)
    parser.add_argument("-ssc","--dssp_converge_cutoff",help="Cutoff of ratio of secodary structures broken",type=float,default=0.1)
    parser.add_argument("-cvk","--cv_knowledge", help="Assign CV using prody-style selection str. Seperate CVs using @@. For example, 'chain A and resnum 0 to 20@@ chain A and resnum 120 to 150'", type=str, default=None)
    parser.add_argument("-tepoch","--train_epoch",help="epoch of cv training",type=int,default=100)
    parser.add_argument("-r0","--r0type",help="The r0 type of each seeded constrained sampling. old or out",type=str,default="out",choices=["old","out"])
    parser.add_argument("-ni","--n_iter_for_cv",help="Use last N iterations for CV selection",type=int,default=-1)
    parser.add_argument("-ri","--ratio_traj_for_cv",help="Use last n%% iterations for CV selection",type=float,default=-1.0)
    parser.add_argument("-nsd","--n_seed_drag",help="Number of seed structures for dragging",type=int,default=20)
    parser.add_argument("-dxys","--dxy_scale",help="scale the movement of each drag",type=float,default=1.0)
    parser.add_argument("-ns","--n_seed",help="Number of seed structures for MD each round",type=int,default=4)
    parser.add_argument("-nso","--n_seed_old",help="Number of seed structures by old local minima",type=int,default=0)
    parser.add_argument("-rc","--rmsd_cutoff",help="RMSD to stop out dragging",type=float,default=1.0)
    parser.add_argument("-rcl","--rmsd_cutoff_least",help="least RMSD to accept a seed",type=float,default=0.5)
    parser.add_argument("-fcp0","--fc_pull0",help="Initial force to drag protein",type=float,default=1000.0)
    parser.add_argument("-dfcp0","--dfc_pull0",help="Delta force to add to drag protein",type=float,default=1000.0)
    parser.add_argument("-of","--other_force",help="Add other forces in dragged simulation. e.g. helix$$resnum 1 to 10@@helix$$resnum 50 to 80",type=str,default="")
    parser.add_argument("-rns","--require_num_seed",help="1 for require num seed dragged > 2 * n_seed for MD",type=int,default=1)
    parser.add_argument("-nstm","--n_sub_try_max",help="Max try when not dragged to required CV",type=int,default=3)
    parser.add_argument("-rcmax","--rmsd_cutoff_max",help="Max of RMSD to stop out dragging",type=float,default=2.0)
    parser.add_argument("-rcmin","--rmsd_cutoff_min",help="Max of RMSD to stop out dragging",type=float,default=1.0)
    parser.add_argument("-rcp","--rmsd_cutoff_pct",help="RMSD to stop out dragging calculated from percentile of now RMSDs ",type=float,default=None)
    parser.add_argument("-rclp","--rmsd_cutoff_least_pct",help="least RMSD to accept a seed calculated from percentile of now RMSDs ",type=float,default=None)
    parser.add_argument("-msr","--msr",help="Methods to split residues. 0 : cluster  1 : cluster & skip ter   2 : top    3 : top & skip ter   4:most related   5:most related&skip ter  Default=0",type=int,default=0,choices=[0,1,2,3,4,5])
    parser.add_argument("-ectp","--ec_type",help="The method to calculate protein energy. implicit or vacuum or sol",type=str,default="sol",choices=["implicit","vacuum","sol"])
    parser.add_argument("-rs","--restart",help=" Round to restart a run.",type=int,default=None)
    parser.add_argument("-subrs","--sub_restart",help="Subround to restart in the restarted run. If all seeded MD is over, you can use a value larger than n_seed.",type=int,default=None)   
    parser.add_argument("-sft","--skip_first_train",help="Set to 1 to skip first train of iterative seeded sampling.",type=int,default=0,choices=[0,1])
    parser.add_argument("-zr","--zrange",help="range to perform final seeded sampling",type=str,default=None)
    parser.add_argument("-ftp","--final_type",help="Type of final seeded sampling",type=str,choices=["US","CMD","no"],default="no")
    parser.add_argument("-nfr","--n_final_replica",help="Number of replica in final seeded sampling",type=int,default=1)
    parser.add_argument("-rsf","--restart_final",help="Restart the final run.",type=int,default=None)
    args = parser.parse_args()
    return args

def get_traj_info(file):
    d = dict()
    f = open(file)
    fli = f.read().split("\n")
    f.close()
    for line in fli:
        tmp = line.strip().split("#")[0].split()
        if len(tmp) >= 2:
            d[tmp[0]] = " ".join(tmp[1:])
            if tmp[0] == "step":
                d[tmp[0]] = int(tmp[1])
            elif os.path.exists(tmp[1]):
                d[tmp[0]] = os.path.abspath(tmp[1])
    return d
        

def ini_checks(args):
    if args.n_seed_old > args.n_seed:
        print("nso should be smaller than ns!")
        exit()
    elif args.n_seed_old > args.n_seed/2:
        print("Warning! More than half of the seed is seed from local minima!")
    if args.sub_restart != None:
        if args.sub_restart < 0:
            print("Invalid restart parameters!")
            exit()

def mk_workdir(args,traj_info):
    workdir = "%sg%s_%s_%s_%s_rt%d_nsd%d_fcp0%d_ec%s_msr%d"%(args.guide_type,args.seed_sample_type,args.method_name,args.high_dim_type,traj_info["name"],args.round_time,args.n_seed_drag,args.fc_pull0,args.ec_type,args.msr) 
    if args.rmsd_cutoff_pct:
        workdir += "_pct%dl%d"%(args.rmsd_cutoff_pct,args.rmsd_cutoff_least_pct)
    else:
        workdir += "_r%.1fl%.1f"%(args.rmsd_cutoff,args.rmsd_cutoff_least)
    if args.res_id_sel:
        workdir += "_" + args.res_id_sel.replace(" ","_").replace('"','')
    elif args.cv_knowledge:
        workdir += "_" + args.cv_knowledge.replace(" ","")
    if args.other_force:
        workdir += "_of" + args.other_force.replace(" ","")
    print(workdir)
    print(args)
    os.system("mkdir %s"%(workdir))
    os.chdir(workdir)
    f = open("args.txt","w")
    f.write(str(args))
    f.close()

def get_rmsd_cutoff(args,net,rmsd_file):
    if args.rmsd_cutoff_pct and args.rmsd_cutoff_least_pct:
        try:
            rmsd_cutoff = np.percentile(net.net.rmsds,args.rmsd_cutoff_pct)
            rmsd_cutoff_least = np.percentile(net.net.rmsds,args.rmsd_cutoff_least_pct)
        except:
            rmsds = np.load(rmsd_file)[2]
            rmsd_cutoff = np.percentile(rmsds,args.rmsd_cutoff_pct)
            rmsd_cutoff_least = np.percentile(rmsds,args.rmsd_cutoff_least_pct)
        print("rmsd cutoff%f(%d%%),%f(%d%%)"%(rmsd_cutoff,args.rmsd_cutoff_pct,rmsd_cutoff_least,args.rmsd_cutoff_least_pct))
        rmsd_cutoff = max(min(args.rmsd_cutoff_max,rmsd_cutoff),args.rmsd_cutoff_min)
        if rmsd_cutoff < rmsd_cutoff_least:
            rmsd_cutoff_least = 0.8 * rmsd_cutoff
            print("%f,%f"%(rmsd_cutoff,rmsd_cutoff_least))
        print("fixed rmsd cutoff%f(%d%%),%f(%d%%)"%(rmsd_cutoff,args.rmsd_cutoff_pct,rmsd_cutoff_least,args.rmsd_cutoff_least_pct))
    else:
        print("direct rmsd cut")
        rmsd_cutoff = args.rmsd_cutoff
        rmsd_cutoff_least = args.rmsd_cutoff_least
    return rmsd_cutoff,rmsd_cutoff_least

def get_further_force(net,r0s,args,traj_info):                
    ff = [{"name":"CV",
            "cv_file":net.fs[i],
            "dimension_id":i,
            "r0":r0s[i],
            "f0":args.fc_pull0,
            "dx2":net.unit_dxy[i]*net.unit_dxy[i]} for i in range(len(net.fs))]
    if args.other_force != "":
        print(args.other_force)
        for f_info in args.other_force.split("@@"):
            if f_info == "":
                continue    
            f_type,sel_str = f_info.split("~~")
            res_id_sel = get_resID_of_sel(traj_info["pdb"],sel_str)
            helix_count = 0
            if f_type == "helix":
                ff.append({"name":"helix_force",
                           "helix_id":helix_count,
                          "fc_pull":1000,
                          "res_id_sel":res_id_sel})
                helix_count += 1
    return ff




def drag(traj_info,input_info,simu_info,cv_info,args,r0s,net_drs,tmpz,coords_ref,
         i,n,rmsd_cutoff,rmsd_cutoff_least,ddxy,repre_zs,repre_names,n_try_max = 10,n_fixed_try=-1):
    fc_pull0 = args.fc_pull0
    d_fc_pull0 = args.dfc_pull0
    n_sub_try_max =  args.n_sub_try_max
    print("window",n,r0s,fc_pull0)
    #create scorer for dragged structures
    if args.ec_type != "sol":
        ec = Energy_Cal(traj_info["implicit_prmtop"],traj_info["pdb"],sel_str=traj_info["pro_sel_str"],t=args.ec_type)
    else:
        if args.guide_type == "mmgbsa":
            ec = Energy_Cal(traj_info["implicit_prmtop"],traj_info["pdb"],sel_str=traj_info["pro_sel_str"],t="implicit")
            ec_r = Energy_Cal(traj_info["implicit_rec_prmtop"],traj_info["pdb"],sel_str=traj_info["pro_rec_sel_str"],t="implicit")
            ec_l = Energy_Cal(traj_info["implicit_lig_prmtop"],traj_info["pdb"],sel_str=traj_info["pro_lig_sel_str"],t="implicit")
            ec_all = Energy_Cal(traj_info["prmtop"],traj_info["pdb"],sel_str="all",t=args.ec_type)
        else:
            ec = Energy_Cal(traj_info["prmtop"],traj_info["pdb"],sel_str="all",t=args.ec_type)
    #start dragging
    md = MD_Engine(input_info)
    fail_drag = False
    n_try = 0
    n_sub_try = 0
    inter_r0s = []
    tmp_pdb = prody.parsePDB("%d/seed%d.pdb"%(i-1,n))
    while n_try < n_try_max:
        if n_fixed_try > 0 and n_try >= n_fixed_try:
            break

        if n_sub_try >= n_sub_try_max:
            sim.saveCheckpoint("%d/tmp/%d.chk"%(i-1,n))
            esb = prody.Ensemble()
            tmp_pdb.setCoords(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
            esb.setCoords(tmp_pdb.select("(%s) and protein and name CA"%(traj_info["sel_str"])).getCoords())
            esb.addCoordset(coords_ref)
            esb.superpose()
            rmsd = esb.getRMSDs().min()
            break

        if rmsd_cutoff == 0 or rmsd_cutoff_least == 0 or fc_pull0 == 0:
            tmp_ffs = [ff for ff in simu_info["further_force"] if ff["name"] != "CV"]
            simu_info["further_force"] = tmp_ffs
            try:
                sim,ffs = md.get_sim(simu_info,pdb_file="%d/seed%d.pdb"%(i-1,n))
                sim.minimizeEnergy()
            except Exception as e:
                print(n,e)
                fail_drag = True
                break
            if len(tmp_ffs) > 0:
                print("run further cv for 49000 steps")
                print(simu_info["further_force"])
                try:
                    sim.step(49000)
                except Exception as e:
                    print(n,e)
                    es = [999999]
                    pots=[999999]
                if args.guide_type == "e":
                    es = []
                    for ii in range(50):
                        sim.step(20)
                        es.append(ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)))
                elif args.guide_type == "ebp":
                    pots = []
                    for ii in range(50):
                        sim.step(20)
                        pots.append(ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)))
                elif args.guide_type == "ss":
                    es = []
                    for ii in range(50):
                        sim.step(20)  
                        tmp_pdb.setCoords(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                        es.append(-get_ss_score(tmp_pdb.select("protein").copy(),traj_info["ss_sel_str"]))
                elif args.guide_type == "mmgbsa":
                    pots = []
                    es = []
                    for ii in range(50):
                        sim.step(20)
                        e_tot = ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                        e_r = ec_r.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                        e_l = ec_l.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                        es.append(e_tot - e_r - e_l)
                        pots.append(ec_all.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)))
                else:
                    sim.step(1000)
            else:
                sim.step(0)           
                if args.guide_type == "e":
                    es = [ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))]
                elif args.guide_type == "ebp":
                    pots = [ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))]
                elif args.guide_type == "ss":
                    tmp_pdb.setCoords(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    es = [-get_ss_score(tmp_pdb.select("protein").copy(),traj_info["ss_sel_str"])]
                elif args.guide_type == "mmgbsa":
                    e_tot = ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    e_r = ec_r.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    e_l = ec_l.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    es = [e_tot - e_r - e_l] 
                    pots = [ec_all.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))]
                else:
                    pass
            sim.saveCheckpoint("%d/tmp/%d.chk"%(i-1,n))
            rmsd = rmsd_cutoff_least + 1
            break



        if n_try == 0:
            try:
                sim,ffs = md.get_sim(simu_info,pdb_file="%d/seed%d.pdb"%(i-1,n))
                sim.minimizeEnergy()
                for f_dict in ffs:
                    if f_dict["name"] == "CV": 
                        sim.context.setParameter(f_dict["f0_name"],fc_pull0/f_dict["dx2"]*f_dict["f_unit"])
            except:
                try:
                    pdb_file = repre_names[np.argmin(np.linalg.norm(repre_zs-r0s,axis=1))]
                    print("window %d use closest local minimum %s"%(n,pdb_file))
                    sim,fns = md.get_sim(simu_info,pdb_file="%d/%s"%(i-1,pdb_file),fc_pull0 = 0,fc_pull1=0,get_sim_only=True)
                    sim.minimizeEnergy()
                    for f_dict in ffs:
                        if f_dict["name"] == "CV": 
                            sim.context.setParameter(f_dict["f0_name"],fc_pull0/f_dict["dx2"]*f_dict["f_unit"])
                except:
                    try:
                        print("window %d use ini pdb file"%(n))
                        sim,fns = md.get_sim(simu_info,pdb_file=traj_info["pdb"],fc_pull0 = 0,fc_pull1=0,get_sim_only=True)
                        sim.minimizeEnergy()
                        for f_dict in ffs:
                            if f_dict["name"] == "CV": 
                                sim.context.setParameter(f_dict["f0_name"],fc_pull0/f_dict["dx2"]*f_dict["f_unit"])
                    except Exception as e:
                        print(n,e)
                        fail_drag = True
                        break
        
        

        ######


        try:
            sim.step(4000)
            if args.guide_type == "e":
                es = []
                for ii in range(50):
                    sim.step(20)
                    es.append(ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)))
            elif args.guide_type == "ebp":
                pots = []
                for ii in range(50):
                    sim.step(20)
                    pots.append(ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)))
            elif args.guide_type == "ss":
                es = []
                for ii in range(50):
                    sim.step(20)  
                    tmp_pdb.setCoords(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    es.append(-get_ss_score(tmp_pdb.select("protein").copy(),traj_info["ss_sel_str"]))
            elif args.guide_type == "mmgbsa":
                pots = []
                es = []
                for ii in range(50):
                    sim.step(20)
                    e_tot = ec.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    e_r = ec_r.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    e_l = ec_l.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
                    es.append(e_tot - e_r - e_l)
                    pots.append(ec_all.get_energy(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom)))
            else:
                sim.step(1000)
        except Exception as e:
            print(n,e)
            fail_drag = True
            break
            #Check position
        for f_dict in ffs:
            if f_dict["name"] == "CV" and f_dict["dimension_id"] == 0:
                r1 = sim.context.getSystem().getForces()[f_dict["f_id"]].getCollectiveVariableValues(sim.context)[0]
                break
        if abs(r1-r0s[0]) > net_drs[0]/(2*cv_info["bins_for_margin"]):
            print("CV1 does not match! Required %f Now %f"%(r0s[0],r1))
            print(abs(r1-r0s[0]),net_drs[0]/(2*cv_info["bins_for_margin"]),net_drs[0],cv_info["bins_for_margin"])
            fc_pull0 += d_fc_pull0
            for f_dict in ffs:
                if f_dict["name"] == "CV": 
                    sim.context.setParameter(f_dict["f0_name"],fc_pull0/f_dict["dx2"]*f_dict["f_unit"])
            print("window",n,r0s,fc_pull0,)
            n_sub_try += 1
            print("n_sub_try",n_sub_try)
            continue
        for f_dict in ffs:
            if f_dict["name"] == "CV" and f_dict["dimension_id"] == 1:
                r2 = sim.context.getSystem().getForces()[f_dict["f_id"]].getCollectiveVariableValues(sim.context)[0]
                break
        if abs(r2-r0s[1]) > net_drs[1]/(2*cv_info["bins_for_margin"]):
            print("CV2 does not match! Required %f Now %f"%(r0s[1],r2))
            print(abs(r2-r0s[1]),net_drs[1]/(2*cv_info["bins_for_margin"]),net_drs[1],cv_info["bins_for_margin"])
            fc_pull0 += d_fc_pull0
            for f_dict in ffs:
                if f_dict["name"] == "CV": 
                    sim.context.setParameter(f_dict["f0_name"],fc_pull0/f_dict["dx2"]*f_dict["f_unit"])
            print("window",n,r0s,fc_pull0)
            n_sub_try += 1
            print("n_sub_try",n_sub_try)
            continue
        n_sub_try = 0
        #check if too close to old
        sim.saveCheckpoint("%d/tmp/%d.chk"%(i-1,n))
        too_close_to_old = False
        for tmpdr in np.abs(r0s - tmpz):
            if tmpdr[0] < net_drs[0]/(2*cv_info["bins_for_margin"]) and tmpdr[1] < net_drs[1]/(2*cv_info["bins_for_margin"]):
                too_close_to_old = True
                break
        tmpz = np.vstack([tmpz,r0s])
        if too_close_to_old:
            inter_r0s.append(np.copy(r0s))
            r0s += ddxy
            for f_dict in ffs:
                if f_dict["name"] == "CV": 
                    sim.context.setParameter("r0%d"%(f_dict["dimension_id"]),r0s[f_dict["dimension_id"]])
            continue
        #calculate RMSD
        n_try += 1
        esb = prody.Ensemble()
        tmp_pdb.setCoords(sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(angstrom))
        esb.setCoords(tmp_pdb.select("%s and protein and name CA"%(traj_info["sel_str"])).getCoords())
        esb.addCoordset(coords_ref)
        esb.superpose()
        if args.guide_type=="mmgbsa":
            esb_lig = prody.Ensemble()
            esb_lig.setCoords(tmp_pdb.select("%s and protein and name CA and %s"%(traj_info["sel_str"],traj_info["pro_lig_sel_str"])).getCoords())
            tmp_pdb2 = tmp_pdb.select("%s and protein and name CA"%(traj_info["sel_str"])).copy()
            tmp_pdb2.setCoords(esb.getCoordsets())
            esb_lig.addCoordset(tmp_pdb2.select("%s and protein and name CA and %s"%(traj_info["sel_str"],traj_info["pro_lig_sel_str"])).getCoordsets())
            rmsd = esb_lig.getRMSDs().min()
        else:
            rmsd = esb.getRMSDs().min()
        print("rmsd",rmsd)
        if rmsd > rmsd_cutoff:
            break
        else:
            inter_r0s.append(np.copy(r0s))
            r0s += ddxy
            for iii in range(len(r0s)):
                sim.context.setParameter("r0%d"%(iii),r0s[iii])
    if fail_drag:
        return []
    if n_fixed_try <= 0 and rmsd <= rmsd_cutoff_least:
        return []
    if args.guide_type in ["bp", "ebp"]:
        if "pro_sel_str" in traj_info.keys():
            prody.writePDB("%d/tmp/draged%d_pro.pdb"%(i-1,n),tmp_pdb.select("protein and %s"%(traj_info["pro_sel_str"])).copy())
        else:
            prody.writePDB("%d/tmp/draged%d_pro.pdb"%(i-1,n),tmp_pdb.select("protein").copy())
        os.system("cd %d/tmp/; %s -r draged%d_pro.pdb -o draged%d_pro.pdbqt"%(i-1,adfr_prepare,n,n))
        os.system("cd %d/tmp/; %s -r draged%d_pro.pdbqt"%(i-1,adfr_autosite,n))
        if os.path.exists("%d/tmp/draged%d_pro/draged%d_pro_summary.csv"%(i-1,n,n)):
            f = open("%d/tmp/draged%d_pro/draged%d_pro_summary.csv"%(i-1,n,n))
            fli = f.read().split("\n")
            f.close()
            es = [float(line.split(",")[1]) for line in fli[1:] if line != ""]
            es.sort()
            es = es[0:int(len(es)*0.5)]
            #es = es[0:3]
        else:
            return []

    prody.writePDB("%d/tmp/draged%d.pdb"%(i-1,n),tmp_pdb)

    if args.guide_type in ["ebp","mmgbsa"]:
        return np.mean(es),np.mean(pots),r0s,inter_r0s
    else:
        return np.mean(es),r0s,inter_r0s


def run_from_chk(input_info,i,simu_info,n,sim_id,net_dxys,args,seed_sample_type,n_seed,n_round_per_seed=1,pdb_file=None):
    print("run from chk",i,n)
    if args.restart != None and args.sub_restart != None and i == args.restart:
        if n < args.sub_restart:
            if os.path.exists("%d/md_%d.dcd"%(i,n)):
                return "%d/md_%d.dcd"%(i,n)
            else:
                return
    md = MD_Engine(input_info)
    try:
        if seed_sample_type == "get_sim":
            sim,fns = md.get_sim(simu_info,pdb_file="%d/lowe_seed%d.pdb"%(i-1,n),fc_pull0 = 2000.0,fc_pull1=500.0,get_sim_only=True)
            sim.loadCheckpoint("%d/tmp/%d.chk"%(i-1,sim_id))
            sim.reporters.append(DCDReporter("%d/md_%d.dcd"%(i,n),200))
            md.get_sim_for_sim(sim,500000,100,net_dxys[n],50000,200)#2nd para is 500000
        elif seed_sample_type == "MD":
            if pdb_file == None:
                pdb_file = "%d/tmp/draged%d.pdb"%(i-1,sim_id)
            for ii in range(n_round_per_seed):
                simu_info["n_steps"] = int(np.ceil(500000 / n_seed / n_round_per_seed * args.round_time))
                simu_info["output_dcd_freq"] = 100 * args.round_time
                simu_info["output_dcd_filename"] = "%d/md_%d_%d.dcd"%(i,n,ii)
                md.unbiased_simulation(simu_info,pdb_file)
            if n_round_per_seed > 1:
                combine_dcd_eq(pdb_file,["%d/md_%d_%d.dcd"%(i,n,ii) for ii in range(n_round_per_seed)],"%d/md_%d.dcd"%(i,n),"all",int(5000/n_round_per_seed))
            else:
                os.system("mv %d/md_%d_%d.dcd %d/md_%d.dcd"%(i,n,ii,i,n))
            for ii in range(n_round_per_seed):
                if os.path.exists("%d/md_%d_%d.dcd"%(i,n,ii)):
                    os.system("rm %d/md_%d_%d.dcd"%(i,n,ii))
            
    except Exception as e:
        print(e)
        if os.path.exists("%d/md_%d.dcd"%(i,n)):
            os.system("rm %d/md_%d.dcd"%(i,n))
        for ii in range(n_round_per_seed):
            if os.path.exists("%d/md_%d_%d.dcd"%(i,n,ii)):
                os.system("rm %d/md_%d_%d.dcd"%(i,n,ii))
        return
    return "%d/md_%d.dcd"%(i,n)


if __name__ == "__main__":
    #1, get args
    args = get_args()
    #2, set some basic environment variables. It is done here to control the GPU used by toech
    if args.cuda_id in ["0","1","2","3"]:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    import torch
    ctx = torch.multiprocessing.get_context("spawn")
    from cv import CV
    from converge import dssp_converge
    #3, checks of args.
    ini_checks(args)
    #4, create workdir
    traj_info = get_traj_info(args.traj_info)
    if args.res_id_sel:
        traj_info["res_id_sel"] = args.res_id_sel
    if args.cv_knowledge:
        traj_info["cv_knowledge"] = args.cv_knowledge
    mk_workdir(args,traj_info)
    #5, start initial simulation
    print("Step 0: Unbiased Simulation.\n%s"%(datetime.datetime.now().strftime("%Y%m%d%H%M")))
    os.system("mkdir 0")
    input_info = {"prmtop":traj_info["prmtop"],
                  "inpcrd":traj_info["inpcrd"],}     
    md = MD_Engine(input_info)
    simu_info = { "seed":111,
                "n_steps":500000 * args.round_time,
                "equil_steps":500000,
                "T":args.T,
                "output_state_filename":"0/md.log",
                "output_state_freq":10000,
                "output_chk_filename":"0/md.chk",
                "output_chk_freq":10000,
                "output_dcd_filename":"0/md.dcd",
                "output_dcd_freq":50 * args.round_time}
    if args.restart == None:
        md.unbiased_simulation(simu_info)###
    #6, initial training
    print("Step 0: Training.\n%s"%(datetime.datetime.now().strftime("%Y%m%d%H%M")))
    
    net = CV("0")
    traj_info["dcd"] = "0/md.dcd"
    traj_info["step"] = 1
    n_seed = args.n_seed
    rerun = 0
    cv_info = {"traj_info":traj_info,
            "method_name": args.method_name,
            "n_train": args.n_train, 
            "n_proc":args.n_proc,
            "epoch":args.train_epoch,
            "batch_size":512,
            "max_nbatch":400,
            "random_seed":111,
            "lr":0.005,
            "n_seed":args.n_seed_drag,
            "schedular_step_size":20,
            "schedular_gamma":0.7,
            "high_dim_type":args.high_dim_type,
            "cluster_margin":True,
            "bins_for_margin":20,
            "r0type":args.r0type,
            "msr":args.msr,}
    if args.restart != None:
        if args.restart == 0:
            net.train(cv_info)
            st_round = 1
            args.sub_restart = None
        else:
            net = CV(str(args.restart-1))
            net.load_net()
            traj_info["dcd"] = "%d/md.dcd"%(args.restart-1)
            st_round = args.restart
    else:
        net.train(cv_info)    
        st_round = 1 
    repre_zs,repre_names = get_repre_conf(net.z0,"%d/md.dcd"%(st_round-1),traj_info["pdb"],"0",traj_info["name"]+"sol",pdb_file_sel_str="all",compare_coord_sel_str="protein and name CA",bx=20,by=20)
    #7, start adaptive sampling
    print("starting from round %d and will end at round %d"%(st_round,args.max_round))
    i = st_round
    while i < args.max_round+1:
        #7.1, Check if only do final
        if args.restart_final != None:
            net = CV(str(i))
            net.load_net()
            traj_info["dcd"] = "%d/md.dcd"%(i)
            break
        #7.2, preprocess for dragging sumulation
        print("Step %d: Adaptive Simulation.\n%s"%(i,datetime.datetime.now().strftime("%Y%m%d%H%M")))
        os.system("mkdir %d"%(i))
        dcd_files = [traj_info["dcd"]]
        if args.sub_restart != None and i == args.restart:###
            further_sim_ids = np.load("%d/further_sim_ids.npy"%(i-1))
        else:
            inter_r0s = []
            os.system("mkdir %d/tmp"%(i-1))
            all_es = []
            z_center = net.z0.mean(0)
            dr = ((net.drs[0]/(2*cv_info["bins_for_margin"]))**2 + (net.drs[1]/(2*cv_info["bins_for_margin"]))**2)**0.5
            #7.3 define cutoff values in drag
            rmsd_cutoff,rmsd_cutoff_least = get_rmsd_cutoff(args,net,'%d/%s_dmatrix.npy'%(i-1,traj_info["name"]))
            #7.4 starting mpi dragging
            pool = ctx.Pool(args.n_proc)
            ps = []
            for n,r0s in enumerate(net.cluster_margin_centers):
                further_force = get_further_force(net,r0s,args,traj_info)
                print(further_force)
                simu_info = {"seed":111,
                            "T":args.T,
                            "further_force":further_force,
                            "output_state_freq":1000,}
                xy2c = r0s - z_center
                xy2c = xy2c / np.linalg.norm(xy2c)
                ddxy = xy2c * net.unit_dxy * args.dxy_scale
                tmpz = net.z0.copy()
                ps.append(pool.apply_async(drag,(traj_info,input_info,simu_info,cv_info,args,r0s,net.drs,tmpz,net.coords,i,n,rmsd_cutoff,rmsd_cutoff_least,ddxy,repre_zs,repre_names,)))
            pool.close()
            for n,p in enumerate(ps):
                res = p.get()
                if len(res) == 0:
                    continue
                if args.guide_type in ["ebp","mmgbsa"]:
                    poc_e,pot,r0s,one_inter_r0s = res
                    all_es.append([n,pot,r0s,poc_e])
                else:
                    es,r0s,one_inter_r0s = res
                    all_es.append([n,np.mean(es),r0s])
                net.cluster_margin_centers[n] = r0s
                inter_r0s += one_inter_r0s
            pool.join()
            pool.terminate()
            #7.5 check the number of valid dragged seeds
            if (len(all_es) < 2*n_seed and i > 1 and args.require_num_seed==1) or (len(all_es)==0 and i > 1):
                rerun += 1
                if rerun >= 3:
                    if len(all_es) < n_seed:
                        os.system("rm -rf %d/tmp"%(i-1))
                        i -= 1
                        break
                    else:
                        rerun = 0
                else:
                    os.system("rm -rf %d/tmp"%(i-1))
                    i -= 1
                    args.restart = i
                    args.sub_restart = 2*n_seed
                    continue
            else:
                rerun = 0
            np.save("%d/cluster_margin_centers.npy"%(i-1),net.cluster_margin_centers)
            #7.6, sort the dragged seeds
            further_sim_ids = []
            further_r0s = []
            all_es.sort(key=lambda x:x[1])
            if args.guide_type == "ebp":
                all_es = all_es[0:2*n_seed]
                all_es.sort(key=lambda x:x[3])
                for n,pot,r0s,poc_e in all_es:
                    further_sim_ids.append(n)
                    further_r0s.append(r0s)
                    os.system("cp %d/seed%d.pdb %d/lowe_seed%d.pdb"%(i-1,n,i-1,len(further_sim_ids)-1))
                    if len(further_sim_ids) == 2*n_seed:
                        break
            elif args.guide_type == "mmgbsa":
                all_es = all_es[0:2*n_seed]
                all_es.sort(key=lambda x:x[3])
                repre_es = []
                for n in range(len(repre_zs)):
                    os.system("cp %d/%s %d/seed-%d.pdb"%(i-1,repre_names[n],i-1,n))
                    poc_e,pot,r0s,one_inter_r0s = drag(traj_info,input_info,simu_info,cv_info,args,repre_zs[n],net.drs,tmpz,net.coords,i,-n,0,0,ddxy,repre_zs,repre_names,)
                    repre_es.append([-n,np.mean(poc_e),repre_zs[n]])
                repre_es.sort(key=lambda x:x[1])
                f = open("%d/repre_es.txt"%(i-1),"w")
                for info in repre_es:
                    f.write(str(info))
                f.close()
                n_repre = 0
                n = 0
                n_repre_in_top_seeds = 0
                while len(further_sim_ids) < 2*n_seed:
                    if all_es[n][3] > 0.5 * repre_es[n_repre][1]:
                        further_sim_ids.append(repre_es[n_repre][0])
                        further_r0s.append(repre_es[n_repre][2])
                        os.system("cp %d/%s %d/lowe_seed%d.pdb"%(i-1,repre_names[n_repre],i-1,len(further_sim_ids)-1))
                        n_repre += 1
                        if len(further_sim_ids) < n_seed:
                            n_repre_in_top_seeds += 1
                    else:
                        further_sim_ids.append(all_es[n][0])
                        further_r0s.append(all_es[n][2])
                        os.system("cp %d/seed%d.pdb %d/lowe_seed%d.pdb"%(i-1,all_es[n][0],i-1,len(further_sim_ids)-1))
                        n += 1
                if n_repre_in_top_seeds >= n_seed:
                    break
            else:
                for n,e,r0s in all_es:
                    further_sim_ids.append(n)
                    further_r0s.append(r0s)
                    os.system("cp %d/seed%d.pdb %d/lowe_seed%d.pdb"%(i-1,n,i-1,len(further_sim_ids)-1))
                    if len(further_sim_ids) == 2*n_seed:
                        break
            print(further_sim_ids)
            np.save("%d/further_sim_ids.npy"%(i-1),np.array(further_sim_ids))
            f = open("%d/all_es.txt"%(i-1),"w")
            for info in all_es:
                f.write(str(info))
            f.close()
            plt.figure()
            inter_r0s = np.array(inter_r0s)
            plt.plot(net.z0[:,0],net.z0[:,1],"o",ms=0.5,color="black")
            plt.plot(net.cluster_margin_centers[:,0],net.cluster_margin_centers[:,1],"o",ms=5)
            print(inter_r0s)
            if len(inter_r0s) > 0 and len(inter_r0s.shape) == 2:
                plt.plot(inter_r0s[:,0],inter_r0s[:,1],"^",ms=8,color="blue")
            plt.plot(net.cluster_margin_centers[further_sim_ids,0],net.cluster_margin_centers[further_sim_ids,1],"*",ms=10,color="red")
            plt.savefig("%d/lowenergy_margin_centers.jpg"%(i-1))
            plt.close()
        #8, start unbiased simulation from dragged seed structures
        n_ok = 0
        pool = ctx.Pool(args.n_proc)
        ps = []
        n_seed_this = min(n_seed,len(further_sim_ids))
        for n in range(n_seed_this):
            sim_id = further_sim_ids[n]
            simu_info = { "seed":111,
                        "n_steps":500000,#500000
                        "equil_steps":5000,
                        "T":args.T,
                        "loop_step":50000,
                        "cvs":net.fs,
                        "dxys":net.unit_dxy,#net.dxys[sim_id]
                        "r0s":net.cluster_margin_centers[sim_id],
                        "update_freq":100,
                        "output_state_filename":"%d/md_%d.log"%(i,n),
                        "output_state_freq":1000,
                        "output_chk_filename":"%d/md_%d.chk"%(i,n),
                        "output_chk_freq":1000,
                        "output_dcd_filename":"%d/md_%d.dcd"%(i,n),
                        "output_dcd_freq":400,}
            ps.append(pool.apply_async(run_from_chk,(input_info,i,simu_info,n,sim_id,net.dxys,args,args.seed_sample_type,n_seed_this,1,"%d/tmp/draged%d.pdb"%(i-1,sim_id))))
        while n_ok < n_seed_this:
            if len(ps) == 0:
                break
            for j in range(len(ps)):
                if ps[j].ready():
                    res = ps[j].get()
                    ps.pop(j)
                    if res:
                        dcd_files.append(res)
                        n_ok += 1
                        print(dcd_files,n_ok,ps,len(ps))
                    else:
                        if n + 1 >= len(further_sim_ids):
                            break
                        n += 1
                        sim_id = further_sim_ids[n]
                        #net.load_net()
                        simu_info = { "seed":111,
                                    "n_steps":500000,
                                    "equil_steps":5000,
                                    "T":300,
                                    "loop_step":50000,
                                    "cvs":net.fs,
                                    "dxys":net.unit_dxy,#net.dxys[sim_id]
                                    "r0s":net.cluster_margin_centers[sim_id],
                                    "update_freq":100,
                                    "output_state_filename":"%d/md_%d.log"%(i,n),
                                    "output_state_freq":1000,
                                    "output_chk_filename":"%d/md_%d.chk"%(i,n),
                                    "output_chk_freq":1000,
                                    "output_dcd_filename":"%d/md_%d.dcd"%(i,n),
                                    "output_dcd_freq":400,}
                        ps.append(pool.apply_async(run_from_chk,(input_info,i,simu_info,n,sim_id,net.dxys,args,args.seed_sample_type,n_seed_this,1,"%d/tmp/draged%d.pdb"%(i-1,sim_id))))
                    break
            time.sleep(0.1)
        pool.close()
        pool.terminate()
        pool.join()
        #9, combine dcd
        n1 = 10000 // (i+1)
        n0 = 10000 - n1
        print(n0,n1,"n0,n1")
        print(dcd_files,"dcd_files")
        combine_dcd("%d/md.dcd"%(i),i,traj_info["pdb"],dcd_files,n0,n1)###
        traj_info["dcd"] = "%d/md.dcd"%(i)
        net.get_cv_values(traj_info,args.high_dim_type,draw_cut = n1)
        #10 training
        print("Step %d: Training.\n%s"%(i,datetime.datetime.now().strftime("%Y%m%d%H%M")))
        net = CV("%d"%(i),)
        traj_info["work_dir"] = "%d"%(i)        
        net.train(cv_info)
        repre_zs,repre_names = get_repre_conf(net.z0,"%d/md.dcd"%(i),traj_info["pdb"],str(i),traj_info["name"]+"sol",pdb_file_sel_str="all",compare_coord_sel_str="protein and name CA",bx=20,by=20)
        if i > 3:
            print("check converge of ",i)
            if dssp_converge(traj_info,str(i),args.dssp_converge_cutoff):
                break
        i += 1    
        for f in dcd_files[1:]:
            os.system("rm %s"%(f))
    #11, final clustering
    i = min(i,args.max_round)
    final_dir = str(i)
    zs = net.z0
    get_repre_conf(zs,"%s/md_aligned.dcd"%(final_dir),traj_info["pdb"],final_dir,traj_info["name"],pdb_file_sel_str=traj_info["sel_str"],compare_coord_sel_str="protein and name CA",bx=20,by=20)
    


