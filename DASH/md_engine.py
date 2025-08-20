from openmm import *
from openmm.app import *
from simtk.unit import *
from openmm.app.metadynamics import *
from openmm import app
import cvpack
import matplotlib.pyplot as plt
from sys import stdout
from openmmtorch import TorchForce
import torch
from torch import nn
import numpy as np
import prody

platform = Platform.getPlatformByName('CUDA')

class MD_Engine():
    def __init__(self,info):
        self.preparations(info)

    def preparations(self,info):
        self.prmtop = AmberPrmtopFile(info["prmtop"])
        self.inpcrd = AmberInpcrdFile(info["inpcrd"])
        
    def unbiased_simulation(self,simu_info,pdb_file=None,no_min=False):
        seed = simu_info["seed"]
        n_steps = simu_info["n_steps"]
        equil_steps = simu_info["equil_steps"]
        T = simu_info["T"]
        output_state_filename = simu_info["output_state_filename"]
        output_state_freq = simu_info["output_state_freq"]
        output_chk_filename = simu_info["output_chk_filename"]
        output_chk_freq = simu_info["output_chk_freq"]
        output_dcd_filename = simu_info["output_dcd_filename"]
        output_dcd_freq = simu_info["output_dcd_freq"]
        system = self.prmtop.createSystem(nonbondedMethod=PME,
         constraints=HBonds,nonbondedCutoff=1*nanometer)
        for f in system.getForces():
            f.setForceGroup(1)
        system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
        integrator = LangevinMiddleIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds)
        integrator.setRandomNumberSeed(seed)
        simulation = Simulation(self.prmtop.topology, system, integrator,platform)
        simulation.context.setPositions(self.inpcrd.positions)
        if pdb_file:
            pdb = app.PDBFile(pdb_file)
            simulation.context.setPositions(pdb.positions)
        else:
            simulation.context.setPositions(self.inpcrd.positions)
        if self.inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
        if not no_min:
            print("Minimization...")
            simulation.minimizeEnergy()
        simulation.reporters.append(StateDataReporter(output_state_filename, output_state_freq,
         step=True,potentialEnergy=True, temperature=True))
        simulation.reporters.append(StateDataReporter(stdout, output_state_freq, step=True,
        potentialEnergy=True, temperature=True))
        print("Equilibration...")
        simulation.step(equil_steps)
        #simulation.reporters.append(CheckpointReporter(output_chk_filename, output_chk_freq, ))
        simulation.reporters.append(DCDReporter(output_dcd_filename,output_dcd_freq))
        print("Production...")
        simulation.step(n_steps)
        return simulation
        
    def metadynamics(self,simu_info,outdir=None,bias_energy=1.0,sf=1000,pdb_file=None,add_pre_pot=False,zs=None,pre_add_scale=10):
        seed = simu_info["seed"]
        equil_steps = simu_info["equil_steps"]
        n_steps = simu_info["n_steps"]
        cvs = simu_info["cvs"]
        update_freq = simu_info["update_freq"]
        T = simu_info["T"]
        output_state_filename = simu_info["output_state_filename"]
        output_state_freq = simu_info["output_state_freq"]
        output_chk_filename = simu_info["output_chk_filename"]
        output_chk_freq = simu_info["output_chk_freq"]
        output_dcd_filename = simu_info["output_dcd_filename"]
        output_dcd_freq = simu_info["output_dcd_freq"]
        bias_energy = bias_energy * kilojoules_per_mole
        system = self.prmtop.createSystem(nonbondedMethod=PME,
         constraints=HBonds,nonbondedCutoff=1*nanometer)
        system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
        meta = Metadynamics(system, cvs, T,sf, bias_energy, update_freq,saveFrequency=output_dcd_freq,biasDir=outdir)
        integrator = LangevinMiddleIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds)
        integrator.setRandomNumberSeed(seed)
        simulation = Simulation(self.prmtop.topology, system, integrator,platform)
        if pdb_file:
            pdb = app.PDBFile(pdb_file)
            simulation.context.setPositions(pdb.positions)
        else:
            simulation.context.setPositions(self.inpcrd.positions)
        if self.inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
        print("Minimization...")
        simulation.minimizeEnergy()
        simulation.reporters.append(StateDataReporter(output_state_filename, output_state_freq, step=True,
        potentialEnergy=True, temperature=True))
        simulation.reporters.append(StateDataReporter(stdout, output_state_freq, step=True,
        potentialEnergy=True, temperature=True))
        print("Equilibration...")
        simulation.step(equil_steps)
        simulation.reporters.append(CheckpointReporter(output_chk_filename, output_chk_freq, ))
        simulation.reporters.append(DCDReporter(output_dcd_filename,output_dcd_freq))
        print("MetaD Production...")
        if add_pre_pot:
            height = bias_energy * pre_add_scale
            for z in zs:
                meta._addGaussian(z,height,simulation.context)
            fes = meta.getFreeEnergy()
            if outdir:
                plt.figure()
                plt.imshow(np.rot90(fes, k=1))
                plt.savefig("%s/fes_ini.jpg"%(outdir))
                plt.close()
        meta.step(simulation, n_steps)
        return meta
        
    def US(self,simu_info,pdb_file,fc_pull0=1000.0,sub_nstep=None):
        seed = simu_info["seed"]
        equil_steps = simu_info["equil_steps"]
        n_steps = simu_info["n_steps"]
        cvs = [TorchForce(cv) for cv in simu_info["cvs"]]
        r0s = simu_info["r0s"]
        dxys = simu_info["dxys"]
        update_freq = simu_info["update_freq"]
        T = simu_info["T"]
        output_state_filename = simu_info["output_state_filename"]
        output_state_freq = simu_info["output_state_freq"]
        output_chk_filename = simu_info["output_chk_filename"]
        output_chk_freq = simu_info["output_chk_freq"]
        output_dcd_filename = simu_info["output_dcd_filename"]
        output_dcd_freq = simu_info["output_dcd_freq"]
        system = self.prmtop.createSystem(nonbondedMethod=PME,
         constraints=HBonds,nonbondedCutoff=1*nanometer)
        system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
        #US Constrain
        for i in range(len(cvs)):
            cv_str = "0.5 * fc_pull%d * (cv%d - r0%d)^2"%(i,i,i)
            pulling_force = mm.CustomCVForce(cv_str)
            pulling_force.addCollectiveVariable("cv%d"%(i),cvs[i])
            pulling_force.addGlobalParameter("fc_pull%d"%(i),fc_pull0/dxys[i]/dxys[i]*unit.kilojoules_per_mole*unit.nanometers**2)
            pulling_force.addGlobalParameter("r0%d"%(i),r0s[i])
            system.addForce(pulling_force)
            print(fc_pull0/dxys[i]/dxys[i]*unit.kilojoules_per_mole*unit.nanometers**2)
        integrator = LangevinMiddleIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds)
        integrator.setRandomNumberSeed(seed)
        simulation = Simulation(self.prmtop.topology, system, integrator,platform)
        pdb = app.PDBFile(pdb_file)
        simulation.context.setPositions(pdb.positions)  
        if self.inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
        print("Minimization...")
        try:
            simulation.minimizeEnergy()
        except:
            pass
        simulation.reporters.append(StateDataReporter(output_state_filename, output_state_freq,
         step=True,potentialEnergy=True, temperature=True))
        simulation.reporters.append(StateDataReporter(stdout, output_state_freq, step=True,
        potentialEnergy=True, temperature=True))
        print("Equilibration...")
        simulation.step(equil_steps)
        simulation.reporters.append(CheckpointReporter(output_chk_filename, output_chk_freq, ))
        simulation.reporters.append(DCDReporter(output_dcd_filename,output_dcd_freq))
        print("Production...")
        if sub_nstep:
            while n_steps - sub_nstep > 0:
                simulation.step(sub_nstep)
                n_steps -= sub_nstep
            simulation.step(n_steps)
        else:
            simulation.step(n_steps)
        return simulation
    

    def get_sim(self,simu_info,pdb_file=None,inpcrd_file=None,):
        seed = simu_info["seed"]
        T = simu_info["T"]

        if "further_force" in simu_info.keys():
            further_force = simu_info["further_force"]
        else:
            further_force = []

        output_state_freq = simu_info["output_state_freq"]
        system = self.prmtop.createSystem(nonbondedMethod=PME,
         constraints=HBonds,nonbondedCutoff=1*nanometer)
        for f in system.getForces():
            f.setForceGroup(1)
        system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))

        for f_dict in further_force:
            if f_dict["name"] == "CV":
                fi = f_dict["dimension_id"]
                cv_str = "0.5 * fc_pull%d * (cv%d - r0%d)^2"%(fi,fi,fi)
                pulling_force = mm.CustomCVForce(cv_str)
                cv_object = TorchForce(f_dict["cv_file"])
                pulling_force.addCollectiveVariable("cv%d"%(fi),cv_object)
                pulling_force.addGlobalParameter("r0%d"%(fi),f_dict["r0"])
                pulling_force.addGlobalParameter("fc_pull%d"%(fi),f_dict["f0"]/f_dict["dx2"]*unit.kilojoules_per_mole*unit.nanometers**2)
                system.addForce(pulling_force)
                f_dict["f_id"] = len(system.getForces())-1
                f_dict["f0_name"] = "fc_pull%d"%(fi)
                f_dict["f_unit"] = unit.kilojoules_per_mole*unit.nanometers**2
                f_dict["cv_object"] = cv_object
                
            elif f_dict["name"] == "helix_force":
                fi = f_dict["helix_id"]
                helix_residues = [x for i,x in enumerate(self.prmtop.topology.residues()) if i in f_dict["res_id_sel"]]
                helix_cv = cvpack.HelixRMSDContent(helix_residues, system.getNumParticles(), normalize=False)
                cv_str = "0.5 * fc_pull_helix%d * (helix_cv%d - helix_r0%d)^2"%(fi,fi,fi)
                pulling_force = mm.CustomCVForce(cv_str)
                pulling_force.addCollectiveVariable("helix_cv%d"%(fi),helix_cv)
                pulling_force.addGlobalParameter("helix_r0%d"%(fi),0)
                pulling_force.addGlobalParameter("fc_pull_helix%d"%(fi),f_dict["fc_pull"])
                system.addForce(pulling_force)
                f_dict["f_id"] = len(system.getForces())-1
                f_dict["r0_name"] = "helix_r0%d"%(fi)
                f_dict["cv_object"] = helix_cv
        
        integrator = LangevinMiddleIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds)
        integrator.setRandomNumberSeed(seed)
        simulation = Simulation(self.prmtop.topology, system, integrator,platform)

        if inpcrd_file:
            inpcrd = AmberInpcrdFile(inpcrd_file)
            simulation.context.setPositions(inpcrd.positions) 
        elif pdb_file:
            pdb = app.PDBFile(pdb_file)
            simulation.context.setPositions(pdb.positions)  

        if self.inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)

        for fi,f_dict in enumerate(further_force):
            if f_dict["name"] == "helix_force":
                print(simulation.context.getSystem().getForces()[f_dict["f_id"]].getCollectiveVariableValues(simulation.context)[0])
                simulation.context.setParameter(f_dict["r0_name"],
                                                min(simulation.context.getSystem().getForces()[f_dict["f_id"]].getCollectiveVariableValues(simulation.context)[0]+3,len(f_dict["res_id_sel"])-5))
        simulation.reporters.append(StateDataReporter(stdout, output_state_freq, step=True,potentialEnergy=True, temperature=True))
        return simulation,further_force
        





    
    def cosUS_for_sim(self,simulation,n_steps,update_freq,dxys,loop_step,fc_pull1):
        fc_pull1 = fc_pull1*unit.kilojoules_per_mole*unit.nanometers**2
        fc_pulls1 = [fc_pull1/dxy/dxy for dxy in dxys]#used for production
        hfc_pulls = [0.5*fc_pull1/dxy/dxy for dxy in dxys]#used for production
        w = 2 * 3.14 / loop_step
        n = 0
        for i in range(len(fc_pulls1)):
            simulation.context.setParameter('fc_pull%d'%(i), fc_pulls1[i])
        while n_steps > update_freq:
            n += update_freq
            n_steps -= update_freq
            simulation.step(update_freq)
            for i in range(len(hfc_pulls)):
                simulation.context.setParameter('fc_pull%d'%(i), hfc_pulls[i]*cos(w*n)+hfc_pulls[i])
        simulation.step(n_steps)
        return simulation

    def pulseUS(self,simu_info,pdb_file,fc_pull0=10.0):
        seed = simu_info["seed"]
        equil_steps = simu_info["equil_steps"]
        n_steps = simu_info["n_steps"]
        cvs = simu_info["cvs"]
        r0s = simu_info["r0s"]
        dxys = simu_info["dxys"]
        T = simu_info["T"]
        update_freq = simu_info["update_freq"]
        fc_pull0 = fc_pull0*unit.kilojoules_per_mole*unit.nanometers**2
        fc_pulls = [fc_pull0/dxy/dxy for dxy in dxys]

        output_state_filename = simu_info["output_state_filename"]
        output_state_freq = simu_info["output_state_freq"]
        output_chk_filename = simu_info["output_chk_filename"]
        output_chk_freq = simu_info["output_chk_freq"]
        output_dcd_filename = simu_info["output_dcd_filename"]
        output_dcd_freq = simu_info["output_dcd_freq"]

        system = self.prmtop.createSystem(nonbondedMethod=PME,
         constraints=HBonds,nonbondedCutoff=1*nanometer)
        system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
        #US Constrain
        for i in range(len(cvs)):
            cv_str = "0.5 * fc_pull%d * (cv%d - r0%d)^2"%(i,i,i)
            pulling_force = mm.CustomCVForce(cv_str)
            pulling_force.addCollectiveVariable("cv%d"%(i),cvs[i])
            pulling_force.addGlobalParameter("r0%d"%(i),r0s[i])
            pulling_force.addGlobalParameter("fc_pull%d"%(i),fc_pulls[i])
            system.addForce(pulling_force)
        integrator = LangevinMiddleIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds)
        integrator.setRandomNumberSeed(seed)
        simulation = Simulation(self.prmtop.topology, system, integrator,platform)

        pdb = app.PDBFile(pdb_file)
        simulation.context.setPositions(pdb.positions)  
        if self.inpcrd.boxVectors is not None:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
        print("Minimization...")
        try:
            simulation.minimizeEnergy()
        except:
            pass
        simulation.reporters.append(StateDataReporter(output_state_filename, output_state_freq,
         step=True,potentialEnergy=True, temperature=True))
        simulation.reporters.append(StateDataReporter(stdout, output_state_freq, step=True,
        potentialEnergy=True, temperature=True))
        print("Equilibration...")
        simulation.step(equil_steps)
        simulation.reporters.append(CheckpointReporter(output_chk_filename, output_chk_freq, ))
        simulation.reporters.append(DCDReporter(output_dcd_filename,output_dcd_freq))
        print("Production...")
        n = 0
        while n_steps > update_freq:
            n += update_freq
            n_steps -= update_freq
            simulation.step(update_freq)
            for i in range(len(hfc_pulls)):
                simulation.context.setParameter('fc_pull%d'%(i), hfc_pulls[i]*cos(w*n)+hfc_pulls[i])
        simulation.step(n_steps)
        return simulation
    



    def get_Energy(self,pdb_file,pbc=False):
        system = self.prmtop.createSystem(nonbondedMethod=CutoffNonPeriodic,
         constraints=None,nonbondedCutoff=1*nanometer)
        integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        simulation = Simulation(self.prmtop.topology, system, integrator,platform)
        print(self.inpcrd.boxVectors)
        if (self.inpcrd.boxVectors is not None) and pbc:
            simulation.context.setPeriodicBoxVectors(*self.inpcrd.boxVectors)
        pdb = app.PDBFile(pdb_file)
        simulation.context.setPositions(pdb.positions)
        return simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoules_per_mole)


                
class Energy_Cal():
    def __init__(self,top_file,pdb_file,sel_str="protein",t="implicit",T=300,inpcrd_file=None):
        prmtop = AmberPrmtopFile(top_file)
        if t == "implicit":
            system = prmtop.createSystem(implicitSolvent=GBn2)
        else:
            system = prmtop.createSystem(constraints=None,nonbondedCutoff=0.8*nanometer)
        integrator = LangevinMiddleIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds)
        self.simulation = Simulation(prmtop.topology, system, integrator)
        
        if inpcrd_file:
            inpcrd = AmberInpcrdFile(inpcrd_file)
            print("inpcrd.boxVectors",inpcrd.boxVectors)
            if (inpcrd.boxVectors is not None):
                self.simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
        self.pdb = prody.parsePDB(pdb_file)
        self.sel_str = sel_str

    def get_energy(self,coords):
        self.pdb.setCoords(coords)
        coords_sel_in_nm = self.pdb.select(self.sel_str).getCoords() / 10
        self.simulation.context.setPositions(coords_sel_in_nm)
        return self.simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilocalorie_per_mole)
        
            
    
    