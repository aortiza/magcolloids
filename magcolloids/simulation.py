from . import parameters
import numpy as np
import string as st
import time as tm
import os
import sys
import pandas as pd
import copy as cp
from . import ureg

 
class sim():
    def __init__(self,
        file_name = "test", dir_name = "",stamp_time = False,
        particles = None, traps = None, world = None, field = None,
        timestep = 1e-3*ureg.s, framerate = 30*ureg.Hz, total_time = 60*ureg.s,
        output = ["x", "y", "z"], processors = 1):
        """
        A sim object contains the parameters defined before. It also requires some extra parameters that define the files in which the simulation scripts and results are stored """
                
        self.file_name = file_name
        self.dir_name = dir_name
        self.stamp_time = stamp_time
        
        self.particles = particles
        
        # particles can be an array or not, but self.particles must be an array. 
        # In the lines below, I ensure that self.particles is set from an array
        try: 
            # If traps can be indexed, they are a list, and they can be assigned to self.traps
            particles[0]
            self.particles=particles
        except TypeError:
            # If traps can't be indexed, they are placed inside a list
            self.particles=[particles]
        
        # same thing for traps
        if not traps is None:
            try:
                # If traps can be indexed, they are a list, and they can be assigned to self.traps
                traps[0]
                self.traps=traps
            except TypeError:
                 # If traps can't be indexed, they are placed inside a list
                self.traps=[traps]
        else:
            self.traps = None
        
        self.world = world
        self.field = field
        
        self.timestep = timestep
        self.framerate = framerate
        self.total_time = total_time
        
        self.output = output
        self.processors = processors
    
    def write_script(self):
        """ Write the lmpin file"""
        with open(self.script_name,'w') as f:
            f.write("### ---preamble--- ###\n")
            f.write("log %s"%self.log_name)
            f.write(self.world.world_def)

            f.write("\n### ---Create Particles and Region--- ###\n")

            f.write("read_data "+self.input_name)
                
            f.write(self.world.group_def)

            f.write("\n### ---Variables--- ###\n") 
   
            f.write("\n## magnetic field\n") 
            f.write(self.field.variable_def)

            if not self.traps is None:
                if any([t.velocity is not None for t in self.traps]):
                    f.write("\n## traps velocities\n") 
                    for t in self.traps:
                        f.write(t.velocity)

            if not self.world.ext_force is None:
                f.write("\n## external force\n") 
    
                f.write(self.world.ext_force.calculation)
    
            f.write("\n### ---Fixes--- ###\n") 

            f.write(self.field.fix_def)
            f.write(self.world.integrator_def)
            f.write(self.world.gravity_def)
            f.write(self.world.wall_def)
            f.write(self.world.enforce2d)
            if not self.traps is None:
                for t in self.traps:
                    f.write(t.velocity_fix)

            if not self.world.ext_force is None:
                f.write(self.world.ext_force.fix_str)
    
            f.write(self.run_def)

    def write_input(self):
        """ Write input file .lmpin"""
        with open(self.input_name,'w') as f:
            f.write("This is the initial atom setup of %s"%self.input_name)
            f.write(self.world.region_def)
        
            f.write("\nAtoms\n\n")
            
            for p in self.particles:
                f.write(p.atom_def)
            
            if not self.traps is None:
                is_there_bonds = [t.cutoff == np.Inf*ureg.um for t in self.traps]
                
                for t in self.traps:
                    f.write(t.atom_def)
        
                if is_there_bonds:
                    f.write("\nBonds\n\n")
                    
                    for t in self.traps:
                        if t.cutoff == np.Inf*ureg.um:
                            f.write(t.bond_def)
                       
                    f.write("\nBond Coeffs\n\n")
                    for t in self.traps:
                        if t.cutoff == np.Inf*ureg.um:
                            f.write(t.bond_params)
                        
            f.write("\nPairIJ Coeffs\n\n")
            
            f.write(self.world.interaction_def)
        
            f.write("\n\n")
    
    def write_run_def(self):
        self.run_steps = int(np.round(self.total_time.to(ureg.s)/self.timestep.to(ureg.s)))
        self.run_def = st.Template("\n" + \
            "### ---Run Commands--- ###\n" + \
            "timestep 	$tmstp \n" + \
            "dump 	3 all custom $sampl $out_name id type $output\n" + \
            "thermo_style 	custom step atoms\n" + \
            "thermo 	100  \n" + \
            "run 	$runtm \n").substitute(out_name = self.output_name,
                             tmstp = (self.timestep.to(ureg.us)).magnitude,
                             sampl = int(np.round(
                                 1/(self.framerate.to(ureg.Hz)*self.timestep.to(ureg.s)))),
                             runtm = self.run_steps,
                             output = "\t ".join(self.output))    
    
    def distribute_ids(self):
        """ Gives an id number to all particles and traps. 
        -------
        This must be done before generating the strings, to give the correct id to each atom and traps. """
        
        atoms_id = 1 # these start at 1 because lammps starts at 1. 
        atom_type = 1
    
        particles = self.particles
        traps = self.traps
        try:
            particles[0]
        except:
            particles = [particles]

        if traps is not None:
            try:
                traps[0]
            except:
                traps = [traps]
        
        for p in particles:
            p.atoms_id = np.array(range(atoms_id, atoms_id + len(p.positions)))
            p.atom_type = atom_type

            atoms_id += len(p.positions)
            atom_type += 1
    
        if traps is not None:  

            for t in traps:
                t.traps_id = np.array(range(atoms_id, atoms_id + len(t.positions)))
                t.atom_type = atom_type

                atoms_id += len(t.positions)
                atom_type +=1

            for t in traps:
                t.bond_to_particles()

            bonds_id = 1
            for t in traps:
                t.bonds_id = np.array(range(bonds_id,bonds_id + len(t.bonds)))
                bonds_id += len(t.bonds)
                                         
    def generate_scripts(self):
        """
        This method generates the input script for the lammps simulation. 
        It accepts some options, which for now include:
        * (future) input_file: boolean, (False) which specifies if the input file is stored separatelly from the main script. 
            This icreases readibility for large amounts of particles
        """

        if not os.path.exists(self.dir_name):
            os.makedirs(self.dir_name)
        
        self.base_name = os.path.join(self.dir_name,self.file_name)
        if self.stamp_time:
            self.base_name = self.base_name + \
                tm.strftime('_%Y_%m_%d_%H_%M_%S')
        self.seed = np.random.randint(1000000)
        #self.seed = 1
                
        self.script_name = self.base_name+'.lmpin'
        self.input_name = self.base_name+'.lmpdata'
        self.output_name =  self.base_name+'.lammpstrj'
        self.log_name =  self.base_name+'.log'
        
        self.distribute_ids()
        
        ### Create strings

        for p in self.particles:
            p.create_string()
            
        if not self.traps is None:
            for t in self.traps:
                t.create_string()
        
        if not self.world.ext_force is None:
            self.world.ext_force.create_string()
                
        self.world.create_string()
        self.field.create_string()

        self.write_run_def()                     
        self.write_script()
        self.write_input()
                    
    def run(self,verbose = False):
        """This function runs an input script named filename in lammps. The input should be located in target_dir"""
        exec_paths = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lammps_executables'))
        
        if sys.platform=='darwin':
            if self.processors>1:
                lmp_exec = "mpirun -np %u "%self.processors + os.path.join(exec_paths,"lmp_mac_mpi")
            else: 
                lmp_exec = os.path.join(exec_paths,"lmp_mac")
            
        elif sys.platform=='linux':
            if self.processors>1:
                lmp_exec = "mpirun -np %u "%self.processors + os.path.join(exec_paths,"lmp_mpi")
            else: 
                lmp_exec = os.path.join(exec_paths,"lmp_serial")
        else:
            if self.processors>1:
                lmp_exec = "mpirun -np %u "%self.processors + os.path.join(exec_paths,"lmp_mingw64-native-mpi")
            else:
                lmp_exec = os.path.join(exec_paths,"lmp_mingw64-native.exe")
            
        
        if verbose:
            print(lmp_exec + " -in "+self.script_name)
            
        self.lmp_exec = lmp_exec
        if self.processors>1:
            os.system(lmp_exec + " -in "+self.script_name)
            
        else:
            os.system(lmp_exec + " -in "+self.script_name)
    
    def load(self,read_trj = False,sl = slice(0,-1,1)):
        """This method creates a lazy read object. The option read_trj = True reads the whole trj file and returns the output"""
        self.lazy_read = trj_lazyread(self.output_name,self.output)
        
        if read_trj:
            trj = self.lazy_read[sl]
            trj['t']=trj.index.get_level_values('frame')*self.timestep.to(ureg.s).magnitude
            frames = trj.index.get_level_values('frame').unique()
            trj.index.set_levels(range(len(frames)), level = "frame", inplace=True)
            return trj
                              
#from collections import Sequence
class trj_lazyread():
    def __init__(self,Filename,output):
        self.T = dict([])
        self.Name = Filename
        item = dict([])
        bounds = dict([])
        
        self.output = output
        
        self.columns = ['frame', 'id', 'type'] + output
        self.formats = ['i8','i8', 'i8']+['float32']*len(output)
        
        with open(Filename) as d:
            line = "d"
            while line:
                line = d.readline()
                
                if 'ITEM: TIMESTEP' in line:
                    line = d.readline()
                    t = int(line)
                    
                if 'ITEM: NUMBER OF ATOMS' in line:
                    line = d.readline()
                    item["atoms"] = int(line)
                    
                if 'ITEM: BOX BOUNDS' in line:
                    line = d.readline()
                    bounds["x"] = np.array([float(i) for i in line.split(' ') if i!='\n'])
                    
                    line = d.readline()
                    bounds["y"] = np.array([float(i) for i in line.split(' ') if i!='\n'])
                    
                    line = d.readline()
                    bounds["z"] = np.array([float(i) for i in line.split(' ') if i!='\n'])
                    
                    item["bounds"] = bounds
                    
                if 'ITEM: ATOMS' in line:
                    item["location"] = d.tell()
                    self.T[t] = cp.deepcopy(item)
     
    def __getitem__(self, sliced):
        return self.read_trj(sliced)
    
    def get_bounds(self,sl=None):
        
        def bound_to_vector(T,frame):
            values = np.array([b for k in T[frame]["bounds"].keys() for b in T[frame]["bounds"][k]])
            names = [k+"_"+name for k in T[frame]["bounds"].keys() for name in ["min","max"]]
            return pd.DataFrame([values],columns=names,index=[frame])
                    
        frames = np.sort(np.array(list(self.T.keys())))
        if sl is not None:
            frames = frames[sl]
        data = pd.concat([bound_to_vector(self.T,frame) for frame in frames])
        data.index.name = "frame"
        return data
            
    def read_frame(self,time):
        
        Atoms = np.zeros(
            int(self.T[time]["atoms"]),
            dtype={
                'names':self.columns[1:],
                'formats':self.formats[1:]})
        j=0
        with open(self.Name) as d:
            d.seek(self.T[time]["location"])
            for i in range(0,int(self.T[time]["atoms"])):
                line = d.readline()
                line = line.replace("-1.#IND","-NaN").replace("1.#IND","NaN")
                line = line.replace("-1.#QNAN","-NaN").replace("1.#QNAN","NaN")
                linearray = np.array([float(i) for i in line.split(' ') if i!='\n'])
                for i,out in enumerate(self.columns[1:]):
                    Atoms[out][j] = linearray[i]
                j=j+1;
        return Atoms
        
    def read_trj(self,*args):
        """reads a trj from the file and returns a pandas object. 
        accepts as first argument a slice object, which allows to read subsets of the file"""
        columns=['frame']+list(self.read_frame(list(self.T.keys())[0]).dtype.names)
        
        frames = np.sort(np.array(list(self.T.keys())))
        if len(args)>0:
            frames = frames[args[0]]
            if args[0].__class__.__name__!="slice":
                frames=frames

        accum = pd.DataFrame(index=[],columns=columns)

        def read_entry(i):
            frame_data = self.read_frame(i)
            entry = pd.DataFrame(data=frame_data)
            entry['frame']=i
            return(entry)

        accum = pd.concat([read_entry(i) for i in frames])
#            for part in frame_data: 
#                data = [np.array([i]+list(part))]
#                entry = pd.DataFrame(data=data,columns=columns)
#                accum = accum.append(entry)
        
        accum = accum.set_index(['frame','id'])
        return accum.sort_index(level='frame')
    
