from . import parameters
import numpy as np
import string as st
import time as tm
import os
import sys
import pandas as pd
import copy as cp
import jsonpickle
from . import ureg


class sim():
    def __init__(self,
        file_name = "test", dir_name = "",stamp_time = False,
        particles = None, traps = None, world = None, field = None,
        timestep = 1e-3*ureg.s, framerate = 30*ureg.Hz, total_time = 60*ureg.s,
        output = ["x", "y", "z"]):
        """ 
        A sim object contains the parameters defined before. It also requires some extra parameters that define the files in which the simulation scripts and results are stored """
        
        
        self.file_name = file_name
        self.dir_name = dir_name
        self.stamp_time = stamp_time
        
        self.particles = particles
        self.traps = traps
        self.world = world
        self.field = field
        
        self.timestep = timestep
        self.framerate = framerate
        self.total_time = total_time
        
        self.output = output

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
        self.pickle_name =  self.base_name+'.jp'
        
        ### Create strings
        self.particles.create_string()
        if not self.traps is None:
            self.traps.create_string()
            
        self.world.create_string()
        self.field.create_string()
        
        self.run_steps = int(np.round(self.total_time.to(ureg.s)/self.timestep.to(ureg.s)))
        
        self.run_def = st.Template(""" 
### ---Run Commands--- ###
timestep 	$tmstp 
dump 	3 all custom $sampl $out_name id type $output
thermo_style 	custom step atoms
thermo 	100  
run 	$runtm
                    """).substitute(out_name = self.output_name,
                             tmstp = (self.timestep.to(ureg.us)).magnitude,
                             sampl = int(np.round(
                                 1/(self.framerate.to(ureg.Hz)*self.timestep.to(ureg.s)))),
                             runtm = self.run_steps,
                             output = "\t ".join(self.output))
                             
                             
        
        ### Write script
        f = open(self.script_name,'w')
        f.write("### ---preamble--- ###\n")
        f.write(self.world.world_def)
        
        f.write("\n### ---Create Particles and Region--- ###\n")
        
        f.write("read_data "+self.input_name)
                        
        f.write(self.world.group_def)
        
        f.write("\n### ---Fixes--- ###\n")    
        f.write(self.field.variable_def)
        
        f.write(self.world.integrator_def)
        f.write(self.world.enforce2d)
        f.write(self.world.gravity_def)
        f.write(self.world.wall_def)
        f.write(self.field.fix_def)
        
        f.write(self.run_def)
        f.close

        ### Write input file
        f = open(self.input_name,'w')
        f.write("This is the initial atom setup of %s"%self.input_name)
        f.write(self.world.region_def)
        
        f.write("\nAtoms\n\n")
        
        f.write(self.particles.atom_def)
        if not self.traps is None:
            f.write(self.traps.atom_def)
        
        f.write("\nBonds\n\n")
        if not self.traps is None:
            f.write(self.traps.bond_def)
        
        f.write("\nBond Coeffs\n\n")
        if not self.traps is None:
            f.write(self.traps.bond_params)
            
        f.write("\nPairIJ Coeffs\n\n")
        f.write(self.world.interaction_def)
        
        f.write("\n\n")
        pk = open(self.pickle_name,'w')
        pk.write(jsonpickle.encode(self))
        pk.close
                    
    def run(self):
        """This function runs an input script named filename in lammps. The input should be located in target_dir"""
        exec_paths = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lammps_executables'))
        
        if sys.platform=='darwin':
            lmp_exec = os.path.join(exec_paths,"lmp_mac")
        elif sys.platform=='linux':
            lmp_exec = os.path.join(exec_paths,"lmp_serial")
        else:
            lmp_exec = os.path.join(exec_paths,"lmp_mingw64.exe")
        
        os.system(lmp_exec + " -in "+self.script_name)
    
    def load(self,read_trj = False):
        """This method creates a lazy read object. The option read_trj = True reads the whole trj file and returns the output"""
        self.lazy_read = trj_lazyread(self.output_name,self.output)
        
        if read_trj:
            trj = self.lazy_read.read_trj()
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
                    
                if 'ITEM: ATOMS' in line:
                    item["location"] = d.tell()
                    self.T[t] = cp.deepcopy(item)
     
    def __getitem__(self, sliced):
        return self.read_trj(sliced)
                        
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
                frames=[frames]

        accum = pd.DataFrame(index=[],columns=columns)

        for i in frames:
            frame_data = self.read_frame(i)
            entry = pd.DataFrame(data=frame_data)
            entry['frame']=i
            accum = accum.append(entry)
#            for part in frame_data: 
#                data = [np.array([i]+list(part))]
#                entry = pd.DataFrame(data=data,columns=columns)
#                accum = accum.append(entry)
        
        accum = accum.set_index(['frame','id'])
        return accum.sort_index(level='frame')
    