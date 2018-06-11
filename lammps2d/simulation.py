import lammps2d.parameters as parameters
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
        particles = None, world = None, field = None,
        timestep = 1e-3*ureg.s, framerate = 30*ureg.Hz, total_time = 60*ureg.s):
        """ 
        A sim object contains the parameters defined before. It also requires some extra parameters that define the files in which the simulation scripts and results are stored """
        
        
        self.file_name = file_name
        self.dir_name = dir_name
        self.stamp_time = stamp_time
        
        self.particles = particles
        self.world = world
        self.field = field
        
        self.timestep = timestep
        self.framerate = framerate
        self.total_time = total_time

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
        self.output_name =  self.base_name+'.lammpstrj'
        self.pickle_name =  self.base_name+'.jp'
        
        ### Create strings
        self.particles.create_string()
            
        self.world.create_string()
        self.field.create_string()
        
        self.run_steps = int(np.round(self.total_time.to(ureg.s)/self.timestep.to(ureg.s)))
        
        self.run_def = st.Template(""" 
### ---Run Commands--- ###
timestep 	$tmstp 
dump 	3 all custom $sampl $out_name id type x y z mu mux muy muz fx fy fz
thermo_style 	custom step atoms
thermo 	100  
run 	$runtm
                    """).substitute(out_name = self.output_name,
                             tmstp = (self.timestep.to(ureg.us)).magnitude,
                             sampl = int(np.round(
                                 1/(self.framerate.to(ureg.Hz)*self.timestep.to(ureg.s)))),
                             runtm = self.run_steps)
                             
                             
        
        ### Write strings
        f = open(self.script_name,'w')
        f.write("### ---preamble--- ###\n")
        f.write(self.world.world_def)
        
        f.write("\n### ---Create Particles and Region--- ###\n")
        f.write(self.world.region_def)
        
        f.write(self.particles.atom_def)
        
        f.write(self.world.group_def)
        
        f.write(self.particles.atom_prop)
        
        f.write("\n### ---Fixes--- ###\n")    
        f.write(self.field.variable_def)
        
        f.write(self.world.integrator_def)
        f.write(self.world.gravity_def)
        f.write(self.world.wall_def)
        f.write(self.field.fix_def)
        
        f.write(self.run_def)
        f.close

        pk = open(self.pickle_name,'w')
        pk.write(jsonpickle.encode(self))
        pk.close        
        
    def generate_interaction_series(self,start,n_points,end = [0,0,0]):
        """
        This method generates a lammps script with no thermostat, 
        which simply moves a particle at a constant velocity through the x axis
        and dumps the force calculated by lammps.
        
        The parameters are, 
        * start (3 number array) a vector of the starting point,
        * n_points = number of points
        * end = [0,0,0] a vector of the end point.
        """
        
        self.base_name = os.path.join(self.sim_parameters.dir,self.sim_parameters.file_name)
        if self.sim_parameters.stamp_time:
            self.base_name = self.base_name + \
                tm.strftime('_%Y_%m_%d_%H_%M_%S')
        self.seed = np.random.randint(1000000)
                
        self.interaction_script_name = self.base_name+'.lmpfieldin'
        self.interaction_output_name =  self.base_name+'.lmpfield'
        
        self.unitconversions()
        
        preamble = st.Template(
            "## ---Preamble--- ## \n" +
            "units micro \n" +
            "atom_style hybrid sphere paramagnet \n" +
            "boundary s s s\n" +
            "neighbor 4.0 nsq \n" +
            "pair_style lj/cut/dipole/cut $lj_cut $dpl_cut\n")
            #"pair_style lj/cut/dipole/cut\n")
            

        preamble = preamble.substitute(
                           lj_cut = self.field_parameters.lj_cutoff,
                           dpl_cut = self.field_parameters.dipole_cutoff
                           )
        
        atoms = st.Template(
            "create_atoms 1 single $x0 $y0 $z0 \n" + 
            "create_atoms 1 single 0 0 0 \n\n")
            
        atoms = atoms.substitute(x0 = start[0],y0 = start[1],z0 = start[2])
            
        world = st.Template(
            "\n## ---Set World-- ## \n" +
            "region space block $spx1 $spx2 $spy1 $spy2 $spz1 $spz2 # this is in microns \n" +
            "create_box 1 space \n\n"+
            atoms + 
            "group Atoms type 1 \n" + 
            "group moving id 1 \n" + 
            "pair_coeff * * $lj_eps $lj_sgm $lj_cut $dp_cut\n")
        
        region = np.array([min(start,end),max(start,end)])
        region[0] = region[0]-10
        region[1] = region[1]+10
        
        world = world.substitute(spx1 = region[0,0],
                        spx2 = region[1,0],
                        spy1 =region[0,1],
                        spy2 = region[1,1],
                        spz1 = region[0,2],
                        spz2 = region[1,2],
                        lj_eps = self.field_parameters.lj_parameters[0],
                        lj_sgm = 2*self.particle_properties[0].radius*self.field_parameters.lj_parameters[1],
                        lj_cut = 2*self.particle_properties[0].radius*self.field_parameters.lj_cutoff,
                        dp_cut = self.field_parameters.dipole_cutoff)

        props = "\n".join(
                [st.Template("set atom $id mass $mass susceptibility $susc diameter $diameter").substitute(
                    mass =m, susc = xi, diameter=2*r, id=i+1) for i,(m,xi,r) in enumerate(zip(self.mass,self.susceptibility,self.radius))])+"\n"
                
        particle_props = st.Template(
            "\n## ---Particle Properties---## \n" +
            "mass * 1 \n" +
            props + 
            "\n")
        particle_props = particle_props.substitute()
         
        field = st.Template(
            "## ---Fixes---## \n" + 
            "variable Bmag atom $Bmag \n" + 
            "variable omega atom $omega \n" + 
            "variable theta atom $angle \n" + 
            "variable fieldx atom v_Bmag*sin(v_omega*time)*sin(v_theta) \n" + 
            "variable fieldy atom v_Bmag*cos(v_omega*time)*sin(v_theta) \n" + 
            "variable fieldz atom v_Bmag*cos(v_theta) \n\n")
        
        field = field.substitute(Bmag = self.field_mag_h,
                        omega = self.frequency*2*np.pi,
                        angle = self.angle)

        fixes = st.Template(
            "fix 	1 moving move linear $velx $vely $velz \n" +
            "fix 	2 Atoms setdipole v_fieldx v_fieldy v_fieldz \n")
        fixes = fixes.substitute(
            velx = -(start[0]-end[0])/n_points,
            vely = -(start[1]-end[1])/n_points,
            velz = -(start[2]-end[2])/n_points)
            
        run = st.Template(
            "\n## ---Run Commands--##\n"
            "timestep 	1 \n" + 
            "dump 	3 all custom 1 $out_name id type x y z mu mux muy muz fx fy fz\n" + 
            "thermo_style 	custom step atoms \n" + 
            "thermo 	100 \n" + 
            "run 	$runtm \n")
        run = run.substitute(out_name = self.interaction_output_name,
                             runtm = n_points)
        
        f = open(self.interaction_script_name,'w')
        f.write(preamble)
        f.write(world)
        f.write(particle_props)
        f.write(field)
        f.write(fixes)
        f.write(run)
        f.close()
        
        if sys.platform=='darwin':
            lmp_exec = "./lmp_mac"
        else:
            lmp_exec = "lmp_mingw64.exe"
            
        print(lmp_exec + " -in "+self.interaction_script_name)

        os.system(lmp_exec + " -in "+self.interaction_script_name)
        
        read_obj = trj_lazyread(self.interaction_output_name)
        
        return read_obj.readtrj()
    
    def unitconversions(self):
        """
        This function converts units from microns seconds piconewtons,
        which is the standard parameter system, to lammps micro units 
        which are microns microseconds and picogram-micrometer/microsecond^2
        """
        # First the easy ones which are the time units
        self.timestep = self.run_parameters.timestep*1e6 # (us/step)
        # The frame rate and the total_time are not in microseconds but in units of timestep.
        self.steps_per_print = 1/(self.run_parameters.framerate*self.run_parameters.timestep) 
        #(1/ts step/s)*(1/fps s/f) = (steps/frame)
        self.run_steps = self.run_parameters.total_time/self.run_parameters.timestep
        
        # now field parameters. 
        
        self.angle = self.field_parameters.angle/180*np.pi #radians
        self.frequency = self.field_parameters.frequency*1e-6
        # (f 1/s)*(1e-6 s/us)*(ts us/step)
        permeability = 4e5*np.pi #pN/A^2
        self.field_mag_h = self.field_parameters.magnitude/permeability*1e9/2.99e8 #I don't know for sure.
        
        # world parameters
        self.temperature = self.sim_parameters.temperature
        #kb = float(4/300) #pN*nm/K
        kb = float(4/300)*1e-6 #pg um^2/us^2/K
        # damping coefficient specification
        # I can only specify a single damping coefficient, which is a parameter to the bd fix
        # To give a different diffusion to each particle, I need to change their mass. 
        mass = 1 # in g
        
        self.diffusion = np.array([p.diffusion*1e-6 for p in self.particle_properties]) # in (um^2/us)
        self.susceptibility = np.array([p.susceptibility for p in self.particle_properties]) # no dimensions
        self.radius = np.array([p.radius for p in self.particle_properties]) # um
        
        D_mean = np.mean(self.diffusion)
        
        self.damp = D_mean*mass/(kb*self.temperature) # this is in 1/us

        self.mass = (kb*self.temperature*self.damp)*1/self.diffusion # this is in picograms
        
    def run(self):
        """This function runs an input script named filename in lammps. The input should be located in target_dir"""
        exec_paths = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lammps_executables'))
        
        if sys.platform=='darwin':
            lmp_exec = os.path.join(exec_paths,"lmp_mac")
        elif sys.platform=='linux':
            lmp_exec = os.path.join(exec_paths,"lmp_serial")
        else:
            lmp_exec = os.path.join(exec_paths,"lmp_mingw64-native.exe")

        os.system(lmp_exec + " -in "+self.script_name)
    
    def load(self,**kargs):
        """This method creates a lazy read object. The option read_trj = True reads the whole trj file and returns the output"""
        self.lazy_read = trj_lazyread(self.output_name)
        
        if "read_trj" in kargs:
            if kargs["read_trj"]==True:
                trj = self.lazy_read.readtrj()
                trj['t']=trj.index.get_level_values('frame')*self.run_parameters.timestep
                return trj
                              
#from collections import Sequence
class trj_lazyread():
    def __init__(self,Filename):
        self.T = dict([])
        self.Name = Filename
        item = dict([])
        
        self.columns=['frame']+['id','type','x','y','z','mu','mux','muy','muz','fx','fy','fz']
        self.formats = ['i8']+['i8','i8','float32','float32','float32','float32','float32','float32','float32','float32','float32','float32']
        
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
        return self.readtrj(sliced)
                        
    def readframe(self,time):
        
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
                linearray = np.array([float(i) for i in line.split(' ') if i!='\n'])
                Atoms['id'][j] = int(linearray[0])
                Atoms['type'][j] = int(linearray[1])
                Atoms['x'][j] = linearray[2]
                Atoms['y'][j] = linearray[3]
                Atoms['z'][j] = linearray[4]
                Atoms['mu'][j] = linearray[5]
                Atoms['mux'][j] = linearray[6]
                Atoms['muy'][j] = linearray[7]
                Atoms['muz'][j] = linearray[8]
                Atoms['fx'][j] = linearray[9]
                Atoms['fy'][j] = linearray[10]
                Atoms['fz'][j] = linearray[11]
                j=j+1;
        return Atoms
    def readtrj(self,*args):
        """reads a trj from the file and returns a pandas object. 
        accepts as first argument a slice object, which allows to read subsets of the file"""
        columns=['frame']+list(self.readframe(list(self.T.keys())[0]).dtype.names)
        
        frames = np.sort(np.array(list(self.T.keys())))
        if len(args)>0:
            frames = frames[args[0]]
            if args[0].__class__.__name__!="slice":
                frames=[frames]

        accum = pd.DataFrame(index=[],columns=columns)

        for i in frames:
            frame_data = self.readframe(i)
            entry = pd.DataFrame(data=frame_data)
            entry['frame']=i
            accum = accum.append(entry)
#            for part in frame_data: 
#                data = [np.array([i]+list(part))]
#                entry = pd.DataFrame(data=data,columns=columns)
#                accum = accum.append(entry)
        
        accum = accum.set_index(['frame','id'])
        return accum.sort_index(level='frame')
    