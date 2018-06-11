import string as st
import numpy as np
from . import ureg

""" Here we should change the abstraction: instead of a collection of particles with parameters, perhaps it would be more adequate to define a collection of particles with a single set of parameters. The program could then create a script from several collections of particles, which would allow different parameters. This is more in line with the way of lammps, and therefore probably simpler """

class particles():
    """ A type of particle to be simulated """
    def __init__(self, positions, 
                radius = 2*ureg.um,
                susceptibility = 1,
                drag = 4e6*ureg.pN/(ureg.um/ureg.s), 
                diffusion = None, temperature = None, 
                density = 1000*ureg.kg/ureg.m**3):
        """
        Initializes a particle type. 
        The diffusion coefficient can be given instead o the drag.
        In that case, the temperature is also needed to calculate the drag. 
        This represents the density mismatch of the particle in the solvent
        """
        
        if diffusion:
            
            KbT = 4*(ureg.pN*ureg.nm)/(300*ureg.K)*temperature
            drag = KbT/diffusion
            
        self.positions = positions.to(ureg.um)
        self.radius = radius.to(ureg.um)
        self.susceptibility = susceptibility
        self.drag = drag.to(ureg.pg/ureg.us)
        self.mass = (density*4/3*np.pi*(radius)**3).to(ureg.pg)
        
        damp = 1e-3*ureg.us
        
        self.drag_mass = (drag*damp).to(ureg.pg)
        
    def create_string(self):
        
        
        self.atom_def = "\n".join(
            [st.Template("create_atoms 1 single $x0 $y0 $z0").substitute(
                x0=pos[0],y0=pos[1],z0=pos[2]) for pos in self.positions.magnitude]
        )
        
        
        self.atom_prop = \
            st.Template("mass * $mass \n").substitute(mass = self.drag_mass.magnitude) + \
            "\n".join(
            [st.Template("set atom $id mass $mass susceptibility $susc diameter $diameter").substitute(
                mass = self.drag_mass.magnitude, susc = self.susceptibility,
                diameter=2*self.radius.magnitude, id=part_id+1) for (part_id,pos) in enumerate(self.positions)]
        )+"\n"
        

class world():
    """ Real world parameters like the temperature. Also the confining walls
    Sets world parameters, like temperture, region, dipole cutoff and such.
    
    the lj and dipole parameters are in units of radius. 
    
    If the dipole_cutoff is not given, the program should calculate a default cutoff 
            as the length when the field is reduced to a fraction of KbT. 
            .. to do::
    """
    def __init__(self, particles,
                temperature = 300*ureg.K, region = [200,200,20]*ureg.um,
                boundaries = ["s","s","f"], walls=[False,False,True],
                dipole_cutoff = None, lj_cutoff = 1, 
                lj_parameters = [1e-2*ureg.pg*ureg.um**2/ureg.us**2, 2**(-1/6)],**kargs):

        self.particles=particles
        
        self.temperature = temperature
        self.region = region
        self.boundaries = boundaries
        self.walls = walls
        self.dipole_cutoff = dipole_cutoff #um
        self.lj_cutoff = lj_cutoff # sigma
        self.lj_parameters = lj_parameters #[pg um^2 us^-2,sigma]
        
        
        if len(self.region)==3:
            self.region = [p*s/2 for s in self.region for p in [-1,1]]
        
        if not dipole_cutoff:
            if "field" in kargs:
                pass
                
    def create_string(self):
        
        self.world_def = st.Template("""
units micro
atom_style hybrid sphere paramagnet
boundary $x_bound $y_bound $z_bound
neighbor 4.0 nsq
pair_style lj/cut/dipole/cut $lj_cut $dpl_cut
""")
        self.world_def = self.world_def.substitute(
                                x_bound = self.boundaries[0],
                                y_bound = self.boundaries[1],
                                z_bound = self.boundaries[2],
                                lj_cut = self.lj_cutoff,
                                dpl_cut = self.dipole_cutoff.magnitude
                                )
        
        
        self.region_def = st.Template("""
region space block $spx1 $spx2 $spy1 $spy2 $spz1 $spz2 # this is in microns
create_box 1 space
""")
            
        self.region_def = self.region_def.substitute(
            spx1 = self.region[0].magnitude,
            spx2 = self.region[1].magnitude,
            spy1 = self.region[2].magnitude,
            spy2 = self.region[3].magnitude,
            spz1 = self.region[4].magnitude,
            spz2 = self.region[5].magnitude)
        
        self.group_def = st.Template("""
group Atoms type 1
pair_coeff * * $lj_eps $lj_sgm $lj_cut $dp_cut 
""").substitute(
                lj_eps=self.lj_parameters[0].magnitude,
                lj_sgm=((2*self.particles.radius)*self.lj_parameters[1]).magnitude,
                lj_cut=((2*self.particles.radius)*self.lj_cutoff).magnitude,
                dp_cut=self.dipole_cutoff.magnitude
                )
        
        damp = 1e-3*ureg.us
        # self.damp = diffusion*mass/(kb*self.temperature)
        # mass = damp/(diffusion/(KbT)) = damp/drag
        
        self.seed = np.random.randint(1000000)
        
        self.integrator_def = st.Template("""
fix 	1 Atoms bd $temp $damp $seed 
""").substitute(temp=self.temperature.magnitude, damp=damp.magnitude, seed=self.seed)
        
        self.gravity = (
            self.particles.mass*(9.8*ureg.m/ureg.s**2)
            ).to(ureg.pg*ureg.um/ureg.us**2)
        
        self.gravity_def = st.Template("""
fix     2 Atoms addforce 0 0 $mg
""").substitute(mg = -self.gravity.magnitude) # pg*um/(us^2) (I hope)
        
        if any(self.walls):
            walls = [
                "%slo EDGE $lj_eps $lj_sgm  $lj_cut %shi EDGE $lj_eps $lj_sgm  $lj_cut "%(r,r)
                if w else "" for (r,w) in zip(["x","y","z"],self.walls)]
            walls = "fix 	3 Atoms wall/lj126 "+"".join(walls)+" \n"
        else: 
            walls = ""
        
        self.wall_def = st.Template(walls).substitute(
                lj_eps=self.lj_parameters[0].magnitude,
                lj_sgm=((self.particles.radius)*self.lj_parameters[1]).magnitude,
                lj_cut=((self.particles.radius)*self.lj_cutoff).magnitude,
                )
    def reset_seed(self, seed = None):
        """ Resets the seed of the world object for it to be used again. If the seed parameter is used, the seed is set to that. If not, it is a random number between 1 and 1000000 """
        
        if seed:
            self.seed = seed
        else:
            self.seed = np.random.randint(1000000)
            
                
        
class field():
    def __init__(self,
                magnitude = 10*ureg.mT,
                frequency = 0*ureg.Hz,
                angle = 0*ureg.degree, 
                fieldx=None, fieldy = None, fieldz = None):
        """
        Characteristics of the field that sets the dipole moment of superparamagnetic particles
        It's normally a precessing field, 
        but every parameter can accept a function as a string in the lammps standard.
        Also, the components of the field can be set to a different function by passing a string 
        to the parameters `fieldx`,`fieldy`,`fieldz`.
        Note however that the field functions should be in lammps units
        """
        # in the future the parameters could be input as a dictionary, This would allow us to input a variable number of parameters, specific to the function being applied.
        
        permeability = (4e5*np.pi)*ureg.pN/ureg.A**2 #pN/A^2
        
        #What is lammps units?
                
        self.magnitude = magnitude
        self.H_magnitude = (magnitude.to(ureg.mT)/permeability)*1e9/2.99e8 #mT to lammps units
        self.frequency = frequency.to(ureg.MHz)
        self.angle = angle.to(ureg.rad)#degrees to radians
                
        self.fieldx = "v_Bmag*sin(v_omega*time)*sin(v_theta)"
        self.fieldy = "v_Bmag*cos(v_omega*time)*sin(v_theta)"
        self.fieldz = "v_Bmag*cos(v_theta)"
        
    def create_string(self):
        self.variable_def = st.Template("""
variable Bmag atom $magnitude
variable omega atom $omega
variable theta atom $theta

variable fieldx atom $fieldx
variable fieldy atom $fieldy
variable fieldz atom $fieldz
""").substitute(
                magnitude = self.H_magnitude.magnitude,
                omega = self.frequency.magnitude*2*np.pi,
                theta = self.angle.magnitude,
                fieldx = self.fieldx,
                fieldy = self.fieldy,
                fieldz = self.fieldz
                )
        
        self
        self.fix_def = """
fix 	4 Atoms setdipole v_fieldx v_fieldy v_fieldz 
"""
                   
