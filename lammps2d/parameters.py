import string as st
class particle():
    """ A type of particle to be simulated
    
    Attributes:
        position (default = [0,0,0])
        radius (default = 4um)
        susceptibility (default = 1)
        drag (default 4e6 pN s/nm)
    
    Methods:
    """
    def __init__(self, position = [0,0,0], radius = 4,
                susceptibility = 1, drag = 4e6, 
                diffusion = None, temperature = None):
        """
        Initializes a single particle. 
        This particle can then be coppied to the positions given by a vector.
        The diffusion coefficient can be given instead o the drag.
        In that case, the temperature is also needed to calculate the drag. 
        """
        
        if diffusion:
            
            KbT = 4/300*temperature
            drag = KbT/diffusion*1e-6 # pN nm / um^2*s (um/1nm)^2
            
        self.position = position
        self.radius = radius
        self.susceptibility = susceptibility
        self.drag = drag
            
    def copy(self, positions):
        """ Copies the atom to a list of atoms. 
        The location of the atoms is given by the positions vector."""
        
        particle_list = [self for p in positions]
        
        for i,part in enumerate(particle_list):
            part.position = positions[i]
            
        return particle_list
        
    def create_string(self):
        
        self.atom_def = st.Template(
            "create_atoms 1 single $x0 $y0 $z0").substitute(
                x0=self.position[0],y0=self.position[1],z0=self.position[2])
        
        self.atom_prop = st.Template(
            "set atom $id mass $mass susceptibility $susc diameter $diameter"
            ).substitute(mass = "$mass", susc = self.susceptibility,
            diameter=2*self.radius, id="$id")
        

class world():
    def __init__(self, temperature = 300, region = [200,200,20],
                boundaries = ["s","s","f"], walls=[False,False,True],
                dipole_cutoff = None, lj_cutoff = 1, 
                lj_parameters = [1e2,2**(-1/6)],**kargs):
        """
        Sets world parameters, like temperture region, dipole cutoff and such.
        the lj and dipole parameters are in units of radius. 
        
        If the dipole_cutoff is not given, the program should calculate a default cutoff 
                as the length when the field is reduced to a fraction of KbT. 
                .. to do::
        """
        self.temperature = temperature
        self.region = region
        self.boundaries = boundaries
        self.walls = walls
        self.dipole_cutoff = dipole_cutoff #um
        self.lj_cutoff = lj_cutoff # sigma
        self.lj_parameters = lj_parameters #[pg um^2 us^-2,sigma]
            
        if len(self.region)==3:
            self.region = [p*s/2 for s in self.region for p in [-1,1]]
        
        if dipole_cutoff:
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
        self.world_def = self.region_def.substitute(
                                x_bound = self.boundaries[0],
                                y_bound = self.boundaries[1],
                                z_bound = self.boundaries[2],
                                lj_cut = self.lj_cutoff,
                                dpl_cut = self.dipole_cutoff
                                )
        
        
        self.region_def = st.Template("""
            region space block $spx1 $spx2 $spy1 $spy2 $spz1 $spz2 # this is in microns +
            create_box 1 space
            """)
            
        self.region_def = self.region_def.substitute(
            spx1 = self.region[0],
            spx2 = self.region[1],
            spy1 = self.region[2],
            spy2 = self.region[3],
            spz1 = self.region[4],
            spz2 = self.region[5])
        
        self.group_def = st.Template("""
            group Atoms type 1
            pair_coeff * * $lj_eps $lj_sgm $lj_cut $dp_cut 
            """).substitute(
                lj_eps=self.lj_parameters[0],
                lj_sgm=self.lj_parameters[1],
                lj_cut=,
                dp_cut=
                )
        
class field():
    def __init__(self,magnitude = 10, frequency = 0, angle = 0):
        """
        Characteristics of the field that sets the dipole moment of superparamagnetic particles
        It's normally a precessing field, 
        but every parameter can accept a function as a string in the lammps standard.
        """
        self.magnitude = magnitude #mT
        self.frequency = frequency #Hz
        self.angle = angle #degrees

        
class run():
    def __init__(self,timestep = 1e-3, framerate = 30, total_time = 60):
        """ 
        This 
        """
        
        self.timestep = 1e-3 #s
        self.framerate = 30 #s
        self.total_time = 60 #s
        
        if 'timestep' in kargs: self.timestep = kargs['timestep']
        if 'framerate' in kargs: self.framerate = kargs['framerate']
        if 'total_time' in kargs: self.total_time = kargs['total_time']
                
class sim():
    def __init__(self,**kargs):
        """
        Optional keyword parameters are:
        temperature (=300K)
        space (={"region":[200,200,20],"boundary":["s","s","f"]})
        file_name (= test)
        dir (="")
        stamp_time (=False) this parameter determines if the file_name is modified by a timestamp.
            This is important to prevent overwriting experimetns
        """
        self.file_name = "test"
        self.dir_name = ""
        self.stamp_time = False
        
        if 'temperature' in kargs: self.temperature = kargs['temperature']
        if 'space' in kargs: self.space = kargs['space']
        if 'file_name' in kargs: self.file_name = kargs['file_name']
        if 'dir_name' in kargs: self.dir_name = kargs['dir_name']
        if 'stamp_time' in kargs: self.stamp_time = kargs['stamp_time']
            
