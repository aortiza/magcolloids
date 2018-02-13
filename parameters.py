class particle():
    def __init__(self,position,**kargs):
        """
        Initializes a set of particle properties. Normally, for each particle we define a particle properties object. 
        the required parameter is the initial position
        the position has to be either the string "random" or a list of three numbers. 
        It the position is specified as "random", the program requires also a space specified. 
        
        Other optional parameters are:
            radius (default = 4um)
            susceptibility (default = 1)
            diffusion (default = 1um^2/s)
        """
        # defaults
        self.radius = 1 #um
        self.susceptibility = 0.1
        self.diffusion = 1 #um^2/s
        
        if (not isinstance(position, (str))) & (len(position)==3):
            self.initial_position = [position[0],position[1],position[2]] #um
        elif position.lower() == "random".lower():
            if 'space' in kargs:
                space = kargs['space']
                if len(space['region'])==3:
                    self.initial_position = [s*r-s/2 for s,r in zip(space['region'],np.random.rand(3))]
                if len(space['region'])==6:
                    size = space['region'][1::2]-space['region'][0::2]
                    center = space['region'][1::2]+space['region'][0::2]/2
                    self.initial_position = [s*r-s/2 for s,r in zip(size,np.random.rand(3))]
                    self.initial_position = [r+c for s,r in zip(center,self.initial_position)]
            else:
                #If we expect a random starting position, we need Space to be defined. 
                #that can be input either by introducing the sim_parameters or the space dictionary.
                raise Exception("I can't place particles randomly if I don't know the coordinates of the space")
        else:
            raise Exception("This is an invalid value for the required argument position")
        
        if 'susceptibility' in kargs: self.susceptibility = kargs['susceptibility']
        
        if 'diffusion' in kargs: self.diffusion = kargs['diffusion']
        elif 'drag' in kargs: 
            drag = kargs['drag'] #pN/(nm/s)
            if 'temperature' in kargs:
                temp = kargs['temperature']
                KbT = 4/300*temp
                self.diffusion = KbT/drag*1e-6 #pN nm / pN*(nm/s) = nm^2/s = 1e-6um^2/nm^2*nm^2/s = 1e-6*um^2/s

        
        if 'radius' in kargs:
            self.radius = kargs['radius']
            if 'diameter' in kargs:
                warn('You have too many particle size specifications')
        elif 'diameter' in kargs:
            self.radius = kargs['diameter']/2
            if 'radius' in kargs:
                warn('You have too many particle size specifications')

class run():
    def __init__(self,**kargs):
        """ 
        Optional keyword parameters are:
        timestep (default = 1e-3 sec)
        framerate (default = 30 sec)
        total_time (default = 60 sec)
        """
        
        self.timestep = 1e-3 #s
        self.framerate = 30 #s
        self.total_time = 60 #s
        
        if 'timestep' in kargs: self.timestep = kargs['timestep']
        if 'framerate' in kargs: self.framerate = kargs['framerate']
        if 'total_time' in kargs: self.total_time = kargs['total_time']
        
class field():
    def __init__(self,**kargs):
        """
        Optional keyword parameters are:
        magnitude (= 10mT)
        frequency (= 10Hz)
        angle (= 30º)
        dipole_cutoff (= 200um)
        lj_cutoff (=10um)
        walls (=[-5um,5um])
        """
        self.magnitude = 10 #mT
        self.frequency = 10 #Hz
        self.angle = 30 #degrees
        self.dipole_cutoff = 200 #um
        self.lj_cutoff = 10 #um
        self.lj_parameters = [1,1] #[pNum,um]
        self.walls = [-5,5] #um
        
        if 'magnitude' in kargs: self.magnitude = kargs['magnitude']
        if 'frequency' in kargs: self.frequency = kargs['frequency']
        if 'angle' in kargs: self.angle = kargs['angle']
        if 'dipole_cutoff' in kargs: self.dipole_cutoff = kargs['dipole_cutoff']
        if 'lj_cutoff' in kargs: self.lj_cutoff = kargs['lj_cutoff']
        if 'lj_parameters' in kargs: self.lj_parameters = kargs['lj_parameters']
        if 'walls' in kargs: self.walls = kargs['walls']
        
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
        self.temperature = 300
        self.space = {"region":[200,200,20],"boundary":["s","s","f"]}
        self.file_name = "test"
        self.dir = ""
        self.stamp_time = False
        
        if 'temperature' in kargs: self.temperature = kargs['temperature']
        if 'space' in kargs: self.space = kargs['space']
        if 'file_name' in kargs: self.file_name = kargs['file_name']
        if 'dir' in kargs: self.dir = kargs['dir']
        if 'stamp_time' in kargs: self.stamp_time = kargs['stamp_time']
            
        if len(self.space["region"])==3:
            self.space["region"] = [p*s/2 for s in self.space["region"] for p in [-1,1]]