import string as st
import numpy as np
import copy as cp
from . import ureg

""" Here we should change the abstraction: instead of a collection of particles with parameters, perhaps it would be more adequate to define a collection of particles with a single set of parameters. The program could then create a script from several collections of particles, which would allow different parameters. This is more in line with the way of lammps, and therefore probably simpler """

class particles():
    """ A type of particle to be simulated """
    def __init__(self, positions,
                atoms_id = None,
                atom_type = 0,
                radius = 2*ureg.um,
                susceptibility = 1,
                drag = 4e6*ureg.pN/(ureg.um/ureg.s), 
                diffusion = None, temperature = None, 
                density = 1000*ureg.kg/ureg.m**3,
                susceptibility_spread = 0):
        """
        Initializes a particle type. 
        The diffusion coefficient can be given instead o the drag.
        In that case, the temperature is also needed to calculate the drag. 
        This represents the density mismatch of the particle in the solvent.
        The `atoms_id` is a list of integers that uniquelly identifies each particle. It must be the same size as the `positions array`.
        The `atom_type` is an integer that uniquely identifies this group of particles. 
        Both arguments are necessary when more than one type of particles is initialized.
        """        
        if diffusion:
            
            kB = (1.38064852e-23*ureg.J/ureg.K).to(ureg.pN*ureg.nm/ureg.K)
            
            KbT = kB*temperature
            drag = KbT/diffusion
            
        self.positions = positions.to(ureg.um)
        self.radius = radius.to(ureg.um)
        self.susceptibility = susceptibility
        self.drag = drag.to(ureg.pg/ureg.us)
        self.mass = (density*4/3*np.pi*(radius)**3).to(ureg.pg)
        
        damp = 1e-3*ureg.us
        
        self.drag_mass = (drag*damp).to(ureg.pg)
        
    def create_string(self):
        """ creates the strings that then are introduced into the lammps scripts"""
        
        density = self.mass/(4/3*np.pi*self.radius**3)
        atom_string = ("atom id|type|x0|y0|z0|d|rho|m|mx|my|mz|chi|trap_id")
        
        ### This is the string pattern for a single atom. The colloid positions are left as a template, as well as the atom id. 
        self.atom_def = st.Template(
            """$atom_ix\t $atom_type\t $center\t $diameter\t $density\t $moment\t $direction\t $susceptibility\t $trap_ix #$string""").substitute(
                atom_ix = "$atom_ix",
                atom_type = self.atom_type,
                center = "$x0\t $y0\t $z0",
                diameter = (self.radius*2).magnitude,
                density = ((self.drag_mass)/(4/3*np.pi*self.radius**3)).magnitude,
                moment = 0.0,
                direction = "0\t 0\t 0",
                susceptibility = self.susceptibility,
                trap_ix = "$atom_trap",
                string = atom_string
        )
        
        # Now we use the string pattern defined before, and define an instance for each trap.
        self.atom_def = "\n".join(
            [st.Template(self.atom_def).substitute(
                atom_ix = ix,
                x0 = c[0], y0 = c[1], z0 = c[2],
                atom_trap = ix)
                    for ix,c in zip(self.atoms_id,self.positions.magnitude)
            ]
        )+"\n"
                
class bistable_trap():
    """ A bistable potential """
    def __init__(self, 
                positions,
                directions,
                particles,
                subsets = None,
                trap_id = None,
                atom_type = 1,
                bonds_id = None,
                distance = 10*ureg.um,
                height = 4 * ureg.pN*ureg.nm,
                stiffness = 1.2e-4 * ureg.pN/ureg.nm,
                height_spread = 0,
                cutoff = np.Inf*ureg.um,
                velocity = None):
        """
        Initializes a group of bistable traps with the same parameters.
        The required arguments are: 
            * positions
            * directions
            * particles (can be an array of id's or a particles object)
        trap_id is an integer that uniquelly identifies each atom. This means that a trap can't have a trap_id if an atom has it. 
        atom_type is the an integer that identifies the trap type. Again, the uniqueness holds also to atoms.
        """
        
        self.positions = positions.to(ureg.um)
        self.directions = np.array(directions)
        
        self.distance = distance.to(ureg.um)
        self.height = height.to(ureg.pg*ureg.um**2/ureg.us**2)
        self.stiffness = stiffness.to(ureg.pg/ureg.us**2)
        self.height_spread = height_spread
        
        self.particles = particles
        
        if subsets is None:
            try: 
                subsets = [slice(None) for p in particles]
            except TypeError:
                subsets = slice(None)
                
        self.subsets = subsets
                  
        self.atom_type = atom_type
        self.cutoff = cutoff
        self.velocity = velocity
    
    def bond_to_particles(self):
        """ Assign bonds from traps to particles. 
        
        Assigns bonds to the particles specified in self.particles.
        self.particles is a particle object or a list of particle objects, but the bonds are defined only on the subset defined by the ids_subset attribute.
        todo: think this through """
        
        try:
            # this happens if particles is only one element
            self.bonds = self.particles.atoms_id[self.subsets]
            self.bonded_atom_type = self.particles.atom_type + np.zeros(len(self.particles.atoms_id))
            
        except AttributeError:
            # this happens if particles is an array
            
            # self.bonds should be an array that contains those atoms to which this trap is bonded. 
            # This is calculated from a subset array which is input to the trap type. 
            
            atom_id = lambda s: self.particles[s[0]].atoms_id[s[1]]
            atom_type = lambda s: self.particles[s[0]].atom_type
            
            self.bonds = np.array([atom_id(s) for s in self.subsets])
            
            self.bonded_atom_type = np.concatenate([atom_type(s) + np.zeros(len(self.particles[s[0]].atoms_id)) for s in self.subsets])
        
    def create_string(self):
        
        # We first define the string pattern that defines a single trap
        atom_string = ("trap id|type|x0|y0|z0|1|1|0|dx|dy|dz|0|atom_id")
        
        self.atom_def = st.Template(
            "$trap_ix\t $atom_type\t $center\t $diameter\t $density\t $moment\t $direction\t $susceptibility\t $atom_ix #$string").substitute(
                trap_ix = "$trap_ix",
                atom_type = self.atom_type,
                center = "$x0\t $y0\t $z0",
                diameter = 1,
                density = 1,
                moment = 0,
                direction = "$dx0\t $dy0\t $dz0",
                susceptibility = 0,
                atom_ix = "$atom_ix",
                string = atom_string)
        
        # Now we use the string pattern defined before, and define an instance for each trap.
        self.atom_def = "\n".join(
            [st.Template(self.atom_def).substitute(
                trap_ix = ix,
                x0 = c[0], y0 = c[1], z0 = c[2],
                dx0 = d[0], dy0 = d[1], dz0 = d[2],atom_ix=a_ix)
                    for ix,c,d,a_ix in zip(
                        self.traps_id, 
                        self.positions.magnitude, self.directions*self.distance.magnitude,
                        self.bonds)])+"\n"
        
        if self.cutoff == np.inf*ureg.um:
            # The bonds are defined as a conection from a trap, to an atom. the bond type is equal to the bond id (bond_ix) because every bond is unique. Only that way can disorder be introduced
            
            self.bond_def = "\n".join([st.Template(
                """$bond_ix $bond_type $trap_ix $atom_ix""").substitute(
                    bond_ix = b,
                    bond_type = b,
                    trap_ix = i,
                    atom_ix = j) for b,i,j in zip(self.bonds_id,self.traps_id,self.bonds)]
            )+"\n"
            # The bond params are defined by bond type. These determine the stiffness and height of the traps
            height_disorder = self.height *(np.random.randn(len(self.traps_id))*self.height_spread+1)
            height_disorder = height_disorder.to(ureg.pg*ureg.um**2/ureg.us**2).magnitude
        
            self.bond_params = "\n".join([st.Template(
                """$bond_type  $stiffness $height_energy""").substitute(
                        bond_type = i, stiffness = self.stiffness.magnitude, height_energy = h
                    ) for i,h in zip(self.bonds_id,height_disorder)])+"\n"
            # If there is no cuttoff, the trap doesn't have a pair interaction. 
                    
        else: 
            # If there is a cuttoff, the trap is defined as a pair interaction between types. For the moment this means the disorder in the height is ignored. 
            
            self.pair_def = st.Template("\n".join([st.Template(
                """$atom_type $trap_type biharmonic $k_outer $k_inner $cutoff""").substitute(
                    atom_type = "$type_i",
                    trap_type = "$type_j",
                    k_outer = self.stiffness.magnitude,
                    k_inner = self.height.to(ureg.pg*ureg.um**2/ureg.us**2).magnitude,
                    cutoff = self.cutoff.to(ureg.um).magnitude)]))
                    
        if self.velocity is not None:
            self.velocity_fix = "fix		7 Traps move variable NULL NULL NULL v_vx v_vy v_vz"
        else:
            self.velocity_fix = ""
            
class ext_force(): 
    
    def __init__(self, calculation, variable = "v_F"):
        
        self.calculation = calculation
        self.variable = variable
                
    def create_string(self):
        
        variables = "%sx %sy %sz"%tuple(3*[self.variable])
        self.fix_str = "fix		8 Atoms addforce "+variables
    
class world():
    def __init__(self, particles,
                traps = None,
                temperature = 300*ureg.K, region = [200,200,20]*ureg.um,
                boundaries = ["s","s","p"], walls=[False,False,True],
                dipole_cutoff = 200*ureg.um, lj_cutoff = 1, 
                lj_parameters = [1e-2*ureg.pg*ureg.um**2/ureg.us**2, 2**(-1/6)],
                gravity = 9.8*ureg.m/ureg.s**2,
                enforce2d = False, ext_force = None):
                
        """ Real world parameters like the temperature. Also the confining walls
        Sets world parameters, like temperture, region, dipole cutoff and such.
    
        the lj and dipole parameters are in units of radius. 
        .. to do::
                If the dipole_cutoff is not given, the program should calculate a default cutoff as the length when the field is reduced to a fraction of KbT. 
        """
        if particles.__class__.__name__ == "particles":
            self.particles=[particles]
        else:
            self.particles=particles
            
        if traps.__class__.__name__ == "bistable_trap":
            self.traps=[traps]
        else:
            self.traps=traps
        
        traps = self.traps
        if traps is None:
            cut_bh = 0
        else:    
            lengths = np.array([t.distance.to(ureg.um).magnitude for t in traps])
            cutoffs = np.array([t.cutoff.to(ureg.um).magnitude for t in traps])
            cut_bh = lengths+cutoffs
            if cut_bh[cut_bh!=np.inf].size==0:
                # no finite traps. cutoff is 0 because traps are defined through bonds, not pairs.
                cut_bh = 0
            else:
                # some finite traps. Global cutoff is the maximum cutoff possible. 
                cut_bh = np.max(cut_bh[cut_bh!=np.inf])
                
        self.bh_cutoff = cut_bh
        self.temperature = temperature
        self.region = region
        self.boundaries = boundaries
        self.walls = walls
        self.dipole_cutoff = dipole_cutoff.to(ureg.um) #um
        self.lj_cutoff = lj_cutoff # sigma
        self.lj_parameters = lj_parameters #[pg um^2 us^-2,sigma]
        self.gravity = gravity.to(ureg.um/ureg.us**2)
        self.enforce2d = enforce2d
                
        if len(self.region)==3:
            self.region = [p*s/2 for s in self.region for p in [-1,1]]
        
        self.ext_force = ext_force
    
    def create_wall_string(self,fx_no):
    
        if any(self.walls):
            walls = [
                "%slo EDGE $lj_eps $lj_sgm  $lj_cut %shi EDGE $lj_eps $lj_sgm  $lj_cut "%(r,r)
                if w else "" for (r,w) in zip(["x","y","z"],self.walls)]
            walls = "fix 	$fx_no Atoms wall/lj126 "+"".join(walls)+" \n"
        else: 
            walls = ""
            
        return st.Template(walls).substitute(
                    fx_no=fx_no,
                    lj_eps=self.lj_parameters[0].magnitude,
                    lj_sgm=((self.particles[0].radius)*self.lj_parameters[1]).magnitude,
                    lj_cut=((self.particles[0].radius)*self.lj_cutoff).magnitude,
                    )

    def header_string(self):
        """ Define the header string of the input and script files. 
        * The enforce 2d string if necessary
        * The units
        * The atom_style, pair_style and bond_style
        * The size of the neighbor skin.
        * region parameters:
            * number of atoms and of traps.
            * number of atom types and trap types. Note that atoms and traps can be assigned different parameters even if they are the same type. The atom and trap types and the trap normally determine what fixes are applied to them. 
            * the region definition. 
        """
        if self.enforce2d:
            dimension = "dimension 2"
        else:
            dimension = ""

        particle_types = len(self.particles)
        total_particles = sum([len(p.positions) for p in self.particles])

        if not self.traps is None:
            trap_types = len(self.traps)
            total_bond_traps = sum([len(t.positions) for t in self.traps if t.cutoff==np.Inf*t.cutoff.units])
            total_pair_traps = sum([len(t.positions) for t in self.traps if t.cutoff<np.Inf*t.cutoff.units])
        else:
            trap_types = 0
            total_bond_traps = 0
            total_pair_traps = 0
        # Traps can be pair traps, or bond traps. 
        # pair traps are finite. They have a cuttoff after which they stop being effective. They also can act on any atom that comes close to them. 

        # bond traps act on a specific set of particles. They can also have a cuttoff, but if a particle leaves the trap, it will keep being associated to it. It cannot then be trapped by an adjoining trap.
        if total_pair_traps>0:  
            self.world_def = st.Template(
            "\n" + \
            "units micro \n" +\
            "atom_style hybrid sphere paramagnet bond \n" +\
            "boundary $x_bound $y_bound $z_bound \n" +\
            "$dimension \n" +\
            "neighbor 4.0 nsq \n" +\
            "pair_style hybrid biharmonic $bh_cut lj/cut/dipole/cut $lj_cut $dpl_cut \n" +\
            "bond_style biharmonic \n")
            
            self.world_def = self.world_def.substitute(
                                x_bound = self.boundaries[0],
                                y_bound = self.boundaries[1],
                                z_bound = self.boundaries[2],
                                bh_cut = self.bh_cutoff,
                                lj_cut = self.lj_cutoff,
                                dpl_cut = self.dipole_cutoff.magnitude,
                                dimension = dimension
                                )
        else:
            self.world_def = st.Template(
            "\n" + \
            "units micro \n" +\
            "atom_style hybrid sphere paramagnet bond \n" +\
            "boundary $x_bound $y_bound $z_bound \n" +\
            "$dimension \n" +\
            "neighbor 4.0 nsq \n" +\
            "pair_style hybrid lj/cut/dipole/cut $lj_cut $dpl_cut \n" +\
            "bond_style biharmonic\n")

            self.world_def = self.world_def.substitute(
                                            x_bound = self.boundaries[0],
                                            y_bound = self.boundaries[1],
                                            z_bound = self.boundaries[2],
                                            bh_cut = self.bh_cutoff,
                                            lj_cut = self.lj_cutoff,
                                            dpl_cut = self.dipole_cutoff.magnitude,
                                            dimension = dimension
                                            )

        self.region_def = st.Template(
        "\n" + \
        "$total_atoms atoms \n" +\
        "$atom_types atom types \n" +\
        "$bonds bonds \n" +\
        "$bonds bond types \n" +\
        "$spx1 $spx2 xlo xhi \n" +\
        "$spy1 $spy2 ylo yhi \n" +\
        "$spz1 $spz2 zlo zhi \n")

        self.region_def = self.region_def.substitute(
            total_atoms = total_particles+total_bond_traps+total_pair_traps,
            atom_types = particle_types+trap_types,
            bonds = total_bond_traps,
            spx1 = self.region[0].magnitude,
            spx2 = self.region[1].magnitude,
            spy1 = self.region[2].magnitude,
            spy2 = self.region[3].magnitude,
            spz1 = self.region[4].magnitude,
            spz2 = self.region[5].magnitude)
    
    def interaction_strings(self):
        """ Create the strings that define the interactions between atoms."""
        
        # This template is used to define interactions between atoms.
        self.interaction_def =st.Template(
            """ 1 1 $lj_eps $lj_sgm $lj_cut $dp_cut""").substitute(
                lj_eps=self.lj_parameters[0].magnitude,
                lj_sgm=((2*self.particles[0].radius)*self.lj_parameters[1]).magnitude,
                lj_cut=((2*self.particles[0].radius)*self.lj_cutoff).magnitude,
                dp_cut=self.dipole_cutoff.magnitude
            )
            
        lj_interaction_def = st.Template(
            """$type_i $type_j lj/cut/dipole/cut $lj_eps $lj_sgm $lj_cut $dp_cut""")
        # particles interact with a bond trap thrnough a null interaction. The interaction is defined by the bond type, not by the interaction 
        null_interaction_def = st.Template(
            """$type_i $type_j none""")
                 
        self.interaction_def = []
        
        # colloidal particle parameters are defined in their string in the lmpdata file. A single atom type can have different parameters, without creating different atom types. Atom types could be useful if we wanted, for example, to drive a subset of atoms. But this is not what we are doing, and we might be able to do it in another way. 
        # In any case, particles should all interact by the same interactions, which is what makes them particles. It makes sense to define different interactions for example for traps, and then we can get a different atom type. 
        
        # UPDATE: Most parameters can be set individually on the atom definition line of the input file. But the solid-like interactions of finite size colloids is given by their Lennard-Jones parameter, which is defined for each interaction. 
        # This is a bit of a pain in the ass from lammps, because the radius which is defined for the volume could be directly used to rescale the LJ potential. But it isn't, and different radius of particles interact with each other through potentials with different parameters. 
        
        
        for i,pi in enumerate(self.particles):
            for j,pj in enumerate(self.particles[i:]):
                self.interaction_def.append(lj_interaction_def.substitute(
                    type_i = pi.atom_type, type_j = pj.atom_type,
                    lj_eps=self.lj_parameters[0].magnitude,
                    lj_sgm=((pj.radius+pi.radius) * self.lj_parameters[1]).magnitude,
                    lj_cut=((pj.radius+pi.radius) * self.lj_cutoff).magnitude,
                    dp_cut = self.dipole_cutoff.magnitude))
             
        for i,pi in enumerate(self.particles):
            if self.traps is not None:
                for j,tj in enumerate(self.traps):
                    if tj.cutoff == np.inf*ureg.um:
                        self.interaction_def.append(null_interaction_def.substitute(
                            type_i = pi.atom_type, type_j = tj.atom_type))
                    else:
                        if pi.atom_type in tj.bonded_atom_type:
                            self.interaction_def.append(tj.pair_def.substitute(
                                type_i = pi.atom_type, type_j = tj.atom_type))
                        else:
                            self.interaction_def.append(null_interaction_def.substitute(
                                type_i = pi.atom_type, type_j = tj.atom_type))
                                
        if self.traps is not None:
            for i,ti in enumerate(self.traps):
                for j,tj in enumerate(self.traps[i:]):
                    self.interaction_def.append(null_interaction_def.substitute(
                            type_i = ti.atom_type, type_j = tj.atom_type))
                    
        self.interaction_def = "\n".join(self.interaction_def)
    
        #atom_group = "group Atoms type $particle_"
        self.group_def = "\n"+"\n".join([
            st.Template(
                """group Atoms type $particle_type \n""").substitute(
                    particle_type = p.atom_type
                ) for p in self.particles
            ])
        if self.traps is not None:    
            self.group_def += "\n".join([
                st.Template(
                    """group Traps type $particle_type \n""").substitute(
                        particle_type = p.atom_type
                        ) for p in self.traps
                    ])
                        
    def create_string(self):
        """ Creates the strings that define the world parameters. These strings will be used by the simulation class to build the LAMMPS script and input file (.lmpin and .lmpdata).
        to do: 
        divide and conquer. 
        """
        self.header_string()
       
        self.interaction_strings()
            
        self.group_def += """mass * 1\n"""
        
        damp = 1e-3*ureg.us
        # self.damp = diffusion*mass/(kb*self.temperature)
        # mass = damp/(diffusion/(KbT)) = damp/drag
        
        self.seed = np.random.randint(1000000)
        
        fx_no = 2
        self.integrator_def = st.Template(
            "\nfix 	$fx_no Atoms bd $temp $damp $seed \n").substitute(
                fx_no = fx_no, temp=self.temperature.magnitude, damp=damp.magnitude, seed=self.seed)
        
        fx_no = 3
        self.gravity_force = (
            self.particles[0].mass*(self.gravity)
            ).to(ureg.pg*ureg.um/ureg.us**2)
        
        fix_no = 4
        self.gravity_def = st.Template("\nfix     $fx_no Atoms addforce 0 0 $mg \n").substitute(fx_no = fx_no, mg = -self.gravity_force.magnitude) # pg*um/(us^2) (I hope)
        
        fx_no = 5
        if self.enforce2d:
            self.enforce2d = "\nfix 	%u all enforce2d_bd\n"%fx_no
        else:
            self.enforce2d = ""

        fx_no = 6
        self.wall_def = self.create_wall_string(fx_no)
        
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
                phase = 0*ureg.degree, 
                fieldx=None, fieldy = None, fieldz = None,
                multibody_iter = 0):
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
        
        try:
            self.frequency = frequency.to(ureg.MHz)
        except TypeError:
            # TypeError ocurs for strings. This is a manual unit conversion for that case. 
            self.frequency = \
                (frequency.magnitude+("*%g" % (1*frequency.units).to(ureg.MHz).magnitude))*ureg.MHz
        
        self.angle = angle.to(ureg.rad)#degrees to radians
        self.phase = phase.to(ureg.rad)#degrees to radians
                
        self.multibody_iter = multibody_iter
        self.fieldx = "v_Bmag*cos(v_freq*time*2*PI+v_phi)*sin(v_theta)"
        self.fieldy = "v_Bmag*sin(v_freq*time*2*PI+v_phi)*sin(v_theta)"
        self.fieldz = "v_Bmag*cos(v_theta)"
        
    def create_string(self):

        self.variable_def = st.Template(
            "variable Bmag atom $magnitude\n"+\
            "variable freq atom $freq\n"+\
            "variable theta atom $theta\n"+\
            "variable phi atom $phase\n"+\
            "\n"+\
            "variable fieldx atom $fieldx\n"+\
            "variable fieldy atom $fieldy\n"+\
            "variable fieldz atom $fieldz\n" ).substitute(
                    magnitude = self.H_magnitude.magnitude,
                    freq = self.frequency.magnitude,
                    theta = self.angle.magnitude,
                    fieldx = self.fieldx,
                    fieldy = self.fieldy,
                    fieldz = self.fieldz,
                    phase = self.phase.magnitude
                    )
        
        self
        self.fix_def = "\nfix 	1 Atoms setdipole v_fieldx v_fieldy v_fieldz %u"%self.multibody_iter
                   
