import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import matplotlib.collections as clt
import pandas as pd
import string as st

def initial_setup(n_of_particles, packing = 0.3, height = 4, radius=1.4):
    """ 
    This function returns an array of initial positions for confined particles, and
    a region where these particles are enclosed with a packing fraction "packing"
    The particles are initially set in a square array, as far from each other as possible.
    """
    part_in_edge = np.round(np.sqrt(n_of_particles))
    n_of_particles = part_in_edge**2

    area_particle = n_of_particles*radius**2*np.pi
    area_region = area_particle/packing

    length_region = np.sqrt(area_region)
    part_separation = length_region/part_in_edge
    
    x_loc = np.linspace(
        -length_region/2+part_separation/2,
        length_region/2-part_separation/2,part_in_edge)
    y_loc = np.linspace(
        -length_region/2+part_separation/2,
        length_region/2-part_separation/2,part_in_edge)

    [X,Y] = np.meshgrid(x_loc,y_loc)
    Z = np.zeros(np.shape(X))

    initial_positions = np.array([[x,y,z] for (x,y,z) in zip(X.flatten(),Y.flatten(),Z.flatten())])
    
    if part_separation<2*radius:
        raise ValueError("packing is too high")

    region = [np.round(length_region),np.round(length_region),height]
    return region, initial_positions

def animate_trj(trj,sim, ax=False, verb=False, start=0, end=False, step = 1, speedup = 1):
    """
    This function animates the trajectory resulting from a confined dimer simulation.
    It displays the z direction as a colormap and the particles in the x and y direction. 
    The simulation is required as argument to obtain parameters like the region size and the 
    particles radius. 
    Optional parameters are:
    * ax: an axis object to use for creating the plot.
    * start: start time of the simulation if not the whole time is required. The default is 0. 
    * end: end time of the simulation. The default is the total simulation time.
    * step = 1. The framerate, so to speak. 
    * verb = False. If verb = True, the routine prints indicators that is running. 
    * speedup allows us to do faster videos. Default is 1, which means normal ratio.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    idx = pd.IndexSlice
    
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,7))

    particles = trj.index.get_level_values('id').unique()
    n_of_particles = len(trj.index.get_level_values('id').unique())
    
    region = [r.magnitude for r in sim.world.region]
    radius = sim.particles.radius.magnitude
    
    framerate = sim.framerate.magnitude
    runtime = sim.total_time.magnitude
    timestep = sim.timestep.magnitude
    
    lammps_time = 1e6;
    
    #dt_data = np.round(1/(timestep*framerate)) # Data timestep in lammps_time
    
    frames = trj.index.get_level_values('frame').unique().values
    
    if not end:
        end = frames[-1]*timestep
    
    frame_min=start/timestep
    frame_max=end/timestep
    
    frame_id_min = np.where(frames>=frame_min)[0][0]
    frame_id_max = np.where(frames<frame_max)[0][-1]
    
    trj = trj.loc[idx[frames[frame_id_min:frame_id_max:step],:]]
    frames = trj.index.get_level_values('frame').unique().values
    dt_video = np.mean(np.diff(frames))*sim.timestep.magnitude*1000/speedup # video timestep in miliseconds
    
    ax.set_xlim(region[0],region[1])
    ax.set_ylim(region[2],region[3])
    ax.set(aspect='equal')
    ax.set_xlabel("$x [\mu{m}]$")
    ax.set_ylabel("$y [\mu{m}]$")
    
    patches = []
    for i,p in enumerate(particles):
        c = plt.Circle((0, 0), radius)
        patches.append(c)

    p = clt.PatchCollection(patches, cmap=plt.cm.RdBu)
    p.set_array(np.zeros(0))
    p.set_clim([region[4]+radius,region[5]-radius])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(p,label='$z [\mu{m}]$',cax=cax)

    def init():
        ax.add_collection(p)
        return p,

    def animate(frame):
        if verb:
            print("frame[%u] is "%frame,frames[frame])
        for (part_id,particle) in enumerate(particles):
            patches[part_id].center = (
                trj.loc[idx[frames[frame],particle],'x'],
                trj.loc[idx[frames[frame],particle],'y'])
        p.set_paths(patches)
        p.set_array(trj.loc[idx[frames[frame],:],'z'].values)
        ax.add_collection(p)
        return p,
    
    if verb: 
        print("started animating")
        print(frames)
        
    anim = anm.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames), interval=dt_video, blit=True);
    plt.close(anim._fig)

    return anim

def draw_trj(trj,sim,iframe=-1,ax=False):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    """ 
    displays a trajectory statically. 
    If iframe is given, it displays particles in the frame specified by the index iframe. 
    If it isn't given, then it displays the last frame
    """
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,7))
    
    idx = pd.IndexSlice
    particles = trj.index.get_level_values('id').unique()
    n_of_particles = len(trj.index.get_level_values('id').unique())
    frames = trj.index.get_level_values('frame').unique()
    
    region = [r.magnitude for r in sim.world.region]
    radius = sim.particles.radius.magnitude
    
    ax.set_xlim(region[0],region[1])
    ax.set_ylim(region[2],region[3])
    ax.set(aspect='equal')
    ax.set_xlabel("$x [\mu{m}]$")
    ax.set_ylabel("$y [\mu{m}]$")
    
    patches = []
    print(frames,iframe)
    for i,p in enumerate(particles):
        c = plt.Circle(
            (trj.loc[idx[frames[iframe],p],'x'],trj.loc[idx[frames[iframe],p],'y']), radius)
        patches.append(c)

    p = clt.PatchCollection(patches, cmap=plt.cm.RdBu)
    p.set_array(trj.loc[idx[frames[iframe],:],'z'].values)
    p.set_clim([region[4]+radius,region[5]-radius])
    ax.add_collection(p)
            
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    plt.colorbar(p,label='$z [\mu{m}]$',cax=cax)

    return ax
    
def display_animation_referenced(sim):
    trj = sim.load(read_trj=True)
    anim = animate_trj(trj,sim)
    anim.save(sim.base_name+".gif",writer = "imagemagick")
    video_html = st.Template(""" <video controls>
          <source src="$name" type="video/mp4">
            </video> """).substitute(name=sim.base_name+".mp4")
    HTML(video_html)
    return anim

def export_animation(sim,*args,**kargs):
    
    if len(args)<1:
        trj = sim.load(read_trj=True)
    else: 
        trj = args[0]
        
    anim = animate_trj(trj,sim,**kargs)
    anim.save(sim.base_name+".gif",writer = "imagemagick")
    return anim
    
def display_animation_direct(sim,*args,**kargs):

    if len(args)<1:
        trj = sim.load(read_trj=True)
        print("reading"+args)
    else: 
        trj = args[0]

    anim = animate_trj(trj,sim,**kargs)
    return anim.to_html5_video()

def draw_exp_phase_diagram(ax=False):
    from scipy.misc import imread

    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(10,8))

    img = imread('ExperimentPhaseDiagram.png')
    img[:,:,3] = img[:,:,3]*0.5
    extent = [0, 0.5, 2.75, 5.7]

    extent_size = [extent[1]-extent[0],extent[3]-extent[2]]
    img_size = np.shape(img)[0:2]
    extent_ratio = extent_size[0]/extent_size[1]
    img_ratio = img_size[0]/img_size[1]


    ax.imshow(img, zorder=0, extent=extent)
    ax.set(aspect=extent_ratio*img_ratio)
    
    ax.set_xlabel("Area Packing Fraction $\phi$")
    ax.set_ylabel("height")
    return ax
    
## Dimer Finding Functions 
def neighbors_within(distance,points,sim):
    """ Calculates, through cKDTree, a list of those pairs of particles that have a distance of less than "distance".
    """
    
    import scipy.spatial as spp
    if any([bc=='p' for bc in sim.sim_parameters.space["boundary"]]):
        
        region = np.array(sim.sim_parameters.space['region'])
        
        # padding a region prevents particles to be consider dimers 
        # when they are close to the boundary of a non periodic dimension
        pad_region = [0 if bc=='p' else distance*2 for bc in sim.sim_parameters.space["boundary"]]
        region[0::2] = region[0::2]-pad_region
        region[1::2] = region[1::2]+pad_region
        
        tree = spp.cKDTree(
            np.mod(
                (points-region[0::2]), # Centers
                region[1::2]-region[0::2]), # mod wraps particles outside the region
            boxsize = region[1::2]-region[0::2]) # boxsize finds neighbors across the borders of a torus. 
    else:
        tree = spp.cKDTree(points) # everything is easier for closed boundaries

    pair_list = list(tree.query_pairs(distance))        
    # pair lists are really more useful as sets, which are not ordered. When comparing sets {a,b}=={b,a}.
    return  [set([i for i in pair]) for pair in pair_list]
    
def nearest_neighbors(points,sim):
    """ Calculates, through cKDTree, the nearest neighbors of each particle in a frame. 
    """
    
    import scipy.spatial as spp
    if any([bc=='p' for bc in sim.sim_parameters.space["boundary"]]):
        
        region = np.array(sim.sim_parameters.space['region'])
        
        # padding a region prevents particles to be consider dimers 
        # when they are close to the boundary of a non periodic dimension
        pad_region = [0 if bc=='p' else distance*2 for bc in sim.sim_parameters.space["boundary"]]
        region[0::2] = region[0::2]-pad_region
        region[1::2] = region[1::2]+pad_region
        
        tree = spp.cKDTree(
            np.mod(
                (points-region[0::2]), # Centers
                region[1::2]-region[0::2]), # mod wraps particles outside the region
            boxsize = region[1::2]-region[0::2]) # boxsize finds neighbors across the borders of a torus. 
    else:
        tree = spp.cKDTree(points) # everything is easier for closed boundaries

    pair_list = list(tree.query())        
    # pair lists are really more useful as sets, which are not ordered. When comparing sets {a,b}=={b,a}.
    return  [set([i for i in pair]) for pair in pair_list]
    
def dimers(trj, sim, distance=False):
    """ This returns a database of dimers in frames"""
    
    if not distance:
        distance = 2*sim.particle_properties[0].radius
        
    idx = pd.IndexSlice
    frames = trj.index.get_level_values('frame').unique()
    pairs_in_frame = []
    
    for i_frame,frame in enumerate(frames):
        points = trj.loc[idx[frame,:]].filter(("x","y","z")).values
        p_id = trj.loc[idx[frame,:]].index.get_level_values('id').values
        
        pair_list = neighbors_within(distance,points,sim)
        # neighbors_within returns the location of the pair members. 
        # we need to convert that into the id of the pair members.
        pair_list = [{p_id[loc] for loc in pair} for pair in pair_list]
        
        if i_frame>0:
            # here I'll store the id's of dimers that I find. 
            # If the dimmer exists in the previous frame I must give it the same id. Otherwise I give it a new id. 
            # It's then straightforward to include this id in the DataFrame
            pairs_index = np.empty(len(pair_list))

            for i_pair,pair in enumerate(pair_list):
                # Now, for each pair in the new frame
                
                pairs_0_id = pairs_in_frame[i_frame-1].index.values

                # "where" is the location of the pair in the previous array. 
                # If the pair is not in the previous array, it returns an empty, which is fine (see later)
                # However, if the pair is in the previous array more than once it should fail.
                where = [pair_id for pair_id, pair_0 in enumerate(pairs_in_frame[i_frame-1].values) if pair_0 == pair]

                if where:
                    pairs_index[i_pair] = pairs_0_id[where]
                else:
                    # if where is empty, I assign a new id to the pair. I then update the newest pair id. 
                    pairs_index[i_pair] = new_pair_id
                    new_pair_id=new_pair_id+1
            pair_df = pd.DataFrame({'members':pair_list},index = pairs_index)
            pair_df.index.name = 'id'
            pairs_in_frame.append(pair_df)

        else: 
            pair_df = pd.DataFrame({'members':pair_list})
            pair_df.index.name = 'id'
            pairs_in_frame.append(pair_df)
            new_pair_id = len(pair_list)
    
    
    return pd.concat(pairs_in_frame,keys = frames).sort_index(level='frame')
    
def dimers_findpositions(dim,trj,sim):
    """
    Finds the positions of both elements of each dimer and stores them in two new fields.
    It also calculates the centers and directions of the dimers. 
    In periodic boundaries, this function needs to unwrap the direction vector of those 
    dimers that have one particle in each side of the boundary. 
    """
    
    p0 = np.zeros([len(dim),3])
    p1 = np.zeros([len(dim),3])

    members = np.array([list(m) for m in dim.members.values])
    frames = dim.index.get_level_values('frame')

    idx = pd.IndexSlice

    for i,m in enumerate(members):
        m0 = trj.loc[idx[frames[i]]].loc[idx[m[0]]]
        m1 = trj.loc[idx[frames[i]]].loc[idx[m[1]]]
        
        p = np.array([m0.filter(("x","y","z")),m1.filter(("x","y","z"))])
        p = p[p[:,2].argsort()]

        p0[i] = p[0]
        p1[i] = p[1]    

    center, direction = unwrap_dimers(p0,p1,sim)
    
    dim['part0'] = list(p0)
    dim['part1'] = list(p1)
    dim['center']= list(center)
    dim['direction']=list(direction)
    
    return dim
    
def unwrap_dimers(p0,p1,sim,tol=False):
    """ 
    Unwraps the dimers that are found across the periodic boundaries periodic boundaries of the region
    The input p0 and p1 are arrays of Nx3, where N is the number of dimers. Each row of p0 is the position in 3D of the first particle (convention is that this is the lower particle)
    Each row of p1 is the position of the second particle.
    """
    region = [r.magnitude for r in sim.world.region]
    
    size = np.array(region)[1::2] - \
            np.array(region)[0::2]
        
    if not tol:
        tol = size/2
    elif not hasattr(tol, "__len__"):
        tol = tol*np.ones(3)
    
    tol = np.array(
        [t if (b=='p') else np.Inf  for b,t in zip(sim.sim_parameters.space['boundary'],tol)])
    
    """
    We stack in 3D both position arrays. This allows us to calculate the direction as a diff.
    In this array then, each element represents a dimer. Each dimer has three arrays which represent the three dimensions, and each of these arrays contains 2 elements, one for each dimer-member.
    That is: the first dimension is the dimers. The second dimension is the three elements of the positions, and the third dimension are the two members of the dimer. The array is therefore Nx3x2
    """
    point_pairs = np.stack([p0,p1],2)
    """Calculate the diferences of each component of every dimer"""
    delta = np.diff(point_pairs,axis=2)
    
    """
    We unwrap only those dimers that have a dimension greater than a tolerance. The tolerance is, by default half the size of the region in the respective direction
    To unwrap a value, we add to it's second point, the size of the region in its direction multiplied by the sign of the direction. 
    """
    needs_unwrap = abs(delta)>np.array(tol).reshape(1,3,1)
    unwrap_amount = np.sign(delta)*size.reshape(1,3,1)
    
    second_points = point_pairs[:,:,1:2].flatten()
    second_points[needs_unwrap.flatten()] -= unwrap_amount[needs_unwrap].flatten()

    point_pairs[:,:,1]=second_points.reshape(np.shape(point_pairs[:,:,1]))
    
    direction = np.diff(point_pairs,axis=2)[:,:,0]
    
    """We calculate like this the center, because the usual formula (p0+p1)/2 would yield something in the center of the region for vectors that were unwraped."""
    center = p0+direction/2
    
    return center, direction
        
def draw_dim(dim,sim,ax=False):
    """ 
    displays a trajectory statically. 
    If iframe is given, it displays particles in the frame specified by the index iframe. 
    If it isn't given, then it displays the last frame
    """
        
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,7))
    iframe=-1
    
    idx = pd.IndexSlice
    
    frames = dim.index.get_level_values('frame').unique()
    dimers = dim.loc[idx[frames[iframe],:]].index.get_level_values('id').unique()
    n_of_dimers = len(dimers)
    
    region = [r.magnitude for r in sim.world.region]
        
    ax.set_xlim(region[0],region[1])
    ax.set_ylim(region[2],region[3])
    ax.set(aspect='equal')
    ax.set_xlabel("$x [\mu{m}]$")
    ax.set_ylabel("$y [\mu{m}]$")
    
    patches = []
    for i,d in enumerate(dimers):
        center = dim.loc[idx[frames[iframe],d],'center']
        direction = dim.loc[idx[frames[iframe],d],'direction']
        p0 = center-direction/2
        c = plt.Arrow(p0[0],p0[1],direction[0],direction[1])
        patches.append(c)

    p = clt.PatchCollection(patches)
    ax.add_collection(p)
            
    return ax
    
def animate_dim(dim ,sim, ax=False, verb=False, start=0, end=False, step = 1, speedup = 1):
    """
    This function animates the dimers resulting from a confined dimer simulation.
    It displays each dimer as a vector that runs from the lower to the higher particles. 
    The simulation is required as argument to obtain parameters like the region size and the 
    particles radius. 
    Optional parameters are:
    * ax: an axis object to use for creating the plot.
    * start: start time of the simulation if not the whole time is required. The default is 0. 
    * end: end time of the simulation. The default is the total simulation time.
    * step = 1. The framerate, so to speak. 
    * verb = False. If verb = True, the routine prints indicators that is running. 
    * speedup allows us to do faster videos. Default is 1, which means normal ratio.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    idx = pd.IndexSlice
    
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,7))

    dimers = dim.index.get_level_values('id').unique()
    n_of_dimers = len(dim.index.get_level_values('id').unique())
    
    region = [r.magnitude for r in sim.world.region]
    radius = sim.particles.radius.magnitude
        
    framerate = sim.framerate.magnitude
    runtime = sim.total_time.magnitude
    timestep = sim.timestep.magnitude
    
    lammps_time = 1e6;
    
    #dt_data = np.round(1/(timestep*framerate)) # Data timestep in lammps_time
    
    frames = dim.index.get_level_values('frame').unique().values
    
    if not frames.size:
        raise ValueError("There are no dimers in the input array")
        
    if not end:
        end = frames[-1]*timestep
    
    frame_min=start/timestep
    frame_max=end/timestep
    
    frame_id_min = np.where(frames>=frame_min)[0][0]
    frame_id_max = np.where(frames<frame_max)[0][-1]
    
    dim = dim.loc[idx[frames[frame_id_min:frame_id_max:step],:]]

    dimers = dim.index.get_level_values('id').unique()
    n_of_dimers = len(dim.index.get_level_values('id').unique())

    frames = dim.index.get_level_values('frame').unique().values
    dt_video = np.mean(np.diff(frames))*sim.timestep.magnitude*1000/speedup # video timestep in miliseconds
    
    ax.set_xlim(region[0],region[1])
    ax.set_ylim(region[2],region[3])
    ax.set(aspect='equal')
    ax.set_xlabel("$x [\mu{m}]$")
    ax.set_ylabel("$y [\mu{m}]$")
    
    patches = []
    for i,p in enumerate(dimers):
        a = plt.Arrow(0, 0, 0, 0)
        patches.append(a)

    p = clt.PatchCollection(patches)
    
    def init():
        ax.add_collection(p)
        return p,

    def animate(frame):
        if verb:
            print("frame[%u] is "%frame,frames[frame])
        
        patches = []
        
        dimers = dim.loc[idx[frames[frame],:]].index.get_level_values('id').unique()
        n_of_dimers = len(dimers)

        for (dim_id,dimer) in enumerate(dimers):

            center = dim.loc[idx[frames[frame],dimer],'center']
            direction = dim.loc[idx[frames[frame],dimer],'direction']
            p0 = center-direction/2
            a = plt.Arrow(p0[0],p0[1],direction[0],direction[1])
            patches.append(a)
            
        p.set_paths(patches)

        ax.add_collection(p)
        return p,
    
    if verb: 
        print("started animating")
        print(frames)
        
    anim = anm.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(frames), interval=dt_video, blit=True);
    plt.close(anim._fig)

    return anim
    
### Order Paramters

def nematic_order(dim,director=False):
    """ 
    Calculates the nematic order parameter $S = <1/2*(3cos(theta)^2-1)>$, where $theta$ is the angle between the director vector and the dimers. If the director is not given, the function returns the order parameter and the director with the largest order with a resolution of one degree. 
    """
    if director:
        direction = np.array(list(dim.direction.values))
        theta = np.arctan2(direction[:,1],direction[:,0])
        theta_0 = np.arctan2(director[1],director[0])
        return np.mean((3*np.cos(theta-theta_0)**2-1)/2)
    else:
        theta_0 = np.linspace(0,2*np.pi,360)

        S = [nematic_order(dim,[np.cos(th),np.sin(th)]) for th in theta_0]
        
        max_order = np.argmax(S)
        director = [np.cos(theta_0[max_order]),np.sin(theta_0[max_order])]
        return S[max_order], director