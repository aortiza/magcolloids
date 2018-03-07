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

def animate_trj(trj,sim,ax=False, verb=False):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    idx = pd.IndexSlice
    
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(7,7))

    particles = trj.index.get_level_values('id').unique()
    n_of_particles = len(trj.index.get_level_values('id').unique())
    
    region = sim.sim_parameters.space["region"] 
    radius = sim.particle_properties[0].radius
    
    framerate = sim.run_parameters.framerate
    runtime = sim.run_parameters.total_time
    timestep = sim.run_parameters.timestep
    
    lammps_time = 1e6;
    
    dt_data = np.round(1/(timestep*framerate)) # Data timestep in lammps_time
    dt_video = 1/framerate*1000 # video timestep in miliseconds
    frames = trj.index.get_level_values('frame').unique().values
    
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

def draw_trj(trj,sim,iframe=-1,ax=False,):
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
    
    region = sim.sim_parameters.space["region"] 
    radius = sim.particle_properties[0].radius
    
    ax.set_xlim(region[0],region[1])
    ax.set_ylim(region[2],region[3])
    ax.set(aspect='equal')
    ax.set_xlabel("$x [\mu{m}]$")
    ax.set_ylabel("$y [\mu{m}]$")
    
    patches = []
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
    
    if args:
        trj = sim.load(read_trj=True)
    else: 
        trj = args[0]
        
    anim = animate_trj(trj,sim,kargs)
    anim.save(sim.base_name+".gif",writer = "imagemagick")
    return anim
    
def display_animation_direct(sim,*args,**kargs):

    if len(args)<1:
        trj = sim.load(read_trj=True)
        print("reading"+args)
    else: 
        trj = args[0]

    anim = animate_trj(trj,sim,kargs)
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