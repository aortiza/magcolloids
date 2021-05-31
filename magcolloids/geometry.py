# Geometry Package
import numpy as np
## Cartesian to Spherical Transformation
def vec_to_spher(vec):
    
    r = np.sum(vec**2, axis=1)    
    phi = np.arctan2(vec[:,1], vec[:,0])
    theta = np.arccos(vec[:,2]/r)
    return r, theta, phi

## Quaternion rotation functions
def quatquat(a,b):
    c = np.empty([len(a),4])*np.NaN
    c[:,0] = a[:,0]*b[:,0] - a[:,1]*b[:,1] - a[:,2]*b[:,2] - a[:,3]*b[:,3]
    c[:,1] = a[:,0]*b[:,1] + b[:,0]*a[:,1] + a[:,2]*b[:,3] - a[:,3]*b[:,2]
    c[:,2] = a[:,0]*b[:,2] + b[:,0]*a[:,2] + a[:,3]*b[:,1] - a[:,1]*b[:,3]
    c[:,3] = a[:,0]*b[:,3] + b[:,0]*a[:,3] + a[:,1]*b[:,2] - a[:,2]*b[:,1]
    
    return c

def quatinv(a):
    c = np.empty([len(a),4])*np.NaN
    c[:,0] = a[:,0]
    c[:,1:] = -a[:,1:]
    return c

def rotate(p,q):
    return quatquat(quatquat(q,p),quatinv(q))
    
def vec_to_quat(vec):
    
    #This function might be optimized by getting rid of the trigonometric functions.     
    r, theta, phi = vec_to_spher(vec)
    
    Qth = np.array(
        [np.cos(theta/2),
         np.zeros(np.shape(theta)),
         np.sin(theta/2),
         np.zeros(np.shape(theta))]).transpose()

    Qphi = np.array(
        [np.cos(phi/2),
         np.zeros(np.shape(phi)),
         np.zeros(np.shape(phi)),
         np.sin(phi/2)]).transpose()
    
    return quatquat(Qphi,Qth)