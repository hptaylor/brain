#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:25:23 2023

@author: patricktaylor
"""
import numpy as np
import vtk
import meshio
from tvtk.api import tvtk, write_data
import nibabel as nib
import surfplot as sp



def read_functional_timeseries(lhfunc,rhfunc,v2s=True):
    l = nib.load(lhfunc).darrays
    r = nib.load(rhfunc).darrays
    timeseries = np.zeros((2*len(l[0].data), len(r)))
    if v2s:
        timeseries=np.concatenate((np.array(l[0].data),np.array(r[0].data)))
        return timeseries
    else:
        for i in range(len(l)):
            lt = np.array(l[i].data)
            rt = np.array(r[i].data)
            tp = np.concatenate((lt, rt))
            timeseries[:, i] = tp
        return timeseries
    
    
def combine_hemis(lhc,rhc,lhi,rhi):
    #concatenates surface coordinates of two hemispheres and creates connectivity array for full surface
    coords=np.vstack((lhc,rhc))
    si=np.vstack((lhi,rhi+len(lhc)))
    return coords, si

def save_eigenvector(filename,points,edges,vecs):
    Cells = {"triangle": edges}
    V={}
#    if len(points)!=len(vecs):
#        vecs=uts.unmask_medial_wall(vecs,bcpmask)
        
    if len(vecs.shape)==1:
        v=vecs
        vecs=np.zeros((len(v),2))
        vecs[:,0]=v
    for i in range (0,len(vecs[0,:])):
        V.update({"ev%d" % (i):vecs[:,i]})

    mesh=meshio.Mesh(points,Cells,V)
    meshio.write(filename,mesh)
    return 

