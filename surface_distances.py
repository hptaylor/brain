#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:04:58 2023

@author: patricktaylor
"""
import numpy as np 
import gdist 
import sklearn as skn 
import utility as uts

def compute_geodesic_distance_matrix(points,triangles):
    distmat = np.zeros((points.shape[0],points.shape[0]))
    for i in range (points.shape[0]):
        dists = gdist.compute_gdist(points,triangles.astype('int32'),np.array([i]).astype('int32'))
        distmat[i,:] = dists
    return dists 

def neighbors(searchedset,queryset,num, radius=None):
    '''
    computes num nearest neighbors of queryset in searchedset and returns numpy arrays size (len(queryset),num) 
    of indices of searched set and distances between neighbors
    '''
    nbrs = skn.NearestNeighbors(n_neighbors=num, algorithm='auto',radius = radius).fit(searchedset)
    distances, indices = nbrs.kneighbors(queryset)
    return indices,distances

def geodesic_distances_from_indices(inds,distmat):
    
    dists = np.zeros(inds.shape)
    for i in range (len(dists)):
        dists[i] = distmat[i,inds[i]]
    
    return dists

def emb_dists_from_phys_neighbors(num_neighbors,grads,distmat):
    
    embdists = np.zeros((distmat.shape[0],num_neighbors))
    physdists = np.zeros((distmat.shape[0],num_neighbors))
    
    for i in range (distmat.shape[0]):
        inds = uts.bottom_k_indices(distmat[i],num_neighbors)
        
        edists = np.linalg.norm(grads[i] - grads[inds],axis=1)
        
        embdists[i,:] = edists
        physdists[i,:] = distmat[i,inds]
    return embdists, physdists
