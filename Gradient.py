#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:53:23 2023

@author: patricktaylor
"""
import numpy as np 
import io as io 
import matrix_comp as mtx 

import brainspace as bs 
from scipy import sparse 
import utility as uts 
import decomp as dcp
import os 

class Gradient:
    
    def __init__(self,grad_path = None, val_path = None, string = None):
        
        if grad_path is not None: 
            self.garray = np.load(grad_path)
            self.nvert = self.garray.shape[0]
            self.ngrad = self.garray.shape[1]
        if val_path is not None: 
            self.varray = np.load(val_path)
            
        else:
            self.array = None
            
        self.string = string 
        
    def compute_FC_from_fMRI(self, lh_fmri_path, rh_fmri_path, threshold = 0.03, chunk_size = 1000, symmetric = True, size = 20484, smoothing_mat = None, cos_sim= True, return_mat = True, verbose = True ):
        
        if type(lh_fmri_path) is str:
            fmri = io.read_functional_timeseries(lh_fmri_path, rh_fmri_path, v2s = False)
            
            zeromask = uts.mask_from_timeseries_zero(fmri)
            fmri = uts.mask_medial_wall_vecs(fmri, zeromask)
            
            
            fcmat = mtx.construct_FC_matrix_row_thresh(fmri, threshold = threshold, chunk_size = chunk_size, symmetric = True)
            fcmat.data[np.isnan(fcmat.data)] = 0.0
            fcmat.eliminate_zeros()
            x = np.arange(fcmat.shape[0])
            fcmat[x,x] = 0.0 
            fcmat.eliminate_zeros()
            fcmat = uts.unmask_connectivity_matrix(fcmat, zeromask)
            
        else:
            zeromask = np.zeros(size)
            fcmat = sparse.csr_matrix((size, size))
            
            n = 0 
            for i in range(len(lh_fmri_path)):
                fmri = io.read_functional_timeseries(lh_fmri_path[i], rh_fmri_path[i], v2s = False)
                zm = uts.mask_from_timeseries_zero(fmri)
                zeromask+= zm 
                fmri = uts.mask_medial_wall_vecs(fmri, zm)
                mat = mtx.construct_FC_matrix_row_thresh(fmri, threshold = threshold, chunk_size = chunk_size, symmetric = True)
                mat.data[np.isnan(mat.data)] = 0.0
                mat.eliminate_zeros()
                x = np.arange(mat.shape[0])
                mat[x,x] = 0.0 
                mat.eliminate_zeros()
                mat = uts.unmask_connectivity_matrix(mat, zm)
                fcmat += mat 
                n += 1
            if verbose: print(f'average FC mat computed from {n} acquisitions with threshold = {threshold}')
            
            zeromask[zeromask>1] = 1
        
        if smoothing_mat is not None:
            
            sm = sparse.load_npz(smoothing_mat)
            
            sm = uts.mask_connectivity_matrix(sm, zeromask)
        
        fcmat = uts.mask_connectivity_matrix(fcmat, zeromask)
        
        fcmat.data=np.nan_to_num(fcmat.data,copy=False,nan=0)
        fcmat.eliminate_zeros()
        fcmat.data=fcmat.data/n
        fcmat.data[fcmat.data>0.999]=0.999
        fcmat=sm.T.dot(fcmat.dot(sm))
        fcmat.data = np.arctanh(fcmat.data)
        if cos_sim:
            fcmat=mtx.cosine_similarity_matrix(fcmat)
        
        if return_mat:
            return fcmat, zeromask 
        else:
            self.fcmat = fcmat 
            
    def compute_grads_from_FC(self, fcmat, zeromask, num_comp = 7, savepath = None):
        
        v,w=dcp.diffusion_embed_brainspace(fcmat,num_comp)
        
        v=uts.unmask_medial_wall_vecs(v,zeromask)
        
        self.garray = v 
        self.varray = w 
        
        if savepath is not None:
            np.save(savepath + '_grads', v)
            np.save(savepath + '_vals', w )
    
    def compute_grads_from_fMRI(self, lh_fmri_path, rh_fmri_path, num_comp = 7, threshold = 0.03, chunk_size = 1000, symmetric = True, size = 20484, smoothing_mat = None, cos_sim= True, return_mat = True, savepath = None, verbose = True):
        
        fcmat, zeromask = self.compute_FC_from_fMRI(lh_fmri_path, rh_fmri_path, threshold, chunk_size, symmetric, size, smoothing_mat, cos_sim, True, verbose)
        
        self.compute_grads_from_FC(self, fcmat, zeromask, num_comp, savepath)
        
        
        
    def compute_dispersion(self, vertex_dist_from_centroid = False):
        centroid = np.mean(self.garray, axis = 0)
        dif = self.garray - centroid
        dists = np.square(np.linalg.norm(dif, axis = 1))
        disp = np.mean(dists)
        self.dispersion = disp
        if vertex_dist_from_centroid:
            self.dists_to_centroid = dists 
    
    def compute_explanation_ratios(self):
        ratios = np.zeros(self.varray.shape)
        valsum = np.sum(self.varray)
        for j in range (self.varray.shape[0]):
            ratios[j]=self.varray[j]/valsum
        self.explanation_ratios = ratios 
        
    def get_neighborhoods(self, num_neighbors = 100):
        
        inds, dists = uts.neighbors(self.garray, self.garray, num_neighbors)
        
        self.neighbor_indices = inds
        self.neighbor_distances = dists 
        
    
        