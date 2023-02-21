#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:53:23 2023

@author: patricktaylor
"""
import numpy as np 
import reading_writing as rw 
import matrix_comp as mtx 

import brainspace as bs 
from scipy import sparse 
import utility as uts 
import decomp as dcp
import os 
import plotting as pltg 
import scipy.stats as ss 

class Gradient:
    
    
    def __init__(self,grad_path = None, val_path = None, gid = None):
        """

        Parameters
        ----------
        grad_path : str, optrwnal
            if supplied, used to load the gradient array. The default is None.
        val_path : str, optrwnal
            path to eigenvalue array. The default is None.
        gid : str, int, optional
            unique identifier for gradient. The default is None.

        Returns
        -------
        None.

        """
        if grad_path is not None: 
            self.garray = np.load(grad_path)
            #self.nvert = self.garray.shape[0]
            #self.ngrad = self.garray.shape[1]
        if val_path is not None: 
            self.varray = np.load(val_path)
            
        else:
            self.array = None
            
        self.gid = gid 
    
    
    def set_gid(self,gid):
        self.gid = gid 
    
    def set_ngrad(self, ngrad):
        
        self.garray = self.garray[:,:ngrad]
        
        
        
    @property
    def nvert(self):
        return self.garray.shape[0]
    
    @property
    def ngrad(self):
        return self.garray.shape[1]
    
    @property
    def grange(self):
        gr = [np.max(self.garray[:,i])-np.min(self.garray[:,i]) for i in range (self.ngrad)]
        return np.array(gr)
    
    @property
    def gvar(self):
        gv = [np.var(self.garray[:,i]) for i in range (self.ngrad)]
        return np.array(gv)

        
    def info(self):
        print(f'ID {self.string}')
        print(f'{self.nvert} vertices, {self.ngrad} gradients')
        return 
    
    def compute_FC_from_fMRI(self, lh_fmri_path, rh_fmri_path, threshold = 0.03, 
                             chunk_size = 1000, symmetric = True, size = 20484, 
                             smoothing_mat = None, cos_sim= True, return_mat = True, 
                             verbose = True ):
        """
    

        Parameters
        ----------
        lh_fmri_path : str or list of str
            path to left hemisphere surface-mapped 
            fmri timeseries.
        rh_fmri_path : str
            path to left hemisphere surface-mapped 
            fmri timeseries.
        threshold : float, optional
            minumum correlation value retained in 
            FC matrix. The default is 0.03.
        chunk_size : int, optional
            number of vertices in chunk of 
            correlation computation and 
            thresholding to conserve memory. 
            The default is 1000.
        symmetric : boolean, optional
            determines whether each acquisition's 
            thresholded FC matrix will be averaged 
            with its transpose to enforce symmetry. 
            The default is True.
        size : int, optional
            number of surface vertices. The default 
            is 20484.
        smoothing_mat : scipy.sparse.csr((size,size)), 
            optional matrix of smoothing weights to be 
            applied to the FC matrix. The default is None.
        cos_sim : boolean, optional
            when true, the cosine similarity kernel is 
            applied after artan transformation. The 
            default is True.
        return_mat : boolean, optional
            if true, final FC matrix is returned. else, 
            it is stored in fcmat attribute The default 
            is True.
        verbose : boolean, optional
            if true, some description is printed. The 
            default is True.

        Returns
        -------
        fcmat : scipy.sparse.csr((size,size))
            thresholded FC matrix.
        zeromask : np.array(size)
            array containing ones for all vertices that 
            did not have fmri signal in every acquisition.

        """
        if type(lh_fmri_path) is str:
            
            #load fMRI timeseries as np array from .gii
            fmri = rw.read_functional_timeseries(
                        lh_fmri_path, rh_fmri_path, v2s = False)
            
            #mask  all vertices with no fMRI
            zeromask = uts.mask_from_timeseries_zero(fmri)
            fmri = uts.mask_medial_wall_vecs(fmri, zeromask)
            
            #compute thresholded pairwise correlation matrix
            fcmat = mtx.construct_FC_matrix_row_thresh(
                        fmri, threshold = threshold, 
                        chunk_size = chunk_size,symmetric = True)
            
            #fix potential bad data and unmask the matrix
            fcmat.data[np.isnan(fcmat.data)] = 0.0
            fcmat.eliminate_zeros()
            x = np.arange(fcmat.shape[0])
            fcmat[x,x] = 0.0 
            fcmat.eliminate_zeros()
            fcmat = uts.unmask_connectivity_matrix(fcmat, zeromask)
            
        else: #we have multiple acquisitions 
            
            zeromask = np.zeros(size)
            #initialize empty sparse mat for average
            fcmat = sparse.csr_matrix((size, size))
            n = 0 
            
            for i in range(len(lh_fmri_path)):
                
                fmri = rw.read_functional_timeseries(lh_fmri_path[i], 
                                                     rh_fmri_path[i], 
                                                     v2s = False)
                
                zm = uts.mask_from_timeseries_zero(fmri)
                zeromask+= zm 
                fmri = uts.mask_medial_wall_vecs(fmri, zm)
                
                mat = mtx.construct_FC_matrix_row_thresh(
                        fmri, threshold = threshold, chunk_size = chunk_size,
                        symmetric = True)
                
                mat.data[np.isnan(mat.data)] = 0.0
                mat.eliminate_zeros()
                x = np.arange(mat.shape[0])
                mat[x,x] = 0.0 
                mat.eliminate_zeros()
                mat = uts.unmask_connectivity_matrix(mat, zm)
                #add this acquisition to running sum across acquisitions
                fcmat += mat 
                n += 1
                
            if verbose: print(f'''average FC mat computed 
                              from {n} acquisitions with 
                              threshold = {threshold}''')
            
            zeromask[zeromask>1] = 1
        
        #load surface-based smoothing matrix
        if smoothing_mat is not None:
            
            sm = sparse.load_npz(smoothing_mat)
            sm = uts.mask_connectivity_matrix(sm, zeromask)
        
        fcmat = uts.mask_connectivity_matrix(fcmat, zeromask)
        
        fcmat.data=np.nan_to_num(fcmat.data,copy=False,nan=0)
        fcmat.eliminate_zeros()
        fcmat.data=fcmat.data/n
        fcmat.data[fcmat.data>0.999]=0.999
        #apply smoothing matrix
        fcmat=sm.T.dot(fcmat.dot(sm))
        #apply arctan transform
        fcmat.data = np.arctanh(fcmat.data)
        
        #apply cosine similarity kernel 
        if cos_sim:
            fcmat=mtx.cosine_similarity_matrix(fcmat)
        
        if return_mat:
            return fcmat, zeromask 
        else:
            self.fcmat = fcmat 
            
    def compute_grads_from_FC(self, fcmat, zeromask, 
                              num_comp = 7, savepath = None):
        """
        

        Parameters
        ----------
        fcmat : scipy.sparse.csr((size,size))
            FC matrix.
        zeromask : np.array(size)
            array containing ones for all vertices that 
            did not have fmri signal in every acquisition.
        num_comp : int, optional
            number of gradients to compute. The default is 7.
        savepath : str, optional
            path to save gradients and evals, naming convention 
            should be /path/to/directory/{grad_ID}. The default 
            is None.

        Returns
        -------
        None. gradients and eigenvalues are stored in garray and 
        varray attributes.

        """
        v,w=dcp.diffusion_embed_brainspace(fcmat,num_comp)
        
        v=uts.unmask_medial_wall_vecs(v,zeromask)
        
        self.garray = v 
        self.varray = w 
        
        if savepath is not None:
            np.save(savepath + '_grads', v)
            np.save(savepath + '_vals', w )
    
    def compute_grads_from_fMRI(self, lh_fmri_path, rh_fmri_path, 
                                num_comp = 7, threshold = 0.03, 
                                chunk_size = 1000, symmetric = True, 
                                size = 20484, smoothing_mat = None, 
                                cos_sim= True, return_mat = True, 
                                savepath = None, verbose = True):
        #bottles compute_grads_from_FC and compute_FC_from_fMRI
        
        fcmat, zeromask = self.compute_FC_from_fMRI(
                                lh_fmri_path, rh_fmri_path, threshold, 
                                chunk_size, symmetric, size, smoothing_mat, 
                                cos_sim, True, verbose)
        
        self.compute_grads_from_FC(self, fcmat, zeromask, num_comp, savepath)
        
    
    def evaluate(self, function, **kwargs):
        """
        

        Parameters
        ----------
        function : function
            any function which takes a gradient array as first positional
            argument followed by any number of keyword arguments.
        **kwargs : 
            keyword argument inputs to function.

        Returns
        -------
        output of function

        """
        return function(self.garray, **kwargs)
        
        
        
        
        
    def compute_dispersion(self, vertex_dist_from_centroid = False, return_result = True):
        """
        

        Parameters
        ----------
        vertex_dist_from_centroid : boolean, optional
            if true, return array of distances from each vertex to centroid 
            of gradient embedding. The default is False.

        Returns
        -------
        dispersion float or dist array if return_result is True/False. 

        """
        centroid = np.mean(self.garray, axis = 0)
        dif = self.garray - centroid
        dists = np.square(np.linalg.norm(dif, axis = 1))
        disp = np.mean(dists)
        self.dispersion = disp
        if vertex_dist_from_centroid:
            self.dists_to_centroid = dists 
        if return_result:
            if vertex_dist_from_centroid: 
                return dists
            else:
                return disp
                
    
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
    
    def compute_spearmanr(self, grad):
        
        mat = ss.spearmanr(self.garray, grad.garray)
        c = mat.correlation[:self.ngrad, self.ngrad : self.ngrad*2]
        p = mat.pvalue[:self.ngrad, self.ngrad : self.ngrad*2]
       
        return  c, p
        
            
            
    def hist(self, gradinds):
        pltg.embed_plot_all_vertices_histogram(self.garray, gradinds, self.gid)
        
        
    
        
    def surface_plot(self, gradind = None):
        
        if gradind is None:
            gradind = np.arange(min(self.ngrad,5))
            
        if len(gradind) < 2:
            pltg.splot(self.garray[:,gradind],title = f'{self.gid} gradient {gradind}')
            #rw.plot_surf(self.garray[:,gradind], lh, rh, title = self.gid)
            
        else:
            for i in gradind:
                pltg.splot(self.garray[:,i],title = f'{self.gid} gradient {gradind[i]}')
                #rw.plot_surf(self.garray[:,i], lh, rh, title = self.gid )
        
    
    