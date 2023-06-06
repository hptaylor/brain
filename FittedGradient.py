#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:58:14 2023

@author: patricktaylor
"""
import scipy.stats as ss 
import reading_writing as rw 
import utility as uts 
import numpy as np 
import Gradient 
import pandas as pd 
from sklearn.decomposition import PCA
import fileops as fps 
import matplotlib.pyplot as plt
import metrics as mts 
import os
class FittedGradient:
    
    def __init__(self, g1_path=None, directory=None, arraypath=None, nvert=20484, ngrad=3, ntimepoints=400, minage=0, maxage=100):
        self.nvert = nvert
        self.ngrad = ngrad
        self.ntimepoints = ntimepoints
        
        if directory is not None:
            self.array = np.load(directory + 'grads_gamm.npy')
            p = directory + 'rsq.npy'
            if os.path.isfile(p):
                self.rsquared = np.load(p)
            p = directory + 'std_err.npy'
            if os.path.isfile(p):
                self.std_err = np.load(p)
            
        if arraypath is not None:
            self.array = np.load(arraypath)

        if g1_path is not None:
            g2_path = g1_path.replace('g1', 'g2')
            g3_path = g1_path.replace('g1', 'g3')
            csvlist = [g1_path, g2_path, g3_path]
            
            self.array = np.zeros((ntimepoints, nvert, ngrad))
            for i, p in enumerate(csvlist):
                df = pd.read_csv(p)
                for j in range(nvert):
                    self.array[:, j, i] = df[f'V{j+1}']
        
        self.ages = np.arange(ntimepoints) / (ntimepoints / (maxage - minage))
        
        self.zarray = np.zeros(self.array.shape)
        
        for i in range(ntimepoints):
            for j in range(ngrad):
                self.zarray[i, :, j] = uts.zscore_surface_metric(self.array[i, :, j])
    
    def load_rsq(self,pathlist):
        rsq=np.zeros((self.nvert,3))
        for i,p in enumerate(pathlist):
            df=pd.read_csv(p)
            rsq[:,i]=df['x'][:]
        self.rsquared = rsq 
    
    def load_standard_error(self,pathlist = None,array_path = None):
        if pathlist is not None:
            std_err=np.zeros((self.ntimepoints,self.nvert,self.ngrad))
            for i,p in enumerate(pathlist):
                df=pd.read_csv(p)
                for j in range (self.nvert):
                    std_err[:,j,i]=df[f'V{j+1}']
            self.std_err = std_err 
        else:
            self.std_err = np.load(array_path)
        
    def get_gradient_at_age(self,age, zscore = True, return_obj = True):
        ind = uts.find_nearest_index(self.ages,age)
        if zscore:
            g = Gradient(garray = self.zarray[ind])
        else:
            g = Gradient(garray = self.array[ind])
        if return_obj:
            return g
        else:
            return g.garray 
    
    def save_vtk_series(self,directory = '/Users/patricktaylor/Documents/lifespan_analysis/scratch/timeseries/',
                        fname = '%s_emb.vtk',feature = None):
        si = np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/si.npy')
        if feature is None:
            feature = self.array
        for i in range (self.ntimepoints):
            rw.save_eigenvector(directory+fname % i, self.array[i],si,self.array[i])
            
    def plot_grad_val_vs_age(self,vertind,gradind=None, zscore = True, 
                             sub_grads = None):
        
        fig, ax = plt.subplots()
        if gradind is None:
            gradind = [i for i in range (self.ngrad)]
        else:
            gradind = [gradind]
            
        for i in gradind:
            ci = self.std_err[:,vertind,i]*1.95
            if zscore:
                y = self.zarray[:,vertind,i]
            else:
                y = self.array[:,vertind,i]
            ax.plot(np.log2(self.ages+1),y ,label = f'g{i+1}')
            ax.fill_between(np.log2(self.ages+1), (y-ci), (y+ci), alpha = 0.5)
        ax.legend()
        
        if sub_grads is not None:
            if gradind is not None:
                subgradvals = sub_grads.grad_arr_list()[:,vertind,gradind]
                subages = sub_grads.ages
                ax.scatter(np.log2(subages+1), subgradvals,s = 0.6)
        ticks = ax.get_xticks()
        ax.set_xticklabels([f'{round(2**i - 1)}' for i in ticks])
        ax.set_xlabel('age (years)')
                
    
        
    def get_ranges(self):
        granges = np.zeros((self.array.shape[0],self.array.shape[2]))
        for i in range (self.ntimepoints):
            for j in range (self.ngrad):
                granges[i,j] = np.percentile(self.array[i,:,j],95)-np.percentile(self.array[i,:,j],5)
        self.granges = granges
    def get_vars(self):
        gvars = np.zeros((self.array.shape[0],self.array.shape[2]))
        for i in range (self.ntimepoints):
            for j in range (self.ngrad):
                gvars[i,j] = ss.median_abs_deviation(self.array[i,:,j])
        self.gvars = gvars
        
    def get_dispersion(self):
        disp = np.array([mts.dispersion_centroid(vecs) for vecs in self.array])
        self.dispersion = disp 