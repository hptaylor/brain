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
import plotting as pltg
class FittedGradient:
    """A class used to represent a FittedGradient."""

    def __init__(
        self, 
        g1_path=None, 
        directory=None, 
        arraypath=None, 
        nvert=20484, 
        ngrad=3, 
        ntimepoints=400, 
        minage=0, 
        maxage=100
    ):
        """
        Initializes the FittedGradient object by loading the gradient data from the provided sources.

        Parameters:
            g1_path : str, optional
                Path to the g1 CSV file.
            directory : str, optional
                Path to the directory containing the gradient data files.
            arraypath : str, optional
                Path to the numpy file containing the gradient data.
            nvert : int, optional
                Number of vertices.
            ngrad : int, optional
                Number of gradients.
            ntimepoints : int, optional
                Number of time points.
            minage : int, optional
                Minimum age.
            maxage : int, optional
                Maximum age.
        """
        self.nvert = nvert
        self.ngrad = ngrad
        self.ntimepoints = ntimepoints

        if directory is not None:
            self.array = np.load(os.path.join(directory, 'grads_gamm.npy'))
            rsquared_path = os.path.join(directory, 'rsq.npy')
            if os.path.isfile(rsquared_path):
                self.rsquared = np.load(rsquared_path)
            std_err_path = os.path.join(directory, 'std_err.npy')
            if os.path.isfile(std_err_path):
                self.std_err = np.load(std_err_path)

        if arraypath is not None:
            self.array = np.load(arraypath)

        if g1_path is not None:
            g2_path = g1_path.replace('g1', 'g2')
            g3_path = g1_path.replace('g1', 'g3')
            csv_list = [g1_path, g2_path, g3_path]

            self.array = np.zeros((ntimepoints, nvert, ngrad))
            for i, path in enumerate(csv_list):
                df = pd.read_csv(path)
                for j in range(nvert):
                    self.array[:, j, i] = df[f'V{j + 1}']

        self.ages = np.arange(ntimepoints) / (ntimepoints / (maxage - minage))
        self.zarray = np.zeros(self.array.shape)

        for i in range(ntimepoints):
            for j in range(ngrad):
                self.zarray[i, :, j] = uts.zscore_surface_metric(self.array[i, :, j])
    
    def load_rsq(self,g1_path,pathlist=None):
        if g1_path is not None:
            g2_path = g1_path.replace('g1', 'g2')
            g3_path = g1_path.replace('g1', 'g3')
            pathlist = [g1_path, g2_path, g3_path]
        rsq=np.zeros((self.nvert,3))
        for i,p in enumerate(pathlist):
            df=pd.read_csv(p)
            rsq[:,i]=df['x'][:]
        self.rsquared = rsq 
    
    def load_standard_error(self,g1_path,pathlist = None,array_path = None):
        if g1_path is not None:
            g2_path = g1_path.replace('g1', 'g2')
            g3_path = g1_path.replace('g1', 'g3')
            pathlist = [g1_path, g2_path, g3_path]
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
    
    def save_vtk_series_emb(self,directory = '/Users/patricktaylor/Documents/lifespan_analysis/scratch/timeseries/',
                        fname = '%s_emb.vtk',feature = None):
        si = np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/si.npy')
        if feature is None:
            feature = self.array
        for i in range (self.ntimepoints):
            rw.save_eigenvector(directory+fname % i, self.array[i],si,feature)
    
    
        
        
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
    def get_dispersion_parc(self,parc,zscore=True):
        disps = np.zeros((self.ntimepoints,np.unique(parc).shape[0]))
        for i in range(disps.shape[1]):
            for j in range(self.ntimepoints):
                inds = np.where(parc == i)[0]
                if not zscore:
                    disps[j,i] = mts.dispersion_centroid(self.garray[j,inds])
                else:
                    disps[j,i] = mts.dispersion_centroid(self.zarray[j,inds])
        return disps 
    
    def get_avg_val_parc(self,parc,zscore=True):
        vals = np.zeros((self.ntimepoints,np.unique(parc).shape[0],3))
        for z in range (vals.shape[1]):
            inds = np.where(parc == z)[0]
            for i in range (vals.shape[0]):
                for j in range (vals.shape[2]):
                    if zscore:
                        m = np.mean(self.zarray[i,inds,j])
                    else:
                        m = np.mean(self.array[i,inds,j])
                    vals[i,z,j] = m
        return vals 
    def plot_avg_val_parc(self,parc,network_names,zscore=True,deviation=True,axes=['SA','VS','MR']):
        vals = self.get_avg_val_parc(parc,zscore)
        for i in range (3):
            if deviation:
                d=np.mean(vals[:,:,i],axis=0)
            else:
                d = 0
            pltg.plot_lines_metric_vs_age_log(self.ages,vals[:,:,i]-d,f'avg {axes[i]} val deviation',network_names)
    
    def get_3d_cmap(self):
        colors = np.zeros(self.zarray.shape)
        for i in range (self.ntimepoints):
            colors[i] = uts.get_3d_cmap(self.zarray[i])
        return colors 