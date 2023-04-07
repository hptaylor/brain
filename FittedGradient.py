#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 13:58:14 2023

@author: patricktaylor
"""
import reading_writing as rw 
import utility as uts 
import numpy as np 
import Gradient 
import pandas as pd 
from sklearn.decomposition import PCA
import fileops as fps 
import matplotlib.pyplot as plt

class FittedGradient:
    
    def __init__(self,arraypath = None, csvlist = None, nvert = 20484, ngrad = 3, ntimepoints = 400, minage = 0, maxage = 100):
        self.nvert = 20484
        self.ngrad = ngrad
        self.ntimepoints = ntimepoints
        if arraypath is not None:
            self.array = np.load(arraypath)
        if csvlist is not None: 
            
            self.array = np.zeros((ntimepoints,nvert,ngrad))
            for i,p in enumerate(csvlist):
                df = pd.read_csv(p)
                for j in range (nvert):
                    self.array[:,j,i] = df[f'V{j+2}']
        
        self.ages = np.arange(ntimepoints)/(ntimepoints/(maxage-minage))
        
        self.zarray = np.zeros(self.array.shape)
        
        for i in range (ntimepoints):
            for j in range (ngrad):
                self.zarray[i,:,j] = uts.zscore_surface_metric(self.array[i,:,j])
    
    def load_rsq(self,pathlist):
        rsq=np.zeros((self.nvert,3))
        for i,p in enumerate(pathlist):
            df=pd.read_csv(p)
            rsq[:,i]=df['x'][1:]
        self.rsquared = rsq 
    
    def load_standard_error(self,pathlist):
        std_err=np.zeros((self.ntimepoints,self.nvert,self.ngrad))
        for i,p in enumerate(pathlist):
            df=pd.read_csv(p)
            for j in range (self.nvert):
                std_err[:,j,i]=df[f'V{j+2}']
        self.std_err = std_err 
        
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
            ax.plot(self.ages,y ,label = f'g{i+1}')
            ax.fill_between(self.ages, (y-ci), (y+ci), alpha = 0.1)
        ax.legend()
        
        if sub_grads is not None:
            if gradind is not None:
                subgradvals = sub_grads.grad_arr_list[:,vertind,gradind]
                subages = sub_grads.ages()
                ax.scatter(subages, subgradvals,s = 0.6)
                
    
        
    
            