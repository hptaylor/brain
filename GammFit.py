#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 14:05:05 2023

@author: patricktaylor
"""
import pandas as pd 
import numpy as np 
import utility as uts 
import plotting as pltg

class GammFit:
    def __init__(self, directory,metricname,maxage=100,minage=0,ntimepoints=400,ndim=1):
        
        self.fit = uts.load_gamm_fit(directory+metricname+'_fit_3M.csv',ndim)
        self.std_error = uts.load_gamm_fit(directory+metricname+'_standard_error_3M.csv',ndim)
        
        self.rsquared = pd.read_csv(directory+metricname+'_rsq_3M.csv')['x'][0]
        self.metric = uts.load_subj_metric_from_csv(directory+metricname+'.csv',ndim)
        self.indages = pd.read_csv(directory+metricname+'.csv')['Age'].to_numpy()
        self.ages = np.arange(ntimepoints)/(ntimepoints/(maxage-minage))
        self.name = metricname
        self.ndim = ndim 
        
        # Check the shapes of the arrays
        assert self.fit.shape[0] == self.std_error.shape[0] == self.ages.shape[0], "Mismatch in shape of input arrays"
        assert self.rsquared.shape == (), "rsquared should be a scalar value"

    def plot_fit(self):
        for i in range (self.ndim):
            pltg.plot_fitted_metric(self.indages,self.metric[:,i],self.ages,self.fit[:,i],self.name + f' {i+1}',self.std_error[:,i])
        
       