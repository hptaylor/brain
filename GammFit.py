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
        
        self.rsquared = pd.read_csv(directory+metricname+'_rsq_3M.csv')['x'].to_numpy()
        self.metric = uts.load_subj_metric_from_csv(directory+metricname+'.csv',ndim)
        self.indages = pd.read_csv(directory+metricname+'.csv')['Age'].to_numpy()
        self.cohort_id = pd.read_csv(directory+metricname+'.csv')['Cohort_ID'].to_numpy()
        self.cohort_effect = uts.load_cohort_effect(directory+metricname+'_cohort_effect.csv')
        self.ages = np.arange(ntimepoints)/(ntimepoints/(maxage-minage))
        self.name = metricname
        self.ndim = ndim 
        
        # Check the shapes of the arrays
        assert self.fit.shape[0] == self.std_error.shape[0] == self.ages.shape[0], "Mismatch in shape of input arrays"
        

    def plot_fit(self,shift=True):
        if not shift:
            for i in range (self.ndim):
                pltg.plot_fitted_metric(self.indages,self.metric[:,i],self.ages,self.fit[:,i],self.name + f' {i+1}',self.std_error[:,i])
        else:
            for i in range (self.ndim):
                pltg.plot_fitted_metric(self.indages,uts.apply_cohort_shift(self.metric[:,i],self.cohort_id,self.cohort_effect[:,i]),self.ages,self.fit[:,i],self.name + f' {i+1}',self.std_error[:,i])
   
    def plot_fits(self):
        pltg.plot_fits_w_ci_one_axis(self.ages, self.fit, self.name, self.std_error)