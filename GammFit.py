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
    """
    A class to fit and plot Generalized Additive Mixed Models (GAMMs).
    
    Attributes:
    ----------
    fit : pandas.DataFrame
        The fitted GAMM
    std_error : pandas.DataFrame
        Standard error of the GAMM
    rsquared : np.ndarray
        R squared value of the GAMM
    metric : pandas.DataFrame
        The metric to which the GAMM is fit
    indages : np.ndarray
        Individual ages in the data
    cohort_id : np.ndarray
        Cohort IDs in the data
    cohort_effect : pandas.DataFrame
        Cohort effect of the GAMM
    ages : np.ndarray
        Array of ages
    name : str
        Name of the metric
    ndim : int
        Number of dimensions in the data
    """

    def __init__(self, directory, metricname, maxage=100, minage=0, ntimepoints=400, ndim=3, cohort = True):
        self.fit = uts.load_gamm_fit(f'{directory}{metricname}_fit_3M.csv', ndim)
        self.std_error = uts.load_gamm_fit(f'{directory}{metricname}_standard_error_3M.csv', ndim)

        self.rsquared = pd.read_csv(f'{directory}{metricname}_rsq_3M.csv')['x'].to_numpy()
        self.metric = uts.load_subj_metric_from_csv(f'{directory}{metricname}.csv', ndim)
        self.indages = pd.read_csv(f'{directory}{metricname}.csv')['Age'].to_numpy()
        
        
        if cohort:
            self.cohort_id = pd.read_csv(f'{directory}{metricname}.csv')['Cohort_ID'].to_numpy()
            self.cohort_effect = uts.load_cohort_effect(f'{directory}{metricname}_cohort_effect.csv')
            for i in range (ndim):
                self.metric[:,i] = uts.apply_cohort_shift(self.metric[:,i],self.cohort_id,self.cohort_effect)
        self.ages = np.arange(ntimepoints) / (ntimepoints / (maxage - minage))
        self.name = metricname
        self.ndim = ndim 

        # Check the shapes of the arrays
        assert self.fit.shape[0] == self.std_error.shape[0] == self.ages.shape[0], "Mismatch in shape of input arrays"
    
    def plot_with_subject_data(self, ylabel = None, metric_names = None, fit_colors = ['r','g','b'],fit_names = ['SA','VS','MR']):
        subject_data = self.metric
        
            
        if self.ndim == 1:
            pltg.plot_fit_with_data(self.fit,self.std_error,subject_data,self.indages,ylabel = ylabel, fit_names = fit_names)
        else:
            pltg.plot_fits_separately(self.fit,self.std_error,subject_data,self.indages,ylabel=ylabel, fit_names = fit_names, fit_colors = fit_colors)
    
    def plot_fits(self, fit_names = ['SA', 'VS', 'MR'], cmap = None, ylabel = None, fit_colors = ['r','g','b']):
        
        if not ylabel:
            ylabel = self.name
        if cmap:
            fit_colors = [cmap(i) for i in range (self.ndim)]
        
        pltg.plot_multiple_fits(self.fit,self.std_error,fit_names=fit_names,fit_colors=fit_colors,ylabel=ylabel)
   