#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 12:25:46 2023

@author: patricktaylor
"""
import numpy as np 
import reading_writing as rw
import utility as uts


def compute_parc_fc(lh_func_path,parc, parc_region_names,return_fmri = False):
    rh_func_path = lh_func_path.replace('lh','rh')
    
    fmri = rw.read_functional_timeseries(lh_func_path, rh_func_path, v2s=False)
    
    parc_fmri = np.zeros((len(set(parc)),fmri.shape[1]))
    
    pinds = []
    for i in range (len(parc_region_names)):
    #    if not parc_region_names[i].endswith('wall'):
        pinds.append(i)
    for i, j in enumerate(pinds):
        
        
        inds = np.where(parc == j)[0]
        parc_fmri[i] = np.mean(fmri[inds],axis=0)
    
    return np.corrcoef(parc_fmri) if not return_fmri else (np.corrcoef(parc_fmri),parc_fmri)

