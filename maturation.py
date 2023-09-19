#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:11:59 2023

@author: patricktaylor
"""
import utility as uts 
import plotting as pltg
import numpy as np 

def maturation_age(trajectory,ages=None,maximum = True,return_index=False):
    if maximum:
        indmax = np.argmax(trajectory)
        return indmax if return_index else ages[indmax]
    else:
        indmin = np.argmin(trajectory)
        return indmin if return_index else ages[indmin]
    
sa_network_behaviors = [
    'min', #medial wall
    'max', #control
    'max', #default
    'min', #DorsAttn
    'max', #limbic
    'min', #SalVenAttn
    'min', #SomMot
    'min' #Vis
    ]
mr_network_behaviors = [
    'min', #medial wall
    'max', #control
    'min', #default
    'max', #DorsAttn
    'max', #limbic
    'max', #SalVentAttn
    'min', #SomMot
    'min' #Vis
    ]
    
def get_maturation_ages_yeo7(trajectories,ages,behaviors):
    mature_ages = []
    
    for i in range (trajectories.shape[1]):
        if behaviors[i] == 'max':
            maximum = True
        else:
            maximum = False
        ma = maturation_age(trajectories[:,i],ages=ages,maximum = maximum)
        mature_ages.append(ma)
    return mature_ages
