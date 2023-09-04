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
    
    