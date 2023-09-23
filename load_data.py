#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 11:24:36 2023

@author: patricktaylor
"""
import brainspace as bs 
from GradientSet import GradientSet
from FittedGradient import FittedGradient
from reading_writing import read_schaefer400_7net_parc
import numpy as np 

lh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii')
rh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii')
gs = GradientSet('/Users/patricktaylor/lifespan_analysis/individual/t10p_cos/grads/')
temp = gs.select_between('age',14,40).compute_pca_template(10,True,True,False,True)/50
reftemp = np.zeros(temp.shape) 
reftemp[:,0] = temp[:,0];reftemp[:,1] = -1*temp[:,2];reftemp[:,2] = temp[:,1];reftemp[:,3:] = temp[:,3:];gs.procrustes_align_noniterative(n_comp=10,scale=True,center=True,template=reftemp)
gs.apply_cohort_shift('/Users/patricktaylor/lifespan_analysis/individual/t10p_cos/dataframes/g1_k6_cohort_effect.csv')

fg = FittedGradient('/Users/patricktaylor/lifespan_analysis/individual/t10p_cos/dataframes/g1_k6_LOG_aligned_cos_GAMM_3M.csv')

parc7 = read_schaefer400_7net_parc('/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Schaefer_Parcellation/Downsampled/Atlas_240Months_lh.schaefer400.7nw.downsampled.L5.label.gii')