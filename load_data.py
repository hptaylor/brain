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

lh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii')
rh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii')

gs = GradientSet('/Users/patricktaylor/lifespan_analysis/individual/t10p_trans/grads/')
temp = gs.compute_pca_template(10,True,True,False,True)/100
gs.procrustes_align_noniterative(n_comp=10,scale=True,center=True,template=temp)
gs.apply_cohort_shift('/Users/patricktaylor/lifespan_analysis/individual/t10p_trans/dataframes/g1_k6_cohort_effect.csv')

fg = FittedGradient('/Users/patricktaylor/lifespan_analysis/individual/t10p_trans/dataframes/g1_k6_LOG_aligned_cos_GAMM_3M.csv')

parc7 = read_schaefer400_7net_parc('/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Schaefer_Parcellation/Downsampled/Atlas_240Months_lh.schaefer400.7nw.downsampled.L5.label.gii')