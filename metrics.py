#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:54:03 2023

@author: patricktaylor
"""
import numpy as np
from scipy import sparse
import utility as uts 
from sklearn.neighbors import kneighbors_graph


def dispersion_centroid(vecs):
    centroid = np.mean(vecs, axis=0)
    dif = vecs - centroid
    squared_dists = np.linalg.norm(dif, axis=1, ord=2)**2
    disp = np.mean(squared_dists)
    return disp

def embedding_degree_centrality(vecs,nn_num = 300):
    inds, dists = uts.neighbors(vecs,vecs,num=nn_num)
    avgdist = np.mean(dists, axis = 1)
    return avgdist 
    