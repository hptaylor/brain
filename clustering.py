#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:45:55 2023

@author: patricktaylor
"""
import numpy as np 
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

def hac(vecs,nclust=7):
    cl=AgglomerativeClustering(n_clusters=nclust, affinity='euclidean', linkage='ward').fit(vecs)
    lab = cl.labels_
    return lab 

def kmeans(vecs, nclust = 7):
    cl = KMeans(n_clusters=nclust, random_state=0, n_init="auto").fit(vecs)
    lab = cl.labels_
    return lab