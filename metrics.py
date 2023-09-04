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

def dists_to_centroid(vecs):
    centroid = np.mean(vecs, axis=0)
    dif = vecs - centroid
    dists = np.linalg.norm(dif, axis=1, ord=2)
    return dists
def dist_to_centroid_var(vecs):
    return np.var(dists_to_centroid(vecs))

def embedding_degree_centrality(vecs,nn_num = 300):
    inds, dists = uts.neighbors(vecs,vecs,num=nn_num)
    avgdist = np.mean(dists, axis = 1)
    return avgdist 

def get_avg_val_parc(grads,parc,std_errors=None):
        vals = np.zeros((grads.shape[0],np.unique(parc).shape[0],3))
        avg_std_errors = np.zeros(vals.shape)
        for z in range (vals.shape[1]):
            inds = np.where(parc == z)[0]
            for i in range (vals.shape[0]):
                for j in range (vals.shape[2]):
                    if std_errors is not None:
                        avg_std_errors[i,z,j] = get_average_std_errors(std_errors[i,inds,j])
                        avg_std_errors[i,z,j] = get_average_std_errors(std_errors[i,inds,j])
                    m = np.mean(grads[i,inds,j])
                    vals[i,z,j] = m
        return vals if std_errors is None else (vals,avg_std_errors)
    
    
def parc_percentile_trajectory(grads,percent,parc):
    '''
    

    Parameters
    ----------
    grads : ndarray (n_timepoints,n_vert,n_grads)
        DESCRIPTION.
    percent : int
        0 < percent < 100.
    parc : ndarray (n_vert)
        DESCRIPTION.

    Returns
    -------
    result : ndarray (n_timepoints,n_networks,n_grads)
        DESCRIPTION.

    '''
    result = np.zeros((grads.shape[0],len(set(parc)),grads.shape[2]))
    
    
    for i in range(len(set(parc))):
        for k in range(grads.shape[2]):
            inds = np.where(parc==i)[0]
            
            for j in range (grads.shape[0]):
                
                result[j,i,k] = np.percentile(grads[j,inds,k],percent)
    return result 


    
    
    
    
def top_k_indices(arr, k):
    # use argpartition to find indices of top k values
    return np.argpartition(arr, -k)[-k:]

def bottom_k_indices(arr, k):
    # use argpartition to find indices of bottom k values
    return np.argpartition(arr, k)[:k]

def get_average_std_errors(ses):
    
    sum_of_variances = 0
    for se in ses:
        var = se**2
        sum_of_variances += var
    
    return np.sqrt(sum_of_variances/len(ses))

def get_grad_poles(grads,percent):
    result = np.zeros(grads.shape)
    keep_num = int(grads.shape[0] * percent/100)
    for i in range (grads.shape[1]):
        topinds = top_k_indices(grads[:,i], keep_num)
        bottominds = bottom_k_indices(grads[:,i], keep_num)
        result[topinds,i] = 1
        result[bottominds,i] = -1 
        result[result[:,i]==0,i] = np.nan
    return result 

def gradient_pole_trajectories(gradlist,percent,ref_grads,std_errors=None,return_range=True):
    result = np.zeros((gradlist.shape[0],gradlist.shape[2],2))
    ##topresult = np.zeros((gradlist.shape[0],gradlist.shape[2]))
    #bottomresult = np.zeros((gradlist.shape[0],gradlist.shape[2]))
    keep_num = int(gradlist.shape[1] * percent/100)
    
    avg_std_errors = np.zeros(result.shape)
    
    for i in range (gradlist.shape[2]):
        
        topinds = top_k_indices(ref_grads[:,i],keep_num)
        bottominds = bottom_k_indices(ref_grads[:,i],keep_num)
        
        for j in range (len(gradlist)):
            if std_errors is not None:
                avg_std_errors[j,i,0] = get_average_std_errors(std_errors[j,topinds,i])
                avg_std_errors[j,i,1] = get_average_std_errors(std_errors[j,bottominds,i])
            result[j,i,0] = np.mean(gradlist[j,topinds,i])
            result[j,i,1] = np.mean(gradlist[j,bottominds,i])
            #topresult[j,i] = np.mean(gradlist[j,topinds,i])
            #bottomresult[j,i] = np.mean(gradlist[j,bottominds,i])
    if not return_range:
        return result
    else:
        if std_errors is not None:
            range_std_error = np.zeros((result.shape[0],result.shape[1]))
            for i in range (result.shape[0]):
                for j in range (result.shape[1]):
                    range_std_error[i,j] = get_average_std_errors(avg_std_errors[i,j])
            
            return result[:,:,0] - result[:,:,1], range_std_error
        else:
            return result[:,:,0] - result[:,:,1]