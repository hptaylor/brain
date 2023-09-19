#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:04:18 2023

@author: patricktaylor
"""
import numpy as np 
import scipy.stats as ss 
import utility as uts 


def clean_data(data, ages = None,sigma = 2):
    std = sigma*np.std(data)
    cleandata = data[((data-np.mean(data))**2)/sigma < std]
    if ages is not None:
        cleanages = ages[((data-np.mean(data))**2)/sigma < std]
    return cleandata if ages is None else (cleandata, cleanages)


def centile_binning(data, percentage,return_indices = True):
    """
    Computes the average of values within each centile bin.

    Args:
    - data (numpy.ndarray): The 1D input array to bin.
    - percentage (float): The percentage width of each bin.

    Returns:
    - bin_means (numpy.ndarray): Array of mean values within each bin.
    - bin_indices (list of numpy.ndarray): List of arrays, each containing the indices from the input data that fall within the corresponding bin.
    """
    
    # Compute the number of bins based on the provided percentage
    n_bins = int(100 / percentage)
    
    n_inds_per_bin = int(len(data)*percentage/100)
    
    sorted_inds = np.argsort(data)
    
    bin_indices = []
    bin_means = [] 
    
    for i in range (n_bins):
        bin_indices.append(sorted_inds[i*n_inds_per_bin:(i+1)*n_inds_per_bin])
    # Calculate the mean value for each bin
    bin_means = np.array([np.mean(data[bin_indices[i]]) for i in range(n_bins)])
    
    # Fetch indices of the input data for each bin
    if return_indices:
    
        return bin_means, bin_indices
    else:
        return bin_means

def compute_confidence_interval(fit,std_error,percentile=97.5):
    z_score = ss.norm.ppf(percentile/100)
    moe = z_score * std_error
    upper_bound = fit + moe
    lower_bound = fit - moe
    return lower_bound, upper_bound

    