#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:31:28 2023

@author: patricktaylor
"""
import numpy as np 


def map_features_to_vertex(features, labels):
    """
    Maps parcellated features to vertices.

    Parameters:
    - features (array-like): The features from parcellation.
    - labels (array-like): The parcellation labels corresponding to each vertex.

    Returns:
    - numpy array: An array with the features mapped to each vertex.
    """
    vertex_features = np.zeros(len(labels))
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        vertex_features[indices] = features[label]
    
    return vertex_features


def map_multi_features_to_vertex(multi_features, labels):
    """
    Maps multi-dimensional parcellated features to vertices.

    Parameters:
    - multi_features (2D array-like): The multi-dimensional features from parcellation.
    - labels (array-like): The parcellation labels corresponding to each vertex.

    Returns:
    - numpy array: A 2D array with the multi-dimensional features mapped to each vertex.
    """
    num_features = multi_features.shape[1]
    vertex_features = np.zeros((len(labels), num_features))
    
    for i in range(num_features):
        vertex_features[:, i] = map_features_to_vertex(multi_features[:, i], labels)
    
    return vertex_features


def compute_parcel_mean(features, labels):
    """
    Computes the mean feature vector for each unique label.

    Parameters:
    - features (2D array-like): The features for each vertex.
    - labels (array-like): The labels corresponding to each vertex.

    Returns:
    - numpy array: A 2D array containing the mean feature vector for each unique label.
    """
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    num_features = features.shape[1]
    mean_vectors = np.zeros((num_labels, num_features))
    
    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)[0]
        mean_vectors[i] = np.mean(features[indices], axis=0)
    
    return mean_vectors
