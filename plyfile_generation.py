#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 17:47:30 2023

@author: patricktaylor
"""
import numpy as np 

from plyfile import PlyData, PlyElement



def save_plyfile_mesh_w_colors(sc,si,colors,savepath):
    vertex_positions = sc 
    mesh_indices = si 
    vertex_colors = colors
    vertex_data = np.empty(vertex_positions.shape[0],
                       dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_data['x'] = vertex_positions[:, 0]
    vertex_data['y'] = vertex_positions[:, 1]
    vertex_data['z'] = vertex_positions[:, 2]
    vertex_data['red'] = vertex_colors[:, 0]
    vertex_data['green'] = vertex_colors[:, 1]
    vertex_data['blue'] = vertex_colors[:, 2]
    
    vertex_element = PlyElement.describe(vertex_data, 'vertex')
    
    # Create the face element
    face_data = np.empty(mesh_indices.shape[0],
                            dtype=[('vertex_indices', 'i4', (3,))])
    
    face_data['vertex_indices'] = mesh_indices
    face_element = PlyElement.describe(face_data, 'face')
    
    # Create the PlyData object and write to file
    ply_data = PlyData([vertex_element, face_element])
    ply_data.write(savepath)
    