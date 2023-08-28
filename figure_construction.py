#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 12:29:26 2023

@author: patricktaylor
"""
import numpy as np
import os 
from PIL import Image
import reading_writing as rw

def save_grid(features,directory,lh,cranges = None, uniform_crange = True,crange_percentile = None,cbar=True):
    n_columns = features.shape[0]
    n_rows = features.shape[1]
    
    if cranges is None and uniform_crange:
        cranges = []
        for i in range (n_rows):
            if crange_percentile is None:
                cr = (np.min(features[:,i,:]),np.max(features[:,i,:]))
            else:
                cr = (np.percentile(features[:,i,:].T,crange_percentile),np.percentile(features[:,i,:].T,100-crange_percentile))
            cranges.append(cr)
    if not uniform_crange:
        cranges = [None]*n_rows
    for i in range (n_columns):
        
        for j in range (n_rows):
            
            rw.plot_surf(features[i,j],lh,crange = cranges[j],cbar = cbar,save = directory + f'{j}_{i}.png')
    


def stitch_images(directory_path,filename = None ):
    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # Extract row and column numbers and sort images accordingly
    sorted_files = sorted(files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

    # Initialize list to store images
    images = []

    # Open images
    for file in sorted_files:
        images.append(Image.open(os.path.join(directory_path, file)))

    # Assuming all images are of the same size
    img_width, img_height = images[0].size

    # Determine the total number of unique rows and columns
    num_rows = len(set([int(f.split('_')[0]) for f in sorted_files]))
    num_cols = len(set([int(f.split('_')[1].split('.')[0]) for f in sorted_files]))

    # Create an empty image with the necessary width and height
    stitched_image = Image.new('RGB', (img_width * num_cols, img_height * num_rows))

    # Paste each image into its position
    for idx, image in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        stitched_image.paste(image, (col * img_width, row * img_height))

    # Save the stitched image
    if filename is None:
        filename = 'stitched.png'
    stitched_image.save(os.path.join(directory_path, filename))
    
    