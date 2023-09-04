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
    


def stitch_images(directory_path):
    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Extract row and column numbers from filenames and sort images
    sorted_files = sorted(files, key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1].split('.')[0])))

    # Initialize list to store images
    images = []

    # Open and crop images
    for file in sorted_files:
        img = Image.open(os.path.join(directory_path, file))

        width, height = img.size
        left_margin = width * 0.10
        right_margin = width * 0.90
        upper_margin = height * 0.20
        lower_margin = height * 0.80

        # Crop the image
        cropped_img = img.crop((left_margin, upper_margin, right_margin, lower_margin))

        images.append(cropped_img)

    # Assuming all cropped images are of the same size
    img_width, img_height = images[0].size

    # Determine number of unique rows and columns based on file count
    num_rows = max(int(file.split('_')[0]) for file in files) + 1
    num_cols = max(int(file.split('_')[1].split('.')[0]) for file in files) + 1

    # Create an empty image with the necessary width and height
    stitched_image = Image.new('RGB', (img_width * num_cols, img_height * num_rows))

    # Paste each cropped image into its position
    for idx, image in enumerate(images):
        row = idx // num_cols
        col = idx % num_cols
        stitched_image.paste(image, (col * img_width, row * img_height))

    # Save the stitched image
    stitched_image.save(os.path.join(directory_path, 'stitched.png'))
