#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:24:11 2023

@author: patricktaylor
"""
import os 
import glob

def run_script_for_all_subjects(path_to_script, subjectlist, parameter_list):
    
    for i, sub in enumerate(subjectlist):
        
        paramstring = ''
        
        for c in parameter_list[i]:
            paramstring = paramstring + c + ' '
            
        
        commandstring = f'sbatch {path_to_script}' + paramstring
        
        os.system(commandstring)
        
    return 

def get_paths_with_pattern(directory,fname_prefix = '', fname_suffix = '', filenames = True):
    
    if not filenames:
        return glob.glob(directory + fname_prefix + '*' + fname_suffix)
    else: 
        paths = glob.glob(directory + fname_prefix + '*' + fname_suffix)
        
        fnames = [f[len(directory):] for f in paths]
        
        return fnames 

def write_subject_text_file(subjlist, savepath):
    with open(savepath, 'a') as file :
        for sub in subjlist:
            file.write(sub+'\n')
    file.close()
    
    return 

        
    