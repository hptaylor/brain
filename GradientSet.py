#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:52:46 2023

@author: patricktaylor
"""

import io
import utility as uts 
import numpy as np 
import Gradient 
import pandas as pd 
class GradientSet:
    
    def __init__(self, pathlist = None, index = None, get_ids = True, get_vals = True, dtable = None):
        
        if dtable is None:
        
            self.dtable = pd.DataFrame()
            
            if index is not None: 
                self.dtable['index'] = index 
            
            glist = []
            ids = []
            if pathlist is not None: 
                
                for (i,p) in enumerate(pathlist): 
                    
                        
                    if get_ids:
                        
                        ind = p.rfind('/') + 1
                        
                        gid = p[ind:-10]
                        
                        if get_vals:
                            val_path = p[:-9] + 'vals.npy'
                            g = Gradient.Gradient(p,val_path, gid)
                        else:
                            g = Gradient.Gradient(p,None, gid)
                        
                        ids.append(gid)
                            
                        
                    
                    else:
                        if get_vals:
                            val_path = p[:-9] + 'vals.npy'
                            g = Gradient.Gradient(p,val_path)
                        else:
                            g = Gradient.Gradient(p)
                    
                
                    
                    
                    glist.append(g)
                
                    
                self.dtable['grads'] = np.array(glist)
                self.dtable['gid'] = np.array(ids)
        
        else:
            self.dtable = dtable 
            
            
        
    def compute_metric_on_grads(self, function, metric_name = None, *args):
        
        reslist = []
        
        for grad in self.dtable['grads']:
            
            g = grad.garray
            
            res = function(g,*args)
            
            reslist.append(res)
            
        reslist = np.array(reslist)
        
        if metric_name is not None:
            
            self.dtable[metric_name] = reslist 
            
        else: 
            return reslist 
        
    def select_equal(self, column, value):
        
        newtable = self.dtable[self.dtable[column] == value]
        
        return GradientSet(dtable = newtable)
    
    def select_between(self, column, lower, upper):
        
        newtable = self.dtable[self.dtable[column].between(lower, upper)]
        
        return GradientSet(dtable = newtable)
    
    
    
    
        
            
        
                