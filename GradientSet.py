#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:52:46 2023

@author: patricktaylor
"""

import reading_writing as rw 
import utility as uts 
import numpy as np 
import Gradient 
import pandas as pd 
from sklearn.decomposition import PCA
import fileops as fps 

class GradientSet:
    
    def __init__(self, pathlist = None, garrays = None, index = None, 
                 get_ids = True, get_vals = True, dtable = None, 
                 aligned = False, get_ages = True):
        """
    
        Parameters
        ----------
        pathlist : str or list of str, optional
            list of paths to gradient .npy files formatted as {ID}_grads.npy, 
            if vals are available in same directory, they will also be loaded.
            if single string is supplied, all files in that directory are 
            searched.
        index : any, optional 
            user-supplied additional index on gradient set 
        get_ids : boolean, optional
            if true, scrape IDs from gradient filenames. The default is True.
        get_vals : boolean, optional
            if true, assume existence/ naming of eigenvalues and load them. 
            The default is True.
        dtable : pandas.DataFrame, optional
            df already containing Gradient objects. if supplied, this will 
            be used to instantiate GradientSet directly. The default is None.

        Returns
        -------
        None. populates dtable attribute with Gradient objects

        """
        if garrays is not None: 
            self.dtable = pd.DataFrame()
            glist = []
            for i,ga in enumerate(garrays):
                g = Gradient.Gradient(garray=ga)
                glist.append(g)
            self.dtable['grads'] = np.array(glist) 
            if index is not None: 
                self.dtable['index'] = index 
        elif dtable is None:
            self.dtable = pd.DataFrame()
            if index is not None: 
                self.dtable['index'] = index 
            glist = []
            ids = []
            if pathlist is not None: 
                if type(pathlist) == str:
                    pathlist = fps.get_paths_with_pattern(pathlist,
                                                          fname_suffix='grads.npy',
                                                          filenames=False)
                for (i,p) in enumerate(pathlist):
                    if get_ids:
                        ind = p.rfind('/') + 1
                        gid = p[ind:-10]
                        if get_vals:
                            val_path = p[:-9] + 'vals.npy'
                            g = Gradient.Gradient(grad_path = p,
                                                  val_path = val_path, gid = gid)
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
        
        self.aligned = aligned 
        
        if get_ages:
            self.get_ages_bcp_hcpd_hcpya_hcpa()
            
            
    @property
    def length(self):
        #number of gradients in set
        return self.dtable.shape[0]
    
    def ages(self):
        if not 'age' in self.dtable:
            self.dtable['age'] = np.nan
        else:
            return self.dtable['age'].to_numpy()
        
    def g(self, ind, return_obj):
        #return grad obj or array at index ind
        if return_obj:
            return self.dtable['grads'][ind]
        else:
            return self.dtable['grads'][ind].garray
    
    def get_g_from_gid(self, gid):
        g = self.dtable[self.dtable['gid'] == gid]['grads'].values[0]
        return g
    @property
    def grad_arr_list(self):
        glist = []
        for g in self.dtable.loc[:,'grads']:
                glist.append(g.garray)                       
        return np.array(glist)
    
    def glist(self):
        #return list of grad arrays
        glist = []
        for g in self.dtable.loc[:,'grads']:
            glist.append(g.garray)
        return glist 
    
    def gwhere_between(self, column, lower, upper):
        #get grad arrays corresponding to column value between lower and upper
        gs = self.select_between(column, lower, upper)
        return gs.glist()
    
    
    def compute_metric_on_grads(self, function, metric_name = None, *args):
        #evaluate any function whose first argument 
        #is gradient array on all gradients in the set 
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
        #return new GradientSet where column field equals value
        newtable = self.dtable[self.dtable[column] == value]
        
        return GradientSet(dtable = newtable)
    
    def select_between(self, column, lower, upper):
        #return new GradientSet where lower < column field < upper
        newtable = self.dtable[self.dtable[column].between(lower, upper)]
        
        return GradientSet(dtable = newtable)
    
    def get_age_from_gid_bcp(self):
        if not 'age' in self.dtable:
            self.dtable['age'] = np.nan
        for i, g in enumerate(self.dtable['grads']):
            if g.gid.startswith('MN') or g.gid.startswith('NC'):
                subage = int(g.gid[12:])/365
                self.dtable.loc[i,'age'] = subage
            
        
    def get_field_from_dataframe(self, dataframe, foreign_fieldname, 
                                 native_fieldname, foreign_key = 'src_subject_id', 
                                 native_key = 'gid', divide_by = 12,intkey = False):
        """
        

        Parameters
        ----------
        dataframe : pandas.DataFrame() or str 
            external dataframe or path to it containing gradient info.
        foreign_fieldname : str
            column name in external dataframe from which to import new data.
        native_fieldname : str
            column name in GradientSet dataframe as destination of new data.
        foreign_key : str, optional
            column in external df to find matching rows. The default is 
            'src_subject_id'.
        native_key : str, optional
            column in GradientSet dataframe to search for in external df. 
            The default is 'gid'.

        Returns
        -------
        adds column native_fieldname to dtable with matching data from 
        external dataframe.

        """
        ff = foreign_fieldname
        nf = native_fieldname
        fk = foreign_key
        nk = native_key
        df = dataframe
        if type(df) == str: 
            if df[-1] == 'v' : 
                df = pd.read_csv(df)
            else: 
                df = pd.read_excel(df)
        if not nf in self.dtable.columns:
            self.dtable[nf] = np.nan
        for i in range (len(self.dtable)):
            n = self.dtable[nk][i]
            if intkey and n.isdigit():
                n = int(n)
            if n in set(df[fk]):
                self.dtable.loc[i, nf] = (df[df[fk] == n][ff].to_numpy()[0])/divide_by
            
    def get_ages_bcp_hcpd_hcpya_hcpa(self):
        
        
        hcpd = '/Users/patricktaylor/Documents/lifespan_analysis/HCPD_subject_infosheet.xls'
        hcpa = '/Users/patricktaylor/Documents/lifespan_analysis/HCPA_subject_infosheet.xls'
        hcpya = '/Users/patricktaylor/Documents/lifespan_analysis/HCPYA/HCPYA_restricted.csv'
        self.get_field_from_dataframe(hcpd,'interview_age', 'age')
        self.get_field_from_dataframe(hcpa, 'interview_age', 'age')
        self.get_field_from_dataframe(hcpya,'Age_in_Yrs', 'age','Subject',
                                      divide_by = 1, intkey = True)
        self.get_age_from_gid_bcp()
        
    def compute_pca_template(self, n_comp = 3,check_sign = True, **kwargs):
        """
        

        Parameters
        ----------
        n_comp : int, optional
            number of components to compute. The default is 3.
        **kwargs : 
            future.

        Returns
        -------
        None. computes principal dimensions of variation across all gradients 
        in the set.

        """
        glist = []
        if 'lower' in kwargs:          
            for i,g in enumerate(self.dtable.loc[:,'grads']):
                if ((kwargs['lower'] <=  self.dtable.loc[i,kwargs['column']]) 
                    and (self.dtable.loc[i,kwargs['column']] <= kwargs['upper'])):
                    glist.append(g)          
            print(len(glist), ' grads used for template')
        else:
            for g in self.dtable.loc[:,'grads']:
                glist.append(g)                       
        gmat=np.zeros((glist[0].nvert,len(glist)*n_comp))
        for i in range (len(glist)):
            gmat[:,n_comp*i:n_comp*(i+1)]=glist[i].garray[:,:n_comp]
        pca=PCA(n_components=3)
        pca.fit(gmat.T)
        v=pca.components_.T
        if check_sign:
            if v[4185,0]<0:
                v[:,0] = -1*v[:,0]
            if v[5890,1]>0:
                v[:,1] = -1*v[:,1]
            if v[186,2]<0:
                v[:,2] = -1*v[:,2]
                
        self.template_grads = v
        
    def procrustes_align(self,replace = True, lower = 14, upper = 40, **kwargs):
        """
        

        Parameters
        ----------
        replace : boolean, optional
            if True, replace garrays with aligned garrays. The default is True.
            if False, return new GradientSet containing aligned gradients.
        **kwargs : 
            arguments for self.compute_pca_template().

        Returns
        -------
        None. computes iterative procrustes alignment to template_grads across 
        all gradients 

        """
        garrlist = []
        for g in self.dtable['grads']:
            garrlist.append(g.garray)
        if not hasattr(self, 'template_grads'):           
            self.compute_pca_template(3,lower = lower, upper = upper, column = 'age')    
        glist = uts.procrustes_alignment(np.array(garrlist)[:,:,:3], 
                                         self.template_grads, n_iter=130, 
                                         tol=1e-20, verbose=True)
        if replace:
            for i,g in enumerate(self.dtable['grads']):
                
                g.garray = glist[i]
            self.aligned = True
        else:
            new_gs = GradientSet(garrays = glist)
            return new_gs
            
    def surface_plot(self, gradind, plotind = None):
        self.g(gradind,True).surface_plot(plotind)
    
    def embed_plots_2d(self, inds = None, ids = None):
        
        if ids is not None:
            if type(ids) ==str:
                ids = [ids]
            for i in ids:
                g = self.get_g_from_gid(i)
                g.embed_plot_2d(age = self.dtable[self.dtable['gid'] == g.gid]['age'].to_numpy()[0])
        else:
            if type(inds) == int:
                inds = [inds]
            for i in inds:
                g = self.g(i,True)
                g.embed_plot_2d(age = self.dtable['age'][i])
            
            
    def save_to_dataframe(self,directory,name_suffix = 'aligned_cos'):
        for k in range(3):
            dataframe=pd.DataFrame()
            dataframe['Name']=self.dtable['gid']
            dataframe['Age']=self.dtable['age']
            gmat = self.grad_arr_list()
            for i in range(self.dtable['grads'][0].nvert):
                dataframe[f'v{i+1}'] = gmat[:,i,k]
                
            #dataframe.to_csv(f'/Users/patricktaylor/Documents/lifespan_analysis/individual/10p_fwhm3/dataframes/g{k+1}_aligned_cos.csv')
            dataframe.to_csv(directory+f'g{k+1}_{name_suffix}.csv')
            
        
        
        
        
    
    
    
    
        
            
        
                