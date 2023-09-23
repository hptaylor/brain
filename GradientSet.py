#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:52:46 2023

@author: patricktaylor
"""
import os 
import reading_writing as rw 
import utility as uts 
import numpy as np 
from brainspace.gradient.alignment import procrustes
from scipy.stats import spearmanr 
import brainspace as bs
import Gradient 
import pandas as pd 
from sklearn.decomposition import PCA
import fileops as fps 
import plotting as pltg
import clustering as clst 
import metrics as mts
import GammFit as gf

class GradientSet:
    
    def __init__(self, pathlist = None, garrays = None, index = None, 
                 get_ids = True, get_vals = True, dtable = None, 
                 aligned = False, get_ages = True, get_cohort = True, get_degrees = True):
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
            degrees = []
            cos_degrees = []
            if pathlist is not None: 
                if type(pathlist) == str:
                    pathlist = fps.get_paths_with_pattern(pathlist,
                                                          fname_suffix='grads.npy',
                                                          filenames=False)
                for (i,p) in enumerate(pathlist):
                    if get_ids:
                        ind = p.rfind('/') + 1
                        gid = p[ind:-10]
                        if get_degrees:
                            cd = np.load(p[:ind]+'degree/' + gid + '_cos_degree.npy')
                            d = np.load(p[:ind]+'degree/' + gid + '_degree.npy')
                            degrees.append(d)
                            cos_degrees.append(cd)
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
                if get_degrees:
                    self.dtable['degree'] = degrees
                    self.dtable['cos_degree'] = cos_degrees
            
        else:
            self.dtable = dtable
        
        self.aligned = aligned 
        
        if get_ages:
            self.get_ages_bcp_hcpd_hcpya_hcpa()
        if get_cohort:
            cohorts = []
            for sid in self.dtable['gid']:
                if sid[0] == 'M' or sid[0] == 'N':
                    cohorts.append(0)
                if sid.startswith('HCD'):
                    cohorts.append(1)
                if sid[0].isdigit():
                    cohorts.append(2)
                if sid.startswith('HCA'):
                    cohorts.append(3)
            self.dtable['cohort_id'] = np.array(cohorts)
            
        self.nvert = self.dtable['grads'][0].nvert
    def __getitem__(self, index):
        return self.dtable['grads'].iloc[index]
    
    def __repr__(self):
        num_gradients = len(self.dtable['grads'])
        return f"GradientSet(num_gradients={num_gradients}, aligned={self.aligned})"

    def __str__(self):
        num_gradients = len(self.dtable['grads'])
        return f"GradientSet with {num_gradients} gradient sets each with {self.ngrad} gradients (aligned: {self.aligned})"
    
    @property
    def length(self):
        #number of gradients in set
        return self.dtable.shape[0]
    @property
    def ages(self):
        if not 'age' in self.dtable:
            self.dtable['age'] = np.nan
        else:
            return self.dtable['age'].to_numpy()
    @property
    def gids(self):
        return self.dtable['gid'].to_numpy()
    
    @property
    def ngrad(self):
        return self.dtable['grads'][0].ngrad
    @property
    def sids(self):
        gids = self.gids
        sids = []
        for i in gids:
            if i.startswith('M') or i.startswith('N'):
                sids.append(i[:11])
            else:
                sids.append(i)
        return np.array(sids)
    
    @property
    def evals(self):
        sub_evals = []
        for g in self.dtable['grads']:      
            sub_evals.append(g.varray)
        return sub_evals
    @property
    def granges(self):
        return self.get_ranges()
    
    @property 
    def gvars(self):
        return self.get_vars()
    
    def g(self, ind, return_obj=True):
        #return grad obj or array at index ind
        if return_obj:
            return self.dtable['grads'][ind]
        else:
            return self.dtable['grads'][ind].garray
    
    def get_g_from_gid(self, gid):
        g = self.dtable[self.dtable['gid'] == gid]['grads'].values[0]
        return g
    
    def grad_arr_list(self):
        glist = []
        for g in self.dtable.loc[:,'grads']:
                glist.append(g.garray)                       
        return np.array(glist)
    def get_explanation_ratios(self):
        ratios = [g.explanation_ratios for g in self.dtable['grads']]
        return np.array(ratios)
    def get_range_ratios(self):
        ratios = [g.range_ratios for g in self.dtable['grads']]
        return np.array(ratios)
    
    def glist(self):
        #return list of grad arrays
        glist = []
        for g in self.dtable.loc[:,'grads']:
            glist.append(g.garray)
        return glist 
    
    def mask_grads(self,mask=None):
        if mask is not None:
            self.mask = mask
            
        for g in self.dtable['grads']:
            g.garray = uts.mask_medial_wall_vecs(g.garray,self.mask)
        
    def unmask_grads(self):
        for g in self.dtable['grads']:
            g.garray = uts.unmask_medial_wall_vecs(g.garray,self.mask)
            
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
    
    def get_dispersions(self,ndim=3,add_column=True):
        disps = []
        for g in self.dtable['grads']:
            disps.append(g.compute_dispersion(ndim))
        if add_column:
            self.dtable['dispersion'] = disps
        else:
            return disps 
    
    def get_dispersions_parc(self,parc,ndim=3):
        disps = np.zeros((self.length,np.unique(parc).shape[0]))
        for i, g in enumerate(self.dtable['grads']):
            disps[i] = g.get_dispersion_parc(parc,ndim)
        return disps 
    def get_mean_value_parc(self,parc,ndim=3):
        vals = np.zeros((self.length,np.unique(parc).shape[0],ndim))
        indslist = []
        for z in range(np.unique(parc).shape[0]):
            inds = np.where(parc == z)[0]
            indslist.append(inds)
        for i, g in enumerate(self.dtable['grads']):
            for z in range (len(indslist)):
                for j in range (ndim):
                    vals[i,z,j] = np.mean(g.garray[indslist[z],j])
        return vals
                
    
    def get_ranges(self):
        granges = np.zeros((self.length,min(10,self.g(0,False).shape[1])))
        for i,g in enumerate(self.dtable['grads']):
            granges[i] = g.grange
        return granges
    def get_vars(self):
        gvars = np.zeros((self.length,self.g(0,False).shape[1]))
        for i,g in enumerate(self.dtable['grads']):
            gvars[i] = g.gvar
        return gvars
    
    def get_pole_ranges(self,percent=25,temp=None):
        if temp is None:
            temp = self.template_grads
        pole_ranges = uts.gradient_pole_trajectories(self.grad_arr_list(),percent,temp)
        return pole_ranges
        
    def select_equal(self, column, value):
        #return new GradientSet where column field equals value
        newtable = self.dtable[self.dtable[column] == value]
        
        return GradientSet(dtable = newtable)
    
    def select_between(self, column, lower, upper):
        #return new GradientSet where lower < column field < upper
        #newtable = self.dtable[self.dtable[column].between(lower, upper)]
        newtable = self.dtable.loc[(self.dtable[column] > lower)]
        newtable = newtable.loc[(newtable[column] < upper )]
        newtable.reset_index(drop=True, inplace = True )
        return GradientSet(dtable = newtable,get_ages=False)
    
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
        
        
        hcpd = '/Users/patricktaylor/lifespan_analysis/HCPD_subject_infosheet.xls'
        hcpa = '/Users/patricktaylor/lifespan_analysis/HCPA_subject_infosheet.xls'
        hcpya = '/Users/patricktaylor/lifespan_analysis/HCPYA/HCPYA_restricted.csv'
        self.get_field_from_dataframe(hcpd,'interview_age', 'age')
        self.get_field_from_dataframe(hcpa, 'interview_age', 'age')
        self.get_field_from_dataframe(hcpya,'Age_in_Yrs', 'age','Subject',
                                      divide_by = 1, intkey = True)
        self.get_age_from_gid_bcp()
        
    def compute_pca_template(self, n_comp = 3,check_sign = True, 
                             scale_by_vals=False, scale_m1_to_1=False, 
                             return_temp=False,return_evals=False, **kwargs):
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
        pca=PCA(n_components=n_comp)
        pca.fit(gmat.T)
        v=pca.components_.T
        if scale_by_vals:
            vals = pca.singular_values_
            for i,val in enumerate(vals):
                v[:,i]*=val 
        if scale_m1_to_1:
            v = uts.scale_vecs_m1_to_1(v)
            
        if check_sign:
            if v[4185,0]<0:
                v[:,0] = -1*v[:,0]
            if v[5890,1]>0:
                v[:,1] = -1*v[:,1]
            if v[186,2]<0:
                v[:,2] = -1*v[:,2]
        if return_temp:
            if return_evals:
                return v, pca.singular_values_
            else:
                return v 
        else:
            self.template_grads = v
    def two_step_template(self,n_comp=10,windows=[0,0.5,1,2,4,8,12,18,25,35,50,100]):
        templates = {}
        gmat = np.zeros((self.nvert, n_comp*(len(windows)-1)))
        
        for i in range(len(windows)-1):
            temp = self.compute_pca_template(n_comp,check_sign=False,
                                             return_temp=True,lower=windows[i],
                                             upper=windows[i+1],column='age',
                                             scale_by_vals=True)
            templates[f'{windows[i]}']=temp 
            gmat[:,n_comp*i:n_comp*(i+1)]=temp
        pca = PCA(n_components=n_comp)
        pca.fit(gmat.T)
        v=pca.components_.T
        if v[4185,0]<0:
            v[:,0] = -1*v[:,0]
        if v[5890,1]>0:
            v[:,1] = -1*v[:,1]
        if v[186,2]<0:
            v[:,2] = -1*v[:,2]
            
        return v, templates
    
    def procrustes_align(self,replace = True, lower = 14, upper = 40, n_comp = 3,return_reference=True, template=True,**kwargs):
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
            if template:
                self.compute_pca_template(n_comp,lower = lower, upper = upper, column = 'age')    
                glist = uts.procrustes_alignment(np.array(garrlist)[:,:,:n_comp], 
                                         self.template_grads, n_iter=25, 
                                         tol=1e-25, verbose=True, return_reference=return_reference)
                if return_reference:
                    glist, reference = glist 
            else:
                glist,reference = bs.gradient.alignment.procrustes_alignment(np.array(garrlist),n_iter=25,tol = 1e-25, return_reference=True)
    
        
            
        self.template_grads = reference
        if replace:
            for i,g in enumerate(self.dtable['grads']):
                
                g.garray = glist[i]
                g.aligned = True 
                
            self.aligned = True
        else:
            new_gs = GradientSet(garrays = glist)
            return new_gs
    
    def procrustes_align_noniterative(self,replace=True, lower = 14, upper = 40, 
                                      n_comp = 3, scale = False, center = False,
                                      scale_template = False,template=None):
        if template is None:
            garrlist = []
            for g in self.dtable['grads']:
                garrlist.append(g.garray)
            if not hasattr(self, 'template_grads'):           
                self.compute_pca_template(n_comp,lower = lower, upper = upper, column = 'age', scale_m1_to_1=scale_template) 
        if template is None:
            glist = [procrustes(g[:,:n_comp] ,self.template_grads,center,scale) for g in garrlist]
        else:
            glist = [procrustes(g[:,:n_comp] ,template,center,scale) for g in self.grad_arr_list()]
        if replace:
            for i,g in enumerate(self.dtable['grads']):
                
                g.garray = glist[i]
                g.aligned = True 
                
            self.aligned = True
        else:
            new_gs = GradientSet(garrays = glist)
            return new_gs
    
    def smooth_grads(self,smoothing_mat):
        
        for g in self.dtable['grads']:
            g.garray = smoothing_mat.dot(g.garray)
            
    def get_cos_sim_w_template(self,ndim=3):
        cossims = np.zeros((self.length,ndim))
        for i,g in enumerate(self.dtable['grads']):
            for j in range (cossims.shape[1]):
                cossims[i,j] = uts.cos_norm(g.garray[:,j],self.template_grads[:,j])
        return cossims 
    def get_spearman_w_template(self,return_p = False,ndim=3):
        spearmans = []
        for i,g in enumerate(self.dtable['grads']):
            sps = []
            for j in range (ndim):
                sps.append(spearmanr(g.garray[:,j],self.template_grads[:,j])[0])
            spearmans.append(sps)
        return np.array(spearmans)
    
    def get_template_clustering(self, nclust = 7, clust_type = 'HAC'):
        if clust_type == 'HAC':
            lab = clst.hac(self.template_grads,nclust=nclust)
            
        if clust_type == 'kmeans':
            lab = clst.kmeans(self.template_grads,nclust=nclust)
        
        self.clustering = lab 
    def plot_grad_ranges(self,n=3):
        granges = self.granges 
        f = pltg.plot_fits_separately(subject_data=granges[:,:3],subject_ages=self.ages,fit_colors=['r','g','b'],fit_names=['SA','VS','MR'],ylabel='grad range')
        #for i in range(n):
        #    pltg.plot_metric_vs_age_log(self.ages,granges[:,i],f'G{i+1} range')
            
    def plot_grad_vars(self,n=3):
        gvars = self.gvars 
        f = pltg.plot_fits_separately(subject_data=gvars[:,:3],subject_ages=self.ages,fit_colors=['r','g','b'],fit_names=['SA','VS','MR'],ylabel='grad variance')
        #for i in range(n):
        #    pltg.plot_metric_vs_age_log(self.ages,gvars[:,i],f'G{i+1} variance')  
    def plot_cos_sims_to_template(self,n=3):
        cossims = self.get_cos_sim_w_template()
        f = pltg.plot_fits_separately(subject_data=cossims[:,:3],subject_ages=self.ages,fit_colors=['r','g','b'],fit_names=['SA','VS','MR'],ylabel='cos sim')
        #for i in range (n):
        #    pltg.plot_metric_vs_age_log(self.ages,cossims[:,i],f'G{i+1} cosine sim')
    def plot_spearmanr_to_template(self,n=3):
        spearmanrs = self.get_spearman_w_template()
        f = pltg.plot_fits_separately(subject_data=spearmanrs[:,:3],subject_ages=self.ages,fit_colors=['r','g','b'],fit_names=['SA','VS','MR'],ylabel='spearman r with template')
        
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
            
            
# =============================================================================
#     def save_to_dataframe(self,directory,name_suffix = 'aligned_cos',cohort = False):
#         for k in range(3):
#             dataframe=pd.DataFrame()
#             dataframe['Name']=self.sids
#             dataframe['Age']=self.dtable['age']
#             if cohort:
#                 dataframe['Cohort_ID'] = self.dtable['cohort_id']
#                 
#             gmat = self.grad_arr_list
#             for i in range(self.dtable['grads'][0].nvert):
#                 dataframe[f'v{i+1}'] = gmat[:,i,k]
#                 
#             #dataframe.to_csv(f'/Users/patricktaylor/Documents/lifespan_analysis/individual/10p_fwhm3/dataframes/g{k+1}_aligned_cos.csv')
#             dataframe.to_csv(directory+f'g{k+1}_{name_suffix}.csv')
# =============================================================================
            
    

    def save_to_dataframe(self, directory, name_suffix='aligned_cos', cohort=True):
        gmat = self.grad_arr_list()
    
        for k in range(3):
            # Create a DataFrame for the vector data
            vector_data = pd.DataFrame(gmat[:, :, k], columns=[f'v{i+1}' for i in range(self.dtable['grads'][0].nvert)])
    
            # Create a DataFrame for the subject info
            subject_info = pd.DataFrame({'Name': self.sids, 'Age': self.dtable['age']})
            
            if cohort:
                subject_info['Cohort_ID'] = self.dtable['cohort_id']
            
            # Combine the subject info and vector data into a single DataFrame
            dataframe = pd.concat([subject_info, vector_data], axis=1)
            
            # Save the DataFrame to a CSV file
            dataframe.to_csv(directory + f'g{k+1}_{name_suffix}.csv', index=False)
    
    def save_metric_to_dataframe(self,metric,directory, name='metric', cohort=True,ndim=3):
        gmat = metric
    
        
            # Create a DataFrame for the vector data
        vector_data = pd.DataFrame(gmat, columns=[f'v{i+1}' for i in range(ndim)])
    
            # Create a DataFrame for the subject info
        subject_info = pd.DataFrame({'Name': self.sids, 'Age': self.dtable['age']})
            
        if cohort:
            subject_info['Cohort_ID'] = self.dtable['cohort_id']
            
            # Combine the subject info and vector data into a single DataFrame
        dataframe = pd.concat([subject_info, vector_data], axis=1)
            
            # Save the DataFrame to a CSV file
        dataframe.to_csv(directory + f'{name}.csv', index=False)
    def apply_cohort_shift(self,g1_path,add_column = False):
        cohort_shift = uts.load_cohort_effect_grads(g1_path)
        shifted_gmat = uts.apply_cohort_shift_grads(self.grad_arr_list()[:,:,:cohort_shift.shape[1]],self.dtable['cohort_id'].to_numpy(),cohort_shift)
        
        if add_column:
            self.dtable['shifted_grads'] = shifted_gmat
        else:
            for i,g in enumerate(self.dtable['grads']):
                
                g.garray = shifted_gmat[i]
                
                
            self.shifted = True
            
    def fit_gamm_metric(self,metric,directory,name,cohort=True,ndim=3,k=10):
        self.save_metric_to_dataframe(metric,directory,name,cohort,ndim)
        if cohort:
            os.system(f'/usr/local/bin/Rscript /Users/patricktaylor/Documents/brain/fit_metric_GAMM_cohort.R {directory} {name} {k}')
        else:
            os.system(f'/usr/local/bin/Rscript /Users/patricktaylor/Documents/brain/fit_metric_GAMM.R {directory} {name} {k}')
        return gf.GammFit(directory,name,ndim = ndim,cohort=cohort)
    
    def plot_eigenvalues(self,lower_age = None, upper_age = None,num_evals=None,exp_ratio=False,range_ratio=False,cmap = 'plasma'):
        if lower_age is not None:
            gs = self.select_between('age',lower_age,upper_age)
        else:
            gs = self 
        if not exp_ratio and not range_ratio:
            pltg.plot_eigenvalues_by_age(gs.evals, gs.ages,num_evals=num_evals,cmap = cmap)
        if exp_ratio:
            pltg.plot_eigenvalues_by_age(gs.get_explanation_ratios(), gs.ages,num_evals=num_evals,ylabel = 'explanation ratio',cmap = cmap)
        if range_ratio:
            pltg.plot_eigenvalues_by_age(gs.get_range_ratios(), gs.ages,num_evals=num_evals,ylabel = 'range ratio', cmap = cmap)
    
    def plot_dispersions_vs_age(self,ndim=3):
        self.get_dispersions(ndim)
        pltg.plot_subject_data(self.dtable['dispersion'],self.ages,ylabel='dispersion')
        
    def plot_gradient_kdes(self, gradind = 0):
        pltg.plot_kde_by_age(self.grad_arr_list()[:,:,gradind],self.ages)
    
        
    
        