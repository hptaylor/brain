#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:27:41 2023

@author: patricktaylor
"""

import sklearn.neighbors as skn
import time
import matplotlib.pyplot as plt
import numpy as np


import os 
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import io as rw
import icc 
import glob
import pandas as pd 
#logage=np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/log_age.npy')
#ages=np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/timepointlist.npy')
#samplesize=np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/numsubs.npy')

def fit_poly(x,y,deg,return_coef=False):
    coef=np.polyfit(x,y,deg)
    poly=np.poly1d(coef)
    new_x=np.linspace(np.min(x),np.max(x))
    new_y=poly(new_x)
    if return_coef:
        return new_x,new_y,coef
    else:
        return new_x,new_y

'''
def polynomial_fit_plot(metric,deg,lab=None,col=None):
    rootns=np.sqrt(samplesize)
    
    coef=np.polyfit(logage,metric,deg)
    poly=np.poly1d(coef)
    new_x=np.linspace(logage[0],logage[-1])
    new_y=poly(new_x)
    
    sigma=np.std(metric)
    
    error=[sigma/rootns[j] for j in range (len(samplesize))]
    
    ercoef=np.polyfit(logage,error,deg)
    erpoly=np.poly1d(ercoef)
    
    ey=erpoly(new_x)
    
    
    plt.plot(new_x,new_y,linewidth=2,label=lab,alpha=0.8,c=col)
    plt.fill_between(new_x,new_y-ey,new_y+ey,alpha=0.4,color=col)
    
    return 
'''        
# =============================================================================
# def add_logage_axis(yaxis=None,labs=None,legend=True):
#     
#     if yaxis is not None:
#         plt.ylabel(yaxis)
#     #plt.xlabel('age')
#     plt.xticks(logage,labels=ages,fontsize=5);
#     plt.locator_params(nbins=10)
#     if legend:
#         if labs is None:
#             plt.legend()
#         else:
#             plt.legend(labs)
#     return 
# 
# def metric_vs_age_plot(metric,deg,yaxis=None,labels=None,colors=None,legend=True,poly=True):
#     metric=np.array(metric)
#     if len(metric.shape)==1:
#         if poly:
#             polynomial_fit_plot(metric,deg)
#         else:
#             plt.plot(logage,metric,linewidth=2,label=labels,alpha=0.8,c=colors)
#         add_logage_axis(yaxis=yaxis)
#     else:
#         for i in range (metric.shape[0]):
#             if poly:
#                 if colors is None:
#                     polynomial_fit_plot(metric[i,:], deg,lab=labels[i])
#                 else:
#                     polynomial_fit_plot(metric[i,:], deg,lab=labels[i],col=colors[i])
#             else:
#                 #if conf:
#                     #plt.fill_between(,alpha=0.4,color=col)
#                 plt.plot(logage,metric[i,:],linewidth=2,label=labels[i],alpha=0.8,c=colors[i])
#                 
#         add_logage_axis(yaxis=yaxis,labs=labels,legend=legend)
#     return 
# =============================================================================
def apply_cohort_shift(data,cohort_ids,shifts):
    shifted_data = np.zeros(data.shape)
    for i in range(len(shifts)):
        inds = np.where(cohort_ids==i)
        shifted_data[inds] = data[inds] - shifts[i]
    return shifted_data 
def load_cohort_effect(path):
    cohort_effect = pd.read_csv(path)
    cohort_effect = cohort_effect.to_numpy()
    cohort_effect = cohort_effect[:,1:]
    return cohort_effect 

def load_cohort_effect_grads(g1_path):
    g2_path = g1_path.replace('g1', 'g2')
    g3_path = g1_path.replace('g1', 'g3')
    pathlist = [g1_path, g2_path, g3_path]
    cohort_effect = np.zeros((4,3,20484))
    for i,p in enumerate(pathlist):
        c = load_cohort_effect(p)
        cohort_effect[:,i] = c
    return cohort_effect

def apply_cohort_shift_grads(gmat,cohort_ids,shifts):
    shifted_data = np.zeros((gmat.shape[0],gmat.shape[1],3))
    
    for i in range(len(shifts)):
        inds = np.where(cohort_ids==i)[0]
        shifted_data[inds] = gmat[inds] - shifts[i].T
        
            
    return shifted_data

def load_gamm_fit(path,ndim=3):
    df=pd.read_csv(path)
    metric = np.zeros((400,ndim))
    for i in range(ndim):
        metric[:,i]=df[f'V{i+1}'].to_numpy()
    return metric
def load_subj_metric_from_csv(path,ndim=3):
    df=pd.read_csv(path)
    metric = np.zeros((df['v1'].size,ndim))
    for i in range(ndim):
        metric[:,i]=df[f'v{i+1}'].to_numpy()
    return metric
def scale_to_range(vec):
    min_val = np.min(vec)
    max_val = np.max(vec)

    # Normalize the vector to range [0, 1]
    normalized_vec = (vec - min_val) / (max_val - min_val)

    # Scale the normalized vector to range [-1, 1]
    scaled_vec = 2 * normalized_vec - 1

    return scaled_vec

def scale_vecs_m1_to_1(vecs):
    scaledvecs = np.zeros(vecs.shape)
    for i in range (vecs.shape[1]):
        scaledvecs[:,i] = scale_to_range(vecs[:,i])
    return scaledvecs 

def cos_norm(v1,v2):
    
    dot = np.dot(v1,v2)
    
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    return np.abs(dot/(n1*n2))

def neighbors(searchedset,queryset,num):
    '''
    computes num nearest neighbors of queryset in searchedset and returns numpy arrays size (len(queryset),num) 
    of indices of searched set and distances between neighbors
    '''
    start=time.time()
    nbrs = skn.NearestNeighbors(n_neighbors=num, algorithm='auto').fit(searchedset)
    distances, indices = nbrs.kneighbors(queryset)
    end=time.time()
    #print('neighbors time=',(end-start))
    return indices,distances

def histogram(data,yRange,numbins=1000,xrange=None,xtitle='values',ytitle='count',title='histogram',save=None):
    plt.figure()
    plt.hist(data, bins=numbins)
    plt.ylabel(ytitle)
    plt.xlabel(xtitle)
    plt.title(title)
    #plt.xlim(0,xRange)
    plt.ylim(0,yRange)
    if xrange is not None:
        plt.xlim(0,xrange)
        
    if save is not None:
        plt.savefig(save)
        
    plt.show()
    return


def hist(data, bins=100,xtitle='values',ytitle='count',title='histogram',saveout=None):
    import matplotlib
    if saveout is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as pyplot
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1,)
    n, bins, patches = ax.hist(data, bins=10, range=(np.min(data), np.max(data)), histtype='bar')
    
    #ax.set_xticklabels([n], rotation='vertical')
    
    
    
    pyplot.title(title)
    pyplot.xlabel(xtitle)
    pyplot.ylabel(ytitle)
    
    if saveout is not None:
        pyplot.savefig(saveout)
    else:
        pyplot.show()
        
    return 


def normalize_zero_to_one(garray):
        normgrads=np.zeros(garray.shape)
        for i in range (3):
            normgrads[:,i]=((garray[:,i]-np.min(garray[:,i]))
                            /(np.max(garray[:,i])-np.min(garray[:,i])))
        return normgrads
    
def get_3d_cmap(garray, a = np.array([0,1.2,0.4]),b = np.array([-0.2,-0.2,0]),
                    c = np.array([1.2,0.5,0])):
        #a = np.array([0,1.2,0.4]),b = np.array([-0.2,-0.2,0]),c = np.array([1.2,0.5,0])
                    
        a = 1.1*a
        b = 1.1*b
        c = 1.1*c
        
        normgrads = normalize_zero_to_one(garray)
        normgrads = normgrads[:,:3]
        colors = np.zeros((garray.shape[0],3))
        
        a = np.repeat(a.reshape((1,3)),garray.shape[0],axis = 0)
        b = np.repeat(b.reshape((1,3)),garray.shape[0],axis = 0)
        c = np.repeat(c.reshape((1,3)),garray.shape[0],axis = 0)
        
        abc = np.linalg.norm(np.cross(c - a, b - a), axis = 1)
        cap = np.linalg.norm(np.cross(c - normgrads, a - normgrads), axis = 1)
        abp = np.linalg.norm(np.cross(a - normgrads, b - normgrads), axis = 1)
        bcp = np.linalg.norm(np.cross(c - normgrads, b - normgrads), axis = 1)
        
        u=np.divide(cap,abc)
        v=np.divide(abp,abc)
        w=np.divide(bcp,abc)
        
        colors[:,0] = v
        colors[:,1] = u
        colors[:,2] = w
        
        return colors

def get_parcellated_cmap(parc,colors,vertex_wise = True):
    pcolors = np.zeros(colors.shape)
    colorlist = np.zeros((len(set(parc)),3))
    
    for i in np.unique(parc):
        inds = np.where(parc == i)[0]
        pcolors[inds] = np.mean(colors[inds],axis = 0)
        colorlist[int(i)] = np.mean(colors[inds],axis = 0)
    if vertex_wise:
        return pcolors
    else:
        return colorlist 


def check_symmetry(a):
    sym_err = a - a.T
    return np.all(np.abs(sym_err.data) < 1e-10)

def unmask_medial_wall(masked_feature,medial_wall_mask=None):
    if medial_wall_mask is None:
        medial_wall_mask=rw.generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_lh.desikan.downsampled.L5.func.gii','/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_rh.desikan.downsampled.L5.func.gii')
    unmasked_feature=np.zeros(len(medial_wall_mask))
    keepinds=np.where(medial_wall_mask==0)[0]
    unmasked_feature[keepinds]=masked_feature
    
    return unmasked_feature

def unmask_medial_wall_vecs(masked_vecs,medial_wall_mask=None):
    if medial_wall_mask is None:
        medial_wall_mask=rw.generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_lh.desikan.downsampled.L5.func.gii','/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_rh.desikan.downsampled.L5.func.gii')
    vecs=np.zeros((len(medial_wall_mask),len(masked_vecs[0,:])))
    for i in range (len(masked_vecs[0,:])):
        vecs[:,i]=unmask_medial_wall(masked_vecs[:,i],medial_wall_mask)
    return vecs
        

def mask_medial_wall(unmasked_feature,medial_wall_mask=None):
    if medial_wall_mask is None:
        medial_wall_mask=rw.generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_lh.desikan.downsampled.L5.func.gii','/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_rh.desikan.downsampled.L5.func.gii')
    keepinds=np.where(medial_wall_mask==0)[0]
    return unmasked_feature[keepinds]

def mask_medial_wall_vecs(unmasked_vecs,medial_wall_mask=None):
    if medial_wall_mask is None:
        medial_wall_mask=rw.generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_lh.desikan.downsampled.L5.func.gii','/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_rh.desikan.downsampled.L5.func.gii')
    keepinds=np.where(medial_wall_mask==0)[0]
    return unmasked_vecs[keepinds]

def mask_timeseries(timeseries_unmasked,medial_wall_mask):
    masked_timeseries=np.zeros((len(np.where(medial_wall_mask==0)[0]),len(timeseries_unmasked[0,:])))
    for i in range (len(timeseries_unmasked[0,:])):
        masked_timeseries[:,i]=mask_medial_wall(timeseries_unmasked[:,i],medial_wall_mask)
    return masked_timeseries

def demean_timeseries(timeseries):
    means=np.mean(timeseries,axis=1)
    demeaned_timeseries=np.zeros(np.shape(timeseries))
    for i in range (len(timeseries[0,:])):
        demeaned_timeseries[:,i]=timeseries[:,i]-means
    return demeaned_timeseries

def mask_connectivity_matrix(matrix,medial_wall_mask)  :
    keep_inds=np.where(medial_wall_mask==0)[0]
    return matrix[keep_inds][:,keep_inds]

def indx_transform(maskinds,correspondence):
    unmaskinds=np.zeros(len(maskinds))
    for i,ind in enumerate(maskinds):
        unmaskinds[i]=correspondence[ind]
    
    return unmaskinds 

def unmask_connectivity_matrix(matrix,medial_wall_mask)  :
    
    keep_inds=np.where(medial_wall_mask==0)[0]
    
    r,c,v=sparse.find(matrix)
    
    nr=keep_inds[r]
    nc=keep_inds[c]
    
    m=sparse.csr_matrix((v,(nr,nc)))
    
    return m
      
def mask_from_timeseries_zero(ts):
    
    mask = np.zeros(len(ts))
    
    zeroinds = np.where(ts[:,0]==0.0)[0]
    naninds = np.isnan(ts[:,0])
    mask[zeroinds]=1
    mask[naninds]=1
    return mask

def mask_array_from_zero(ts):
    mask=mask_from_timeseries_zero(ts)
    return mask_timeseries(ts,mask)

def stack_vecs_hor(vecs):
    
    vlist=[]
    for i in range (len(vecs)):
        vlist.append(vecs[i])
    return np.hstack(tuple(vlist))

def parc_agglo_save(sc,si,vecs=None,nclust=None,save=True,savevecs=None,lab=None,return_lab=True,surfpath='/Users/patricktaylor/Documents/lifespan_analysis/scratch/%s_clust.vtk',embpath='/Users/patricktaylor/Documents/lifespan_analysis/scratch/emb_clust.vtk'):
    if vecs is not None:
        cl=AgglomerativeClustering(n_clusters=nclust, affinity='euclidean', linkage='ward').fit(vecs)
        lab = cl.labels_
    rw.save_surface_to_hems(surfpath,sc,si,lab) 
    if save:
        if savevecs is None:
            rw.save_surface(embpath,vecs,si,lab)
        else:
            rw.save_surface(embpath,savevecs,si,lab)
    
    if return_lab:
        return lab
    else:
        return 
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

    
from sklearn import cluster as clst 

def spectral_cluster(vecs,nclust):
    clust = clst.SpectralClustering(n_clusters=nclust,degree=0.5,affinity='nearest_neighbors',n_neighbors=10,n_jobs=-1).fit(vecs)
    
    lab = clust.labels_
    
    return lab 

def cluster_within_network(grads,parc,ind,nclust):
    inds = np.where(parc==ind)[0]
    g = grads[inds]
    
    subparc = agglo(g,nclust)
    
    res = np.zeros(20484)
    res[inds] = subparc + 1
    
    return res 
    
def zscore_surface_metric(metric):
    mu=np.mean(metric)
    sig=np.std(metric)
    
    zm=(metric-mu)
    zscore=np.divide(zm,sig)
    return zscore 

def dominant_set_sparse(s, k, is_thresh=False, norm=False):
    """Compute dominant set for a sparse matrix."""
    if is_thresh:
        mask = s > k
        idx, data = np.where(mask), s[mask]
        s = sparse.coo_matrix((data, idx), shape=s.shape)

    else:  # keep top k
        nr, nc = s.shape
        idx = np.argpartition(s, nc - k, axis=1)
        col = idx[:, -k:].ravel()  # idx largest
        row = np.broadcast_to(np.arange(nr)[:, None], (nr, k)).ravel()
        data = s[row, col].ravel()
        s = sparse.coo_matrix((data, (row, col)), shape=s.shape)

    if norm:
        s.data /= s.sum(axis=1).A1[s.row]

    return s.tocsr(copy=False)

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def load_all_files_in_directory(directory,extension=None):
    
    l=[]
    for f in os.listdir(directory):
        if extension is not None:
            if f.endswith(extension):
                file=np.load(directory+f)
                l.append(file)
        else:
            file=np.load(directory+f)
            l.append(file)
    
    return l 

def load_all_files_in_directory_sparse(directory,extension='.npz',return_sum=True):
    listofobjects=[]
    for file in os.listdir(directory):
        if file.endswith(extension):
            m=sparse.load_npz(directory+file)
            listofobjects.append(m)
    sum_mat=sparse.csr_matrix(np.shape(m))
    for mat in listofobjects:
        sum_mat+=mat
    return sum_mat


    
def agglo(Vecs,nClust,ConnectivityMat=None):
    '''
    performs hierarchical agglomerative clustering on 'nvecs' laplacians matrix eigenvectors 'Vecs'. 
    -Connectivity mat is used to guide the tree computation for the clustering. The surface matrix + IHC works well. 
    -nClust controls the number of clusters outputted in the label
    -returns tuple of labels, clusters. labels is a nvertex long vector containing vertex label between 0 and nClust-1 for each vertex. 
    '''
    clusters=AgglomerativeClustering(n_clusters=nClust, affinity='euclidean', linkage='ward',connectivity=ConnectivityMat).fit(Vecs)
    labels=np.array(clusters.labels_)
    return labels

def parc_feat_to_vertex(pf,parc):
    vf = np.zeros(len(parc))
    
    u = np.unique(parc)
    
    for i in range (len(u)):
        inds=np.where(parc==u[i])[0]
        vf[inds]=pf[i]
    return vf

#def lab_vecs_to_vertex(lv,lab):
#    vf=np.zeros((len(lab),len(lv.T)))
    


def parcel_mean_vecs(v,lab):
    mv=np.zeros((len(set(lab)),len(v.T)))
    u=np.unique(lab)
    for i in range (len(u)):
        inds = np.where(lab==u[i])[0]
        mv[i] = np.mean(v[inds],axis=0)
        
    return mv

def get_distances(v):
    
    dists=np.zeros((len(v),len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            dists[i][j]=np.linalg.norm(v[i,:]-v[j,:])
    return dists    

def parc_av_spectral_dist(vecs,nclust,surf):
    lab,c=agglo(vecs,nclust,surf)
    mv=parcel_mean_vecs(vecs,lab)
    d=get_distances(mv)
    D=np.mean(d,axis=1)
    dv=parc_feat_to_vertex(D,lab)
    return dv
    
def get_ICC(measures):
    
    return icc.icc(measures,model='twoway',type='consistency',unit='single')

def sparse_memory_size(a):
    bs=a.data.nbytes + a.indptr.nbytes + a.indices.nbytes
    gbs=bs*1e-9
    print(f'matrix is {gbs} GB')
    return 

def get_all_vecs(directory,extension='.npy',discard=False):

    evlist=[]
    
    for file in glob.glob(directory+f'*.npy'):
            
        v=np.load(file)
        if discard:
            evlist.append(v[:,1:])
        else:
            evlist.append(v)
    return evlist

def gradient_magnitude_at_point(function,pointindex,indices):
    
    difs=[]
    
    for i in range(len(indices[pointindex])):
        d=np.abs(function[pointindex]-function[indices[i]])
    
        difs.append(d)
        
    return np.max(difs)


def get_gradient_around_zero(function,masksc,tol,returnmag=True):
    inds,dists=neighbors(masksc,masksc,7)
    zeros=np.where(np.abs(function)<tol)[0]
    grads=np.zeros(len(function))
    for i in range(len(zeros)):
        grads[zeros[i]]=gradient_magnitude_at_point(function,zeros[i],inds)
    if not returnmag:
        return grads
    else:
        return np.linalg.norm(grads)
    

    
def find_k_maxima_indices(vec,k):
    return np.argpartition(vec,-k)[-k:]
    
def find_k_minima_indices(vec,k):
    return np.argpartition(vec,k)[:k]
    
    
    
    
def descent_on_vector(vec,masksc):
    path=np.zeros(len(vec))
    start=np.argmax(vec)
    inds,dists=neighbors(masksc,masksc,20)
    path[start]=1
    cur=start
    curval=vec[cur]
    end=np.argmin(vec)
    for i in range (100):
        print(cur)
        adjinds=inds[cur][1:]
        adjvals=vec[adjinds]
        difs=(curval-adjvals)
        cur=adjinds[np.argmax(difs)]
        curval=vec[cur]
        
        path[cur]=1
    return path
        

def compute_gradient_at_each_vertex(f,sc):
    ind,dist=neighbors(sc,sc,7)
    
    grad=np.zeros(len(f))
    
    for i in range(len(f)):
        neighborsF=f[ind[i][1:]]
        neighborDist=dist[i][1:]
        dfdx=(f[i]-neighborsF)
        grad[i]=np.sum(dfdx)
    return grad

def second_deriv(f,sc):
    ind,dist=neighbors(sc,sc,7)
    
    d2=np.zeros(len(f))
    
    for i in range(len(f)):
        neigh=f[ind[i][1:]]
        maxN=np.max(neigh)
        minN=np.min(neigh)
        d2[i]=maxN-2*f[i]+minN
    return d2
    

def avg_steepness(f,sc):
    ind,dist=neighbors(sc,sc,100)
    
    res=np.zeros(len(f))
    
    for i in range(len(f)):
        neighborsF=f[ind[i][1:]]
        neighborDist=dist[i][1:]
        dfdx=(f[i]-neighborsF)/neighborDist
        res[i]=np.sum(np.abs(dfdx))
    return res 

def cos_norm(v1,v2):
    
    dot = np.dot(v1,v2)
    
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    return np.abs(dot/(n1*n2))

def spectral_change_in_angle(vecs,sc,nnNum):
    
    ind,dist=neighbors(sc,sc,nnNum)
    
    res=np.zeros(len(vecs))
    for i in range(len(vecs)):
        nvecs=vecs[ind[i][1:],:]
        
        cossum=0
        for j in range(nnNum-1):
            cossum+=cos_norm(vecs[i,:],nvecs[j,:])
        
        cossum=cossum/nnNum
        
        res[i]=cossum
    
    return res
        
        

def get_specific_harmonic_lifespan_dict(vecdict,harmnum):
    
    vecs=np.zeros((list(vecdict.values())[0].shape[0],len(vecdict)))
    
    i=0
    for key in vecdict:
        vecs[:,i]=vecdict[key][:,harmnum]
        i+=1
    
    return vecs
        
    
    
def sign_flip_vecs(veclist):
    if type(veclist) is list:
        flippedlist=[]
        for i in range(1,len(veclist)):
            flippedvecs=np.zeros(veclist[i].shape)
            for j in range(veclist[i].shape[1]):
                c=np.dot(veclist[i][:,j],veclist[i-1][:,j])
                if c<0:
                    flippedvecs[:,j]=-1*veclist[i][:,j]
                else:
                    flippedvecs[:,j]=veclist[i][:,j]
            flippedlist.append(flippedvecs)
    
        return flippedlist
    
    if type(veclist) is dict:
        flippedlist=veclist
        keys=list(veclist.keys())
        
        for i in range(1,len(keys)):
            
            flippedvecs=np.zeros(veclist[keys[i]].shape)
            for j in range(veclist[keys[i]].shape[1]):
                c=np.dot(flippedlist[keys[i]][:,j],flippedlist[keys[i-1]][:,j])
                if c<0:
                    flippedvecs[:,j]=-1*veclist[keys[i]][:,j]
                else:
                    flippedvecs[:,j]=veclist[keys[i]][:,j]
            
            flippedlist[keys[i]]=flippedvecs
                    
            
    
        return flippedlist
        
def flip_vecs(veclist,reference,return_indices=False):
    flippedlist=[]
    
    matching_indices=[]
    for i in range (len(veclist)):
        flippedvecs=np.zeros(veclist[i].shape)
        inds=[]
        for j in range (veclist[0].shape[1]):
            cosnorms=[]
            for k in range (veclist[0].shape[1]):
                c=cos_norm(veclist[i][:,j],reference[:,k])
                cosnorms.append(c)
            matchind=np.argmax(cosnorms)
            inds.append(matchind)
            d=np.dot(veclist[i][:,j],reference[:,matchind])
            
            if d<0:
                flippedvecs[:,j]=-1*veclist[i][:,j]
            else:
                flippedvecs[:,j]=veclist[i][:,j]
            
        flippedlist.append(flippedvecs)
        matching_indices.append(inds)
    
    if return_indices:
        return flippedlist,matching_indices
    else:
        return flippedlist
            
    
    
def color_points_embed(v1,v2):
    rgb=[]
    
    v1=v1-np.min(v1)
    v2=v2-np.min(v2)
    
    v1max=np.max(v1)
    #v1min=np.min(v1)
    
    #v1dif=v1max-v1min
    
    v2max=np.max(v2)
    #v2min=np.min(v2)
    
    #v2dif=v2max-v2min
    
    
    
    for i in range (len(v1)):
        
        r=min(v1[i]/v1max,1.0)
        
        g=max((v2[i]-v2max/2)/(v2max/2),0)
        #g=max(v2[i]/v2max-0.1,0)
        
        b=max(1-(v2[i]/(v2max/2)),0)
        
        
        rgb.append((r,g,b))
        
    return rgb
        

def embed_plot(vecs,age,save=None):
    v1=vecs[:,0]
    v2=vecs[:,1]
    
    rgb=color_points_embed(v1,v2)
    
    plt.figure()
    plt.scatter(v2,v1,c=rgb,s=0.6)
    plt.xlabel('Gradient 2')
    plt.ylabel('Gradient 1')
    plt.title(f'2D Embedding {age}')
    
    if save is not None:
        plt.savefig(save)
    return 

def embed_plot_network_colors(vecs,age,netmasks,mask,save=None):
    v1=vecs[:,0]
    v2=vecs[:,1]
    netcolorlist=['red','orange','pink','purple','blue','green']
    netnames=['DMN', 'FPC', 'SAL', 'LIM', 'VIS', 'SOM']
    
    #rgb=color_points_embed(v1,v2)
    netmasks=mask_medial_wall_vecs(netmasks,mask)
    plt.figure()
    
    for i in range (6):
        inds=np.where(netmasks[:,i]==1)[0]
        plt.scatter(v2[inds],v1[inds],c=netcolorlist[i],label=netnames[i],s=0.4)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('Gradient 2')
    plt.ylabel('Gradient 1')
    plt.title(f'{age}')
    plt.legend(markerscale=9)
    if save is not None:
        plt.savefig(save,dpi=400)
    return 

import seaborn as sns
import pandas as pd 

def embed_plot_histogram(vecs,age,parc,save=None):
    #df=pd.DataFrame(vecs[:,:2],columns=['SA','VS'])
    df=pd.DataFrame(np.vstack((vecs[:,0],vecs[:,1])).T,columns=['SA','VS'])
    #netassignments=np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/my_vertex_network_labels_ext.npy',allow_pickle=True)
    
    df['network']=parc
    
    #palette={}
    #netcolorlist=['red','orange','pink','purple','blue','green']
    #netnames=['DMN', 'FPC', 'SAL', 'LIM', 'VIS', 'SOM']
    
    #netnames=['DM','ATN','FPC','VIS','SOM']
    #netcolorlist=['red','green','yellow','purple','blue']
    #netnames=['DM','AT1','LIM','V2','AT2','SOM','FPC','V1']
    #netcolorlist=['red','green','tan','indigo','pink','steelblue','gold','purple']
    
    
    #p=sns.jointplot(data=df,x='VS',y='SA',hue='network',palette=palette,s=5,alpha=0.8)
    #p=sns.jointplot(data=df,x='VS',y='SA',hue='network',palette=plt.cm.tab10,s=5,alpha=0.8)
    p=sns.jointplot(data=df,x='VS',y='SA',s=5,alpha=0.8,scatter=False)
    p.ax_joint.scatter(X,y, c=classes)
    p.fig.suptitle(age)
    
    p.fig.tight_layout()
    #p.fig.subplots_adjust(top=0.95)
    
    #plt.suptitle(age)
    #p.ax_marg_x.set_xlim(-2.75,2.75)
    #p.ax_marg_y.set_ylim(-2.75,2.75)
    #plt.xlabel('Gradient 2')
    #plt.ylabel('Gradient 1')
    #sns.title(f'{age}')
    #plt.legend(markerscale=9)
    if save is not None:
        plt.savefig(save)
    return 


def embed_plot_all_vertices_histogram(vecs,age,save=None):
    #df=pd.DataFrame(vecs[:,:2],columns=['SA','VS'])
    df=pd.DataFrame(vecs,columns=['SA','VS'])
    #netassignments=np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/my_vertex_network_labels_ext.npy',allow_pickle=True)
    
    #df['network']=mask_medial_wall(np.array(netassignments),mask)
    
    #palette={}
    #netcolorlist=['red','orange','pink','purple','blue','green']
    #netnames=['DMN', 'FPC', 'SAL', 'LIM', 'VIS', 'SOM']
    
    #netnames=['DM','ATN','FPC','VIS','SOM']
    #netcolorlist=['red','green','yellow','purple','blue']
    #netnames=['DM','AT1','LIM','V2','AT2','SOM','FPC','V1']
    #netcolorlist=['red','green','tan','indigo','pink','steelblue','gold','purple']
    
    #for i in range (len(netnames)):
    #    palette[netnames[i]]=netcolorlist[i]
    
    #p=sns.jointplot(data=df,x='VS',y='SA',hue='network',palette=palette,s=5,alpha=0.8)
    p=sns.jointplot(data=df,x='VS',y='SA',s=5,alpha=0.8)
    p.fig.suptitle(age)
    
    p.fig.tight_layout()
    #p.fig.subplots_adjust(top=0.95)
    
    #plt.suptitle(age)
    #p.ax_marg_x.set_xlim(-2.75,2.75)
    #p.ax_marg_y.set_ylim(-2.75,2.75)
    #plt.xlabel('Gradient 2')
    #plt.ylabel('Gradient 1')
    #sns.title(f'{age}')
    #plt.legend(markerscale=9)
    if save is not None:
        plt.savefig(save)
    return 
    
def embed_plot_network_colors_histogram(vecs,age,netmasks,mask,save=None):
    df=pd.DataFrame(vecs[:,:2],columns=['SA','VS'])
    netassignments=np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/vertex_network_labels.npy',allow_pickle=True)
    df['network']=mask_medial_wall(np.array(netassignments),mask)
    
    palette={}
    netcolorlist=['red','orange','pink','purple','blue','green']
    netnames=['DMN', 'FPC', 'SAL', 'LIM', 'VIS', 'SOM']
    
    #netnames=['DM','ATN','FPC','VIS','SOM']
    #netcolorlist=['red','green','yellow','purple','blue']
    
    
    for i in range (len(netnames)):
        palette[netnames[i]]=netcolorlist[i]
    
    p=sns.jointplot(data=df,x='VS',y='SA',hue='network',palette=palette,s=5,alpha=0.8)
    p.fig.suptitle(age)
    
    p.fig.tight_layout()
    #p.fig.subplots_adjust(top=0.95)
    
    #plt.suptitle(age)
    #p.ax_marg_x.set_xlim(-2.75,2.75)
    #p.ax_marg_y.set_ylim(-2.75,2.75)
    #plt.xlabel('Gradient 2')
    #plt.ylabel('Gradient 1')
    #sns.title(f'{age}')
    #plt.legend(markerscale=9)
    if save is not None:
        plt.savefig(save)
    return 

import matplotlib.colors as cs

def embed_plot_feature_color(vecs,age,feature,save=None,colorbar=False,axes=[0,2]):
    axisnames=['SA','VS','MR']
    df=pd.DataFrame(np.vstack((vecs[:,axes[0]],vecs[:,axes[1]])).T,columns=[axisnames[axes[0]],axisnames[axes[1]]])
    df['col']=feature
    fig,ax=plt.subplots(1,1,figsize=(7,7))
    ax.set_aspect('equal')
    pallete=sns.color_palette('turbo',as_cmap=True)
    p=sns.scatterplot(data=df,x=axisnames[axes[1]],y=axisnames[axes[0]],hue='col',hue_norm=cs.Normalize(np.min(feature),np.max(feature)),palette=pallete, s=5,alpha=0.8,legend=False)
    plt.title(age)
    if colorbar:
        points = plt.scatter([], [], c=[], vmin=np.min(feature), vmax=np.max(feature), cmap=pallete)
        plt.colorbar(points)
    if save is not None:
        plt.savefig(save)
    return 

def embed_plot_color_by_gradval(vecs,age,save=None,axes=[0,1],fontsize=20,title=True):
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 36
    hfont = {'fontname':'Times New Roman'}
    axisnames=['SA','VS','MR']
    df=pd.DataFrame(np.vstack((vecs[:,axes[0]],vecs[:,axes[1]])).T,columns=[axisnames[axes[0]],axisnames[axes[1]]])
    
    fig,ax=plt.subplots(1,1,figsize=(7,7))
    ax.set_aspect('equal')
    colors=np.zeros((len(vecs),3))
    normgrads=np.zeros((len(vecs),2))
    
    for j,i in enumerate(axes):
        normgrads[:,j]=(vecs[:,i]-np.min(vecs[:,i]))/(np.max(vecs[:,i])-np.min(vecs[:,i]))
    
    if axes[1]==1:
        for i in range(len(vecs)):
            
            colors[i,0]=normgrads[i,0]
            #if normgrads[i,1]>0.5:
            colors[i,2]=max(0,normgrads[i,1]-normgrads[i,0])
            #else:
            colors[i,1]=max(0,1-normgrads[i,1]-normgrads[i,0])
    if axes[1]==2:
        for i in range(len(vecs)):
            
            colors[i,0]=max(0,normgrads[i,0]-normgrads[i,1]/2)
            #if normgrads[i,1]>0.5:
            colors[i,1]=max(0,normgrads[i,1]*normgrads[i,0])
            colors[i,2]=max(0,min(1,normgrads[i,1]-(0.5-normgrads[i,0])))
            #else:
            #colors[i,2]=max(1,1-normgrads[i,1]-normgrads[i,0])
        
    p=sns.scatterplot(data=df,x=axisnames[axes[1]],y=axisnames[axes[0]],c=colors, s=5,legend=False)
    if title:
        plt.title(f'{age}Y',fontdict={'fontsize': fontsize},**hfont)
    sns.despine()
    plt.tight_layout()
    #if colorbar:
    #    points = plt.scatter([], [], c=[], vmin=np.min(feature), vmax=np.max(feature), cmap=pallete)
    #    plt.colorbar(points)
    if save is not None:
        plt.savefig(save,format='eps')
    return 
from matplotlib import ticker

def embed_plot_color_by_gradval_row(grads,indlist,save=None,axisinds=[0,1],fontsize=20,title=True):
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 18
    hfont = {'fontname':'Times New Roman'}
    axisnames=['SA','VS','MR']
    
    
    fig,axs=plt.subplots(1,6,sharey=True,sharex=True,figsize=(63,6))
    for k,ax in enumerate(axs):
        ax.set_aspect('equal')
        M = 3
        yticks = ticker.MaxNLocator(M)

        ax.xaxis.set_major_locator(yticks)
        r=ax.spines['right']
        r.set_visible(False)
        t=ax.spines['top']
        t.set_visible(False)
        if k>0:
            l=ax.spines['left']
            l.set_visible(False)
            ax.yaxis.set_visible(False)
        vecs=grads[indlist[k]*4]
        colors=np.zeros((len(vecs),3))
        normgrads=np.zeros((len(vecs),2))
        df=pd.DataFrame(np.vstack((vecs[:,axisinds[0]],vecs[:,axisinds[1]])).T,columns=[axisnames[axisinds[0]],axisnames[axisinds[1]]])
# =============================================================================
#         for j,i in enumerate(axisinds):
#             normgrads[:,j]=(vecs[:,i]-np.min(vecs[:,i]))/(np.max(vecs[:,i])-np.min(vecs[:,i]))
#         
#         if axisinds[1]==1:
#             for i in range(len(vecs)):
#                 
#                 colors[i,0]=normgrads[i,0]
#                 #if normgrads[i,1]>0.5:
#                 colors[i,2]=max(0,normgrads[i,1]-normgrads[i,0])
#                 #else:
#                 colors[i,1]=max(0,1-normgrads[i,1]-normgrads[i,0])
#         if axisinds[1]==2:
#             for i in range(len(vecs)):
#                 
#                 colors[i,0]=max(0,normgrads[i,0]-normgrads[i,1]/2)
#                 #if normgrads[i,1]>0.5:
#                 colors[i,1]=max(0,normgrads[i,1]*normgrads[i,0])
#                 colors[i,2]=max(0,min(1,normgrads[i,1]-(0.5-normgrads[i,0])))
#                 #else:
#                 #colors[i,2]=max(1,1-normgrads[i,1]-normgrads[i,0])
# =============================================================================
        #colors=embed_colormap_3D(vecs,age=indlist[k]*4,unmask=False,op=15)
        colors = cmap3d_bary(vecs)
        sns.scatterplot(data=df,x=axisnames[axisinds[1]],y=axisnames[axisinds[0]],c=colors, s=1,legend=False,ax=ax,linewidth=0,edgecolor='black')
    plt.tight_layout()
    if save is not None:
        plt.savefig(save,format='eps')
    return 

import math as m
  
def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, m.cos(theta),-m.sin(theta)],
                   [ 0, m.sin(theta), m.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ m.cos(theta), 0, m.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-m.sin(theta), 0, m.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def embed_plot_custom_colors(vecs,ai=[0,1],op=1,age=None,rot=None,rotcol=None,a=None,b=None,c=None):
    plt.rcParams["font.family"] = "Helvetica"
    plt.rcParams["font.size"] = 18
    axisnames=['SA','VS','MR']
    colors=embed_colormap_3D(vecs,unmask=False,op=op,age=age,rot=rot,rotcol=rotcol,a=a,b=b,c=c)
    plt.figure()
    colors[colors>1]=1
    colors[colors<0]=0
    plt.scatter(vecs[:,ai[1]],vecs[:,ai[0]],s=2,c=colors)
    if age is not None:
        plt.title(f'{age}Y')
    plt.show()
    return 
    
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R

def embed_colormap_3D(grads,age=None,unmask=False,op=1,rot=None,rotcol=None,a=None,b=None,c=None):
    colors=np.zeros((len(grads),4))
    normgrads=np.zeros((len(grads),3))
    vecs=grads.copy()
    
    if rot is not None:
        for i in range (len(vecs)):
            vecs[i,:]=rot.dot(vecs[i,:].T)
    for i in range (3):
        normgrads[:,i]=(vecs[:,i]-np.min(vecs[:,i]))/(np.max(vecs[:,i])-np.min(vecs[:,i]))
    
    if op==13:
        colors=color_barycentric_vs(vecs,normgrads)
        
    if op==10:
        if a is None:
            cind=np.argmax(normgrads[:,0])
            aind=np.argmax(normgrads[:,2])
            bind=np.argmax(normgrads[:,1])
            b=normgrads[bind]
            c=normgrads[cind]
            a=normgrads[aind]
        for i in range (len(vecs)):
            colors[i,:]=space2color(normgrads[i,:],op=10,a=a,b=b,c=c)
    else:
        for i in range (len(vecs)):
            #colors[i,0]=normgrads[i,0]
            #colors[i,1]=normgrads[i,1]
            #colors[i,2]=normgrads[i,2]np.sqrt(max(0,grads[i,j,0]**2+ grads[i,j,2]**2-2*grads[i,j,1]**2))
            #t=max(0.4,min(1,np.sqrt(max(0,normgrads[i,0]**2+ normgrads[i,2]**2-2*normgrads[i,1]**2))))
            #colors[i,3]=t
            #colors[i,0]=normgrads[i,0]
            #colors[i,2]=normgrads[i,2]
            #colors[i,1]=1-(normgrads[i,0]+normgrads[i,2])/2
            
            colors[i,:]=space2color(normgrads[i,:],op=op,age=age)
            if rotcol is not None:
                colors[i,:3]=rotcol.dot(colors[i,:3].T)
        
    if op==10:
        
        #inds=np.where((colors[:,1]>colors[:,0]) & (colors[:,1]>colors[:,2]*1.3))[0]
        #inds2=np.where()[0]
        #inds=np.unique(np.concatenate((inds,inds2)))
        inds=np.arange(20484)
        for i in inds:
            h,s,v=colorsys.rgb_to_hsv(colors[i,0],colors[i,1],colors[i,2])
            v=min(1,max(0.3,max(normgrads[i,1],normgrads[i,0],normgrads[i,2])))
            r,g,b=colorsys.hsv_to_rgb(h,s,v)
            colors[i,0]=r
            colors[i,1]=g
            colors[i,2]=b
# =============================================================================
#         c=np.zeros((colors.shape[0],3))
#         vs=np.zeros(len(vecs))
#         negativeinds = np.where(vecs[:,1]<0)[0]
#         positiveinds = np.where(vecs[:,1]>0)[0]
#         
#         nvec=vecs[negativeinds,1]
#         pvec=vecs[positiveinds,1]
#         
#         vs[negativeinds]=(np.abs(nvec)-np.min(np.abs(nvec)))/(np.max(np.abs(nvec))-np.min(np.abs(nvec)))
#         
#         vs[positiveinds]=(np.abs(pvec)-np.min(np.abs(pvec)))/(np.max(np.abs(pvec))-np.min(np.abs(pvec)))
#         for i in negativeinds:
#             
#             if normgrads[i,0]<1 :
#                 colors[i,1]=colors[i,1]/1.5
# =============================================================================
# =============================================================================
#                 h,s,v=colorsys.rgb_to_hsv(colors[i,0],colors[i,1],colors[i,2])
#                 
#                 v=min(1,1-vs[i]/2)
#                 r,g,b=colorsys.hsv_to_rgb(h,s,v)
#                 
#                 colors[i,0]=r
#                 colors[i,1]=g
#                 colors[i,2]=b
# =============================================================================
    if op==15:
        colors=cmap3d_bary(grads)
        colors=np.append(colors,np.ones((colors.shape[0],1)),1)
    if not unmask:
        return colors 
    else: 
        return unmask_medial_wall_vecs(colors)

            
        #norm=plt.Normalize(0,len(vecs)-1)
        #cmap=ListedColormap(colors,name='from_list')
        #c=np.linspace(0,len(vecs)-1,len(vecs))
        

    

#def scale_zero_to_one()
import colorsys



def space2color(g,age=None,op=1,c=None,a=None,b=None):
    
    c=np.zeros(4)
    if op==1:
        #t=max(0.4,min(1,np.sqrt(max(0,g[0]**2+ g[2]**2-2*g[1]**2))))
        t=1
        c[3]=t
        c[0]=g[0]**(1.5)
        c[2]=g[2]**(1.5)
        c[1]=1-(g[0]**(1/2)+g[2]**(1/2))/2
    
    if op==2:
        t=max(0.4,(min(1,np.sqrt(max(0,g[2]**2+ g[0]**2-2*g[1]**2))))**(1/3))
        c[3]=1
        c[0]=g[2]**0.5
        c[2]=g[0]**0.5
        c[1]=1-(g[0]+g[2])/2
        #c[1]=g[1]
    if op==3:
        #t=
        t=1
        c[3]=t
        c[0]=min(1,g[0]**3+np.exp(-20*g[2]))
        c[2]=g[2]**1.3
        c[1]=1-max(0.6,(min(1,np.sqrt(max(0,g[2]**2+ g[0]**2-2*g[1]**2))))**(1/12))
    if op==4:
        #t=
        t=1
        c[3]=t
        c[0]=min(1,g[0]+np.exp(-20*g[2]))
        c[2]=g[2]**1.3
        c[1]=1-max(0.6,(min(1,np.sqrt(max(0,g[2]**2+ g[0]**2-2*g[1]**2))))**(1/12))
        
    if op==5:
        #t=
        t=1
    
        c[3]=t
        c[0]=g[0]**2
        c[2]=g[2]**2
        c[1]=1-min(1,np.sqrt(g[2]**(1/2)+g[0]**2))
    if op==6:
        #t=
        t=1
    
        c[3]=t
        c[0]=min(1,g[0]**2+g[2]/8)
        c[2]=g[2]*(1-g[1])
        c[1]=2*np.abs(0.5-g[1])
    if op==7:
        #t=
        t=1
    
        c[3]=t
        c[0]=g[0]*g[2]
        c[2]=np.abs(g[0]-g[2])
        c[1]=2*np.abs(0.5-g[1])
    if op==8:
        #t=
        t=1
        c[0]=1-g[0]*g[2]
        c[1]=g[0]
        c[2]=max(g[2],g[0]**0.5)
        
        c[0],c[1],c[2]=colorsys.hsv_to_rgb(c[0],c[1],c[2])
        
        c[3]=t
        #c[0]=g[0]*g[2]
        #c[2]=np.abs(g[0]-g[2])
        #c[1]=2*np.abs(0.5-g[1])
    if op==9:
        #r = R.from_rotvec(np.pi/8 * np.array([0, 0, 1])).as_matrix()
        #g=r.dot(g.T)
        c[0]=g[0]
        c[1]=2*np.abs(0.5-g[1]**0.5)
        c[2]=g[2]**0.5
        c[3]=1
    if op==10:
        c=color_barycentric(g)
        
        #c[:3]=r.dot(c[:3].T)
        #c[0]=min(1,max(0,c[0]))
        #c[1]=min(1,max(0,c[1]))
        #c[2]=min(1,max(0,c[2]))
    if op==11:
        c=color_glasser(g)
    if op==12:
        c=color_glasser_hsv(g)
        
    if op==14:
        c[:3]=g
        c[3]=1
        
    
        
    
        
    return c

def color_barycentric(g,c=np.array([1,0,0]),a=np.array([0.7,0,0.95]),b=np.array([0,0,0.34])):
    
    p=[g[0],g[1],g[2]]
    abc=np.linalg.norm(np.cross(c-a,b-a))
    cap=np.linalg.norm(np.cross(c-p,a-p))
    abp=np.linalg.norm(np.cross(a-p,b-p))
    bcp=np.linalg.norm(np.cross(c-p,b-p))
    
    u=cap/abc
    v=abp/abc
    w=bcp/abc
    c=np.zeros(4)
    
    c1=np.array([0,0,1])
    c1=c1/np.linalg.norm(c1)
    c2=np.array([1,0,0])
    c2=c2/np.linalg.norm(c2)
    c3=np.array([0,1,0.2])
    c3=c3/np.linalg.norm(c3)
    c[:3]=c1*w+c2*v+c3*u
    c[:3]=c[:3]/np.linalg.norm(c[:3])
    #c[0]=min(1,v)
    #c[1]=min(1,u)
    #c[2]=min(1,w)
    c[3]=1
    return c 

def color_barycentric_vs(vec):
    colors=np.zeros((len(vec),4))
    normgrads=np.zeros(vec.shape)
    for i in range (3):
        normgrads[:,i]=(vec[:,i]-np.min(vec[:,i]))/(np.max(vec[:,i])-np.min(vec[:,i]))
    vs=np.zeros(len(vec))
    negativeinds = np.where(vec[:,1]<0)[0]
    positiveinds = np.where(vec[:,1]>0)[0]
    
    nvec=vec[negativeinds,1]
    pvec=vec[positiveinds,1]
    
    vs[negativeinds]=(np.abs(nvec)-np.min(np.abs(nvec)))/(np.max(np.abs(nvec))-np.min(np.abs(nvec)))
    
    vs[positiveinds]=(np.abs(pvec)-np.min(np.abs(pvec)))/(np.max(np.abs(pvec))-np.min(np.abs(pvec)))
    
    for i in range (len(vec)):
        c=color_barycentric(normgrads[i])
        
        colors[i,:]=c
    
    g2w=white_to_colormap(vec,cind=1)
    
    for i in range (len(vec)):
        c=g2w[i]
        if c[0]<0.3 and c[2]<0.3:
            colors[i,:3]=c
    return colors
    
def color_glasser(g):
    
    c=np.zeros(4)
    
    if g[0]>0.5 or g[2]>0.5:
        
        if g[0]>g[2]:
            bw = 1-g[0]
            c[:3]=bw
        else:
            bw = g[2]
            c[:3]=bw
        
    else: 
        if g[1]<0.5:
            c[1]=1-g[1]

        else:
            c[2]=g[1]
        
        
    c[3]=1
    return c

def color_glasser_hsv(g):
    #hue green=0.33
    #hue blue =0.66
    
    #sat 0 --> white
    #val 0 --> black
    c=np.zeros(4)
    c[3]=1
    
    h=0.33 + 0.33*g[1]
    
    s = max(0, (1-np.sqrt(g[0]**3+g[2]**3)))

    v = max(0, max(g[2], np.abs( 0.5-g[1])*2) - g[0]**4/4)
    
    c[0],c[1],c[2]=colorsys.hsv_to_rgb(h,s,v)
    
    return c

def barycentric(g,a=np.array([0,1,0.5]),b=np.array([-0.2,-0.2,0]),c=np.array([1.2,0.7,0])):
    #c=np.array([1,0.7,0.35])np.array([0,1,0.4])np.array([1,1,0.5])
    p=[g[0],g[1],g[2]]
    abc=np.linalg.norm(np.cross(c-a,b-a))
    cap=np.linalg.norm(np.cross(c-p,a-p))
    abp=np.linalg.norm(np.cross(a-p,b-p))
    bcp=np.linalg.norm(np.cross(c-p,b-p))
    
    u=cap/abc
    v=abp/abc
    w=bcp/abc
    cc=np.zeros(3)
    
    c1=np.array([0,0,1])
    c1=c1/np.linalg.norm(c1)
    c2=np.array([1,0,0])
    c2=c2/np.linalg.norm(c2)
    c3=np.array([0,1,0])
    c3=c3
    cc=c1*w+c2*v+c3*u
    cc=cc
    return cc 

def cmap3d_bary(grads):
    point=np.array([1,0.7,0.35])
    colors=np.zeros(grads.shape)
    ng=norm_vecs(grads)
    for i in range (len(grads)):
        colors[i]=barycentric(ng[i])
        #h,s,v=colorsys.rgb_to_hsv(colors[i,0],colors[i,1],colors[i,2])
        #v=min(1,np.linalg.norm(ng[i]-point))
        #s=1-ng[i,0]
        #r,g,b=colorsys.hsv_to_rgb(h,s,v)
        #r,g,b=colorsys.hsv_to_rgb(colors[i,0],colors[i,1],colors[i,2])
        #colors[i]=np.array([r,g,b])
    return norm_vecs(colors)
        
    
    
def test_color_scheme(grads,age,op,rot=None,rotcol=None,a=None,b=None,c=None):
    from io import plot_custom_colormap
    plot_custom_colormap(grads,age=age,op=op,rot=rot,a=a,b=b,c=c)
    embed_plot_custom_colors(grads[:10242],ai=[0,1],op=op,age=age,rot=rot,rotcol=rotcol,a=a,b=b,c=c)
    embed_plot_custom_colors(grads[:10242],ai=[0,2],op=op,age=age,rot=rot,rotcol=rotcol,a=a,b=b,c=c)
    
def procrustes(source, target, center=False, scale=False):
    """Align `source` to `target` using procrustes analysis.
    Parameters
    ----------
    source : 2D ndarray, shape = (n_samples, n_feat)
        Source dataset.
    target : 2D ndarray, shape = (n_samples, n_feat)
        Target dataset.
    center : bool, optional
        Center data before alignment. Default is False.
    scale : bool, optional
        Remove scale before alignment. Default is False.
    Returns
    -------
    aligned : 2D ndarray, shape = (n_samples, n_feat)
        Source dataset aligned to target dataset.
    """

    # Translate to origin
    if center:
        ms = source.mean(axis=0)
        mt = target.mean(axis=0)

        source = source - ms
        target = target - mt

    # Remove scale
    if scale:
        ns = np.linalg.norm(source)
        nt = np.linalg.norm(target)
        source /= ns
        target /= nt

    # orthogonal transformation: rotation + reflection
    u, w, vt = np.linalg.svd(target.T.dot(source).T)
    
    
    t = u.dot(vt)
    
    #print(np.shape(t))
    # Recover target scale
    if scale:
        t *= w.sum() * nt

    aligned = source.dot(t)
    if center:
        aligned += mt
    return aligned

def procrustes_alignment(data, reference=None, n_iter=10, tol=1e-5,
                         return_reference=False, verbose=False):
    """Iterative alignment using generalized procrustes analysis.
    Parameters
    ----------
    data :  list of ndarray, shape = (n_samples, n_feat)
        List of datasets to align.
    reference : ndarray, shape = (n_samples, n_feat), optional
        Dataset to use as reference in the first iteration. If None, the first
        dataset in `data` is used as reference. Default is None.
    n_iter : int, optional
        Number of iterations. Default is 10.
    tol : float, optional
        Tolerance for stopping criteria. Default is 1e-5.
    return_reference : bool, optional
        Whether to return the reference dataset built in the last iteration.
        Default is False.
    verbose : bool, optional
        Verbosity. Default is False.
    Returns
    -------
    aligned : list of ndarray, shape = (n_samples, n_feat)
        Aligned datsets.
    mean_dataset : ndarray, shape = (n_samples, n_feat)
        Reference dataset built in the last iteration. Only if
        ``return_reference == True``.
    """

    if n_iter <= 0:
        raise ValueError('A positive number of iterations is required.')

    if reference is None:
        # Use the first item to build the initial reference
        aligned = [data[0]] + [procrustes(d, data[0]) for d in data[1:]]
        reference = np.mean(aligned, axis=0)
    else:
        aligned = [None] * len(data)
        reference = reference.copy()

    dist = np.inf
    for i in range(n_iter):
        # Align to reference
        aligned = [procrustes(d, reference) for d in data]

        # Compute new mean
        new_reference = np.mean(aligned, axis=0)

        # Compute distance
        new_dist = np.square(reference - new_reference).sum()

        # Update reference
        reference = new_reference

        if verbose:
            print('Iteration {0:>3}: {1:.6f}'.format(i, new_dist))

        if dist != np.inf and np.abs(new_dist - dist) < tol:
            break

        dist = new_dist

    return (aligned, reference) if return_reference else aligned


def color_by_network(colors,networkmask,netcolor):
    
    inds=np.where(networkmask==1)[0]
    
    #colors[inds]=netcolor
    for i in range (len(inds)):
        colors[inds[i]]=netcolor
    return colors

    
from scipy.interpolate import make_interp_spline

def interp_pts(x,y,num=50):
    X_Y_Spline = make_interp_spline(x, y)
 
    # Returns evenly spaced numbers
# over a specified interval.
    X_ = np.linspace(x.min(), x.max(), num)
    Y_ = X_Y_Spline(X_)
    return X_,Y_

def load_ind_gradients(directory):
    allsubnames=np.load('/Users/patricktaylor/Documents/lifespan_analysis/individual/5p_fwhm2/allsub_names.npy')
    allsubages=np.load('/Users/patricktaylor/Documents/lifespan_analysis/individual/5p_fwhm2/allsub_ages.npy')
    agedict={}
    for i in range(len(allsubnames)):
        agedict[allsubnames[i]]=allsubages[i]
    gradlist=[]
    ages=[]
    names=[]
    for f in os.listdir(directory):
        if f.endswith('grads.npy'):
            g=np.load(directory+f)
            g=np.nan_to_num(g)
            if np.sum(np.abs(g[:100,0]))>0:
                gradlist.append(g)
            
                if f.startswith('MNBCP') or f.startswith('NCBCP'): 
                    a=float(f[12:-10])/365
                    ages.append(a)
                    names.append(f[:11])
                else:
                    
                    ages.append(agedict[f[:-10]])
                    if f.startswith('H'):
                        names.append(f[:10])
                    else:
                        names.append(f[:6])
        
    return gradlist, ages,names
            
def gauss_av_weight(x,mu,sig):
    weight=np.exp(-1*((x-mu)**2)/2/sig**2)
    return weight

#import io as rw

#bcpmask=rw.generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_lh.desikan.downsampled.L5.func.gii','/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_rh.desikan.downsampled.L5.func.gii')
bcpmask =0 

def agglo_by_hem(vecs,nclust,mask=bcpmask):
    parc=np.zeros(len(vecs))
    
    vecs=unmask_medial_wall_vecs(vecs,mask)
    
    h=int(len(vecs)/2)
    
    lm=mask[:h]
    rm=mask[h:]
    
    ms=[lm,rm]
    
    A=sparse.load_npz('/Users/patricktaylor/Documents/lifespan_analysis/misc/surface_distance_matrix.npz')
    
    #A=unmask_connectivity_matrix(A,mask)
    
    lh=vecs[:h]
    rh=vecs[h:]
    
    As=[A[:9281,:9281],A[9281:,9281:]]
    
    labs=[]
    
    for i,hem in enumerate([lh,rh]):
        
        clusters=AgglomerativeClustering(n_clusters=nclust, linkage='ward',connectivity=As[i]).fit(mask_medial_wall_vecs(hem,ms[i]))

        la=np.array(clusters.labels_)+1
        
        labs.append(la)
    
    parc[:len(labs[0])]=labs[0]
    parc[len(labs[0]):]=labs[1]+np.max(labs[0])
    
    return parc
        
def compare_embed_physical_dist(vecs,sc,num=50):
    
    surfinds,surfdists=neighbors(sc,sc,num)
    
    embedinds,embeddists=neighbors(vecs,vecs,num)
    
    overlap=np.zeros(len(vecs))
    
    for i in range (len(vecs)):
        
        s=set(surfinds[i][1:])
        e=set(embedinds[i][1:])
        
        intersect=s.intersection(e)
        
        overlap[i]=len(intersect)/num
        
    return overlap
    
def embed_local_density(vecs,num=50):
    
    embedinds,embeddists=neighbors(vecs,vecs,num)
    
    density=np.zeros(len(vecs))
    
    for i in range (len(vecs)):
        density[i]=np.sum(embeddists[i])/num
    return density 
    
def embed_distance_ratio(vecs,n1,n2):
    
    embedinds,embeddists=neighbors(vecs,vecs,n2)
    
    density=np.zeros(len(vecs))
    
    for i in range (len(vecs)):
        density[i]=np.sum(embeddists[i][:n1])/np.sum(embeddists[i][n1:])
        
    return density 
    

def netmasks_from_parc(parc):
    u=np.unique(parc)
    netmasks=np.zeros((len(parc),len(u)))
    for i in range (len(u)):
        inds=np.where(parc==i)[0]
        
        netmasks[inds,i]=1
    return netmasks

def parc_from_netmasks(masks):
    parc=np.zeros(len(masks))
    for i in range(masks.shape[1]):
        inds=np.where(masks[:,i]==1)[0]
        parc[inds]=i
    return parc 

from scipy.spatial import distance

def dice_reorder(p1,p2,return_indices=True):
    m1=netmasks_from_parc(p1)
    m2=netmasks_from_parc(p2)
    u1=np.unique(p1)
    #u2=np.unique(p2)
    
    #mat=np.zeros((len(u1),len(u2)))
    
    d1s=[]
    nm=np.zeros(m2.shape)
    for i in range(len(u1)):
        dices=[]
        used=[]
        
        for j in range(len(u1)):
            d=distance.dice(m1[:,i],m2[:,j])
            dices.append(d)
        a=np.argmax(dices)
        
        
        nm[:,i]=m1[:,a]
    return nm

from sklearn.metrics.cluster import adjusted_rand_score
# =============================================================================
# def rand_matrix(target,reference):
#     mtarg=netmasks_from_parc(target)
#     mref=netmasks_from_parc(reference)
#     
#     mre=np.zeros(mtarg.shape)
#     
#     u=np.unique(target)
#     
#     ur= np.unique(reference)
#     rands=np.zeros((len(u),len(ur)))
#     
#     for i in range (len(u)):
#         for j in range (len(ur)):
#             rands[i][j]=adjusted_rand_score(mtarg[:,i],mref[:,j])
#     return rands
# =============================================================================

def rand_matrix(template,lab,return_reordered=False):
    ut= np.unique(template)
    ul = np.unique(lab)
    mat = np.zeros((len(ut),len(ul)))
    
    tnetmask = netmasks_from_parc(template)
    lnetmask = netmasks_from_parc(lab)
    
    for i in range (len(ut)):
        for j in range (len(ul)):
            mat[i][j] = adjusted_rand_score(tnetmask[:,i], lnetmask[:,j])
            
    if return_reordered:
        rlab = np.zeros(len(lab))
        
        for j in range (len(ul)):
            
            new = np.argmax(mat[:,j])
            
            rlab[lab==j] = new 
            
        return rlab
    
    else:
        return mat 
    
def rand_reorder(target,reference):
    mtarg=netmasks_from_parc(target)
    mref=netmasks_from_parc(reference)
    
    mre=np.zeros(mtarg.shape)
    
    u=np.unique(target)
    
    rands=np.zeros((len(u),len(u)))
    
    for i in range (len(u)):
        for j in range (len(u)):
            rands[i][j]=adjusted_rand_score(mtarg[:,i],mref[:,j])
    
    refmaxinds=np.zeros(len(u))-1
    used=[]
    for i in range (len(u)):
        ind=np.argmax(rands[:,i])
        if ind not in used:
            mre[:,i]=mtarg[:,ind]
        else:
            ind=np.argsort(rands[:,i])[-2]
            if ind not in used:
                mre[:,i]=mtarg[:,ind]
            else:
                ind=np.argsort(rands[:,i])[-3]
                mre[:,i]=mtarg[:,ind]
        used.append(ind)
    
    return parc_from_netmasks(mre)
        
        
    
from sklearn.metrics.cluster import contingency_matrix

import munkres

def translateLabels(masterList, listToConvert):    
  #contMatrix = contingency_matrix(masterList, listToConvert)
  contMatrix = rand_matrix(masterList, listToConvert)
  labelMatcher = munkres.Munkres()
  labelTranlater = labelMatcher.compute(contMatrix.max() - contMatrix)

  uniqueLabels1 = list(set(masterList))
  uniqueLabels2 = list(set(listToConvert))

  tranlatorDict = {}
  for thisPair in labelTranlater:
    tranlatorDict[uniqueLabels2[thisPair[1]]] = uniqueLabels1[thisPair[0]]

  return [tranlatorDict[label] for label in listToConvert]

def relabel_to_max_rand(temp,move):
    randmat= rand_matrix(temp, move)
    
    relabel = np.zeros(len(move))
    
    for i in range(randmat.shape[0]):
        
        inds= np.where(move==i)[0]
        relabel[inds]= np.argmax(randmat[i])
        
    return relabel
    
def get_grad_mask_by_zscore(grad,zlow,zhigh):
    zgrad=zscore_surface_metric(grad)
    
    mask=np.zeros(len(grad))
    for i in range (len(grad)):
        if zlow<zgrad[i]<zhigh:
            mask[i]=zgrad[i]
    return mask
    
def plot_n_largest(grad,num):
    inds=np.argsort(grad)
    
    a=np.zeros(18463)
    a[inds[-num:]]=grad[inds[-num:]]
    #rw.plot_surf(a,lh)
    return 
    

def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]

def make_path_feature(Pr,i,j,pathvec=None):
    if pathvec is None:
        pathvec=np.zeros(20484)
    pathinds=get_path(Pr,i,j)
    for j,i in enumerate(pathinds):
        pathvec[i]=j
    return pathvec

def get_path_distance(vecs,path):
    dist = 0
    for i in range (1,len(path)):
        d = np.linalg.norm(vecs[i]-vecs[i-1])
        dist += d
    return dist 

def get_parc_velocity(parc,grads):
    velocities = np.zeros((400,np.max(parc),3))
    for j in range (1,grads.shape[0]):
        for i in range (np.max(parc)):
            v = grads[j,parc==i] - grads[j-1,parc == i]
            velocities[j,i] = np.mean(v,axis = 0)
    return velocities 

def plot_gamm_fit(grads_gamm,subgrads,error,ages,ind,gradind=0):
    fig, ax = plt.subplots()
    ax.scatter(ages,subgrads[:,ind,gradind],s=0.5,color='black')
    ax.plot(np.arange(400)/4,grads_gamm[:,ind,gradind],color='red')
    ax.fill_between(np.arange(400)/4, (grads_gamm[:,ind,gradind]-2*error[:,ind,gradind]), (grads_gamm[:,ind,gradind]+2*error[:,ind,gradind]), color='b', alpha=.1)
    return
    
    
    
def get_grad_range_and_vars(grads):
    gradranges=np.zeros((400,3))
    gradvars=np.zeros((400,3))
    for i in range (400):
        for j in range (3):
            gradranges[i,j]=np.max(grads[i,:,j])-np.min(grads[i,:,j])
            gradvars[i,j]=np.var(grads[i,:,j])
    return gradranges,gradvars
        
def zscore_gradients(grads_gamm):
    grads_zscore=np.zeros(grads_gamm.shape)
    for i in range (len(grads_gamm)):
        for j in range (3):
            grads_zscore[i,:,j]=zscore_surface_metric(grads_gamm[i,:,j])
    return grads_zscore


import matplotlib as mpl 
def plot_pdfs(grads,ind,zscore=True,save=None):
    if zscore:
        axisnames=['SA z-score','VS z-score','MR z-score']
    else:
        axisnames=['SA','VS','MR']
        
    tps=np.arange(100)*4
    cm=sns.color_palette('rainbow', n_colors=len(tps))
    cmap = plt.get_cmap('rainbow', len(tps))
    norm=mpl.colors.Normalize(vmin=tps[0],vmax=tps[-1]/4)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    #ccm=sns.color_palette('icefire', n_colors=len(tps),as_cmap=True)
    #fig, ax = plt.subplots()
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.figure(figsize=(8,8))
    for k in range (len(tps)):
        i=tps[k]
        index=int(i*4)
        sns.kdeplot(grads[i,:,ind],cumulative=False,label=f'{index}Y',color=cm[k],cbar=True)
    plt.colorbar(sm)
    plt.xlabel(axisnames[ind])
    if save is not None:
        plt.savefig(save)
    return
    
def load_gamm_gradients_from_dataframe(p1,p2,p3,nvert=20484):
    gradients=np.zeros((400,nvert,3))
    for i,p in enumerate([p1,p2,p3]):
        df=pd.read_csv(p)
        for j in range (nvert):
            gradients[:,j,i]=df[f'V{j+2}']
    return gradients
    
def load_standard_error_from_dataframe(p1,p2,p3,nvert=20484):
    std_err=np.zeros((400,nvert,3))
    for i,p in enumerate([p1,p2,p3]):
        df=pd.read_csv(p)
        for j in range (nvert):
            std_err[:,j,i]=df[f'V{j+2}']
    return std_err

def load_r_squared_from_dataframe(p1,p2,p3,nvert=20484):
    rsq=np.zeros((nvert,3))
    for i,p in enumerate([p1,p2,p3]):
        df=pd.read_csv(p)
        rsq[:,i]=df['x'][1:]
    return rsq

from matplotlib.ticker import ScalarFormatter

from matplotlib import ticker

def log_scale_plot_vs_age(metric,labels=None):
    if type(metric) is list:
        metric=np.array(metric)
    fig, ax = plt.subplots()
    if len(metric.shape)==1:
        ax.plot(np.arange(400)/4, metric)
    else:
        for i in range (metric.shape[1]):
            ax
    #ax.set_xticks([0,0.5,1,2,5,12,20,40,80])
    #ax.set_xticks([0,0.5,1,2,5,12,20,40,80])
    ax.set_xscale('log')
    locs = np.array([0,0.5,1,2,5,12,20,40,80])
    ax.xaxis.set_minor_locator(ticker.FixedLocator(locs.astype('int64')))
    ax.xaxis.set_major_locator(ticker.NullLocator())

    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    return 



def white_to_colormap(vec,cind=0):
    
# =============================================================================
#     if cind!=1:
#         vmax=np.max(vec)
#         vmin=np.min(vec)
#         
#         v=(vec-vmin)/(vmax-vmin)
#         
#         colors=np.zeros((len(vec),3))+1
#     
#     
#         for i in range (len(v)):
#             
#             for j in range (3):
#                 if j!=cind:
#                     colors[i,j]-=v[i]
#                     
#         return np.power(colors,1/4)
# =============================================================================

    #else:
        #vmax=np.max(np.abs(vec[:,1]))
        #vmin=np.min(np.abs(vec[:,1]))
        
        #v=(np.abs(vec[:,1])-vmin)/(vmax-vmin)
    
    colors= embed_colormap_3D(vec,25,unmask=False,op=10)[:,:3]
    
    
    if cind==1:
        inds=np.where((colors[:,1]<colors[:,0]) | (colors[:,1]<colors[:,2]))[0]
        
        normgrads=np.zeros(vec.shape)
         
        vs=np.zeros(len(vec))
        negativeinds = np.where(vec[:,1]<0)[0]
        positiveinds = np.where(vec[:,1]>0)[0]
         
        nvec=vec[negativeinds,1]
        pvec=vec[positiveinds,1]
        vs[negativeinds]=(np.abs(nvec)-np.min(np.abs(nvec)))/(np.max(np.abs(nvec))-np.min(np.abs(nvec)))
         
        vs[positiveinds]=(np.abs(pvec)-np.min(np.abs(pvec)))/(np.max(np.abs(pvec))-np.min(np.abs(pvec)))
        for i in range (3):
            normgrads[:,i]=(vec[:,i]-np.min(vec[:,i]))/(np.max(vec[:,i])-np.min(vec[:,i]))
            
        for i in positiveinds:
            colors[i,0]=min(1,1-np.power(vs[i],1/2)+normgrads[i,0]/2+normgrads[i,2]/2)
            colors[i,1]=1
            colors[i,2]=min(1,1-np.power(vs[i],1/2)+normgrads[i,0]/2+normgrads[i,2]/2)
        for i in negativeinds:
            colors[i,0]=colors[i,2]=min(1,1-np.power(vs[i],1/2)+normgrads[i,0]/2+normgrads[i,2]/2)
            colors[i,1]=1
            if colors[i,1]>colors[i,0]:
                h,s,v=colorsys.rgb_to_hsv(colors[i,0],colors[i,1],colors[i,2])
                v=min(1,max(0.4,1-vs[i]+normgrads[i,0]/2+normgrads[i,2]/3))
                r,g,b=colorsys.hsv_to_rgb(h,s,v)
                colors[i,0]=r
                colors[i,1]=g
                colors[i,2]=b
    
# =============================================================================
#     normgrads=np.zeros(vec.shape)
#     
#     vs=np.zeros(len(vec))
#     negativeinds = np.where(vec[:,1]<0)[0]
#     positiveinds = np.where(vec[:,1]>0)[0]
#     
#     nvec=vec[negativeinds,1]
#     pvec=vec[positiveinds,1]
#     
#     
#     vs[negativeinds]=(np.abs(nvec)-np.min(np.abs(nvec)))/(np.max(np.abs(nvec))-np.min(np.abs(nvec)))
#     
#     vs[positiveinds]=(np.abs(pvec)-np.min(np.abs(pvec)))/(np.max(np.abs(pvec))-np.min(np.abs(pvec)))
#     for i in range (3):
#         normgrads[:,i]=(vec[:,i]-np.min(vec[:,i]))/(np.max(vec[:,i])-np.min(vec[:,i]))
#     if cind==1:
# # =============================================================================
# #         for i in range (len(vec)):
# #             r=colors[i,0]
# #             colors[i,0]=1-np.power(colors[i,1],2)
# #             colors[i,2]=1-np.power(colors[i,1],2)
# #         colors[:,1]=1
# # =============================================================================
#         colors=np.zeros(colors.shape)
#         #white=np.array([1,1,1])
#         #green=np.array([0,1,0])
#         for i in positiveinds:
#             colors[i,0]=min(1,1-np.power(vs[i],1/2)+normgrads[i,0]/2+normgrads[i,2]/2)
#             colors[i,1]=1
#             colors[i,2]=min(1,1-np.power(vs[i],1/2)+normgrads[i,0]/2+normgrads[i,2]/2)
#         for i in negativeinds:
#             #colors[i,0]=min(1,1-np.power(vs[i],1/2)+normgrads[i,0]/2+normgrads[i,2]/2)
#             #colors[i,2]=min(1,1-np.power(vs[i],1/2)+normgrads[i,0]/2+normgrads[i,2]/2)
#             
#             i#f colors[i,0]>0.99:
#              #   colors[i,1]=1
#             
#             #else:
#             #    colors[i,1]=1-0.4*vs[i]
#             #if vs[i]>0.1:
#             #    colors[i,1]=1-np.power(0.4*vs[i],1/2)
#             #else:
#             #    colors[i,1]=1
#             colors[i,0]=colors[i,2]=1-vs[i]
#             colors[i,1]=1-0.6*vs[i]
# =============================================================================
            
            
# =============================================================================
#         negc=colors[negativeinds]
#         for i in range (len(negc)):
#             if (negc[i,0]>0.67) and (negc[i,1]<negc[i,0]) and (negc[i,1]<negc[i,2]): 
#                 negc[i,1]=min(1,negc[i,0]*1.2)
#             #if (negc[i,1]==negc[i,0]) and negc[i,1]>0.83:
#             #    negc[i,1]=1
#         colors[negativeinds]=negc
# =============================================================================
# =============================================================================
#         negc=colors[negativeinds]
#         ind=np.where(negc[:,0]<0.6)
#         negc[ind,1]=0.6
#         for i in range (len(negc)):
#             if negc[i,1]<0.2:
#                 if negc[i,0]>negc[i,1]:
#                     negc[i,0]=negc[i,1]
#                 if negc[i,2]>negc[i,1]:
#                     negc[i,2]=negc[i,1]
#         colors[negativeinds]=negc
# =============================================================================
    if cind==0:
        for i in range (len(vec)):
            colors[i,1]=1-np.power(colors[i,0],2)
            colors[i,2]=1-np.power(colors[i,0],2)
        colors[:,0]=1
    if cind==2:
        for i in range (len(vec)):
            colors[i,0]=1-np.power(colors[i,2],2)
            colors[i,1]=1-np.power(colors[i,2],2)
        colors[:,2]=1
        #colors[:,0]=1
        #colors[:,2]=1
        #a=np.where(colors[:,1]<0.95)[0]
        #colors[a,1]=0.95
    #else:
    #    colors=colors*0.9
        #for i in range (len(v)):
        #    for j in range (3):
        #        if j!=cind:
        #            colors[i,j]=max(0,colors[i,j]-v[i])
        
    return colors
    
def black_to_colormap(vec,cind=0):
    

    colors = np.zeros((len(vec),3))
    normvec = norm_vecs(vec)
    
    if cind==0:
        colors[:,0] = normvec[:,0]
    if cind==1:
        ninds = np.where(vec[:,1]<0)[0]
        pinds = np.where(vec[:,1]>0)[0]
        
        nvp = norm_vecs(vec[pinds])
        nvn = norm_vecs(-vec[ninds])
        
        colors[pinds,2] = np.power(nvp[:,1],1/2)
        colors[ninds,1] = nvn[:,1]
        
    if cind==2:
        
        colors[:,0] = colors[:,1] = colors[:,2] = normvec[:,2]
        
    return colors

def norm_vecs(vec):
    normgrads=np.zeros(vec.shape)
    for i in range (3):
        normgrads[:,i]=(vec[:,i]-np.min(vec[:,i]))/(np.max(vec[:,i])-np.min(vec[:,i]))
    return normgrads

def colormap_embed(grads,cind=None):
    
    colors=np.zeros(grads.shape)
    
    #for i in range 
        
    return colors
                
        
def cb(g,c=np.array([1,0,0]),a=np.array([0.7,0,0.95]),b=np.array([0,0,0.34])):
    
    p=[g[0],0,g[2]]
    abc=np.linalg.norm(np.cross(c-a,b-a))
    cap=np.linalg.norm(np.cross(c-p,a-p))
    abp=np.linalg.norm(np.cross(a-p,b-p))
    bcp=np.linalg.norm(np.cross(c-p,b-p))
    
    u=cap/abc
    v=abp/abc
    w=bcp/abc
    c=np.zeros(4)
    
    c1=np.array([0,0,1])
    c1=c1/np.linalg.norm(c1)
    c2=np.array([1,0,0])
    c2=c2/np.linalg.norm(c2)
    c3=np.array([0,1,0.2])
    c3=c3/np.linalg.norm(c3)
    c[:3]=c1*w+c2*v+c3*u
    c[:3]=c[:3]/np.linalg.norm(c[:3])
        
    return c

import decomp as dct 

def cluster_by_temporal_change(grads,nclust=10,save='/Users/patricktaylor/Documents/lifespan_analysis/scratch/'+'%s_tempclust.vtk',sc=None,si=None):
    dif = grads[:,:,:]#-np.mean(grads[:,:,:],axis=0)
    sdif = np.vstack((dif[:,:,0],dif[:,:,1],dif[:,:,2]))
    #clust=AgglomerativeClustering(n_clusters=nclust).fit(dif.T)
    clust=KMeans(n_clusters=nclust).fit(sdif.T)
    rw.save_surface_to_hems(save, sc, si, clust.labels_)
# =============================================================================
#     for i in range (nclust):
#         plt.figure()
#         plt.plot(np.mean(grads[:,clust.labels_==i],1))
#         plt.title(i)
#         plt.show()
# =============================================================================
    for i in range (400):
        rw.save_eigenvector(f'/Users/patricktaylor/Documents/lifespan_analysis/scratch/timeseries/{i}_clust.vtk',grads[i],si,clust.labels_)
        
    return clust.labels_
        
        

from sklearn.metrics.cluster import contingency_matrix

def align_cluster_index(ref_cluster, map_cluster):
    """
    remap cluster index according the the ref_cluster.
    both inputs must be nparray and have same number of unique cluster index values.
    
    Xin Niu Jan-15-2020
    """
    
    ref_values = np.unique(ref_cluster)
    map_values = np.unique(map_cluster)
    
    #print(ref_values)
    #print(map_values)
    
    num_values = ref_values.shape[0]
    
    if ref_values.shape[0]!=map_values.shape[0]:
        print('error: both inputs must have same number of unique cluster index values.')
        return()
    
    switched_col = set()
    while True:
        cont_mat = contingency_matrix(ref_cluster, map_cluster)
        #print(cont_mat)
        # divide contingency_matrix by its row and col sums to avoid potential duplicated values:
        col_sum = np.matmul(np.ones((num_values, 1)), np.sum(cont_mat, axis = 0).reshape(1, num_values))
        row_sum = np.matmul(np.sum(cont_mat, axis = 1).reshape(num_values, 1), np.ones((1, num_values)))
        #print(col_sum)
        #print(row_sum)
    
        cont_mat = cont_mat/(col_sum+row_sum)
        #print(cont_mat)
    
        # ignore columns that have been switched:
        cont_mat[:, list(switched_col)]=-1
    
        #print(cont_mat)
    
        sort_0 = np.argsort(cont_mat, axis = 0)
        sort_1 = np.argsort(cont_mat, axis = 1)
    
        #print('argsort contmat:')
        #print(sort_0)
        #print(sort_1)
    
        if np.array_equal(sort_1[:,-1], np.array(range(num_values))):
            break
    
        # switch values according to the max value in the contingency matrix:
        # get the position of max value:
        idx_max = np.unravel_index(np.argmax(cont_mat, axis=None), cont_mat.shape)
        #print(cont_mat)
        #print(idx_max)
    
        if (cont_mat[idx_max]>0) and (idx_max[0] not in switched_col):
            cluster_tmp = map_cluster.copy()
            #print('switch', map_values[idx_max[1]], 'and:', ref_values[idx_max[0]])
            map_cluster[cluster_tmp==map_values[idx_max[1]]]=ref_values[idx_max[0]]
            map_cluster[cluster_tmp==map_values[idx_max[0]]]=ref_values[idx_max[1]]
    
            switched_col.add(idx_max[0])
            #print(switched_col)
    
        else:
            break
    
    #print('final argsort contmat:')
    #print(sort_0)
    #print(sort_1)
    
    ##print('final cont_mat:')
    cont_mat = contingency_matrix(ref_cluster, map_cluster)
    col_sum = np.matmul(np.ones((num_values, 1)), np.sum(cont_mat, axis = 0).reshape(1, num_values))
    row_sum = np.matmul(np.sum(cont_mat, axis = 1).reshape(num_values, 1), np.ones((1, num_values)))
    cont_mat = cont_mat/(col_sum+row_sum)
    
    #print(cont_mat)
    
    return(map_cluster)

def exclude_elements(a, b):
    '''

    Parameters
    ----------
    a : numpy array 
        list to be searched and modified by excluding elements in b.
    b : numpy array
        list of elements to exclude from a.

    Returns
    -------
    numpy array.

    '''
    c = a[ ~np.in1d(a,b)]
    return c
        