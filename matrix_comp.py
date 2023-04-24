#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 18:27:26 2020

@author: patricktaylor
"""
from scipy import sparse
import numpy as np 
import utility as ut
from sklearn.metrics import pairwise_distances_chunked
import time
from scipy.sparse import csgraph

def construct_surface_matrix(SC,SI):
    # SC- surface coordinates (used for determining size of surface matrix only, can be empty array of length=len(SC))
    # SI - array of vertex connections. Each row of SI contains indices of vertices in SC that form a triangle in the mesh
    M=sparse.lil_matrix((len(SC),len(SC))) #initialize
    #set each M_ij, where i,j are vertices connected in the mesh, equal to 1.
    M[SI[:,0],SI[:,1]]=1
    M[SI[:,1],SI[:,0]]=1
    M[SI[:,1],SI[:,2]]=1
    M[SI[:,2],SI[:,1]]=1
    M[SI[:,2],SI[:,0]]=1
    M[SI[:,0],SI[:,2]]=1
    return M.tocsr()


def construct_inter_hemi_matrix(SC,tol=4):
    '''
    creates interhemisphere connection matrix IHC for a given surface mesh with coordinates SC by connecting all vertices on the midline fissure 
    to their nearest neighbor on the opposite hemisphere. IHC has dimension (len(SC),len(SC)).
    '''
    half=int(len(SC)/2)
    li,ld=ut.neighbors(SC[:half],SC[half:],1)
    ri,rd=ut.neighbors(SC[half:],SC[:half],1)
    IHC=sparse.lil_matrix((half*2,half*2))
    #R=sparse.lil_matrix((half,half))
    for i in range (half):
        if ld[i]<tol:
            IHC[i+half,li[i]]=1
            IHC[li[i],i+half]=1
    for i in range (half):
        if rd[i]<tol:
            IHC[ri[i]+half,i]=1
            IHC[i,ri[i]+half]=1
    IHC=IHC.tocsr()
    return IHC





def construct_structural_connectivity_matrix(SC,EC,tol=3,NNnum=45,binarize=False):
    ind,dist=ut.neighbors(SC,EC,1)
    bad=[]
    c=np.arange(len(dist))
    even=c[::2]
    odd=c[1::2]
    for i in range (int(len(dist)/2)):
        if (dist[even[i]]>=tol or dist[odd[i]]>=tol):
            bad.append(even[i])
            bad.append(odd[i])
    newEC=np.delete(EC,bad,axis=0)
    s2eInd, s2eDist=ut.neighbors(SC,newEC,1)
    Rind,Rdist=ut.neighbors(newEC,SC,NNnum)
    OtherEndInd=np.zeros(np.shape(Rind))
    for i in range(len(Rind)):
        for j in range (NNnum):
            if Rind[i][j]%2==0 :
                OtherEndInd[i][j]=int(s2eInd[Rind[i][j]+1])
            else: 
                OtherEndInd[i][j]=int(s2eInd[Rind[i][j]-1])
    
    M=sparse.lil_matrix((len(SC),len(SC)))
    x=np.arange(len(SC))
    for i in range (NNnum):
        AccSurfInd=np.column_stack((x,OtherEndInd[:,i]))
        U,C=np.unique(AccSurfInd,axis=0,return_counts=True)
        M[U[:,0],U[:,1]]+=C
        M[U[:,1],U[:,0]]+=C
    print(M.nnz)

    x=np.arange(len(SC))
    M[x,x]=0
    print(M.nnz)
    M=M.tocsr()
    if binarize:
        M[M>=1]=1
    return M.tocsr()

def construct_struc_conn(sc,ec,tol=2):
    ind,dist=ut.neighbors(sc,ec,1)
    indstart=ind[::2][:,0]
    indend=ind[1::2][:,0]
    diststart=dist[::2][:,0]
    distend=dist[1::2][:,0]
    
    tolmask= (diststart>tol) | (distend>tol)
    
    M=sparse.lil_matrix((len(sc),len(sc)))
    
    for i in range (len(indend)):
        
        if not tolmask[i]:
            
            M[indstart[i],indend[i]]+=1
    
    return M.tocsr()

def construct_incidence_matrices(sc,ec,tol=2):
    start=time.time()
    ind,dist=ut.neighbors(sc,ec,1)
    indstart=ind[::2][:,0]
    indend=ind[1::2][:,0]
    diststart=dist[::2][:,0]
    distend=dist[1::2][:,0]
    
    tolmask= (diststart>tol) | (distend>tol)
    
    startinc=sparse.lil_matrix( (len(sc), (len(tolmask) -tolmask.sum()) ), dtype=np.float32)
    endinc=sparse.lil_matrix( (len(sc), (len(tolmask) -tolmask.sum()) ), dtype=np.float32)
    
    good_inds=[i for i in range (len(tolmask)) if not tolmask[i]]
    x=np.arange(len(good_inds))
    startinc[indstart[good_inds],x]=1
    endinc[indend[good_inds],x]=1
    
    #for j,i in enumerate(good_inds):
        
    #    startinc[indstart[i],j]=1
    #    endinc[indend[i],j]=1
    end=time.time()
    print(f'{end-start} seconds taken for incidence matrix construction')
    return startinc.tocsr(), endinc.tocsr()
    
def construct_smoothing_matrix(sc,si,mask=None,sigma=2,epsilon=0.05):
    start=time.time()
    h=int(len(sc)/2)
    hsi=int(len(si)/2)
    lg=construct_smoothing_matrix_one_hem(sc[:h], si[:hsi],sigma=sigma,epsilon=epsilon)
    rg=construct_smoothing_matrix_one_hem(sc[h:], si[hsi:]-h ,sigma=sigma,epsilon=epsilon)
    
    sm=sparse.vstack((sparse.hstack((lg,sparse.csr_matrix((lg.shape[0], rg.shape[1]), dtype=lg.dtype))).tocsr(),sparse.hstack((sparse.csr_matrix((rg.shape[0], lg.shape[1]), dtype=lg.dtype),rg)).tocsr()))
    if mask is not None:
        sm=uts.mask_connectivity_matrix(sm,mask)
    sm=sklearn.preprocessing.normalize(sm, norm='l1')
    end=time.time()
    print (end-start, 'seconds taken for smoothing matrix construction')
    return sm

import gdist 

def construct_smoothing_matrix_one_hem(sc,si,sigma=2,epsilon=0.05):
    maxd=sigma * (-2 * np.log(epsilon)) ** (1 / 2)
    dists=gdist.local_gdist_matrix(sc.astype(np.float64), si.astype(np.int32),maxd )
    dists[dists > maxd] = 0
    dists = dists.minimum(dists.T)
    dists.eliminate_zeros()
    dists = dists.tolil()
    dists.setdiag(0)
    dists = dists.tocsr()
    
    g = -(dists.power(2) / (2 * (sigma ** 2)))
    np.exp(g.data, out=g.data)
    g += sparse.eye(g.shape[0], dtype=g.dtype).tocsr()

    return g
    
def smooth_incidence_matrices(start, end, coefs,binarize=False,return_unsmoothed=False):
    
    if binarize:
        M=start.dot(end.T)
        if return_unsmoothed:
            mat=M
        M[M>1]=1
        
        M=coefs.T.dot(M.dot(coefs))
    else:
        smooth_start = start.T.dot(coefs).T
        smooth_end = end.T.dot(coefs).T
        M = smooth_start.dot(smooth_end.T)
    if return_unsmoothed:
        return M+M.T,mat+mat.T
    else:
        return M+M.T

def construct_smoothed_connectivity_matrix(sc,si,ec,mask,tol=2,sigma=3,epsilon=0.2,binarize=False,return_unsmoothed=False):
    start=time.time()
    
    starti,endi=construct_incidence_matrices(uts.mask_medial_wall_vecs(sc,mask), ec, tol)
    print('incidence matrices computed')
    
    smoothing_coefs=construct_smoothing_matrix(sc,si,mask,sigma,epsilon)
        
    print('smoothing coefficients computed')
    if return_unsmoothed:
        smoothA,A=smooth_incidence_matrices(starti,endi,smoothing_coefs,binarize=binarize,return_unsmoothed=return_unsmoothed)
        end=time.time()
        print(f'{end-start} seconds taken')
        return smoothA,A
    else:
        A=smooth_incidence_matrices(starti,endi,smoothing_coefs,binarize=binarize)
    
        end=time.time()
        print(f'{end-start} seconds taken')
        return A



    
def construct_smoothed_connectivity_matrix_other(sc,ec,lhfile,rhfile,tol=2,sigma=2,epsilon=0.01,binarize=False):
    start=time.time()
    
    starti,endi=construct_structural_connectivity_by_incidence(sc, ec,tol=tol)
    print('incidence matrices computed')
    l=get_cortical_local_distances(nib.load(lhfile),nib.load(rhfile),max_smoothing_distance(sigma,epsilon,2),'/Users/patricktaylor/Documents/neural-identity-master/data/templates/cifti/ones.dscalar.nii')
    print('local cortical distances computed')
    smoothing_coefs=local_distances_to_smoothing_coefficients(l,2)
    print('smoothing coefficients computed')
    A=get_smoothed_adjacency_from_unsmoothed_incidence(starti,endi,smoothing_coefs)
    end=time.time()
    print(f'{end-start} seconds taken')
    return A


def normalize_time_series(ts):
    ts=ts.T
    nts=(ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)
    return nts.T

def construct_FC_matrix_with_sparsity_mask(ts,chunk_size=1000):
    
    sparse_mask=sparse.load_npz('/Users/patricktaylor/Documents/HCP_func_gradient/testing/functional_sparse_mask_1%_density.npz')[:59412,:59412]
    
    nts = normalize_time_series(ts).T

    sparse_chunck_list = []

    for i in range(0, int(nts.shape[1]), chunk_size):
        # portion of connectivity
        pcon = (np.matmul(nts[:, i:i + chunk_size].T, nts) / nts.shape[0])

        # sparsified connectivity portion
        spcon = sparse_mask[i:i + chunk_size, :].multiply(pcon)
        
        
        
        sparse_chunck_list.append(spcon)

    scon = sparse.vstack(sparse_chunck_list)
    
    scon=scon.tolil()
    scon[scon<0]=0
    scon=scon.tocsr()
    scon=(scon+scon.T)/2

    return scon

# def truncate_top_k(x, k, inplace=False):
#     m, n = x.shape
#     # get (unsorted) indices of top-k values
#     topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
#     # get k-th value
#     rows, _ = np.indices((m, k))
#     kth_vals = x[rows, topk_indices].min(axis=1)
#     # get boolean mask of values smaller than k-th
#     is_smaller_than_kth = x < kth_vals[:, None]
#     # replace mask by 0
#     if not inplace:
#         return np.where(is_smaller_than_kth, 0, x)
#     x[is_smaller_than_kth] = 0
#     return x

def truncate_top_k(x, k, inplace=False):
    m, n = x.shape
    topk_indices = np.argsort(x, axis=1)[:, -k:]
    kth_vals = np.partition(x, k-1, axis=1)[:, k-1]
    is_smaller_than_kth = x < kth_vals[:, None]
    if inplace:
        x[is_smaller_than_kth] = 0
    else:
        x = np.where(is_smaller_than_kth, 0, x)
    x.flags.writeable = False
    return x
# =============================================================================
# def truncate_top_k(x, k, inplace=False):
#     m, n = x.shape
#     # get (unsorted) indices of top-k values
#     topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
#     # get k-th value
#     rows, _ = np.indices((m, k))
#     kth_vals = x[rows, topk_indices].min(axis=1)
#     # get boolean mask of values smaller than k-th
#     is_smaller_than_kth = x < kth_vals[:, None]
#     # replace mask by 0
#     if not inplace:
#         return np.where(is_smaller_than_kth, 0, x)
#     x[is_smaller_than_kth] = 0
#     return x
# =============================================================================

def construct_FC_matrix_row_thresh(ts, threshold=0.01, chunk_size=1000, symmetric=True):
    nts = normalize_time_series(ts).T
    keep_num = int(np.round(len(ts) * threshold))
    print(keep_num, 'elements per row retained')

    scon_list = []
    for i in range(0, nts.shape[1], chunk_size):
        chunk = nts[:, i:i + chunk_size]
        pcon = chunk.T @ nts / nts.shape[0]
        pcon[:, :i] = 0
        if i + chunk_size < nts.shape[1]:
            pcon[:, i + chunk_size:] = 0
        spcon = sparse.csr_matrix(truncate_top_k(pcon, keep_num, inplace=True))
        scon_list.append(spcon)

    scon = sparse.vstack(scon_list).tocsr()
    if symmetric:
        scon = (scon + scon.T) / 2

    return scon

# =============================================================================
# def construct_FC_matrix_row_thresh(ts,threshold=0.01,chunk_size=1000,normalize=True,verbose=False):
#     
#     start=time.time()
#     
#     if normalize:
#         nts = normalize_time_series(ts).T
#     else:
#         nts =ts.T
# 
#     sparse_chunck_list = []
#     
#     keep_num=int(np.round(len(ts)*threshold))
#     print(keep_num, 'elements per row retained')
#     
#     for i in range(0, int(nts.shape[1]), chunk_size):
#         # portion of connectivity
#         pcon = (np.matmul(nts[:, i:i + chunk_size].T, nts) / nts.shape[0])
#         
#         # sparsified connectivity portion
#         spcon = sparse.csr_matrix(truncate_top_k(pcon,keep_num,inplace=True))
#         
#         
#         
#         sparse_chunck_list.append(spcon)
#         
#         if verbose:
#             print(f'rows {i}:{i+chunk_size} done')
#         
#     scon = sparse.vstack(sparse_chunck_list)
#     
#     #scon=scon.tolil()
#     #scon[scon<0]=0
#     scon=scon.tocsr()
#     scon=(scon+scon.T)/2
#     end = time.time()
#     
#     x=np.arange(len(ts))
#     scon[x,x]=0
#     print(f'{end-start} seconds taken')
#     return scon.astype('float32')
# =============================================================================

def construct_microstructure_matrix_inverse(msvecs, mc, SM, IHC,k=20):

    M = sparse.lil_matrix((len(mc), len(mc)))  # initialize empty result matrix
    r, c, v = sparse.find(SM + IHC)  # extract and flatten nonzero indices of SM
    dif = msvecs[r] - msvecs[c]
    dist = np.linalg.norm(dif, axis=1)
    for i in range(len(r)):

        M[r[i], c[i]] = (1 / (1 + k*dist[i]))

    x = np.arange(len(mc))
    M[x, x] = 0
    return M.tocsr()

def connectivity_similarity(M,chunk_size):
    sim=np.zeros((M.shape[1]))
    
    for i in range(0, int(M.shape[1]), chunk_size):
        # portion of connectivity
        
        sim[i] = np.linalg.norm((np.matmul(np.array(M[:, i:i + chunk_size].T), np.array(M)) / M.shape[0]),axis=1)
        
        # sparsified connectivity portion
        
        
    return sim

def construct_functional_connectivity_matrix_chunked(timeseries,threshold=0.01):
    starttime=time.time()
    #dense_covar=np.nan_to_num(np.corrcoef(timeseries))
    dense_covar=(np.corrcoef(timeseries))
    #flat_dense_covar=np.reshape(dense_covar,(len(dense_covar)*len(dense_covar)))
    #flat_dense_covar=np.nan_to_num(flat_dense_covar)
    #flat_zscore=scipy.stats.zscore(flat_dense_covar_nonan)
    #dense_covar_mat=np.reshape(flat_dense_covar,np.shape(dense_covar)) 
    #x=np.arange(len(dense_covar))
    #dense_covar[x,x]=0
    print('covariance done')
    sparse_threshold_covar=sparse.lil_matrix(np.shape(dense_covar))
    keep_num=int(np.round(len(dense_covar)*threshold))
    for i in range (len(dense_covar)):
        #print(i)
        row=np.nan_to_num(dense_covar[i,:])
        keep_inds=np.argpartition(row,-keep_num)[-keep_num:]
        sparse_threshold_covar[i,keep_inds]=row[keep_inds]
    
    sparse_threshold_covar=sparse_threshold_covar.tocsr()
    gen = pairwise_distances_chunked(sparse_threshold_covar,metric='cosine',n_jobs=-1)
    cosine_similarity=sparse.lil_matrix((len(timeseries),len(timeseries)))
    start=0
    for item in gen:
        end=start+len(item)
        g=1-item
        print(np.shape(g))
        cosine_similarity[start:end,:]=g
        start=end
    #cosine_similarity=1-cosine_distance
    x=np.arange(len(dense_covar))
    cosine_similarity[x,x]=0
    #cosine_similarity_sparse=sparse.csr_matrix(cosine_similarity)
    #symmetric_sparse_threshold_covar=(sparse_threshold_covar+sparse_threshold_covar.T)/2
    endtime=time.time()
    print(endtime-starttime)
    return cosine_similarity.tocsr()



def construct_thresholded_correlation(timeseries,threshold=0.01):
    dense_covar=(np.corrcoef(timeseries))
    #flat_dense_covar=np.reshape(dense_covar,(len(dense_covar)*len(dense_covar)))
    #flat_dense_covar=np.nan_to_num(flat_dense_covar)
    #flat_zscore=scipy.stats.zscore(flat_dense_covar_nonan)
    #dense_covar_mat=np.reshape(flat_dense_covar,np.shape(dense_covar)) 
    #x=np.arange(len(dense_covar))
    #dense_covar[x,x]=0
    print('correlation done')
    sparse_threshold_covar=sparse.lil_matrix(np.shape(dense_covar))
    keep_num=int(np.round(len(dense_covar)*threshold))
    for i in range (len(dense_covar)):
        #print(i)
        row=np.nan_to_num(dense_covar[i,:])
        keep_inds=np.argpartition(row,-keep_num)[-keep_num:]
        sparse_threshold_covar[i,keep_inds]=row[keep_inds]
    
    sparse_threshold_covar=sparse_threshold_covar.tocsr()
    return sparse_threshold_covar

def threshold_sparse_matrix(mat,threshold=0.05):
    #newmat=sparse.lil_matrix(np.shape(mat))
    orignum = mat.nnz
    new = mat.copy
    dat = new.data
    dat[dat<=threshold]=0
    new.data = dat
    
    fnum = new.nnz
    
    #print(100*fnum/orignum, '% of nonzero elements remaining')
    
    return new

'''
def construct_k_neighbors_fc_matrix(ts, numneighbors=100, metric='minkowski'):
    cor = np.corrcoef(ts)
    
    neighbors_graph = kneighbors_graph(cor, 100 , mode ='distance', metric='minkowski',n_jobs = -1)
'''   
    
    
    

def construct_functional_connectivity_matrix_chunked_transpose(timeseries,threshold=0.01):
    starttime=time.time()
    #dense_covar=np.nan_to_num(np.corrcoef(timeseries))
    dense_covar=(np.corrcoef(timeseries))
    #flat_dense_covar=np.reshape(dense_covar,(len(dense_covar)*len(dense_covar)))
    #flat_dense_covar=np.nan_to_num(flat_dense_covar)
    #flat_zscore=scipy.stats.zscore(flat_dense_covar_nonan)
    #dense_covar_mat=np.reshape(flat_dense_covar,np.shape(dense_covar)) 
    #x=np.arange(len(dense_covar))
    #dense_covar[x,x]=0
    print('correlation done')
    sparse_threshold_covar=sparse.lil_matrix(np.shape(dense_covar))
    keep_num=int(np.round(len(dense_covar)*threshold))
    for i in range (len(dense_covar)):
        #print(i)
        row=np.nan_to_num(dense_covar[i,:])
        keep_inds=np.argpartition(row,-keep_num)[-keep_num:]
        sparse_threshold_covar[i,keep_inds]=row[keep_inds]
    
    sparse_threshold_covar=sparse_threshold_covar.tocsr()
    
    x=np.arange(np.shape(sparse_threshold_covar)[0])
    sparse_threshold_covar[x,x]=0
    
    #cosine_similarity_sparse=sparse.csr_matrix(cosine_similarity)
    #symmetric_sparse_threshold_covar=(sparse_threshold_covar+sparse_threshold_covar.T)/2
    endtime=time.time()
    print(endtime-starttime)
    return (sparse_threshold_covar+sparse_threshold_covar.T)/2

def construct_single_scalar_matrix_inverse(scalar, SM, ihc):
    M = sparse.lil_matrix((len(scalar), len(scalar)))
    r, c, v = sparse.find(SM + ihc)
    dif = scalar[r] - scalar[c]
    dot = np.square(dif)
    dist = np.sqrt(dot)
    for i in range(len(r)):
        M[r[i], c[i]] = (1 / (1 + dist[i]))
    M.data = np.nan_to_num(M.data)
    x = np.arange(len(scalar))
    M[x, x] = 0
    return M.tocsr()

def construct_microstructure_matrix_gaussian(msvecs,SM,IHC,kappa=.5): 
    '''
    msvecs- microstructure vertex-wise feature vectors
    mc- surface coordinates 
    SM- surface matrix used to define local neighborhood of each vertex
    IHC- interhemicon matrix, same use as SM
    kappa- used in kernelized weight calculation: w_ij=C*exp(kappa*(<v_i,v_j>)^2), 
        where w_ij is the connection weight and <v_i,v_j> is the cosine norm between the feature vectors of vertex i and vertex j
    
    '''
    M=sparse.lil_matrix((len(msvecs),len(msvecs))) #initialize empty result matrix
    r,c,v=sparse.find(SM) #extract and flatten nonzero indices of SM
    a=np.einsum('ij, ij->i', msvecs[r], msvecs[r]) # compute magnitude of feature vectors
    b=np.einsum('ij, ij->i', msvecs[c], msvecs[c]) # ^^^^      ^^^^     ^^
    d=np.einsum('ij, ij->i',msvecs[r] ,msvecs[c])  # compute unormalized dot product between all vertices connected by SM
    M[r,c]=np.nan_to_num((1/np.exp(kappa))*np.exp(-kappa*np.square(np.true_divide(d,np.multiply(np.sqrt(a),np.sqrt(b)))))) #populate result matrix M using gaussian kernel of cosine norm
    R,C,V=sparse.find(IHC) #repeat above process with IHC connection matrix
    A=np.einsum('ij, ij->i', msvecs[R], msvecs[R])
    B=np.einsum('ij, ij->i', msvecs[C], msvecs[C])
    D=np.einsum('ij, ij->i',msvecs[R] ,msvecs[C])
    M[R,C]=np.nan_to_num((1/np.exp(kappa))*np.exp(-kappa*np.square(np.true_divide(D,np.multiply(np.sqrt(A),np.sqrt(B))))))
    
    M.data=np.nan_to_num(M.data)
    M=np.nan_to_num(M)
    rr,cc,vv=sparse.find(M)
    print('mean val=',np.mean(vv),'median val=',np.median(vv))
    print('max val=',np.max(vv),'min val=',np.min(vv)) 
    M=M.tocsr()
    
    return M

def diffusion_matrix(A,t=0):
    Lap,D_sqrt_vec=csgraph.laplacian(A,normed=True,return_diag=True)
    D_inv_sqrt_vec=1/D_sqrt_vec
    D_inv_sqrt=sparse.lil_matrix(np.shape(Lap))
    x=np.arange(np.shape(Lap)[0])
    D_inv_sqrt[x,x]=D_inv_sqrt_vec
    L_sqrt=(D_inv_sqrt.dot(A)).dot(D_inv_sqrt)
    Lap,D_alpha=csgraph.laplacian(L_sqrt,return_diag=True)
    D_alpha_mat=sparse.lil_matrix(np.shape(Lap))
    D_alpha_mat[x,x]=1/D_alpha
    P=D_alpha_mat.dot(L_sqrt)
    print('diffusion matrix computed')
    if t==0:
        return P
    else:
        return P**t

def cosine_similarity_matrix(M):
    gen = pairwise_distances_chunked(M,metric='cosine',n_jobs=-1)
    cosine_similarity=sparse.lil_matrix(np.shape(M))
    start=0
    for item in gen:
        end=start+len(item)
        g=1-item
        #g=np.where(g<0.1,0,g)
        print(np.shape(g))
        cosine_similarity[start:end,:]=g
        start=end
    #cosine_similarity=1-cosine_distance
    x=np.arange(np.shape(M)[0])
    cosine_similarity[x,x]=0
    return cosine_similarity

def cosine_similarity_matrix_threshold(M,thresh=0.1):
    gen = pairwise_distances_chunked(M,metric='cosine',n_jobs=-1)
    cosine_similarity=sparse.lil_matrix(np.shape(M))
    start=0
    for item in gen:
        end=start+len(item)
        g=1-item
        #g=np.where(g<0.1,0,g)
        print(np.shape(g))
        g[g<thresh]=0
        cosine_similarity[start:end,:]=g
        start=end
    #cosine_similarity=1-cosine_distance
    x=np.arange(np.shape(M)[0])
    cosine_similarity[x,x]=0
    return cosine_similarity

import os

def cosine_similarity_matrix_memory_save(M,directory):
    if os.path.isdir(directory)==False:
        os.mkdir(directory)
    gen = pairwise_distances_chunked(M,metric='cosine',n_jobs=-1)
    #cosine_similarity=sparse.lil_matrix(np.shape(M))
    start=0
    running=0
    for item in gen:
        end=start+len(item)
        g=1-item
        #g=np.where(g<0.1,0,g)
        print(np.shape(g))
        #cosine_similarity[start:end,:]=g
        start=end
        np.save(directory+'/comp_%d' % running, g)
        running+=1
    #cosine_similarity=1-cosine_distance
    #x=np.arange(np.shape(M)[0])
    #cosine_similarity[x,x]=0
    return running

def combine_chunks(directory,size,num):
    
    m=sparse.lil_matrix((size,size))
    #l=[]
    s=0
    for i in range (num):
        chunk=np.load(directory+'/comp_%d.npy' % i)
        e=s+np.shape(chunk)[0]
        m[s:e,:]=chunk
        s=e
        
    return m.tocsr()

# =============================================================================
# import sparse_dot_mkl
# 
# def get_cosine_matrix(mat):
#     mat = mat / np.linalg.norm(mat, 2, 0)
#     
#     return sparse_dot_mkl.dot_product_transpose_mkl(mat)
# =============================================================================

def parcel_average_timeseries(ts,parc):
    #h = int(len(ts)/2)
    #parc[h:]+=1
    labels=np.unique(parc)
    pts=np.zeros((len(labels),np.shape(ts)[1]))
    
    for i in range (len(labels)):
        inds=np.where(parc==labels[i])[0]
        pts[i,:]=np.mean(ts[inds],0)
    
    return pts

def parcel_average_timeseries_sep_hem(ts,parc):
    h = int(len(ts)/2)
    lts = ts[:h]
    rts=ts[h:]
    
    lp=parc[:h]
    rp=parc[h:]
    plts = parcel_average_timeseries(lts,lp)
    prts = parcel_average_timeseries(rts,rp)
    
    pts = np.vstack((plts,prts))
    
    return pts




def parcelwise_connectivity_from_vertexwise(mat,parc):
    
    labels = np.unique(parc)
    
    pmat = np.zeros((len(labels),len(labels)))
    
    for i in range (len(labels)):
        inds = np.where(parc==labels[i])[0]
        
        chunk = mat[:,inds]
        
        sumchunk = np.sum(chunk,1)
        
        for j in range (len(labels)):
            jnds = np.where(parc==labels[j])[0]
            if i == j :
                pmat[i,j]=np.sum(sumchunk[jnds])/2
            else:
                pmat[i,j]=np.sum(sumchunk[jnds])
                
    return pmat
        
        
    
    
    
def cos_norm(v1,v2):
    
    dot = np.dot(v1,v2)
    
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    return np.abs(dot/(n1*n2))
    
    
    
    
    
    
def temporal_multilayer(m1,m2,gamma=0.1):
    identity=sparse.eye(np.shape(m1)[0])*gamma
    
    return sparse.bmat([[m1,identity],[identity,m2]])

def make_multilayer(matlist,gamma=0.1):
    identity=sparse.eye(np.shape(matlist[0])[0])*gamma
    rowlist=[]
    
    for i in range(len(matlist)):
        
        row=[]
        
        for j in range(len(matlist)):
            
            if i==j:
                
                row.append(matlist[i])
            else:
                row.append(identity)
        
        rowlist.append(row)
    
    return sparse.bmat(rowlist)

        

def normalize_evec(vecs):
    nvecs=np.zeros(np.shape(vecs))
    for i in range(len(vecs.T)):
        nvecs[:,i]=(vecs[:,i]-np.mean(vecs[:,i]))/np.std(vecs[:,i])
    
    return nvecs

from decomp import LapDecomp

def get_multilayer_flexibility(ts,windowlength,num=7,gamma=15):
    
    nwindows=int(len(ts.T)/windowlength)
    difs=np.zeros((len(ts),nwindows-2))
    
    ts1=ts[:,0:windowlength]
    ts2=ts[:,windowlength:windowlength*2]
    m1=construct_FC_matrix_row_thresh(ts1,normalize=False)
    m2=construct_FC_matrix_row_thresh(ts2,normalize=False)
    mlg=temporal_multilayer(m1,m2,gamma=gamma)
    vals,vecs=LapDecomp(mlg,num)
        
    vecs1=vecs[:len(ts),1:]
    vecs2=vecs[len(ts):2*len(ts),1:]
    
    vecs1=normalize_evec(vecs1)
    vecs2=normalize_evec(vecs2)
        
    d=vecs1-vecs2
        
    difs[:,0]=np.linalg.norm(d,axis=1)
    
    for i in range (1,nwindows-2):
        
        m1=m2
        ts2=ts[:,windowlength*(i+1):windowlength*(i+2)]
        m2=construct_FC_matrix_row_thresh(ts2,normalize=False)
        mlg=temporal_multilayer(m1,m2,gamma=gamma)
        vals,vecs=LapDecomp(mlg,num)
        
        vecs1=vecs[:len(ts),1:]
        vecs2=vecs[len(ts):2*len(ts),1:]
        
        vecs1=normalize_evec(vecs1)
        vecs2=normalize_evec(vecs2)
        
        d=vecs1-vecs2
        
        difs[:,i]=np.linalg.norm(d,axis=1)
        
    return difs, np.linalg.norm(difs,axis=1)
        


        
        
#import gdist
import nibabel as nib
    
import sklearn

def get_streamline_incidence(start_dists, start_indices, end_dists, end_indices, node_count, threshold=2):
    """
    returns a couple of half incidence matrices in a sparse format after
    filtering the streamlines that are far (>2mm) from their closest vertex.
    """
    # mask points that are further than the threshold from all surface coordinates
    outlier_mask = (start_dists > threshold) | (end_dists > threshold)
    
    # create a sparse incidence matrix
    start_dict = {}
    end_dict = {}
    indices = (i for i in range(len(outlier_mask)) if not outlier_mask[i])
    for l, i in enumerate(indices):
        start_dict[(start_indices[i], l)] = start_dict.get((start_indices[i], l), 0) + 1
        end_dict[(end_indices[i], l)] = end_dict.get((end_indices[i], l), 0) + 1

    start_inc_mat = sparse.dok_matrix( (node_count, (len(outlier_mask) -outlier_mask.sum()) ), dtype=np.float32)

    for key in start_dict:
        start_inc_mat[key] = start_dict[key]

    end_inc_mat = sparse.dok_matrix( (node_count, (len(outlier_mask) -outlier_mask.sum()) ), dtype=np.float32)

    for key in end_dict:
        end_inc_mat[key] = end_dict[key]


    return (start_inc_mat.tocsr(), end_inc_mat.tocsr())

import utility as uts

def get_incidence_matrix(ec,sc,threshold=2,return_indcidence=True):
    inds,dists=uts.neighbors(sc,ec,1)
    inds=inds[:,0]
    dists=dists[:,0]
    startinds=inds[::2]
    endinds=inds[1::2]
    startdists=dists[::2]
    enddists=dists[1::2]
    starti,endi=get_streamline_incidence(startdists,startinds,enddists,endinds,len(sc),threshold=threshold)
    
    if return_indcidence:
        return starti,endi
    else:
        return starti.dot(endi.T)

    
def _diagonal_stack_sparse_matrices(m1, m2):
    """
    Inputs are expected to be CSR matrices

    this is what the output looks like:

    | M1  0 |
    | 0  M2 |

    """
    return sparse.vstack((
        sparse.hstack((
            m1,
            sparse.csr_matrix((m1.shape[0], m2.shape[1]), dtype=m1.dtype)
        )).tocsr(),
        sparse.hstack((
            sparse.csr_matrix((m2.shape[0], m1.shape[1]), dtype=m1.dtype),
            m2
        )).tocsr()
    ))


    
    
def local_geodesic_distances(max_distance, vertices, triangles):
    distances = gdist.local_gdist_matrix(vertices.astype(np.float64), triangles.astype(np.int32), max_distance)
    

    # make sure maximum distance is applied
    distances[distances > max_distance] = 0
    distances = distances.minimum(distances.T)
    distances.eliminate_zeros()
    distances = distances.tolil()
    distances.setdiag(0)
    distances = distances.tocsr()
    return distances


def local_geodesic_distances_on_surface(surface, max_distance):
    vertices = surface.darrays[0].data
    triangles = surface.darrays[1].data
    retval = local_geodesic_distances(max_distance, vertices, triangles)
    return retval


def trim_and_stack_local_distances(left_local_distances,
                                    right_local_distances,
                                    sample_cifti_file=None):
    # load a sample file to read the mapping from
    if sample_cifti_file is None:
        sample_cifti_file='/Users/patricktaylor/Documents/neural-identity-master/data/templates/cifti/ones.dscalar.nii'
        
    cifti = nib.load(sample_cifti_file)

    # load the brain models from the file (first two models are the left and right cortex)
    brain_models = [x for x in cifti.header.get_index_map(1).brain_models]

    # trim left surface to cortex
    left_cortex_model = brain_models[0]
    left_cortex_indices = left_cortex_model.vertex_indices[:]
    left_cortex_local_distance = left_local_distances[left_cortex_indices, :][:, left_cortex_indices]

    # trim right surface to cortex
    right_cortex_model = brain_models[1]
    right_cortex_indices = right_cortex_model.vertex_indices[:]
    right_cortex_local_distance = right_local_distances[right_cortex_indices, :][:, right_cortex_indices]

    # concatenate local distances with diagonal stacking
    return _diagonal_stack_sparse_matrices(left_cortex_local_distance, right_cortex_local_distance)


def get_cortical_local_distances(left_surface, right_surface, max_distance, sample_cifti_file=None):
    """
    This function computes the local distances on the cortical surface and returns a sparse matrix
    with dimensions equal to cortical brainordinates in the cifti file.
    """
    left_local_distances = local_geodesic_distances_on_surface(left_surface, max_distance)
    right_local_distances = local_geodesic_distances_on_surface(right_surface, max_distance)
    return trim_and_stack_local_distances(left_local_distances, right_local_distances, sample_cifti_file)


def local_distances_to_smoothing_coefficients(local_distance, sigma):
    """
    Takes a sparse local distance symmetric matrix (CSR) as input,
    Generates an assymetric coefficient sparse matrix where each
    row i, has the coefficient for smoothing a signal from node i,
    therefore, each row sum is unit (1). sigma comes from the smoothing
    variance.
    """
    # apply gaussian transform
    gaussian = -(local_distance.power(2) / (2 * (sigma ** 2)))
    np.exp(gaussian.data, out=gaussian.data)

    # add ones to the diagonal
    gaussian += sparse.eye(gaussian.shape[0], dtype=gaussian.dtype).tocsr()

    # normalize rows of matrix
    return sklearn.preprocessing.normalize(gaussian, norm='l1')


def get_smoothed_adjacency_from_unsmoothed_incidence(start_inc_mat, end_inc_mat, local_smoothing_coefficients):
    """
    Return a smoothed sparse adjacency matrix from the two halfs of incidence matrix.
    The smoothing is done at the network level, that is the incidence matrices are
    smoothed before creation of the adjacency.
    """
    #print('shape of local smoothing coefficients matrix is',np.shape(local_smoothing_coefficients))
    smoothed_start_inc_mat = start_inc_mat.T.dot(local_smoothing_coefficients).T
    smoothed_end_inc_mat = end_inc_mat.T.dot(local_smoothing_coefficients).T
    A = smoothed_start_inc_mat.dot(smoothed_end_inc_mat.T)
    return A + A.T
    
    
    
    
def max_smoothing_distance(sigma, epsilon, dim):
    """
    return the distance of the smoothing kernel that will miss a epsilon proportion of the
    smoothed signal energy
    """
    # return sigma * (-stats.norm.ppf((1 - (1 - epsilon) ** (1 / dim)) / 2))
    return sigma * (-2 * np.log(epsilon)) ** (1 / dim)
    
    
    
def get_smoothing_matrix(lhfile,rhfile,epsilon,sigma):
    start=time.time()
    localdist=get_cortical_local_distances(nib.load(lhfile), nib.load(rhfile), max_smoothing_distance(sigma, epsilon, 2), sample_cifti_file=None)
    smoothing_coefs=local_distances_to_smoothing_coefficients(localdist, sigma)
    end=time.time()
    print(end-start, 'seconds taken for smoothing matrix construction')
    return smoothing_coefs

    
    
    
def construct_structural_connectivity_by_incidence(sc,ec,tol=2,binarize=False):
    start=time.time()
    inds,dists=uts.neighbors(sc,ec,1)
    inds=inds[:,0]
    dists=dists[:,0]
    startinds=inds[::2]
    endinds=inds[1::2]
    startdists=dists[::2]
    enddists=dists[1::2]
    outlier_mask = (startdists > tol) | (enddists > tol)
   

    # create a sparse incidence matrix
    start_dict = {}
    end_dict = {}
    indices = (i for i in range(len(outlier_mask)) if not outlier_mask[i])
    for l, i in enumerate(indices):
        start_dict[(startinds[i], l)] = start_dict.get((startinds[i], l), 0) + 1
        end_dict[(endinds[i], l)] = end_dict.get((endinds[i], l), 0) + 1

    start_inc_mat = sparse.dok_matrix(
        (
            len(sc),
            (len(outlier_mask) - outlier_mask.sum())
        ),
        dtype=np.float32
    )

    for key in start_dict:
        start_inc_mat[key] = start_dict[key]

    end_inc_mat = sparse.dok_matrix(
        (
            len(sc),
            (len(outlier_mask) - outlier_mask.sum())
        ),
        dtype=np.float32
    )

    for key in end_dict:
        end_inc_mat[key] = end_dict[key]

    end=time.time()
    print(end-start, 'seconds taken for incidence matrix construction')
    return (start_inc_mat.tocsr(), end_inc_mat.tocsr())
    
    
    
    
    


