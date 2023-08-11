#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:36:03 2020

@author: patricktaylor
"""
from scipy.sparse.linalg import eigsh
import scipy.sparse.linalg
from scipy.sparse import csgraph
import time
import numpy as np
import mapalign as mp
from sklearn.decomposition import PCA
from scipy import sparse
from sklearn.utils import check_random_state
import brainspace as bs

def LapDecomp(m,num,normed=True,alternate=False,random_state=None):
    #m- csr format sparse adjacency matrix
    #num - number of laplacian eigenmodes to compute
    #returns tuple of eigenvalues (vals) and eigenvectors (vecs) as arrays of size (dim(m),1) and  size (dim(lap),num)
    start = time.time()
    
    
    #l=l.astype('float32')
    if not alternate:
        lap=csgraph.laplacian(m,normed=True).astype('float32')
        vals,vecs=eigsh(lap,num,which='SM')
    else:
        lap,dd=csgraph.laplacian(m,normed=True,return_diag=True)
        lap=lap.astype('float32')
        dd=dd.astype('float32')
        rs = check_random_state(random_state)
        lap*= -1
        v0 = rs.uniform(-1, 1, lap.shape[0])
        w, v = eigsh(lap, k=num + 1, sigma=1, which='LM', tol=0, v0=v0)
    
        # Sort descending and change sign of eigenvalues
        w, v = -w[::-1], v[:, ::-1]
        
        if normed:
            v /= dd[:, None]
    
        # Drop smallest
        w, v = w[1:], v[:, 1:]
    
        # Consistent sign (s.t. largest value of element eigenvector is pos)
        v *= np.sign(v[np.abs(v).argmax(axis=0), range(v.shape[1])])
    end = time.time()
    print('decomposition time=', (end-start),'seconds, which is', (end - start)/60, 'minutes')
    return vals,vecs



def get_pca_comp(evlist,num=None,discard=True):
    if num is None:
        num=evlist[0].shape[1]
    if discard:
        tempmat = np.zeros((np.shape(evlist[0])[0],num*len(evlist)))
        for i in range (len(evlist)):
            tempmat[:,i*num:(i+1)*num]=evlist[i][:,1:num+1]
    else:
        tempmat = np.zeros((np.shape(evlist[0])[0],num*len(evlist)))
        for i in range (len(evlist)):
            tempmat[:,i*num:(i+1)*num]=evlist[i][:,:num]
    pca = PCA(n_components=num)
    pca.fit(tempmat.T)
    components = pca.components_
    components = components.T

    return components 

def laplacian_embedding(vals,vecs):
    lapemb=np.zeros(vecs.shape)
    for i in range (len(vals)):
        lapemb[:,i]=vecs[:,i]/np.sqrt(vals[i])
    return lapemb

def diffusion_embed(mat,num):
    embed=mp.embed.compute_diffusion_map(mat,n_components=num,skip_checks=True)
    return embed

def diffusion_embed_brainspace(mat,num=12):
    v,w=bs.gradient.diffusion_mapping(mat,n_components=num)
    return v,w

#def projec
def svd_solver(p_matrix,num_vecs):
    u,singular_values,right_evecs=scipy.sparse.linalg.svds(p_matrix,k=num_vecs,which='LM',return_singular_vectors=True)
    
    return singular_values[::-1],right_evecs[::-1,:].T

def get_group_pca_comp(evlist,num):
    tempmat = np.zeros((np.shape(evlist[0])[0],num*len(evlist)))
    for i in range (len(evlist)):
        tempmat[:,i*num:(i+1)*num]=evlist[i][:,1:num+1]
        
    pca = PCA(n_components=num)
    pca.fit(tempmat.T)
    components = pca.components_
    components = components.T

    return components 

def projection_coefficient_timeseries(vecs,ts):
    return np.matmul(vecs.T,ts)

def projected_timeseries(vecs,ts):
    proj=projection_coefficient_timeseries(vecs,ts)
    return np.matmul(proj.T,vecs.T).T

def alignment(vecs,ts):
    filt=projected_timeseries(vecs,ts)
    filtered_norm=np.sum(np.linalg.norm(filt,axis=1))/len(ts)
    unfiltered_norm=np.sum(np.linalg.norm(ts,axis=1))/len(ts)
    return filtered_norm/unfiltered_norm

def liberality(timeseries,vecs):
    start=time.time()
    zeromean = np.zeros(np.shape(timeseries))
    for i in range(len(timeseries)):
        zeromean[i, :] = (timeseries[i, :] - np.mean(timeseries[i, :]))
    #spectrum=np.zeros(len(vecs[0,:]))
    filtered_timeseries=np.zeros(np.shape(zeromean))
    for tp in range (len(zeromean[0,:])):
        timeslice=zeromean[:,tp]
        
        for k in range (len(vecs[0,:])):
            v=vecs[:,k] 
            filtered_timeseries[:,tp]+= v*np.dot(v,timeslice)
    
    filtered_norm=np.sum(np.linalg.norm(filtered_timeseries,axis=1))/len(timeslice)
    unfiltered_norm=np.sum(np.linalg.norm(zeromean,axis=1))/len(timeslice)
    end=time.time()
    print(start-end, 'seconds taken')
    return filtered_norm/unfiltered_norm


def dynamic_energy_spectrum(timeseries,vecs,vals):
    zeromean = np.zeros(np.shape(timeseries))
    for i in range(len(timeseries)):
        zeromean[i, :] = (timeseries[i, :] - np.mean(timeseries[i, :]))
    spectrum=np.zeros((len(vecs[0,:]),len(timeseries[0,:])))
    for k in range (len(spectrum)):
        print(k)
        v=vecs[:,k]
        for tp in range (len(zeromean[0,:])):
            spectrum[k][tp]=(np.abs(np.dot(v,zeromean[:,tp]))**2)*vals[k]**2
    return spectrum 

def dynamic_power_spectrum(timeseries,vecs):
    zeromean = np.zeros(np.shape(timeseries))
    for i in range(len(timeseries)):
        zeromean[i, :] = (timeseries[i, :] - np.mean(timeseries[i, :]))
    spectrum=np.zeros((len(vecs[0,:]),len(timeseries[0,:])))
    for k in range (len(spectrum)):
        #print(k)
        v=vecs[:,k]
        for tp in range (len(zeromean[0,:])):
            spectrum[k][tp]=np.abs(np.dot(v,zeromean[:,tp]))
    return spectrum

def dynamic_reconstruction_spectrum(timeseries,vecs):
    zeromean = np.zeros(np.shape(timeseries))
    for i in range(len(timeseries)):
        zeromean[i, :] = (timeseries[i, :] - np.mean(timeseries[i, :]))
    spectrum=np.zeros((len(vecs[0,:]),len(timeseries[0,:])))
    for k in range (len(spectrum)):
        #print(k)
        v=vecs[:,k]
        for tp in range (len(zeromean[0,:])):
            spectrum[k][tp]=np.dot(v,zeromean[:,tp])
    return spectrum

def mean_energy_spectrum(timeseries,vecs,vals):
    zeromean = np.zeros(np.shape(timeseries))
    for i in range(len(timeseries)):
        zeromean[i, :] = (timeseries[i, :] - np.mean(timeseries[i, :]))
    spectrum=np.zeros(len(vecs[0,:]))
    for k in range (len(spectrum)):
        #print(k)
        v=vecs[:,k]
        for tp in range (len(zeromean[0,:])):
            spectrum[k]+=(np.abs(np.dot(v,zeromean[:,tp]))**2)*(vals[k]**2)/len(zeromean[0,:])
    return spectrum 

def mean_power_spectrum(timeseries,vecs):
    zeromean = np.zeros(np.shape(timeseries))
    for i in range(len(timeseries)):
        zeromean[i, :] = (timeseries[i, :] - np.mean(timeseries[i, :]))
    spectrum=np.zeros(len(vecs[0,:]))
    for k in range (len(spectrum)):
        #print(k)
        v=vecs[:,k]
        for tp in range (len(zeromean[0,:])):
            spectrum[k]+=np.abs(np.dot(v,zeromean[:,tp]))/len(zeromean[0,:])
    return spectrum 

def normalized_power_spectrum(timeseries,vecs):
    zeromean = np.zeros(np.shape(timeseries))
    for i in range(len(timeseries)):
        zeromean[i, :] = (timeseries[i, :] - np.mean(timeseries[i, :]))
    spectrum=np.zeros(len(vecs[0,:]))
    for k in range (len(spectrum)):
        #print(k)
        v=vecs[:,k]
        for tp in range (len(zeromean[0,:])):
            spectrum[k]+=np.abs(np.dot(v,zeromean[:,tp]))/len(zeromean[0,:])/np.linalg.norm(v)/np.linalg.norm(zeromean[:,tp])
    return spectrum 


def cos_norm(v1,v2):
    
    dot = np.dot(v1,v2)
    
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    
    return np.abs(dot/(n1*n2))

def reconstruct_vec(vtarg,vbasis):
    vrecon=np.zeros(np.shape(vtarg))
    for i in range(len(vbasis[0,:])):
        c = np.dot(vtarg,vbasis[:,i])
        
        vrecon+=c*vbasis[:,i]
        
    return vrecon

def reconstruct_basis(vecs,basisvecs):
    reconvecs=np.zeros(np.shape(vecs))
    for i in range (len(vecs[0,:])):
        reconvecs[:,i]=reconstruct_vec(vecs[:,i], basisvecs)
    
    return reconvecs


def get_projection_coefs(vtarg,vbasis,metric='cos'):
    coefs=np.zeros(len(vbasis[0,:]))
    
    for i in range(len(vbasis[0,:])):
        if metric=='cos':
            coefs[i]=cos_norm(vtarg,vbasis[:,i])
        if metric=='dot':
            coefs[i]=np.dot(vtarg,vbasis[:,i])
    
    return coefs

def recon_npercent_best(vtarg,vbasis,percent):
    coscoefs=get_projection_coefs(vtarg, vbasis)
    dotcoefs=get_projection_coefs(vtarg, vbasis,metric='dot')
    
    numcoefs=int(len(coscoefs)*(percent/100))
    
    sortinds=np.argsort(coscoefs)[::-1]
    sorted_coscoefs=coscoefs[sortinds]
    sorted_dotcoefs=dotcoefs[sortinds]
    sorted_vbasis=vbasis[:,sortinds]
    
    resultvec=np.zeros(len(vtarg))
    
    for i in range(numcoefs):
        resultvec+=sorted_dotcoefs[i]*sorted_vbasis[:,i]
    
    return resultvec

def get_recon_as_function_of_number_of_vecs_used(vtarg,vbasis):
    percents=np.arange(1,100)
    reconquals=[]
    for p in percents:
        recvec=recon_npercent_best(vtarg,vbasis,p)
        q = cos_norm(recvec,vtarg)
        reconquals.append(q)
    return reconquals

def get_number_of_vecs_needed(vtarg,vbasis,tol=.01):
    
    reconquals=get_recon_as_function_of_number_of_vecs_used(vtarg, vbasis)
    
    finalval=reconquals[-1]
    for i in range(len(reconquals)-1):
        if (finalval-reconquals[i])<=tol:
            return i

    
def get_av_num_vecs_needed(vecstarg,vecsbasis):
    num=0
    for i in range (len(vecstarg[0,:])):
        n=get_number_of_vecs_needed(vecstarg[:,i], vecsbasis)
        print(n)
        num+=n
    
    return num/len(vecstarg[0,:])

        
    
def subspace_distance(v1,v2):
    #p1=np.matmul(v1,v1.T)
    #p2=np.matmul(v2,v2.T)
    p1=np.matmul(v1.T,v2)
    p2=np.matmul(v2.T,v1)
    dif=p1-p2
    return np.linalg.norm(dif,ord='fro')

def temporal_derivative_timeseries(ts,method='sum difference'):
    
    dts=np.zeros((len(ts),len(ts.T)-1))
    dsum=np.zeros(len(ts))
    
    if method=='sum difference':
        for i in range (len(ts.T)-1):
            d=ts[:,i]-ts[:,i+1]
            dsum+=np.abs(d)
            dts[:,i]=d
        return dts, (dsum-np.mean(dsum))/np.std(dsum)
    
    if method=='differences':
        for i in range (len(ts.T)-1):
            d=ts[:,i]-ts[:,i+1]
            dts[:,i]=d
        return dts

def projection_matrix(basis):
    a=np.einsum('ij,jk->ik', basis.T, basis)
    a=np.linalg.inv(a)
    a=np.einsum('ij,jk->ik', a, basis.T)
    a=np.einsum('ij,jk->ik', basis, a)
    return a

def truncate_top_k(x, k, inplace=False):
    m, n = x.shape
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = x[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = x < kth_vals[:, None]
    # replace mask by 0
    if not inplace:
        return np.where(is_smaller_than_kth, 0, x)
    x[is_smaller_than_kth] = 0
    return x


def matmul_sparsify_percent(a,b,threshold=0.01,chunk_size=1000):
    keep_num=int(np.round(a.shape[0]*threshold))
    #print(keep_num, 'elements per row retained')
    sparse_chunck_list=[]
    for i in range(0, int(a.shape[0]), chunk_size):
        # portion of connectivity
        pcon = (np.matmul(a[i:i + chunk_size,: ], b))
        
        # sparsified connectivity portion
        spcon = sparse.csr_matrix(truncate_top_k(pcon,keep_num,inplace=True))
        
        
        
        sparse_chunck_list.append(spcon)
        
        #print(f'rows {i}:{i+chunk_size} done')
        
    scon = sparse.vstack(sparse_chunck_list)
    
    #scon=scon.tolil()
    #scon[scon<0]=0
    scon=scon.tocsr()
    return scon



def matmul_sparsify_absolute(a,b,threshold=0.01,chunk_size=1000):
    keep_num=int(np.round(a.shape[0]*threshold))
    #print(keep_num, 'elements per row retained')
    sparse_chunck_list=[]
    for i in range(0, int(a.shape[0]), chunk_size):
        pcon = (np.matmul(a[i:i + chunk_size,: ], b))
        spcon = sparse.csr_matrix(pcon[pcon>threshold])
        sparse_chunck_list.append(spcon)
        
    scon = sparse.vstack(sparse_chunck_list)

    scon=scon.tocsr()
    return scon
    
    
def projection_matrix_sparsify(basis,threshold=0.01):
    #basis=basis
    #a=matmul_sparsify_percent(basis.T,basis,threshold)
    #print(a)
    #a=sparse.linalg.inv(a.tocsc()).tocsr()
    a=np.matmul(basis.T,basis)
    a=np.linalg.inv(a)
    #a=matmul_sparsify_percent( a, basis.T,threshold)
    #print(a.shape)
    #a=a.dot(sparse.csr_matrix(basis.T))
    a=np.matmul(a,basis.T)
    #a=sparse.csr_matrix(basis).dot(a)
    a=matmul_sparsify_percent(basis,a,threshold)
    #a=matmul_sparsify_percent(basis, a,threshold)
    
    return a
    
    
def subspace_distance_projection(v1,v2,threshold=0.01):
    start=time.time()
    p1=projection_matrix_sparsify(v1,threshold)
    p2=projection_matrix_sparsify(v2,threshold)
    dif=p1-p2
    end=time.time()
    #print(f'{end-start} seconds taken')
    return sparse.linalg.norm(dif)


def subspace_similarity_for_rotation(angle,p1,p2,basis,threshold):
    rotMat=rotation_from_angle_and_plane(angle,p1,p2)
    rotBasis=np.matmul(rotMat,basis)
    
    return subspace_distance_projection(basis,rotBasis,threshold)

def subspace_dist_angle_sweep(n,m,threshold):
    angles=np.linspace(0,2*np.pi,30)
    plane=np.random.rand(n,2)
    basis=np.zeros((n,m))
    for i in range(len(basis.T)):
        basis[i,i]=1
    
    dists=[]
    for i in range (len(angles)):
        d=subspace_similarity_for_rotation(angles[i],plane[:,0],plane[:,1],basis,threshold)
        dists.append(d)
    return dists




import matplotlib.pyplot as plt


def subspace_dist_n_m_sweep_fixed_angle(angle,ms,ns,threshold,plot=True):
    
    dists=np.zeros((len(ns),len(ms)))
    plane=np.random.rand(ns[-1],2)
    
    for i in range (len(ns)):
        for j in range(len(ms)):
            if ns[i]>ms[j]:
                basis=np.zeros((ns[i],ms[j]))
                x=np.arange(ms[j])
                basis[x,x[:ms[j]]]=1
                dists[i,j]=subspace_similarity_for_rotation(angle,plane[:ns[i],0],plane[:ns[i],1],basis,threshold)
    
    if plot:
        plt.figure()
        im = plt.imshow(dists, cmap='hot')
        plt.xlabel('n')
        plt.ylabel('m')
        plt.colorbar(im, orientation='horizontal',label='subspace_dist')
        plt.show()
    return dists
            
def subspace_dist_m_angle_sweep(ms,n,threshold,plot=True):
    
    plane=np.random.rand(n,2)
    angles=np.linspace(0,2*np.pi,30)
    dists=np.zeros((len(angles),len(ms)))
    X,Y=np.meshgrid(angles,ms)
    
    for i in range (len(angles)):
        for j in range(len(ms)):

            basis=np.zeros((n,ms[j]))
            x=np.arange(ms[j])
            basis[x,x[:ms[j]]]=1
            dists[i,j]=subspace_similarity_for_rotation(angles[i],plane[:n,0],plane[:n,1],basis,threshold)
            
    if plot:
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(X, Y, dists.T, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('subspace distance as function of rotation angle and subspace dim');
        plt.xlabel('angle (radians)')
        plt.ylabel('vector subspace dimension m')
        #plt.zlabel('subspace distance')
        plt.show()
        
    return
    
    
    
    
    
def plot_angle_sweep(distlists,mnvals,m_or_n,mnval,thresh):
    if (m_or_n=='m'):
        other='n'
    else:
        other='m'
    plt.figure()
    plt.title(f'varying angle for different {m_or_n} values; {other}={mnval} thresh={thresh}')
    for i in range (len(distlists)):
        plt.plot(np.linspace(0,2*np.pi,30),distlists[i],label=f'{mnvals[i]}')
    plt.legend()
    plt.xlabel('angle (radians)')
    plt.ylabel('subspace distance')
    
    return 
    
    

import math

def rotation_from_angle_and_plane(angle, vector1, vector2, abs_tolerance=1e-10):
    '''
    generates an nxn rotation matrix from a given angle and
    a plane spanned by two given vectors of length n:
    https://de.wikipedia.org/wiki/Drehmatrix#Drehmatrizen_des_Raumes_%7F'%22%60UNIQ--postMath-0000003B-QINU%60%22'%7F
    The formula used is
    .. math::
        M = 𝟙 + (\cos\\alpha-1)\cdot(v_1\otimes v_1 + v_2\otimes v_2) - \sin\\alpha\cdot(v_1\otimes v_2 - v_2\otimes v_1)
    with :math:`M` being the returned matrix, :math:`v_1` and :math:`v_2` being the two
    given vectors and :math:`\\alpha` being the given angle. It differs from the formula
    on wikipedia in that it is the transposed matrix to yield results that are consistent
    with the 2D and 3D cases.
    :param angle: the angle by which to rotate
    :type angle: float
    :param vector1: one of the two vectors that span the plane in which to rotate
                    (no normalisation required)
    :type vector1: array like
    :param vector2: the other of the two vectors that span the plane in which to rotate
                    (no normalisation required)
    :type vector2: array like
    :param abs_tolerance: The absolute tolerance to use when checking if vectors have length 0 or are parallel.
    :type abs_tolerance: float
    :returns: the rotation matrix
    :rtype: an nxn :any:`numpy.ndarray`
    '''

    vector1 = np.asarray(vector1, dtype=np.float)
    vector2 = np.asarray(vector2, dtype=np.float)

    vector1_length = np.linalg.norm(vector1)
    if math.isclose(vector1_length, 0., abs_tol=abs_tolerance):
        raise ValueError(
            'Given vector1 must have norm greater than zero within given numerical tolerance: {:.0e}'.format(abs_tolerance))

    vector2_length = np.linalg.norm(vector2)
    if math.isclose(vector2_length, 0., abs_tol=abs_tolerance):
        raise ValueError(
            'Given vector2 must have norm greater than zero within given numerical tolerance: {:.0e}'.format(abs_tolerance))

    vector2 /= vector2_length
    dot_value = np.dot(vector1, vector2)

    if abs(dot_value / vector1_length ) > 1 - abs_tolerance:
        raise ValueError(
            'Given vectors are parallel within the given tolerance: {:.0e}'.format(abs_tolerance))

    if abs(dot_value / vector1_length ) > abs_tolerance:
        vector1 = vector1 - dot_value * vector2
        vector1 /= np.linalg.norm(vector1)
    else:
        vector1 /= vector1_length


    vectors = np.vstack([vector1, vector2]).T
    vector1, vector2 = np.linalg.qr(vectors)[0].T

    V = np.outer(vector1, vector1) + np.outer(vector2, vector2)
    W = np.outer(vector1, vector2) - np.outer(vector2, vector1)

    return np.eye(len(vector1)) + (math.cos(angle) - 1)*V - math.sin(angle)*W


from scipy.spatial.distance import cdist

import utility as uts

from scipy.spatial.distance import euclidean as euc

from scipy.spatial.distance import pdist

from scipy.spatial.distance import cdist

def segregation_harmonic(vecs,chunk_size=100):
    start=time.time()
    seg=np.zeros(len(vecs))
    
    for i in range (0,len(vecs),chunk_size):
        
        dchunk=cdist(vecs[i:i+chunk_size,:],vecs)
        
        seg[i:i+chunk_size]=np.mean(dchunk,axis=1)
    end=time.time()
    
    print(f'{end-start} seconds taken')
    return seg 

def within_network_dispersion(vecs,networkmask,mask):
    #network=uts.mask_medial_wall(networkmask,mask)
    #v=uts.mask_medial_wall_vecs(vecs,mask)
    
    network=networkmask
    v=vecs
    netinds=np.where(network==1)[0]
    
    vnet=v[netinds]
    
    centroid=np.mean(vnet,axis=0)
    
    sumsquare=0
    
    for vi in vnet:
        sumsquare+=(euc(vi,centroid))**2

    return sumsquare

def disp_within_longitudinal(vecs,parc):
    disps=np.zeros((len(np.unique(parc)),vecs.shape[0]))
    for i in range (vecs.shape[0]):
        disps[:,i]=disp_within(vecs[i],uts.netmasks_from_parc(parc))
    return disps

def disp_within(vecs,netmasks,normed=True):
    disps=np.zeros((netmasks.shape[1]))
    dd=dispersion_centroid(vecs)
    for i in range (netmasks.shape[1]):
        netinds=np.where(netmasks[:,i]==1)[0]
        d=1
        if normed is not None:
            d=dd
            disps[i]=dispersion_centroid(vecs[netinds])/d
    
    return disps
    

def between_network_dispersion(v,net1,net2):
    #net1=uts.mask_medial_wall(net1,mask)
    #net2=uts.mask_medial_wall(net2,mask)
    #v=uts.mask_medial_wall_vecs(vecs,mask)
    
    net1inds=np.where(net1==1)[0]
    net2inds=np.where(net2==1)[0]
    
    centroid1=np.mean(v[net1inds],axis=0)
    centroid2=np.mean(v[net2inds],axis=0)
    
    disp=euc(centroid1,centroid2)
    
    return disp 



def regionwise_dispersion(coords):
    
    return np.mean(pdist(coords))


def pairwise_dispersion(coords1, coords2):
    
    return np.mean(cdist(coords1,coords2))


def dispersion_centroid(vecs):
    
    centroid=np.mean(vecs,axis=0)
    
    dif=vecs-centroid
    
    dists=np.square(np.linalg.norm(dif,axis=1))
    
    disp=np.mean(dists)
    
    return disp
    

def compute_explanation_ratios(evals):
    ratios=np.zeros(evals.shape)
    
    for i in range (evals.shape[0]):
        valsum=np.sum(evals[i])
        for j in range (evals.shape[1]):
            ratios[i][j]=evals[i][j]/valsum
    return ratios
    
def network_eccentricity(vecs,netmasks):
    centrality=np.zeros(netmasks.shape[1])
    cen=np.mean(vecs,axis=0)
    for i in range (netmasks.shape[1]):
        inds=np.where(netmasks[i]==1)[0]
        
        netcentroid=np.mean(vecs[inds],axis=0)
        
        centrality[i]=np.square(np.linalg.norm(netcentroid-cen))
    
    return centrality 

def segregation_embedding(vecs,netmasks):
    segregations=np.zeros(netmasks.shape[1])
    
    disp_center=dispersion_centroid(vecs)
    for i in range (netmasks.shape[1]):
        dists=[]
        i1=np.where(netmasks[:,i]==1)[0]
        for j in range (netmasks.shape[1]):
            i2=np.where(netmasks[:,j]==1)[0]
            if i!=j:
                d=np.square(np.linalg.norm(np.mean(vecs[i1],axis=0)-np.mean(vecs[i2],axis=0)))/disp_center
                dists.append(d)
        segregations[i]=np.mean(dists)
    
    return segregations
        
        
def grad_variance_network(vecs,ind,netmasks):
    variance=np.zeros(netmasks.shape[1])
    
    for i in range (netmasks.shape[1]):
        i1=np.where(netmasks[:,i]==1)[0]
        variance[i]=np.var(vecs[i1,ind])
    return variance 

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

def network_distance_proportion(vecs,netmasks,nclust=10,lab=None):
    
    if lab is None:
        clusters=AgglomerativeClustering(n_clusters=nclust,linkage='ward').fit(vecs)

        lab=np.array(clusters.labels_)
    
    numverts=np.zeros(nclust)
    for i in range (nclust):
        numverts[i]=len(np.where(lab==i)[0])
    
    meansize=int(np.mean(numverts))
    print('mean size=',meansize)
    
    props=np.zeros(len(vecs))
    
    inds,dists=uts.neighbors(vecs,vecs,int(np.max(numverts)))
    inds=inds[:,1:]
    dists=dists[:,1:]
    
    for i in range (vecs.shape[0]):
        
        vert_allegiance=lab[i]
        
        
        num_in=[]
        num_out=[]
        #for j in range (len(np.where(lab==vert_allegiance)[0])):
        nv=int(numverts[lab[i]])
        for j in range (nv-1):
            net=lab[inds[i][j]]
            if net==vert_allegiance:
                num_in.append(dists[i][j])
            else:
                num_out.append(dists[i][j])
        if np.mean(num_in)>0.0001:
            props[i]=np.sum(num_out)/np.sum(num_in)
        
    return props 

def embed_local_dist(vecs,radius=0.1,numNN=None):
    if numNN is not None:
        inds,dists=uts.neighbors(vecs,vecs,numNN)
        inds=inds[:,1:]
        dists=dists[:,1:]
        meandist=np.mean(dists,axis=1)
        return meandist
    else:
        
        inds,dists=uts.neighbors(vecs,vecs,1000)
        inds=inds[:,1:]
        dists=dists[:,1:]
        meandist=np.zeros(len(vecs))
        num_in_radius=np.zeros(len(vecs))
        for i in range (len(vecs)):
            i1=np.where(dists[i]<radius)[0]
            meandist[i]=np.mean(dists[i][i1])
            num_in_radius[i]=len(i1)
        return np.sqrt(num_in_radius)
    
    
    
    
    #close_dist=
    
def mean_temporal_displacement_magnitude_embed(vecs):
    dists=[]
    for i in range (1,vecs.shape[0]-1):
        d=vecs[i]-vecs[i+1]
        dists.append(np.mean(np.linalg.norm(d,axis=1)))
        
    return np.array(dists)

def temporal_displacement_embed(vecs,netmasks,mag=True):
    difs=np.zeros((8,vecs.shape[2],vecs.shape[0]-1))
    difmags=np.zeros((8,vecs.shape[0]-1))
    for k in range (8):
        if netmasks is not None:
            inds=np.where(netmasks[:,k]==1)[0]
        for i in range (vecs.shape[0]-1):
            if netmasks is not None:
                d=vecs[i,inds]-vecs[i+1,inds]
            else:
                d=vecs[i,:]-vecs[i+1,:]
            #print(np.shape(d))
            difs[k,:,i]=np.mean(d,axis=0)
            difmags[k,i]=np.linalg.norm(difs[k,:,i])
            
    if mag:
        return difs,np.nan_to_num(difmags)


def temporal_displacement_vertex_embed(vecs,parc=None):
    #difs=np.zeros((1,vecs.shape[2],vecs.shape[0]-1))
    difmags=np.zeros((18463,vecs.shape[0]-1))
    for t in range (vecs.shape[0]-1):
        for i in range (18463):
    
                d=vecs[t,i]-vecs[t+1,i]
            #print(np.shape(d))
            
                difmags[i,t]=np.linalg.norm(d)
    
    if parc is None:
        return difmags
    
    else:
        mags=np.zeros((len(np.unique(parc)),vecs.shape[0]-1))
        for i in range(len(np.unique(parc))):
            inds=np.where(parc==i)[0]
            mags[i,:]=np.mean(difmags[inds,:],axis=0)
        return mags
            
            
    
    
    
from sklearn.decomposition import PCA

def isotropy_vecset(vecs,parc):
    anisotropy=np.zeros(len(np.unique(parc)))
    for i in range (len(np.unique(parc))):
        inds=np.where(parc==i)[0]
        pca = PCA(n_components=3)
        pca.fit(vecs[inds]-np.mean(vecs[inds],axis=0))
        exp=pca.explained_variance_ratio_
        anisotropy[i]=exp[0]/(exp[1]+exp[2])
    return anisotropy

def vector_anisotropy(vecs):
    anisotropy=np.zeros(len(vecs))
    inds,dists=uts.neighbors(vecs,vecs,num=1000)
    inds=inds[:,1:]
    dists=dists[:,1:]
    
    for i in range (len(vecs)):
        v=vecs[i]
        difs=vecs[inds[i][100:500]]-v
        pca = PCA(n_components=3)
        pca.fit(difs)
        exp=pca.explained_variance_ratio_
        anisotropy[i]=exp[0]/(exp[1]+exp[2])
    return anisotropy 
        

def embed_PCA(vecs):
    
    
    pca = PCA(n_components=3)
    pca.fit(vecs[::10])
    exp=pca.explained_variance_ratio_
    simplicity=exp[0]/(exp[1]+exp[2])
    
    return  simplicity


#displacements=mean_temporal_displacement_embed(grads_gammfit_masked[:,:,:3])
def avg_dif_mag(vecs,n=500):
    inds,dists=uts.neighbors(vecs,vecs,num=n)
    inds=inds[:,1:]
    mag=np.zeros(18463)
    for i in range(18463):
        v=vecs[i]
        dif= -1*(vecs[inds[i]]-v)
        meandif=np.mean(dif,axis=0)
        mag[i]=np.linalg.norm(meandif)
    return mag 

def detect_ridge(vecs):
    inds,dists=uts.neighbors(vecs,vecs,num=2000)
    inds=inds[:,1:]
    ridgecheck=np.zeros(18463)
    for i in range(18463):
        v=vecs[i]
        dif= -1*(vecs[inds[i]]-v)
        meandif=np.mean(dif,axis=0)
        ridgecheck[i]=(meandif[2]**2+meandif[0]**2)*(vecs[i,0]/np.max(vecs[:,0])+max(vecs[i,2],0)/np.max(vecs[:,2]))
    return ridgecheck

def physical_gradient_dist(vecs,sc,nnum=100):
    result=np.zeros(len(vecs))
    inds,dists=uts.neighbors(vecs,vecs,num=nnum)
    inds=inds[:,1:]
    dists=dists[:,1:]
    
    pinds,pdists=uts.neighbors(sc,sc,num=nnum)
    pinds=pinds[:,1:]
    pdists=pdists[:,1:]
    
    for i in range (len(vecs)):
        result[i]=len(set(inds[i]).intersection(set(pinds[i])))/nnum
    return result

def embed_velocity(vecs):
    velocitysum=np.zeros(vecs.shape[1])
    
    vecsum=np.zeros((vecs.shape[1],vecs.shape[2]))
    for i in range (vecs.shape[0]-1):
        d=vecs[i+1]-vecs[i]
        vecsum+=d
        velocitysum+=np.linalg.norm(d,axis=1)
    return velocitysum,np.linalg.norm(vecsum,axis=1)

def embed_angular_velocity(vecs):
    lsum=np.zeros(vecs.shape[1])
    for i in range (vecs.shape[0]-1):
        cent=np.mean(vecs[i],axis=0)
        rs=vecs[i]-cent
        vs=vecs[i+1]-vecs[i]
        lsum+=np.linalg.norm(np.cross(rs,vs),axis=1)
    return lsum 

def compute_transmodality(grads):
    transmodality=np.zeros((grads.shape[0],grads.shape[1]))
    for i in range (400):
        for j in range (grads.shape[1]):
            transmodality[i][j]=np.sqrt(max(0,grads[i,j,0]**2+ grads[i,j,2]**2-2*grads[i,j,1]**2))
    return transmodality

#phys_dist=np.load('/Users/patricktaylor/Documents/lifespan_analysis/intermediary/surf_dist_mat.npy')
# def compute_avg_physical_distance_in_neighborhood(grads,nn=1000,dthresh=0.4,phys_dist=phys_dist):
#     glh=grads[:10242]
#     grh=grads[10242:]
    

#     lhpdist=phys_dist[:10242,:10242]
#     rhpdist=phys_dist[10242:,10242:]
#     ds=[lhpdist,rhpdist]
#     fs=[]
#     for i, g in enumerate([glh,grh]):
#         f=np.zeros(len(g))
#         inds,dists=uts.neighbors(g,g,nn)
        
#         for j in range (len(inds)):
#             f[j]=np.mean(ds[i][j,inds[j][dists[j]<dthresh]])
#         fs.append(f)
#     fs=np.hstack((fs[0],fs[1]))
#     return fs
    
# def loc_metric(grads,nn=100,pd=phys_dist):
    
#     f=np.zeros((len(grads)))
#     for i in range (len(grads)):
#         inds=np.argpartition(pd[i], nn)[:nn]
        
#         d=np.linalg.norm(grads[i]-grads[inds],axis=1)
#         f[i]=np.mean(d)
#     return f 
    
    
    
def compute_average_distance_bt_pts(grads):
    
    inds,dists=uts.neighbors(grads,grads,10)
    
    return np.mean(np.mean(dists[:,1:]))