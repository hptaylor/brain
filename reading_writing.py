#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 14:25:23 2023

@author: patricktaylor
"""

import numpy as np
import vtk
import meshio
from tvtk.api import tvtk, write_data
import nibabel as nib
import surfplot as sp
import utility as uts 
import brainspace as bs 

#define globals 
scrpath='/Users/patricktaylor/Documents/lifespan_analysis/scratch/'
axisnames=['SA','VS','MR']
lhp = '/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii' 
rhp = '/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii' 
        

def load_surf_objs():
    global lh 
    global rh 
    lh=bs.mesh.mesh_io.read_surface(lhp)
    rh=bs.mesh.mesh_io.read_surface(rhp)
    return 




def plot_custom_colormap(vecs,col=None, age=None,save=None,return_colors=False,op=1,rot=None,rotcol=None,a=None,b=None,c=None,unmask=False):
    from utility_functions import embed_colormap_3D
    if col is None:
        if unmask:
            colors=embed_colormap_3D(uts.unmask_medial_wall_vecs(vecs)[:10242],unmask=False,op=op,age=age,rot=rot,rotcol=rotcol,a=a,b=b,c=c)
        else:
            colors=embed_colormap_3D(vecs[:10242],unmask=False,op=op,age=age,rot=rot,rotcol=rotcol,a=a,b=b,c=c)
        c=colors*255
    else:
        
        colors = col 
        if colors.shape[1]==3:
            colors = np.hstack((colors[:10242],np.ones((10242,1))))
        c = colors*255
    #c=np.append(colors,np.ones((10242,1)),1)*255
    from brainspace.plotting.colormaps import colormaps
    c=c.astype(np.uint8)
    colormaps['me']=c
    if age is not None:
        lh=load_surface_atlas(age)[0]
    else:
        lh = load_surface_atlas(25)[0]
    p=sp.Plot(lh,layout='row',size=(800,400),zoom=1.2)
    p.add_layer(np.arange(10242)+1,cmap='me',cbar=False)
    fig=p.build()
    if save is not None:
        fig.savefig(save)
    if return_colors:
        return colors

def save_parc_3d_cmap(path, sc, si, parc,grads,emb=False,saveone=False,both=False):
    
    colors = uts.cmap3d_bary(grads)
    
    u = np.unique(parc)
    
    parc_colors = np.zeros(colors.shape)
    if saveone:
        for i in u:
            parc_colors[:,:]=1
            inds= np.where(parc==i)[0]
            parc_colors[inds] = np.mean(colors[inds],axis=0)#+np.random.randint(-100,100)/1000
            if not emb:
                save_surface_to_hems(path[:-4]+f'{i}.vtk',sc,si,parc,color=parc_colors)
            else:
                save_surface(path[:-4]+f'{i}.vtk',grads,si,parc,colors=parc_colors)
             
        
    else:
        for i in u:
            inds = np.where(parc==i)[0]
            parc_colors[inds] = np.mean(colors[inds],axis=0)#+np.random.randint(-100,100)/1000
            
        if not emb and not both:
            save_surface_to_hems(path,sc,si,parc,color=parc_colors)
        else:
            save_surface(path % 'emb' ,grads,si,parc,colors=parc_colors)
        if both:
            save_surface_to_hems(path,sc,si,parc,color=parc_colors)
        return 

def save_within_net_parc_3d_cmap(path, sc, si, parclist, grads):
    colors = uts.cmap3d_bary(grads)
    for i in range (len(parclist)):
        
        u = np.unique(parclist[i])
        
        parc_colors = np.zeros(colors.shape)
        parc_colors[:,:]=1 
        for j in u[1:]: 
            
            inds = np.where(parclist[i]==j)[0]
            parc_colors[inds] = np.mean(colors[inds],axis=0)*(j+4)/(len(u)+4)+np.random.rand(3)/30*np.random.choice([-1,1])
        savepath = path[:-4] % 'emb' 
        save_surface(savepath + f'_{i}.vtk' ,grads,si,parclist[i],colors=parc_colors)
        savepath = path[:-4] + f'_{i}.vtk'
        save_surface_to_hems(savepath,sc,si,parclist[i],color=parc_colors)
    return 

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def load_surface_atlas(age,white=False,tonumpy=False):
    mn=months.astype('int32')
    yrs=mn/12
    ind=find_nearest_index(yrs,age)
    monthsurf=months[ind]
    lp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_{(monthsurf)}Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii'
    rp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_{(monthsurf)}Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii'
    if white:
        lp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Surface/Atlas_{monthsurf}Months_L.white.ver2.downsampled.L5.surf.gii'
        rp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Surface/Atlas_{monthsurf}Months_R.white.ver2.downsampled.L5.surf.gii'
    lh=bs.mesh.mesh_io.read_surface(lp)
    rh=bs.mesh.mesh_io.read_surface(rp)
    
    if not tonumpy:
        return lh,rh
    else:
        sc=np.zeros((20484,3))
        sc[:10242]=lh.GetPoints()
        sc[10242:]=rh.GetPoints()
        return sc 

def plot_grad_list_on_surf(gradlist,whichgrad,start,stop,surflist=None,savelist=None):
    
    for i in range (start,stop):
        if savelist is not None:
            plot_surf(gradlist[i][:,whichgrad],lhlist[i],rhlist[i],save=savelist[i])
        else:
            plot_surf(gradlist[i][:,whichgrad],lhlist[i],rhlist[i])
            
    return 


def save_surface(filename,points,edges,feature=None,colors=None):
    mesh = tvtk.PolyData(points=points, polys=edges)
    if feature is not None:
        mesh.point_data.scalars=feature
    if colors is not None:
        mesh.point_data.vectors=colors
    write_data(mesh, filename)
    return

def save_points(filename,points,feature=None,colors=None):
    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])
        
    if colors is not None:
        Colors = vtk.vtkUnsignedCharArray()
        Colors.SetNumberOfComponents(3)
        Colors.SetName('colors')
        for i in range (len(colors)):
            Colors.InsertNextTuple3(colors[i,0]*255,colors[i,1]*255,colors[i,2]*255)
    
        
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)
    if colors is not None:
        vpoly.GetPointData().SetVectors(Colors)
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(vpoly)
    writer.SetFileName(filename)
    writer.Write()
    return 

def save_surface_with_vector(filename,points,edges,feature=None):
    mesh = tvtk.PolyData(points=points, polys=edges)
    if feature is not None:
        mesh.point_data.vectors=feature
    write_data(mesh, filename)
    return

def save_surface_with_vector_at_cells(filename,points,edges,feature=None):
    mesh = tvtk.PolyData(points=points, polys=edges)
    if feature is not None:
        mesh.cell_data.vectors=feature
    write_data(mesh, filename)
    return

def save_surface_with_scalar_at_cells(filename,points,edges,feature=None):
    mesh = tvtk.PolyData(points=points, polys=edges)
    if feature is not None:
        mesh.cell_data.scalars=feature
    write_data(mesh, filename)
    return

def save_surface_to_hems(filename,points,edges,feature,color=None):
    half=int(len(points)/2)
    lhsc=points[:half]
    rhsc=points[half:]
    lhsi=edges[:int(len(edges)/2)]
    rhsi=edges[int(len(edges)/2):]-half
    lhvec=feature[:half]
    rhvec=feature[half:]
    if color is None:
        
        save_surface(filename %'lh',lhsc,lhsi,lhvec)
        save_surface(filename %'rh',rhsc,rhsi,rhvec)
    else:
        lc=color[:half]
        rc=color[half:]
        save_surface(filename %'lh',lhsc,lhsi,lhvec,lc)
        save_surface(filename %'rh',rhsc,rhsi,rhvec,rc)
        
    return


def save_eigenvector(filename,points,edges,vecs):
    Cells = {"triangle": edges}
    V={}
    
        
    if len(vecs.shape)==1:
        v=vecs
        vecs=np.zeros((len(v),2))
        vecs[:,0]=v
    for i in range (0,len(vecs[0,:])):
        V.update({"ev%d" % (i):vecs[:,i]})

    mesh=meshio.Mesh(points,Cells,V)
    meshio.write(filename,mesh)
    return  

def save_eigenvector_to_hems(filename,points,edges,vecs):
    half=int(len(points)/2)
    lhsc=points[:half]
    rhsc=points[half:]
    lhsi=edges[:int(len(edges)/2)]
    rhsi=edges[int(len(edges)/2):]-half
    lhvec=vecs[:half]
    rhvec=vecs[half:]
    save_eigenvector(filename %'lh',lhsc,lhsi,lhvec)
    save_eigenvector(filename %'rh',rhsc,rhsi,rhvec)
    return

def parc_feat_to_vertex(pf,parc):
    vf = np.zeros(len(parc))
    
    u = np.unique(parc)
    
    for i in range (len(u)):
        inds=np.where(parc==u[i])[0]
        vf[inds]=pf[i]
    return vf


def save_parc_feature_to_surf_vecs(filename,points,edges,vecs,parc,mask=False):
    h=int(len(points)/2)
    hp = int(len(vecs)/2)
    lvertvecs=np.zeros((h,np.shape(vecs)[1]))
    rvertvecs=np.zeros((h,np.shape(vecs)[1]))
    lp = parc[:h]
    rp= parc[h:]
    
    for i in range(np.shape(vecs)[1]):
        lvertvecs[:,i]=parc_feat_to_vertex(vecs[:hp,i],lp)
        rvertvecs[:,i]=parc_feat_to_vertex(vecs[hp:,i],rp)
    
    vertvecs = np.vstack((lvertvecs,rvertvecs))
    if mask:
        vertvecs=uts.unmask_medial_wall_vecs(vertvecs,mask)
    save_eigenvector(filename,points,edges,vertvecs)
    return


def read_vtk_feature(filename,featurename):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.ReadAllScalarsOn()
    reader.Update()
    data = reader.GetOutput()
    scalars=data.GetPointData()
    Scalars=scalars.GetArray(featurename)
    feature=np.zeros((data.GetNumberOfPoints(),))
    for i in range (data.GetNumberOfPoints()):
        feature[i]=Scalars.GetValue(i)
    return feature

def read_vtk_feature_both_hem(lfile,rfile,featurename):
    '''
    

    Parameters
    ----------
    lfile : TYPE
        DESCRIPTION.
    rfile : TYPE
        DESCRIPTION.
    featurename : TYPE
        DESCRIPTION.

    Returns
    -------
    feature : TYPE
        DESCRIPTION.

    '''
    l=read_vtk_feature(lfile,featurename)
    r=read_vtk_feature(rfile,featurename)
    feature=np.hstack((l,r))
    return feature 
    

def read_vtk_surface(filename):
    #reads a vtk surface mesh and returns the coordinates of vertices (nvert,3), and the connections definiing the mesh (ncon,3) as numpy arrays
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    CellArray = data.GetPolys()
    Polygons = CellArray.GetData()
    edges=np.zeros((CellArray.GetNumberOfCells(),3))
    for i in range (0, CellArray.GetNumberOfCells()):
        edges[i,:]=[Polygons.GetValue(j) for j in range (i*4+1,i*4+4)]
    points=np.zeros((data.GetNumberOfPoints(),3))
    for i in range(data.GetNumberOfPoints()):
            points[i,:] = data.GetPoint(i)
    return points, edges

def read_vtk_surface_both_hem(Lfile,Rfile):
    #takes full path+file name of two hemispheres of a surface, loads them, and combines the coordinates and connections from both into 2 full surface arrays
    lhc,lhi=read_vtk_surface(Lfile)
    rhc,rhi=read_vtk_surface(Rfile)
    coords,si=combine_hemis(lhc,rhc,lhi,rhi)
    return coords,si

def read_gifti_surface(filename):
    data=nib.load(filename)
    a1=data.darrays[0].data
    a2=data.darrays[1].data
    if a1.dtype=='int32':
        edges=a1
        points=a2
    if a2.dtype=='int32':
        edges=a2
        points=a1
    return points,edges
    
def save_ev_timeseries(positions,scalars,savepath=scrpath+'timeseries/%s_an.vtk'):
    
    for i in range (positions.shape[0]):
        if len(scalars.shape)>1:
            save_eigenvector(savepath % i, positions[i],si,scalars[i])
        else:
            save_eigenvector(savepath % i, positions[i],si,scalars)
    return 
def read_gifti_surface_both_hem(Lfile,Rfile,hcp=False):
    lhc,lhi=read_gifti_surface(Lfile)
    rhc,rhi=read_gifti_surface(Rfile)
    points,edges=combine_hemis(lhc,rhc,lhi,rhi)
    return points,edges

def read_functional_timeseries(lhfunc,rhfunc,v2s=True):
    l = nib.load(lhfunc).darrays
    r = nib.load(rhfunc).darrays
    timeseries = np.zeros((2*len(l[0].data), len(r)))
    if v2s:
        timeseries=np.concatenate((np.array(l[0].data),np.array(r[0].data)))
        return timeseries
    else:
        for i in range(len(l)):
            lt = np.array(l[i].data)
            rt = np.array(r[i].data)
            tp = np.concatenate((lt, rt))
            timeseries[:, i] = tp
        return timeseries
    
def read_timeseries(file,*args):
    '''loads N_vert x N_timepoints matrix of surface mapped fmri 
    values for .gii or .nii (cifti). For .gii, both left and right
    hemisphere paths must be provided in that order. For cifti 
    grayordinates,will return masked timeseries 59k x N_timepoints.'''
    if file.endswith('.gii'):
        l = nib.load(file).darrays
        r = nib.load(args[0]).darrays
        timeseries = np.zeros((2*len(l[0].data), len(r)))
        if len(l[0].data)<len(l):
            
            timeseries=np.concatenate((np.array(l[0].data),np.array(r[0].data)))
            return timeseries
        else:
            for i in range(len(l)):
                lt = np.array(l[i].data)
                rt = np.array(r[i].data)
                tp = np.concatenate((lt, rt))
                timeseries[:, i] = tp
            return timeseries
            
    if file.endswith('.nii') or file.endswith('.nii.gz'):
        time_series=nib.load(file)
        time_series=np.array(time_series.dataobj)[:,:59412]
        return time_series.T
        
def read_and_concatenate_timeseries(lh1,rh1,lh2,rh2):
    ts1=read_timeseries(lh1,rh1)
    m1=mask_from_timeseries_zero(ts1)
    ts2=read_timeseries(lh2,rh2)
    m2=mask_from_timeseries_zero(ts2)
    mask=m1+m2
    mask[mask==2]=1
    ts1=uts.mask_medial_wall_vecs(ts1,mask)
    ts2=uts.mask_medial_wall_vecs(ts2,mask)
    ts1=normalize_ts(ts1)
    ts2=normalize_ts(ts2)
    return np.hstack((ts1,ts2)),mask

def read_cifti_timeseries_masked(file):
    time_series=nib.load(file)
    time_series=np.array(time_series.dataobj)[:,:59412]
    return time_series.T


def read_streamline_endpoints(filename):
    #reads endpoint locations of vtk file containing only the endpoints of a tractogram. returns numpy array of size (nEndpoints,3). 
    #endpoint 0 and endpoint 1 correspond to the same fiber. endpoint 2, endpoint 3 correspond to the same fiber... etc 
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
    points=np.zeros((data.GetNumberOfPoints(),3))
    for i in range(data.GetNumberOfPoints()):
            points[i,:] = data.GetPoint(i)
    return points

def combine_hemis(lhc,rhc,lhi,rhi):
    #concatenates surface coordinates of two hemispheres and creates connectivity array for full surface
    coords=np.vstack((lhc,rhc))
    si=np.vstack((lhi,rhi+len(lhc)))
    return coords, si

def split_vtk_feature_to_hems(path,fname):
    sc,si=read_vtk_surface(path+fname)
    feature=read_vtk_feature(path+fname,'scalars')
    lsc=sc[:int(len(sc)/2)]
    rsc=sc[int(len(sc)/2):]
    lsi=si[:int(len(si)/2)]
    rsi=si[int(len(si)/2):]-int(len(sc)/2)
    save_surface(path+'lh_'+fname,lsc,lsi,feature[:int(len(sc)/2)])
    save_surface(path+'rh_'+fname,rsc,rsi,feature[int(len(sc)/2):])
    return 

def save_gifti_surface(sc,si,fname,hcp=False):
    pmeta = {
        'description': 'anything',      # brief info here
        'GeometricType': 'Anatomical',  # an actual surface; could be 'Inflated', 'Hull', etc
        'AnatomicalStructurePrimary': 'CortexLeft', # the specific structure represented
        'AnatomicalStructureSecondary': 'GrayWhite', # if the above field is not specific enough
    }
    
    # Prepare the coordinate system. 
    # The general form is GiftiCoordSystem(X,Y,A) where X is the intial space, 
    # Y a destination space, and A the affine transformation between X and Y. 
    # See GIFTI spec, page 9
    # By default A = I(4), so X and Y can have the same values, as follows:
    # 0: NIFTI_XFORM_UNKNOWN
    # 1: NIFTI_XFORM_SCANNER_ANAT
    # 2: NIFTI_XFORM_ALIGNED_ANAT
    # 3: NIFTI_XFORM_TALAIRACH
    # 4: NIFTI_XFORM_MNI_152
    pcoord = nib.gifti.GiftiCoordSystem(1,1)    # surface is in world mm coordinates
    
    parray = nib.gifti.GiftiDataArray(sc, 
        intent='NIFTI_INTENT_POINTSET',                     # represents a set of points
        coordsys=pcoord,                                    # see above
        datatype='NIFTI_TYPE_FLOAT32',                      # float type data 
        meta=nib.gifti.GiftiMetaData.from_dict(pmeta)   # again, see above. 
    ) 
    
    # The triangles array
    # Triangle metadata dict
    tmeta = {
        'TopologicalType': 'Closed',    # a closed surface, could be 'Open', see spec
        'Description': 'anything',      # brief info here
    }
    
    # Triangle coordinate system. 
    # As the triangles are not point data, we put it into NIFTI_XFORM_UNKNOWN space
    # for consistency with the behaviour of nibabel.load() on a GIFTI surface
    tcoord = nib.gifti.GiftiCoordSystem(0,0)
    
    tarray = nib.gifti.GiftiDataArray(si, 
        intent='NIFTI_INTENT_TRIANGLE',                     # triangle surface elements
        coordsys=tcoord,                                    # see above
        datatype='NIFTI_TYPE_INT32',                        # integer indices
        meta=nib.gifti.GiftiMetaData.from_dict(tmeta)   # see above
    )
    
    # Finally, create the GiftiImage object and save
    if hcp:
        gii = nib.gifti.GiftiImage(darrays=[parray, tarray])
    else:
        gii = nib.gifti.GiftiImage(darrays=[tarray, parray])
    nib.save(gii, fname)
    return 
    
def save_gifti_scalar(scalar,fname):
    pmeta = {
        'description': 'anything',      # brief info here
        'GeometricType': 'Anatomical',  # an actual surface; could be 'Inflated', 'Hull', etc
        'AnatomicalStructurePrimary': 'CortexLeft', # the specific structure represented
        'AnatomicalStructureSecondary': 'GrayWhite', # if the above field is not specific enough
    }
    
    pcoord = nib.gifti.GiftiCoordSystem(1,1)    # surface is in world mm coordinates
    if len(scalar.shape)==2:
        scalar=scalar[:,0]
    parray = nib.gifti.GiftiDataArray(scalar, 
        intent='NIFTI_INTENT_POINTSET',                     # represents a set of points
        coordsys=pcoord,                                    # see above
        datatype='NIFTI_TYPE_FLOAT32',                      # float type data 
        meta=nib.gifti.GiftiMetaData.from_dict(pmeta)   # again, see above. 
    ) 
    
    # Finally, create the GiftiImage object and save
    
    gii = nib.gifti.GiftiImage(darrays=[parray])
    nib.save(gii, fname)
    return 

def gifti_to_scalar(L,R): 
    l=L.darrays
    r=R.darrays
    La=np.array([l[0].data]).T
    Ra=np.array([r[0].data]).T
    scal=np.vstack((La,Ra))
    return scal

def gifti2scal(L): 
    l=L.darrays

    La=np.array([l[0].data]).T
    
    return La

def read_gifti_feature(fname):
    f=nib.load(fname)
    f=gifti2scal(f)
    return f

def read_gifti_feature_both_hem(lfname,rfname):
    L=nib.load(lfname)
    R=nib.load(rfname)
    featurevec=gifti_to_scalar(L,R)
    return featurevec

def normalize_ts(ts):
    ts=ts.T
    nts=(ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)
    return nts.T

def combine_pe(ts_lr, ts_rl):
    ts_lr_n = normalize_ts(ts_lr)
    ts_rl_n = normalize_ts(ts_rl)
    return np.hstack((ts_lr_n, ts_rl_n))


#from matrix_compute import construct_inter_hemi_matrix
#from scipy import sparse
def mask_from_parc_bcp(parc):
    if len(parc.shape)==2:
        parc=parc[:,0]
    mask=np.zeros(len(parc))
    mask[parc==-100]=1
    return mask

def generate_mask_from_parc_bcp(lhparc,rhparc):
    parc=read_gifti_feature_both_hem(lhparc,rhparc).astype('int32')
    inds1=np.where(parc==1639705)[0]
    inds3=np.where(parc==1639704)[0]
    inds2=np.where(parc==3294840)[0]
    inds4=np.where(parc==3294839)[0]
    mask=np.zeros(len(parc))
    mask[inds1]=1
    mask[inds2]=1
    mask[inds3]=1
    mask[inds4]=1
    return mask 

#bcpmask=generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_lh.desikan.downsampled.L5.func.gii','/Users/patricktaylor/Documents/lifespan_analysis/misc/Atlas_12Months_rh.desikan.downsampled.L5.func.gii')

#def get_bcp_mask():
#     bcpmask=generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/BCP_Bin_SC/Atlas_12Months_lh.parcellation.downsampled.func.gii','/Users/patricktaylor/Documents/BCP_Bin_SC/Atlas_12Months_rh.parcellation.downsampled.func.gii')
#     return bcpmask
 
def generate_mask_from_parc_hcp(lhparc,rhparc):
    parc=read_gifti_feature_both_hem(lhparc,rhparc).astype('int32')
    inds1=np.where(parc==-100)[0]
    inds2=np.where(parc==3250)[0]
    mask=np.zeros(len(parc))
    mask[inds1]=1
    mask[inds2]=1
    return mask 

hcpmask=generate_mask_from_parc_hcp('/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.DesikanParc.ver2.L5.func.gii','/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.DesikanParc.ver2.L5.func.gii')


def get_hcp_mask():
    mask = generate_mask_from_parc_hcp('/Volumes/OrangeDrive/HCP_orig_Structural/105923/MNINonLinear/fsaverage_LR32k/105923.L.aparc.32k_fs_LR.label.gii','/Volumes/OrangeDrive/HCP_orig_Structural/105923/MNINonLinear/fsaverage_LR32k/105923.R.aparc.32k_fs_LR.label.gii')
    return mask

def mask_from_timeseries_zero(ts):
    
    mask = np.zeros(len(ts))
    
    zeroinds = np.where(ts[:,0]==0.0)[0]
    
    mask[zeroinds]=1
    return mask


    

def read_microstructure_vecs(path,types=['SMSI_ECVF','SMSI_ICVF','SMSI_IVF'],myelin=None):
    mslist=[]
    for Type in types:
        lpath = path % ('lh', Type)
        rpath = path % ('rh', Type)
        vec = read_gifti_feature_both_hem(lpath, rpath)
        vec= np.nan_to_num(vec)
        mslist.append(vec)
    if myelin is not None:
        my=read_gifti_feature_both_hem(myelin % 'lh',myelin %'rh')
        mslist.append(my)
    msvecs = np.hstack(tuple(mslist))
    return msvecs
    
def save_parcellation_with_color_map(filename,refPoints,savepoints,edges,Labels,indices,distances,blackout,tol=3.5):
    '''
    Saves whole-surface color-mapped parcellation 'Labels' vtk file on surface given by savepoints and edges. 
    Uses EucDist, Cmap, triangle_area, and get_vertex_color
    '''
    colors=color_map(Labels,refPoints)
    if blackout:
        for i in range (len(distances)):
            if distances[i]<tol:
                colors[indices[i]]=np.array([0,0,0])
    mesh = tvtk.PolyData(points=savepoints, polys=edges)
    Colors=tvtk.UnsignedCharArray()
    Colors.from_array(colors)
    mesh.point_data.scalars = Colors
    mesh.point_data.scalars.name = 'colors'
    labs=tvtk.UnsignedCharArray()
    labs.from_array(Labels)
    mesh.point_data.add_array(labs)
    write_data(mesh,filename)
    return

#from utility_functions import neighbors
from sklearn.preprocessing import label_binarize

def save_colormapped_parcellation_to_hems(path,filename,refPoints,savepoints,edges,Labels,blackout=False):
    '''
    Saves left and right hemisphere of color-mapped parcellation 'Labels' vtk file on surface given by savepoints and edges. 
    Uses EucDist, Cmap, triangle_area, and get_vertex_color
    '''
    #colors=color_map(Labels,refPoints)
    half=int(len(Labels)/2)
    li,ld=neighbors(refPoints[:half],refPoints[half:],1)
    ri,rd=neighbors(refPoints[half:],refPoints[:half],1)
    H=int(len(edges)/2)
    #c1=colors[:half]
    #c2=colors[half:]
    p1=refPoints[:half]
    p2=refPoints[half:]
    k1=savepoints[:half]
    k2=savepoints[half:]
    Filename=path+'L_'+filename
    save_parcellation_with_color_map(Filename,p1,k1,edges[:H],Labels[:half],li,ld,blackout)
    Filename=path+'R_'+filename
    save_parcellation_with_color_map(Filename,p2,k2,edges[:H],Labels[half:],ri,rd,blackout)

    return 

def eucdist(u,v):
    dist=((u[0]-v[0])**2+(u[1]-v[1])**2+(u[2]-v[2])**2)**.5
    return dist
    

def color_map(Labs,Asc):
    half=int(len(Asc)/2)
    x=np.arange(0,max(Labs)+1)
    #color map can be altered by varying the location of the "reference locations", given by variables Front, Par, Temp
    Front=np.array([76,135,174])
    Par=np.array([77,46,216])
    Temp=np.array([81,1,139])
    Bin=label_binarize(Labs,x)
    colors=np.zeros((half*2,3))
    for i in range (len(Bin[0,:])):
        nz=np.nonzero(Bin[:,i])
        meanPos=(np.mean(Asc[nz,:],1)).T
        Fd=eucdist(meanPos,Front)
        Pd=eucdist(meanPos,Par)
        Td=eucdist(meanPos,Temp)
        l=[Fd,Pd,Td]
        f2p=eucdist(Front,Par)
        p2t=eucdist(Par,Temp)
        f2t=eucdist(Front,Temp)
        r,g,b=get_vertex_color(f2p,p2t,f2t,Fd,Pd,Td)
        colors[nz,0]=r
        colors[nz,1]=g
        colors[nz,2]=b
    return colors
        
def triangle_area(len1, len2, len3):
    p = (len1 + len2 + len3)/2
    area = (p*(p-len1)*(p-len2)*(p-len3))**(0.5)
    return area

def get_vertex_color(f2p, p2t, f2t, v2f, v2p, v2t): #f2p: frontal to parietal distance, scalar
                                                    #p2t: parietal to temporal distance, scalar
                                                    #f2t: frontal to temporal distance, scalar
                                                    #v2f: current vertex to frontal distance, scalar
                                                    #v2p: current vertex to parietal distance, scalar
                                                    #v2f: current vertex to temporal distance, scalar
    area_whole = triangle_area(f2p,p2t,f2t)
    area_f = triangle_area(p2t, v2p, v2t)
    area_p = triangle_area(f2t, v2f, v2t)
    area_t = triangle_area(f2p, v2f, v2p)
    
    red = min(area_f/area_whole,1)*255
    green = min(area_p/area_whole,1)*255
    blue = min(area_t/area_whole,1)*255
    
    return red,green,blue

#import utility_functions as uts

def save_evec_to_vtk(filename,agerange,atlasmonth):
    valsvecs=np.load(filename,allow_pickle=True)
    vals=valsvecs[0]
    vecs=valsvecs[1]
    mask=generate_mask_from_parc_bcp('/Users/patricktaylor/Documents/bcp_cosine_FC_vecs/Atlas_12Months_lh.parcellation.downsampled.func.gii','/Users/patricktaylor/Documents/bcp_cosine_FC_vecs/Atlas_12Months_rh.parcellation.downsampled.func.gii')
    unmaskvecs=uts.unmask_medial_wall_vecs(vecs,mask)
    
    sc,si=read_vtk_surface_both_hem('/Users/patricktaylor/Documents/bcp_cosine_FC_vecs/Atlas_%sMonths_lh.InflatedSurf.nifti.newscans.downsampled.vtk' % atlasmonth, '/Users/patricktaylor/Documents/bcp_cosine_FC_vecs/Atlas_%sMonths_rh.InflatedSurf.nifti.newscans.downsampled.vtk' % atlasmonth)
    name1='/Users/patricktaylor/Documents/bcp_cosine_FC_vecs/%s.'
    name2='%s_cosine_vecs.vtk' % agerange
    save_eigenvector_to_hems(name1+name2 ,sc,si,unmaskvecs)
    return

#from pymatreader import read_mat 
from scipy import sparse

import h5py 

def load_matlab_mat(filename):
    hf=h5py.File(filename,'r')
    d=hf.get('connectivity')
    r=d[0]-1
    c=d[1]-1
    v=d[2]
    
    m=sparse.csr_matrix((v,(r,c)),shape=(81924,81924))
    
    return m 
    

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir = np.asarray(ds['ir'])
            jc = np.asarray(ds['jc'])
            out = sparse.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out 


'''


def load_matlab_sparse_matrix(filename):
    dictionary = read_mat(filename)
    
    d=dictionary['connectivity']['data']
    r=dictionary['connectivity']['ir']
    c=dictionary['connectivity']['jc']
    m=sparse.csc_matrix((d,(r,c)),shape=(81924,81924))
    
    return m 

'''

def load_hcp_most_prevalent_harmonics():
    vecdic={}
    groups=['40','80sec','2min','3.5min','retest_40','retest_80sec','retest_2min','retest_3.5min']
    
    for g in groups:
        
        vecdic[g]=np.load(f'/Users/patricktaylor/Documents/HCP_func_gradient/results/{g}_group_pca_harmonics.npy')
    
    return vecdic


def load_lifespan_most_prevalent_harmonics():
    vecdic={}
    groups=['/Users/patricktaylor/Documents/HCPD/results/6Y_most_prevalent_harmonics.npy','/Users/patricktaylor/Documents/HCPD/results/7Y_most_prevalent_harmonics.npy','/Users/patricktaylor/Documents/HCPD/results/8Y_most_prevalent_harmonics.npy','/Users/patricktaylor/Documents/HCPD/results/12Y_most_prevalent_harmonics.npy','/Users/patricktaylor/Documents/HCPD/results/16Y_most_prevalent_harmonics.npy','/Users/patricktaylor/Documents/HCPD/results/20Y_most_prevalent_harmonics.npy','/Users/patricktaylor/Documents/HCPA/results/40Y_2min_window_evecs.npy','/Users/patricktaylor/Documents/HCPA/results/60Y_2min_window_evecs.npy','/Users/patricktaylor/Documents/HCPA/results/80Y_2min_window_evecs.npy','/Users/patricktaylor/Documents/HCPA/results/100Y_2min_window_evecs.npy']
    names=['6Y','7Y','8Y','12Y','16Y','20Y','40Y','60Y','80Y','100Y']
    for i in range(len(groups)):
        
        vecdic[names[i]]=np.load(groups[i])
    
    return vecdic
    
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres
import brainspace as bs

def plot_surface_brainspace(feature,mask,unmask=True,filename=None,hcpd=True,bcp=False,veryinflated=False):
    if hcpd:
        surf_lh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii')
        surf_rh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii')
            
    if bcp:
        surf_lh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/Documents/BCP_preprocessed_fMRI/newfiles/Atlas_12Months_L.veryinflated.white.ver2.L5.surf.gii')
        surf_rh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/Documents/BCP_preprocessed_fMRI/newfiles/Atlas_12Months_R.veryinflated.white.ver2.L5.surf.gii')
    if veryinflated:
        surf_lh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/Documents/lifespan_analysis/105923.L.very_inflated_MSMAll.32k_fs_LR.surf.gii')
        surf_rh=bs.mesh.mesh_io.read_surface('/Users/patricktaylor/Documents/lifespan_analysis/105923.R.very_inflated_MSMAll.32k_fs_LR.surf.gii')
    else:
        surf_lh, surf_rh = load_conte69()
    
    if feature.shape[1]>1:
        if unmask:
            feature=uts.unmask_medial_wall_vecs(feature,mask)
        feature=[feature[:,i] for i in range (feature.shape[1])]
    
    else:
        if unmask:
            feature=uts.unmask_medial_wall(feature,mask)
    
    if filename:
        screenshot=True
        scale=1
    else:
        screenshot=False
        scale=1
    if screenshot:
        transparent_bg=True
    else:
        transparent_bg=False
    
    plot_hemispheres(surf_lh,surf_rh,array_name=feature, cmap='jet',color_bar=True ,screenshot=screenshot,filename=filename,transparent_bg=transparent_bg,scale=scale)
    
    return
        
        


    
def read_surface(surfpath,*args):
    '''loads in the coordinates and polygon information for surface mesh of type .vtk or .gii . 
    If two paths are given, it will treat these as left and right hemispheres and return the 
    concatenated coordinates and polygons.'''
    
    if len(args)==0:

        if surfpath.endswith('.gii'):
            data=nib.load(surfpath)
            a1=data.darrays[0].data
            a2=data.darrays[1].data

            
            if a1.dtype=='int32':
                edges=a1
                points=a2
            if a2.dtype=='int32':
                edges=a2
                points=a1
            return points,edges

        if surfpath.endswith('.vtk'):
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(surfpath)
            reader.Update()
            data = reader.GetOutput()
            CellArray = data.GetPolys()
            Polygons = CellArray.GetData()
            edges=np.zeros((CellArray.GetNumberOfCells(),3))
            for i in range (0, CellArray.GetNumberOfCells()):
                edges[i,:]=[Polygons.GetValue(j) for j in range (i*4+1,i*4+4)]
            points=np.zeros((data.GetNumberOfPoints(),3))
            for i in range(data.GetNumberOfPoints()):
                points[i,:] = data.GetPoint(i)
            return points, edges

    elif len(args)==1:
        if surfpath.endswith('.gii'):

            ldata=nib.load(surfpath)
            a1=ldata.darrays[0].data
            a2=ldata.darrays[1].data
            if a1.dtype=='int32':
                ledges=a1
                lpoints=a2
            if a2.dtype=='int32':
                ledges=a2
                lpoints=a1

            rdata=nib.load(args[0])
            a1=rdata.darrays[0].data
            a2=rdata.darrays[1].data
            if a1.dtype=='int32':
                redges=a1
                rpoints=a2
            if a2.dtype=='int32':
                redges=a2
                rpoints=a1

            points=np.vstack((lpoints,rpoints))
            edges=np.vstack((ledges,redges+len(lpoints)))
            return points,edges

        if surfpath.endswith('.vtk'):
            reader = vtk.vtkDataSetReader()
            reader.SetFileName(surfpath)
            reader.Update()
            data = reader.GetOutput()
            CellArray = data.GetPolys()
            Polygons = CellArray.GetData()
            ledges=np.zeros((CellArray.GetNumberOfCells(),3))
            for i in range (0, CellArray.GetNumberOfCells()):
                ledges[i,:]=[Polygons.GetValue(j) for j in range (i*4+1,i*4+4)]
            lpoints=np.zeros((data.GetNumberOfPoints(),3))
            for i in range(data.GetNumberOfPoints()):
                lpoints[i,:] = data.GetPoint(i)

            reader = vtk.vtkDataSetReader()
            reader.SetFileName(args[0])
            reader.Update()
            data = reader.GetOutput()
            CellArray = data.GetPolys()
            Polygons = CellArray.GetData()
            redges=np.zeros((CellArray.GetNumberOfCells(),3))
            for i in range (0, CellArray.GetNumberOfCells()):
                redges[i,:]=[Polygons.GetValue(j) for j in range (i*4+1,i*4+4)]
            rpoints=np.zeros((data.GetNumberOfPoints(),3))
            for i in range(data.GetNumberOfPoints()):
                rpoints[i,:] = data.GetPoint(i)
                    
            points=np.vstack((lpoints,rpoints))
            edges=np.vstack((ledges,redges+len(lpoints)))

            return points, edges

sc,si=read_surface('/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii','/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii')

def downsample_scalar_sahar(lhpath,rhpath,save=None):
    
    l=read_gifti_feature(lhpath)
    r=read_gifti_feature(rhpath)
    
    if save is not None:
        save_gifti_scalar(l[:10242],save % 'L')
        save_gifti_scalar(r[:10242],save % 'R')
    else:
        return l[:10242],r[:10242]
    
def downsample_surface_sahar(lhpath,rhpath,save=None):
    
    l,sil=read_gifti_surface(lhpath)
    r,ril=read_gifti_surface(rhpath)
    
    if save is not None:
        save_gifti_surface(l[:10242],si[:int(len(si)/2)],save % 'L')
        save_gifti_surface(r[:10242],si[int(len(si)/2):]-10242,save % 'R')
    else:
        return l[:10242],r[:10242]

    
    
    

