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
scrpath='/Users/patricktaylor/lifespan_analysis/scratch/'
axisnames=['SA','VS','MR']
lhp = '/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii' 
rhp = '/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii' 
        

def load_surf_objs():
    global lh 
    global rh 
    lh=bs.mesh.mesh_io.read_surface(lhp)
    rh=bs.mesh.mesh_io.read_surface(rhp)
    return 


def plot_surf(data,lh,rh=None,age=None,white=False,size=(800,200),title=None,crange=None,cmap='turbo',save=None,show=True,monthsurf=None,right=False,cbar=True,interactive=False):
    
    if len(data)==18463:
        data=uts.unmask_medial_wall(data,bcpmask)
    if monthsurf is not None:
        lh=bs.mesh.mesh_io.read_surface(f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_{(monthsurf)}Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii')
        if rh is not None:
            rh=bs.mesh.mesh_io.read_surface(f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_{(monthsurf)}Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii')
    if age is not None:
        if rh is None:
            lh=load_surface_atlas(age,white=white)[0]
        else:
            lh,rh=load_surface_atlas(age,white=white)
    if rh is not None:
        if not right:
            p = sp.Plot(lh,rh,size=size,zoom=1.2, layout='row')
            p.add_layer(data,color_range=crange,cmap=cmap,zero_transparent=False,cbar_label=title,cbar=cbar)
        if right:
            p = sp.Plot(rh,size=size,zoom=1.2, layout='row',cmap=cmap)
            p.add_layer(data[int(data.shape[0]/2):],color_range=crange,cmap='jet',zero_transparent=False,cbar_label=title,cbar=cbar)
    else:
        if size==(800,200):
            size=(400,200)
            
        p = sp.Plot(lh,size=size,zoom=1.2, layout='row')
        if len(data)==20484:
            p.add_layer(data[:int(data.shape[0]/2)],color_range=crange,cmap=cmap,zero_transparent=False,cbar_label=title,cbar=cbar)
        else:
            p.add_layer(data,color_range=crange,cmap=cmap,zero_transparent=False,cbar_label=title,cbar=cbar)
    #p.show(interactive=interactive)
    fig = p.build()
    #fig.title(title)
    if show:
        fig.show()
    if save is not None:
        fig.savefig(save)
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

def save_surface(filename, points, edges, labels=None, features=None, feature_names='feature', colors=None,to_hems=True):
    if to_hems:
        p = filename.rsplit('/',1)
        lhpath = p[0] + '/lh_' + p[1]
        rhpath = p[0] + '/rh_' + p[1]
        
        h = int(len(points)/2)
        lsc = points[:h]
        rsc = points[h:]
        lsi = si[:int(len(si)/2)]
        rsi = si[int(len(si)/2):] - len(lsc)
        if labels is not None:
            lhlabels = labels[:h]
            rhlabels = labels[h:]
        else:
            lhlabels = None
            rhlabels = None
        if features is not None:
            lfeatures = features[:h]
            rfeatures = features[h:]
        else:
            lfeatures = None
            rfeatures = None
        if colors is not None:
            lcol = colors[:h]
            rcol = colors[h:]
        else:
            lcol=None
            rcol = None
            
        save_surface(lhpath,lsc,lsi,lhlabels,lfeatures,feature_names,lcol,False)
        save_surface(rhpath,rsc,rsi,rhlabels,rfeatures,feature_names,rcol,False)
        
        return 
    # Create a polydata object
    mesh = vtk.vtkPolyData()

    # Set the points
    points_vtk = vtk.vtkPoints()
    for point in points:
        points_vtk.InsertNextPoint(point)
    mesh.SetPoints(points_vtk)

    # Set the polygons
    polys = vtk.vtkCellArray()
    for edge in edges:
        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(edge))
        for i, vertex in enumerate(edge):
            polygon.GetPointIds().SetId(i, vertex)
        polys.InsertNextCell(polygon)
    mesh.SetPolys(polys)

    # Set the colors if available
    if colors is not None:
        colors_vtk = vtk.vtkDoubleArray()
        colors_vtk.SetNumberOfComponents(3)
        colors_vtk.SetName('color')
        for color in colors:
            colors_vtk.InsertNextTuple(color)
        mesh.GetPointData().AddArray(colors_vtk)

    # Set the labels if available
    if labels is not None:
        labels_vtk = vtk.vtkStringArray()
        labels_vtk.SetName('label')
        for label in labels:
            labels_vtk.InsertNextValue(label)
        mesh.GetPointData().AddArray(labels_vtk)

    # Set the features if available
    if features is not None:
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        if len(features.shape) == 1:  # if only one feature, expand dimension
            features = np.expand_dims(features, axis=0)
        for f, name in zip(features, feature_names):
            feature_vtk = vtk.vtkDoubleArray()
            feature_vtk.SetName(name)
            for value in f:
                feature_vtk.InsertNextValue(value)
            mesh.GetPointData().AddArray(feature_vtk)

    # Write to file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.Write()

    return

# =============================================================================
# 
# def save_surface(filename,points,edges,feature=None,colors=None):
#     mesh = tvtk.PolyData(points=points, polys=edges)
#     if feature is not None:
#         mesh.point_data.scalars=feature
#     if colors is not None:
#         mesh.point_data.vectors=colors
#     write_data(mesh, filename)
#     return
# =============================================================================

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
    
def read_timeseries(file,keep_num= 59412, *args):
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
        time_series=np.array(time_series.dataobj)[:,:keep_num]
        return time_series.T
   

def mask_from_timeseries_zero(ts):
    
    mask = np.zeros(len(ts))
    
    zeroinds = np.where(ts[:,0]==0.0)[0]
    
    mask[zeroinds]=1
    return mask
     
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

def read_cifti_arr(file):
    data=nib.load(file)
    data=np.array(data.dataobj)
    return data.T
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

def save_gifti_surface(sc,si,fname,hcp=False,hem = 'lh'):
    if hem == 'lh':
        anat_structure = 'CortexLeft'
    if hem == 'rh':
        anat_structure = 'CortexRight'
    pmeta = {
    	'description': 'anything',      # brief info here
    	'GeometricType': 'Anatomical', 	# an actual surface; could be 'Inflated', 'Hull', etc
        'AnatomicalStructurePrimary': anat_structure, # the specific structure represented
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
    	intent='NIFTI_INTENT_POINTSET', 	                # represents a set of points
    	coordsys=pcoord,	                                # see above
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

def save_gifti_func(func,fname,hem='lh'):
    if hem == 'lh':
        anat_structure = 'CORTEX_LEFT'
    if hem == 'rh':
        anat_structure = 'CORTEX_RIGHT'
    pmeta = {
    	'description': 'anything',      # brief info here
    	'GeometricType': 'Anatomical', 	# an actual surface; could be 'Inflated', 'Hull', etc
        'AnatomicalStructurePrimary': anat_structure, # the specific structure represented
        'AnatomicalStructureSecondary': 'GrayWhite', # if the above field is not specific enough
    }
    
    pcoord = nib.gifti.GiftiCoordSystem(1,1)    # surface is in world mm coordinates
    plist = []
    for i in range (func.shape[1]):
        parray = nib.gifti.GiftiDataArray(func[:,i], 
        	intent='NIFTI_INTENT_POINTSET', 	                # represents a set of points
        	coordsys=pcoord,	                                # see above
        	datatype='NIFTI_TYPE_FLOAT32',                      # float type data 
        	meta=nib.gifti.GiftiMetaData.from_dict(pmeta)   # again, see above. 
        ) 
        plist.append(parray)
    
    # Finally, create the GiftiImage object and save
    
    gii = nib.gifti.GiftiImage(darrays=plist)
    nib.save(gii, fname)
    return 

def downsample_fmri_gifti(filename,savepath,hem,keep_num = 10242):
    f = nib.load(filename)
    func = np.array([d.data[:keep_num] for d in f.darrays]).T
    
    save_gifti_func(func,filename,hem)
    
    
    
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

def read_schaefer400_17net_parc(lhpath):
    
    rhpath = lhpath.replace('lh', 'rh')
    
    lhg = nib.load(lhpath)
    rhg = nib.load(rhpath)
    
    lhparc = gifti2scal(lhg)[:,0]
    rhparc = gifti2scal(rhg)[:,0]
    
    reg_parcs = [lhparc,rhparc]
    
    lhdict = lhg.labeltable.get_labels_as_dict()
    rhdict = rhg.labeltable.get_labels_as_dict()
    
    lhregions = []
    rhregions = []
    
    lhnetworks = []
    rhnetworks = []
    
    lhdict = [lhdict[i][14:] for i in range(1,201)]
    
    rhdict =  [rhdict[i][14:] for i in range(1,201)]
    
    lhdict = [[' medial wall']]+[lhdict[i].rsplit('_') for i in range (200)]
    rhdict = [[' medial wall']] + [rhdict[i].rsplit('_') for i in range (200)]
    
    lhnetworks = [lhdict[i][0] for i in range (201) ]
    rhnetworks = [rhdict[i][0] for i in range (201) ]
    
    for i in range (201):
        if len(lhdict[i]) == 3:
            lhregname = 'lh_'+lhdict[i][1] + '_' + lhdict[i][2]
        elif len(lhdict[i]) == 2:
            lhregname = 'lh_'+lhdict[i][0] + '_' + lhdict[i][1]
        elif len(lhdict[i]) == 1:
            lhregname = 'lh_' +lhdict[i][0]
        if len(rhdict[i]) == 3:
            rhregname = 'rh_'+rhdict[i][1] + '_' + rhdict[i][2]
        elif len(rhdict[i]) == 2:
            rhregname = 'rh_'+rhdict[i][0] + '_' + rhdict[i][1]
        elif len(rhdict[i]) == 1:
            rhregname = 'rh_' +rhdict[i][0]
        lhregions.append(lhregname)
        rhregions.append(rhregname)
    region_names = np.array(lhregions+rhregions)
    
    lhnetworks = np.array(lhnetworks)
    rhnetworks = np.array(rhnetworks)
    network_assignments = [lhnetworks,rhnetworks]
    
    networknames = np.unique(lhnetworks)
    
    lhnetparc = np.zeros(len(lhparc))
    rhnetparc = np.zeros(len(rhparc))
    
    for z,parc in enumerate([lhnetparc,rhnetparc]):
        
        for i,name in enumerate(networknames):
            
            parcinds = np.where(network_assignments[z] == name)[0]
            
            for pnum in parcinds:
                vertinds = np.where(reg_parcs[z] == pnum)[0]
                parc[vertinds] = i
    netparc = np.concatenate((lhnetparc,rhnetparc))
    netparclabs = np.empty(netparc.shape,dtype='<U12')
    for i in range (len(networknames)):
        inds = np.where(netparc == i)[0]
        netparclabs[inds] = networknames[i]
    regparclabs = np.empty(netparc.shape,dtype='<U12')
    regparc = np.concatenate((lhparc,rhparc+201))
    for i in range (len(region_names)):
        inds = np.where(regparc == i)[0]
        regparclabs[inds] = region_names[i]
    parc_dict= {}
    
    parc_dict['parc'] = regparc
    parc_dict['region_labels'] = regparclabs
    parc_dict['region_names'] = region_names
    parc_dict['net_parc'] = netparc
    parc_dict['net_labels'] = netparclabs
    parc_dict['net_names'] = networknames
    return parc_dict

def read_schaefer400_7net_parc(lhpath):
    
    rhpath = lhpath.replace('lh', 'rh')
    
    lhg = nib.load(lhpath)
    rhg = nib.load(rhpath)
    
    lhparc = gifti2scal(lhg)[:,0]
    rhparc = gifti2scal(rhg)[:,0]
    
    reg_parcs = [lhparc,rhparc]
    
    lhdict = lhg.labeltable.get_labels_as_dict()
    rhdict = rhg.labeltable.get_labels_as_dict()
    
    lhregions = []
    rhregions = []
    
    lhnetworks = []
    rhnetworks = []
    
    lhdict = [lhdict[i][13:] for i in range(1,201)]
    
    rhdict =  [rhdict[i][13:] for i in range(1,201)]
    
    lhdict = [[' medial wall']]+[lhdict[i].rsplit('_') for i in range (200)]
    rhdict = [[' medial wall']] + [rhdict[i].rsplit('_') for i in range (200)]
    
    lhnetworks = [lhdict[i][0] for i in range (201) ]
    rhnetworks = [rhdict[i][0] for i in range (201) ]
    
    for i in range (201):
        if len(lhdict[i]) == 3:
            lhregname = 'lh_'+lhdict[i][1] + '_' + lhdict[i][2]
        elif len(lhdict[i]) == 2:
            lhregname = 'lh_'+lhdict[i][0] + '_' + lhdict[i][1]
        elif len(lhdict[i]) == 1:
            lhregname = 'lh_' +lhdict[i][0]
        if len(rhdict[i]) == 3:
            rhregname = 'rh_'+rhdict[i][1] + '_' + rhdict[i][2]
        elif len(rhdict[i]) == 2:
            rhregname = 'rh_'+rhdict[i][0] + '_' + rhdict[i][1]
        elif len(rhdict[i]) == 1:
            rhregname = 'rh_' +rhdict[i][0]
        lhregions.append(lhregname)
        rhregions.append(rhregname)
    region_names = np.array(lhregions+rhregions)
    
    lhnetworks = np.array(lhnetworks)
    rhnetworks = np.array(rhnetworks)
    network_assignments = [lhnetworks,rhnetworks]
    
    networknames = np.unique(lhnetworks)
    
    lhnetparc = np.zeros(len(lhparc))
    rhnetparc = np.zeros(len(rhparc))
    
    for z,parc in enumerate([lhnetparc,rhnetparc]):
        
        for i,name in enumerate(networknames):
            
            parcinds = np.where(network_assignments[z] == name)[0]
            
            for pnum in parcinds:
                vertinds = np.where(reg_parcs[z] == pnum)[0]
                parc[vertinds] = i
    netparc = np.concatenate((lhnetparc,rhnetparc))
    netparclabs = np.empty(netparc.shape,dtype='<U12')
    for i in range (len(networknames)):
        inds = np.where(netparc == i)[0]
        netparclabs[inds] = networknames[i]
    regparclabs = np.empty(netparc.shape,dtype='<U12')
    regparc = np.concatenate((lhparc,rhparc+201))
    for i in range (len(region_names)):
        inds = np.where(regparc == i)[0]
        regparclabs[inds] = region_names[i]
    parc_dict= {}
    
    parc_dict['parc'] = regparc
    parc_dict['region_labels'] = regparclabs
    parc_dict['region_names'] = region_names
    parc_dict['net_parc'] = netparc
    parc_dict['net_labels'] = netparclabs
    parc_dict['net_names'] = networknames
    return parc_dict

def normalize_ts(ts):
    ts=ts.T
    nts=(ts - np.mean(ts, axis=0)) / np.std(ts, axis=0)
    return nts.T

def combine_pe(ts_lr, ts_rl):
    ts_lr_n = normalize_ts(ts_lr)
    ts_rl_n = normalize_ts(ts_rl)
    return np.hstack((ts_lr_n, ts_rl_n))


#from matrix_compute import construct_inter_hemi_matrix

        


    
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

sc,si=read_surface('/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii','/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii')

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

    
    
    

