#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:56:27 2023

@author: patricktaylor
"""
import numpy as np
import vtk
import meshio
from tvtk.api import tvtk, write_data
import brainspace as bs 

class Surface:
    def __init__(self, coordspath = '/Users/patricktaylor/Documents/lifespan_analysis/misc/sc.npy' , indicespath = '/Users/patricktaylor/Documents/lifespan_analysis/misc/si.npy'):
        self.coords = np.load(coordspath)
        self.indices = np.load(indicespath)
        self.type = 'smooth'
        
    def load_age(self, age, white = False):
        self.age = age
        def find_nearest_index(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
        months = np.load('/Users/patricktaylor/Documents/lifespan_analysis/misc/monthlist.npy')
        half=int(len(self.coords)/2)
        mn = months.astype('int32')
        yrs=mn/12
        ind=find_nearest_index(yrs,age)
        monthsurf=months[ind]
        lp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_{(monthsurf)}Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii'
        rp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_{(monthsurf)}Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii'
        if white:
            self.type = 'white'
            lp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Surface/Atlas_{monthsurf}Months_L.white.ver2.downsampled.L5.surf.gii'
            rp=f'/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Surface/Atlas_{monthsurf}Months_R.white.ver2.downsampled.L5.surf.gii'
        self.coords[:half] = bs.mesh.mesh_io.read_surface(lp).GetPoints()
        self.coords[half:] = bs.mesh.mesh_io.read_surface(rp).GetPoints()
    
    
        
    def save_vtk(self, path = '/Users/patricktaylor/Documents/lifespan_analysis/scratch/', fname = 'surf.vtk', tohems = True, feature = None, vectors = None, ev = False):
        
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

        def save_eigenvector_to_hems(path, fname, points,edges,vecs):
            half=int(len(points)/2)
            lhsc=points[:half]
            rhsc=points[half:]
            lhsi=edges[:int(len(edges)/2)]
            rhsi=edges[int(len(edges)/2):]-half
            lhvec=vecs[:half]
            rhvec=vecs[half:]
            save_eigenvector(path + 'lh' + fname, lhsc,lhsi, lhvec)
            save_eigenvector(path + 'rh' + fname, rhsc, rhsi, rhvec)
            return
    
        if not tohems:
            if not ev:
                mesh = tvtk.PolyData(points = self.coords, polys =self.indices)
                if feature is not None:
                    mesh.point_data.scalars = feature
                if vectors is not None:
                    mesh.point_data.vectors = vectors
                write_data(mesh, path+fname)
            else:
                save_eigenvector(path + fname, self.coords, self.indices, feature)
                
        else:
            if not ev:
                half=int(len(self.coords)/2)
                lhsc = self.coords[:half]
                rhsc = self.coords[half:]
                lhsi = self.indices[:int(len(self.indices)/2)]
                rhsi = self.indices[int(len(self.indices)/2):]-half
                lhvec = feature[:half]
                rhvec = feature[half:]
                
                hems = ['lh', 'rh']
                cds = [lhsc,rhsc]
                sis = [lhsi, rhsi]
                fts = [lhvec, rhvec]
                if vectors is not None:
                    cols = [vectors[:half], vectors[half:]]
                for i in range (2):
                    mesh = tvtk.PolyData(points = cds[i], polys = sis[i])
                    mesh.point_data.scalars = fts[i]
                    if vectors is not None:
                        mesh.point_data.vectors = cols[i]
                        
                    write_data(mesh, path + hems[i] + '_' + fname)
            else:
                save_eigenvector_to_hems(path, fname, self.coords, self.indices, feature)
        
            
            
                
                    
                
                
            
        
    