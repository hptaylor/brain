#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:51:44 2023

@author: patricktaylor
"""
import reading_writing as rw 
import utility as uts 
import brainspace as bs 
import surfplot as sp 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

scrpath='/Users/patricktaylor/Documents/lifespan_analysis/scratch/'
axisnames=['SA','VS','MR']
lhp = '/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii' 
rhp = '/Users/patricktaylor/Documents/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii' 
        



def splot(data, cmap='turbo', **kwargs):
    # size=(800,200), age=None,white=False, ,crange=None, title=None, save=None,show=True,monthsurf=None,right=False,cbar=True,interactive=False,
    lh=bs.mesh.mesh_io.read_surface(lhp)
    rh=bs.mesh.mesh_io.read_surface(rhp)
    if 'age' in kwargs:
            lh, rh = rw.load_surface_atlas(kwargs['age'], white = kwargs['white'])
            
    if 'rh' in kwargs:
        
        p = sp.Plot(lh, rh, layout = 'row', zoom = 1.2, size = (800,200))
    
    else: 
        data = data[:int(len(data)/2)]
        p = sp.Plot(lh, layout = 'row', zoom = 1.2, size = (400,200))
        
    p.add_layer(data, cmap = cmap)

    fig = p.build()
    
    if 'title' in kwargs:
        fig.suptitle(kwargs['title'])
        
    if 'returnfig' in kwargs:
        return fig 

def embed_plot_all_vertices_histogram(vecs,ginds,title, columns = None ,save=None):
    if columns is None:
        columns = ginds
    df=pd.DataFrame(vecs[:,ginds],columns=columns)
    
    p=sns.jointplot(data=df,x=columns[1],y=columns[0],s=5,alpha=0.8)
    p.fig.suptitle(title)
    
    p.fig.tight_layout()

    plt.xlabel(columns[1])
    plt.ylabel(columns[0])

    if save is not None:
        plt.savefig(save)
    return 