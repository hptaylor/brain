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
import matplotlib.cm as cm
import numpy as np 
from cycler import cycler
import matplotlib.colors as mcolors

scrpath='/Users/patricktaylor/lifespan_analysis/scratch/'
axisnames=['SA','VS','MR']
lhp = '/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii' 
rhp = '/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii' 
        



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

def plot_eigenvalues_by_age(eigenvalues, ages,num_evals = None, ylabel = 'Eigenvalue',point = False,cmap = 'viridis'):
    n_sub = len(eigenvalues)
    if num_evals is None:
        num_evals = len(eigenvalues[0])
    fig, ax = plt.subplots()
    colormap = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(ages), vmax=max(ages)))
    colormap.set_array([])
    for i in range(n_sub):
        if not point:
            ax.plot(eigenvalues[i][:num_evals], color=colormap.to_rgba(ages[i]),linewidth=0.5)
        else:
            ax.plot(eigenvalues[i][:num_evals], 'o', color=colormap.to_rgba(ages[i]),ms = 0.3)
    ax.set_xlabel('gradient index')
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    cbar = plt.colorbar(colormap)
    cbar.ax.set_ylabel('Age')
    plt.show()

def plot_kde_by_age(data_list, age_list):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('inferno')
    age_norm = plt.Normalize(min(age_list), max(age_list))

    for i in range(len(data_list)):
        data = data_list[i]
        age = age_list[i]
        color = cmap(age_norm(age))
        sns.kdeplot(data, ax=ax, color=color)

    ax.set_xlabel('Gradient Value')
    ax.set_ylabel('Kernel Density Estimate')
    ax.set_title('Distribution of Values by Age')
    ax.legend()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=age_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
   

    plt.show()

def plot_metric_vs_age_log_scale_lifespan(ages, metric,metriclabel = 'metric'):
    fig, ax = plt.subplots()
    ax.scatter(np.log2(ages + 1), metric, s= 0.5)
    ax.set_xticks([0,1, 2, 3, 4, 5,6])
    ax.set_xticklabels(['0', '2', '4', '8', '16','32','64'], 
                   fontsize=18)
    ax.set_xlabel('age (years)')
    ax.set_ylabel(metriclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    return 


def plot_metric_vs_age_log(ages, metric, metriclabel = 'metric'):
    fig, ax = plt.subplots()
    ax.scatter(np.log2(ages + 1), metric, s= 0.5)
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('age (years)')
    ax.set_ylabel(metriclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    
def plot_line_metric_vs_age_log(ages, metric, metriclabel = 'metric'):
    fig, ax = plt.subplots()
    ax.plot(np.log2(ages + 1), metric)
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('age (years)')
    ax.set_ylabel(metriclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()

def plot_lines_metric_vs_age_log(ages, metric, metriclabel = 'metric',keys=['SA','VS','MR'],subtract_mean = False,cmap=None,legend=True,ci=None):
    fig, ax = plt.subplots()
    if cmap is None:
        if metric.shape[1]>10:
            cmap = mcolors.ListedColormap(plt.get_cmap('tab20').colors)
        else:
            cmap = mcolors.ListedColormap(plt.get_cmap('tab10').colors)
        # Define the color cycle based on tab20 colormap
        color_cycle = cycler(color=cmap.colors)
        # Update the default rc settings
        plt.rcParams['axes.prop_cycle'] = color_cycle
        for i in range(metric.shape[1]):
            if subtract_mean:
                ax.plot(np.log2(ages + 1), metric[:,i] - np.mean(metric[:,i]),label=keys[i])
            else:
                ax.plot(np.log2(ages + 1), metric[:,i] ,label=keys[i])
    else:
        for i in range(metric.shape[1]):
            if subtract_mean:
                ax.plot(np.log2(ages + 1), metric[:,i] - np.mean(metric[:,i]),label=keys[i],color = cmap(i))
            else:
                ax.plot(np.log2(ages + 1), metric[:,i] ,label=keys[i],color = cmap(i))
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('age (years)',fontsize=20)
    ax.set_ylabel(metriclabel,fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:  
        ax.legend(fontsize=15)
    plt.show()

def plot_lines_metric_vs_age_from_dict(fg,metric,name_list,net_names,metric_label='metric',subtract_mean=False,cmap=None,legend=True):
    #inds = np.where(np.isin(name_list,np.sort(np.array(net_names))))[0]
    inds = [np.where(name_list == s)[0][0] for s in net_names]
    if cmap == 'yeo7':
        cmap = ListedColormap(yeo7_colors[inds])
    elif cmap == 'yeo17':
        cmap = ListedColormap(yeo17_colors[inds])
    if len(metric.shape)>2: 
        for i in range (metric.shape[2]):
            plot_lines_metric_vs_age_log(fg.ages,metric[:,inds,i],metric_label,net_names,subtract_mean,cmap,legend=legend)
    else:
        plot_lines_metric_vs_age_log(fg.ages,metric[:,inds],metric_label,net_names,subtract_mean,cmap,legend=legend)
        
        
def plot_fits_w_ci_one_axis(fitages,fitmetrics,metric_name,std_error,metric_labels=['SA','VS','MR'],annotate_max=True,annotate_offset = 5):
    fig, ax = plt.subplots()
    #cmap = mcolors.ListedColormap(plt.get_cmap('tab10').colors)
    # Define the color cycle based on tab20 colormap
    #color_cycle = cycler(color=cmap.colors)
    # Update the default rc settings
    #plt.rcParams['axes.prop_cycle'] = color_cycle
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    if metric_labels[0] == 'SA':
        colors = ['red','green','gray']
    for i in range(fitmetrics.shape[1]):
        ax.plot(np.log2(fitages + 1), fitmetrics[:,i],label=metric_labels[i],c = colors[i])
        
        ax.fill_between(np.log2(fitages+1), fitmetrics[:,i]-std_error[:,i]*1.96, fitmetrics[:,i]+std_error[:,i]*1.96,  alpha=0.2,color = colors[i])
        if annotate_max:
            if i!=1:
                x = np.log2(fitages + 1)
                y = fitmetrics[:,i]
                xpos = np.where(y == max(y))
                xmax = x[xpos]
                ax.annotate(f'{fitages[xpos][0]} y', xy = (xmax,max(y)),xytext = (xmax,max(y)+annotate_offset), arrowprops=dict(arrowstyle='wedge',facecolor='black'),fontsize=18,color = 'black')
    # Specify the tick positions and labels
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.xlabel('X-axis Label', fontsize=20)
    plt.ylabel('Y-axis Label', fontsize=20)
    ax.set_xlabel('age (years)')
    ax.set_ylabel(metric_name)
    ax.legend(fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
def plot_fitted_metric(indages, indmetric, fitages, fitmetric, metriclabel, std_error=None,annotate_max=True,shift=200):
    """
    This function plots individual metric data along with fitted metric values over age. 
    The x-axis values (age) are transformed using log2.
    
    Parameters:
    indages (array-like): Individual ages corresponding to individual metric values.
    indmetric (array-like): Individual metric values to be plotted.
    fitages (array-like): Ages corresponding to the fitted metric values.
    fitmetric (array-like): Fitted metric values to be plotted.
    metriclabel (str): Label for the y-axis representing the metric.
    std_error (array-like, optional): Standard errors corresponding to the fitted metric values. 
                                      If provided, these will be used to add a shaded region around the fit line. 
                                      Defaults to None.
    
    Returns:
    None. The function directly plots the data using matplotlib.
    """
    fig, ax = plt.subplots()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    ax.plot(np.log2(fitages + 1), fitmetric, c='blue')
    if std_error is not None:
        ax.fill_between(np.log2(fitages+1), fitmetric-std_error*1.96, fitmetric+std_error*1.96, color='blue', alpha=0.2)
    ax.scatter(np.log2(indages + 1), indmetric, s=0.5, c='black',marker=',')

    # Specify the tick positions and labels
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    plt.xlabel('X-axis Label', fontsize=20)
    plt.ylabel('Y-axis Label', fontsize=20)
    ax.set_xlabel('age (years)')
    ax.set_ylabel(metriclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if annotate_max:
        x = np.log2(fitages + 1)
        y = fitmetric
        xpos = np.where(y == max(y))
        xmax = x[xpos]
        ax.annotate(f'{fitages[xpos][0]} y', xy = (xmax,max(y)),xytext = (xmax,max(y)+shift), arrowprops=dict(arrowstyle='wedge',facecolor='red'),fontsize=18,color = 'red')
    plt.show()
from plotnine import ggplot, aes, geom_point, scale_x_continuous, ggtitle, geom_line, scale_color_gradientn
from mizani.transforms import trans
    
class Log1pTrans(trans):
    @staticmethod
    def transform(x):
        return np.log1p(x)

    @staticmethod
    def inverse(x):
        return np.expm1(x)


# Function to create scatter plot
def scatterplot_log_x_plus_1(ages, metric, metriclabel = 'metric'):
    data = {
        'age (years)': ages,
        metriclabel: metric,
    }
    df = pd.DataFrame(data)

    # Generate 10 equally spaced tick labels between min and max x values
    x_ticks = np.linspace(np.min(ages), np.max(ages), 6)

    scatterplot = (
        ggplot(df, aes(x='age (years)', y=metriclabel))
        + geom_point(size=0.5)
        + scale_x_continuous(trans=Log1pTrans(), breaks=x_ticks, labels=x_ticks.astype(int))
    )

    return scatterplot

def line_plot_log_x_plus_1( metric, metriclabel = 'metric'):
    ages=np.arange(400)/4
    data = {
        'age (years)': ages,
        metriclabel: metric,
    }
    df = pd.DataFrame(data)

    # Generate 10 equally spaced tick labels between min and max x values
    x_ticks = np.linspace(np.min(ages), np.max(ages), 6)

    plot = (
        ggplot(df, aes(x='age (years)', y=metriclabel))
        + geom_line(size=1)
        + scale_x_continuous(trans=Log1pTrans(), breaks=x_ticks, labels=x_ticks.astype(int))
    )

    return plot

def line_plots_log_x_plus_1(metrics, metric_labels):
    ages = np.arange(400) / 4

    # Create a DataFrame with the age column and each metric column
    data = {'age (years)': ages}
    for metric, metric_label in zip(metrics, metric_labels):
        data[metric_label] = metric
    df = pd.DataFrame(data)

    # Melt the DataFrame to have a long format suitable for ggplot
    df_melted = pd.melt(df, id_vars=['age (years)'], value_vars=metric_labels, var_name='metric', value_name='value')

    # Generate 10 equally spaced tick labels between min and max x values
    x_ticks = np.linspace(np.min(ages), np.max(ages), 6)

    plot = (
        ggplot(df_melted, aes(x='age (years)', y='value', color='metric'))
        + geom_line(size=1)
        + scale_x_continuous(trans=Log1pTrans(), breaks=x_ticks, labels=x_ticks.astype(int))
    )

    return plot
#from plotnine import ggplot, aes, geom_line, ggtitle, scale_color_gradientn
#import matplotlib.cm as cm

def plot_eigenvalues_by_age_gg(eigenvalues, ages):
    # Create a pandas DataFrame from eigenvalues and ages
    n_subjects, n_vals = eigenvalues.shape
    data = {
        'x': np.tile(np.arange(n_vals), n_subjects),
        'y': eigenvalues.flatten(),
        'age': np.repeat(ages, n_vals),
        'subject': np.repeat(np.arange(n_subjects), n_vals)
    }
    df = pd.DataFrame(data)

    # Get the 'magma' colormap from matplotlib and convert it to a list of colors
    magma_colors = cm.get_cmap('plasma', 256)
    colors = [magma_colors(i) for i in range(magma_colors.N)]

    # Create a ggplot plot with eigenvalues as lines color-coded by subject age
    plot = (
        ggplot(df, aes(x='x', y='y', color='age', group='subject'))
        + geom_line()
        + ggtitle('Eigenvalues as Lines Color-coded by Subject Age')
        + scale_color_gradientn(colors=colors)
    )

    return plot

def plot_subject_data_and_fit(subject_data, subject_ages, fitted_data, age_atlas):
    if len(subject_data) != len(subject_ages):
        raise ValueError("Length of subject_data and subject_ages must be the same (n_subjects).")
    
    if len(fitted_data) != len(age_atlas):
        raise ValueError("Length of fitted_data and age_atlas must be the same (n_timepoints).")
    
    # Create a scatter plot of subject_data vs subject_ages
    plt.scatter(subject_ages, subject_data, label='Subject Data')
    
    # Create a line plot of fitted_data vs age_atlas
    plt.plot(age_atlas, fitted_data, label='Fitted Data', color='red')
    
    # Add labels, title, and legend
    plt.xlabel("Age")
    plt.ylabel("Data Value")
    plt.title("Subject Data vs Age and Fitted Data")
    plt.legend()
    
    # Show the plot
    plt.show()

def plot_network_kdes(grads,net_names,net_labels,colormap,title='kde plots',x_min=-1,x_max = 1.5,sort_by_rank=True):
    num_rows = len(net_names)
    num_cols = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 2*len(net_names)),sharex=True)
    data_list = []
    for name in net_names:
        inds = np.where(net_labels==name)[0]
        data_list.append(grads[inds])
    if sort_by_rank:
        means = [np.mean(d) for d in data_list]
        sorted_index = np.argsort(means)[::-1]
        for i, sr in enumerate(sorted_index):
            color = colormap(sr)  # Get color based on the row index
            sns.kdeplot(data_list[sr], ax=axs[i], color=color,fill=True)
            axs[i].set_ylabel(net_names[sr])
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xlim(x_min, x_max)
    else:
        for row_index, data in enumerate(data_list):
            color = colormap(row_index)  # Get color based on the row index
            sns.kdeplot(data, ax=axs[row_index], color=color,fill=True)
            axs[row_index].set_ylabel(net_names[row_index])
            axs[row_index].spines['top'].set_visible(False)
            axs[row_index].spines['right'].set_visible(False)
            axs[row_index].set_xlim(x_min, x_max)
    fig.suptitle(title,y = 0.99, fontsize = 16, fontweight = 'bold')
    
    plt.tight_layout()
    
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

def plot_network_kdes_longitudinal(grads,net_names,net_labels,colormap,title='kde plots',x_min=-1,x_max = 1.5,sort_by_rank=True,timepoints = np.arange(50)*8,agemin=0, agemax=25,sorted_indices=None,return_indices=False):
    num_rows = len(net_names)-1
    num_cols = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 2*len(net_names)),sharex=True)
    data_list = []
    for name in net_names:
        if name == ' medial wall':
            continue
        inds = np.where(net_labels==name)[0]
        data_list.append(grads[:,inds])
    if sort_by_rank:
        if sorted_indices is None:
            means = [np.mean(d[-1]) for d in data_list]
            sorted_index = np.argsort(means)[::-1]
        else:
            sorted_index=sorted_indices
        for i, sr in enumerate(sorted_index):
            
            color = colormap(sr+1)  # Get color based on the row index
            agecmap = LinearSegmentedColormap.from_list('grey_to_default',['lightgrey',color])
            for j in range(len(data_list[sr])):
                sns.kdeplot(data_list[sr][j], ax=axs[i], color=agecmap(j/len(data_list[sr])),fill=False)
            #axs[i,1].axis('off')
            #axs[i,1].imshow([[0,1]],cmap=agecmap,aspect='auto')
            #axs[i,1].set_yticks([])
            #axs[i,1].set_xticks([])
            axs[i].set_ylabel(net_names[sr+1])
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xlim(x_min, x_max)
            
            norm = Normalize(vmin=agemin,vmax=agemax)
            sm = plt.cm.ScalarMappable(cmap=agecmap,norm=norm)
            sm.set_array([])
            fig.colorbar(sm,ax=axs[i],orientation='vertical')
# =============================================================================
#             cb_ax = fig.add_axes([0.88, axs[i, 0].get_position().y0, 0.02, axs[i, 0].get_position().height])
#             cb = ColorbarBase(cb_ax, cmap=agecmap)
#             cb_ax.yaxis.set_ticks_position('left')
#             cb_ax.yaxis.set_label_position('left')
#             cb_ax.spines['left'].set_position(('outward', 5))
#             cb_ax.set_ylabel('Colorbar')
# =============================================================================
# =============================================================================
#     else:
#         for row_index, data in enumerate(data_list):
#             color = colormap(row_index)  # Get color based on the row index
#             sns.kdeplot(data, ax=axs[row_index], color=color,fill=True)
#             axs[row_index].set_ylabel(net_names[row_index])
#             axs[row_index].spines['top'].set_visible(False)
#             axs[row_index].spines['right'].set_visible(False)
#             axs[row_index].set_xlim(x_min, x_max)
# =============================================================================
    fig.suptitle(title,y = 0.99, fontsize = 16, fontweight = 'bold')
    
    plt.tight_layout()
    if return_indices:
        return sorted_index
    
    
yeo7_colors = np.array([
    (0, 0, 0),  # medial wall
    (230, 148, 34),  # cont
    (205, 62, 78),  # default
    (0, 118, 14),  # DorsAttn
    (220, 248, 26),  # Limbic
    (196, 58, 250),  # SalVenAttn
    (70, 130, 180),  # SomMot
    (120, 18, 134)   # Vis
])/255

yeo7_colors_3d =np.array([
    (0, 0, 0),  # medial wall
    (200, 200, 200),  # cont
    (255, 80, 80),  # default
    (130, 100, 200),  # DorsAttn
    (220, 128, 85),  # Limbic
    (50, 100, 50),  # SalVenAttn
    (0, 255, 0),  # SomMot
    (0, 0, 255)   # Vis
])/255

 
yeo17_colors =np.array([
    [  0.,   0.,   0.],
       [230., 148.,  34.],
       [135.,  50.,  74.],
       [119., 140., 176.],
       [255., 255.,   0.],
       [205.,  62.,  78.],
       [  0.,   0., 130.],
       [ 74., 155.,  60.],
       [  0., 118.,  14.],
       [200., 248., 164.],
       [122., 135.,  50.],
       [196.,  58., 250.],
       [255., 152., 213.],
       [ 70., 130., 180.],
       [ 42., 204., 164.],
       [ 12.,  48., 255.],
       [120.,  18., 134.],
       [255.,   0.,   0.]])/255

#from matplotlib import cm

def plot_colorbar(cmap):
    fig = plt.figure(figsize=(5, 1))
    cax = fig.add_axes([0.05, 0.5, 0.9, 0.4])
    cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), cax=cax, orientation='horizontal', ticks=[])
    plt.show()
from matplotlib.colors import ListedColormap
yeo17_cmap = ListedColormap(yeo17_colors)
yeo7_cmap3d = ListedColormap(yeo7_colors_3d)

yeo7_cmap = ListedColormap(yeo7_colors)

def get_color_tuples_from_cmap(netparc,cmap):
    colors = np.zeros((len(netparc),4))
    
    for i in range (len(set(netparc))):
        inds = np.where(netparc == i)[0]
        colors[inds] = cmap(i)
    return colors[:,:3]

def get_listed_cmap3d_from_parc(parc,colors):
    parc_cmap = uts.get_parcellated_cmap(parc, colors,False)
    cmap = ListedColormap(parc_cmap)
    return cmap 

def make_region_mask(parc,inds):
    mask = np.zeros(len(parc))
    
    for j,i in enumerate(inds):
        pinds = np.where(parc==i)[0]
        mask[pinds] = j
    return mask 

def plot_corr_mat_upper_diag(mat,xlabels= ['SA','VS','MR'],ylabels = ['G1','G2','G3'],title='corr'):
    corr_matrix = mat

# Mask the lower triangle of the matrix (excluding the main diagonal)
    #mask = np.tril(np.ones_like(corr_matrix, dtype=bool), k=-1)
    #corr_matrix_masked = np.where(mask, np.nan, corr_matrix)
    corr_matrix_masked = mat
    # Plot the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(corr_matrix_masked, cmap="coolwarm", vmin=0, vmax=1)
    
    # Display the colorbar
    plt.colorbar(cax)
    
    # Set ticks to the top
    ax.xaxis.tick_top()
    
    # Set tick labels
    ax.set_xticks(np.arange(corr_matrix.shape[1]))
    ax.set_yticks(np.arange(corr_matrix.shape[0]))
    ax.set_xticklabels(xlabels)
    ax.set_yticklabels(ylabels)
    
    # Loop over data dimensions and create text annotations.
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            #if not mask[i, j]:
            text = ax.text(j, i, round(corr_matrix[i, j], 2),
                               ha="center", va="center", color="black")
    
    # Remove bounding box
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Display the plot
    ax.set_title(title)
    plt.show()
    
from matplotlib import cm 
import utility as uts

def get_vertex_cmap_tuples(grads,cmap_name='jet',gradind = 0):
    cmap = cm.get_cmap(cmap_name)
    normgrad = uts.norm_vecs(grads)[:,gradind]
    
    tups = [cmap(normgrad[i]) for i in range (len(grads))]
    return np.array(tups)[:,:3]*255

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
    ng=uts.norm_vecs(grads)
    for i in range (len(grads)):
        colors[i]=barycentric(ng[i])
        #h,s,v=colorsys.rgb_to_hsv(colors[i,0],colors[i,1],colors[i,2])
        #v=min(1,np.linalg.norm(ng[i]-point))
        #s=1-ng[i,0]
        #r,g,b=colorsys.hsv_to_rgb(h,s,v)
        #r,g,b=colorsys.hsv_to_rgb(colors[i,0],colors[i,1],colors[i,2])
        #colors[i]=np.array([r,g,b])
    return uts.norm_vecs(colors)

def parc_avg_cmap(grads,parc):
    colors3d = cmap3d_bary(grads)
    colors3d_parc = uts.get_parcellated_cmap(parc,colors3d)
    
    return colors3d_parc