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
    plt.show()

def plot_lines_metric_vs_age_log(ages, metric, metriclabel = 'metric',keys=['SA','VS','MR']):
    fig, ax = plt.subplots()
    for i in range(metric.shape[1]):
        ax.plot(np.log2(ages + 1), metric[:,i],label=keys[i])
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('age (years)')
    ax.set_ylabel(metriclabel)
    ax.legend()
    plt.show()

def plot_fits_w_ci_one_axis(fitages,fitmetrics,metric_name,std_error,metric_labels=['SA','VS','MR']):
    fig, ax = plt.subplots()
    for i in range(fitmetrics.shape[1]):
        ax.plot(np.log2(fitages + 1), fitmetrics[:,i],label=metric_labels[i])
        
        ax.fill_between(np.log2(fitages+1), fitmetrics[:,i]-std_error[:,i]*1.96, fitmetrics[:,i]+std_error[:,i]*1.96,  alpha=0.2)

    # Specify the tick positions and labels
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('age (years)')
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
def plot_fitted_metric(indages, indmetric, fitages, fitmetric, metriclabel, std_error=None):
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
    ax.plot(np.log2(fitages + 1), fitmetric, c='blue')
    if std_error is not None:
        ax.fill_between(np.log2(fitages+1), fitmetric-std_error*1.96, fitmetric+std_error*1.96, color='blue', alpha=0.2)
    ax.scatter(np.log2(indages + 1), indmetric, s=0.5, c='black')

    # Specify the tick positions and labels
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]  # Desired x-axis labels
    tick_positions = np.log2(np.array(tick_labels) + 1)  # Corresponding x-axis positions

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('age (years)')
    ax.set_ylabel(metriclabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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