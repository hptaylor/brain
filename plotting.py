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
from brainspace.plotting import plot_hemispheres
import stat_utils as stu 

scrpath='/Users/patricktaylor/lifespan_analysis/scratch/'
axisnames=['SA','VS','MR']
lhp = '/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_L.veryinflated.white.ver2.downsampled.L5.surf.gii' 
rhp = '/Users/patricktaylor/lifespan_analysis/Lifespan_Atlases/Atlas_420Months_R.veryinflated.white.ver2.downsampled.L5.surf.gii' 
        

def plot_histogram(data,title='histogram'):
    """
    Plot a histogram of the data with 100 bins.

    Parameters:
    - data (list or array-like): The data to be plotted.
    """
    plt.figure()
    plt.hist(data, bins=100, edgecolor='k', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_histograms_same_axis(datas,labels,title='histogram'):
    """
    Plot histograms of multiple datasets on the same axis.

    Parameters:
    - datas (array-like): The data to be plotted. Expected shape (N_datapoints, n_datasets).
    - labels (list): List of labels for the datasets. Expected length n_datasets.
    - title (str): Title for the plot.
    """
    plt.figure()
    
    # Ensure datas and labels match in the second dimension
    if datas.shape[1] != len(labels):
        raise ValueError("Number of datasets does not match the number of labels.")

    # Plot each dataset
    for i in range(datas.shape[1]):
        plt.hist(datas[:, i], bins=100, alpha=0.5, label=labels[i], edgecolor='k')

    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def plot_surf_grid(data_list,lh,rh,data_labels = None,cmap = 'jet', save = None, width = 800, zoom = 1.2, cbar = True):
    
    height = int(len(data_list)*width/4)
    size = (width,height)
    
    if save is not None:
        transparent_bg = True
        screenshot = True
    else:
        transparent_bg = False
        screenshot = False
    f = plot_hemispheres(lh, rh, array_name = data_list, label_text = data_labels, 
                     interactive = False, screenshot = screenshot, 
                     filename = save, size = size, zoom = zoom, 
                     background = (255,255,255), 
                     transparent_bg = transparent_bg, cmap = cmap, color_bar = cbar)
    return f

def prepare_log_axis_plot(figsize=(800,800)):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up the log axis ticks
    tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]
    tick_positions = np.log2(np.array(tick_labels) + 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('age (years)', fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig, ax

def plot_subject_data(subject_data, subject_ages, color='black', marker='o', marker_size=0.5, fig=None, ax=None, **kwargs):
    """
    Scatter plot subject-wise data on a prepared log axis.

    Args:
    - subject_data (array): Subject-wise data values.
    - subject_ages (array): Corresponding ages for each subject.
    - color (str, optional): Color of the plot points. Defaults to 'black'.
    - marker (str, optional): Marker style for data points. Defaults to 'o' (circle).
    - fig (matplotlib.figure.Figure, optional): Figure object to plot on. If None, a new figure is created.
    - ax (matplotlib.axes.Axes, optional): Axis object to plot on. If None, a new axis is created.

    Returns:
    - fig, ax: Figure and axis objects with the plotted data.
    """

    # Check if fig and ax are provided. If not, create new ones using prepare_log_axis_plot
    if fig is None or ax is None:
        fig, ax = prepare_log_axis_plot()

    # Convert ages to log2 scale for plotting
    log_ages = np.log2(np.array(subject_ages) + 1)

    # Plot the data using scatter
    ax.scatter(log_ages, subject_data, color=color, marker=marker, s=marker_size)
    
    if 'title' in  kwargs:
        ax.set_title(kwargs['title'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    return fig, ax


def plot_fit_with_data(fit, std_error=None, subject_data=None, subject_ages=None, 
                       fit_color='r', ci_alpha=0.2, age_factor=4, ylabel=None,
                       max_hdi_bounds=None, max_marker_color='black',fig=None, ax=None, **kwargs):
    """
    Plots a fit (and optionally its confidence interval) on a prepared log axis. 
    If subject data and ages are provided, they are plotted as a scatter plot.

    Args:
    - fit (array): Fit values.
    - std_error (array, optional): Standard error for each fit value. If provided, confidence interval is plotted.
    - subject_data (array, optional): Subject-wise data values.
    - subject_ages (array, optional): Corresponding ages for each subject.
    - fit_color (str, optional): Color of the fit line. Defaults to 'r' (red).
    - ci_color (str, optional): Color of the confidence interval shading. Defaults to 'r' (red).
    - ci_alpha (float, optional): Transparency level of the confidence interval shading. Defaults to 0.2.
    - age_factor (int, optional): Factor to adjust the age-to-log2 conversion. Defaults to 4.
    - ylabel (str, optional): Y-axis label.
    - fig, ax (optional): Existing figure and axis objects. If not provided, new ones are created.

    Returns:
    - fig, ax: Figure and axis objects with the plotted data and fit.
    """

    # Simplifying the dimensions of fit and std_error if they are 2D with single column
    if len(fit.shape) > 1:
        fit = fit[:, 0]
    if std_error is not None and len(std_error.shape) > 1:
        std_error = std_error[:, 0]
        
    # If subject data is provided, first plot it
    if subject_data is not None and subject_ages is not None:
        fig, ax = plot_subject_data(subject_data, subject_ages, fig=fig, ax=ax, **kwargs)
    elif fig is None or ax is None:
        fig, ax = prepare_log_axis_plot()

    # Convert ages to log2 scale for plotting the fit
    log_ages = np.log2(np.arange(len(fit)) / age_factor + 1)

    # Plot the fit
    ax.plot(log_ages, fit, color=fit_color)

    # If std_error is provided, shade the confidence interval
    if std_error is not None:
        lower_bound, upper_bound = stu.compute_confidence_interval(fit, std_error)
        ax.fill_between(log_ages, lower_bound, upper_bound, color=fit_color, alpha=ci_alpha)
    if max_hdi_bounds is not None:
        max_age_index = uts.find_nearest_index(np.arange(len(fit))/4,(max_hdi_bounds[0]+max_hdi_bounds[1])/2)
        
        ax.plot(log_ages[max_age_index], fit[max_age_index], 'o', 
                color=max_marker_color, markersize=8, label='_nolegend_')
        
        # Convert the HDI bounds to log2 scale
        log_lower_bound = np.log2(max_hdi_bounds[0] + 1)
        log_upper_bound = np.log2(max_hdi_bounds[1] + 1)
        
        # Calculate the errors for lower and upper bounds relative to the maximum age index
        lower_error = log_ages[max_age_index] - log_lower_bound
        upper_error = log_upper_bound - log_ages[max_age_index]
    
        # Draw horizontal error bars for the HDI of the age at which maximum fit occurs
        ax.errorbar(log_ages[max_age_index], fit[max_age_index], xerr=[[lower_error], [upper_error]], 
                    fmt='o', color=max_marker_color, capsize=10,  elinewidth= 3)
    # If ylabel is provided, set the y-axis label
    if ylabel:
        ax.set_ylabel(ylabel)
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
        
    return fig, ax

def plot_multiple_fits(fits, std_errors=None, subject_data=None, subject_ages=None, 
                       fit_names=None, fit_colors=None, age_factor=4, 
                       ylabel=None, max_hdi_bounds = None, **kwargs):
    """
    Plot multiple fits (and optionally their confidence intervals) on the same axis.

    Args:
    - fits (array): Array of shape (N_timepoints x N_fits) containing the fit values.
    - std_errors (array, optional): Array of shape (N_timepoints x N_fits) containing standard errors.
    - subject_data (array, optional): Subject-wise data values.
    - subject_ages (array, optional): Corresponding ages for each subject.
    - fit_names (list, optional): List of names for the fits.
    - fit_colors (list, optional): List of colors for the fits.
    - age_factor (int, optional): Factor to adjust the age-to-log2 conversion. Defaults to 4.
    - ylabel (str, optional): Y-axis label.

    Returns:
    - fig, ax: Figure and axis objects with the plotted data and fits.
    """
    bounds = max_hdi_bounds[0] if max_hdi_bounds is not None else None
    
    # Create the initial plot (with the first fit or just the subject data)
    fig, ax = plot_fit_with_data(fits[:, 0], 
                                 std_errors[:, 0] if std_errors is not None else None,
                                 subject_data, subject_ages, 
                                 fit_color=fit_colors[0] if fit_colors is not None else 'r',
                                 ylabel=ylabel,
                                 max_hdi_bounds=bounds,
                                 age_factor=age_factor, 
                                 **kwargs)
    
    # Plot the remaining fits
    for i in range(1, fits.shape[1]):
        color = fit_colors[i] if fit_colors else 'r'
        bounds = max_hdi_bounds[i] if max_hdi_bounds is not None else None
        plot_fit_with_data(fits[:, i], 
                           std_errors[:, i] if std_errors is not None else None,
                           fig=fig, ax=ax, 
                           fit_color=color,
                           max_hdi_bounds=bounds,
                           age_factor=age_factor, 
                           **kwargs)
    
    # Add legend if fit names are provided
    if fit_names is not None:
        ax.legend(fit_names)
    
    return fig, ax

def plot_fits_separately(fits=None, std_errors=None, subject_data=None, subject_ages=None, 
                         fit_names=None, fit_colors=None, age_factor=4, 
                         ylabel=None, max_hdi_bounds = None, **kwargs):
    """
    Plot each fit (and optionally its confidence interval) on separate axes.

    Args:
    - fits (array): Array of shape (N_timepoints x N_fits) containing the fit values.
    - std_errors (array, optional): Array of shape (N_timepoints x N_fits) containing standard errors.
    - subject_data (array, optional): Subject-wise data values.
    - subject_ages (array, optional): Corresponding ages for each subject.
    - fit_names (list, optional): List of names for the fits.
    - fit_colors (list, optional): List of colors for the fits.
    - age_factor (int, optional): Factor to adjust the age-to-log2 conversion. Defaults to 4.
    - ylabel (str, optional): Y-axis label.

    Returns:
    - fig: Figure object with the plotted data and fits on separate axes.
    """
    if fits is not None:
        n_fits = fits.shape[1]
    else:
        n_fits = subject_data.shape[1]
        
    
    # Create a new figure with n_fits number of axes arranged horizontally
    fig, axes = plt.subplots(nrows=1, ncols=n_fits, figsize=(5*n_fits, 5))
    
    # Ensure axes is an array (useful for the case of a single fit)
    if n_fits == 1:
        axes = [axes]
    
    # For each fit, plot it on its corresponding axis
    for i in range(n_fits):
        color = fit_colors[i] if fit_colors else 'r'
        bounds = max_hdi_bounds[i] if max_hdi_bounds is not None else None
        if fits is not None:
            _, ax = plot_fit_with_data(fits[:, i], 
                                       std_errors[:, i] if std_errors is not None else None,
                                       subject_data[:,i], subject_ages, 
                                       fit_color=color, ylabel=ylabel if i==0 else None,
                                       max_hdi_bounds=bounds,
                                       age_factor=age_factor, 
                                       fig=fig, ax=axes[i], 
                                       **kwargs)
            tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]
            tick_positions = np.log2(np.array(tick_labels) + 1)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            
            ax.tick_params(axis='both', which='major')
            ax.set_xlabel('age (years)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            _, ax = plot_subject_data(subject_data[:,i],subject_ages,color,fig=fig,ax=axes[i],title=fit_names[i],ylabel = ylabel if i==0 else None)
            tick_labels = [0, 1, 2, 4, 10, 18, 30, 50, 80]
            tick_positions = np.log2(np.array(tick_labels) + 1)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            
            ax.tick_params(axis='both', which='major')
            ax.set_xlabel('age (years)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        # Set title for each axis if fit_names are provided
        if fit_names is not None:
            ax.set_title(fit_names[i])
    
    # Adjust the layout to prevent overlap
    fig.tight_layout()
    
    return fig


def plot_centiles_vs_age_log(ages, metric, metriclabel='metric', subtract_mean=False, cmap='jet',cmap_age=25):
    """
    Plot centile curves versus age on a logarithmic axis.
    
    Args:
    - ages (numpy.ndarray): 1D array containing age values.
    - metric (numpy.ndarray): 2D array containing metric values. Each column represents a set of measurements.
    - metriclabel (str, optional): Y-axis label. Defaults to 'metric'.
    - subtract_mean (bool, optional): If True, subtract the mean of each metric set from its values. Defaults to False.
    - cmap (str or callable, optional): Colormap name or callable for plotting. Defaults to 'jet'.
    - show_cbar (str, optional): If provided, the name of a colormap to display alongside as a colorbar.
    
    Raises:
    - ValueError: If the ages and metric arrays do not have compatible shapes.
    
    Returns:
    - None
    """
    
    if len(ages) != metric.shape[0]:
        raise ValueError("The length of `ages` array must match the number of rows in the `metric` array.")
    
    if not callable(cmap):
        cmap_func = construct_centile_colormap(metric[4*cmap_age],cmap)
    else:
        cmap_func = cmap
    
    fig, ax = prepare_log_axis_plot()
    ax.set_ylabel(metriclabel, fontsize=20)
    
    for i in range(metric.shape[1]):
        if subtract_mean:
            ax.plot(np.log2(ages + 1), metric[:,i] - np.mean(metric[:,i]), color=cmap_func(i))
        else:
            ax.plot(np.log2(ages + 1), metric[:,i], color=cmap_func(i))

    
    mappable = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap))
    mappable.set_array([])  # To ensure the mappable's data range covers 0 to 1
    cbar = plt.colorbar(mappable, ax=ax)
    
    # Adjust colorbar ticks to span from 0 to 100
    cbar_ticks = np.linspace(0, 1, 11)  # e.g., [0, 0.1, 0.2, ... 1]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{int(100*tick)}" for tick in cbar_ticks])
    plt.show()

def construct_centile_colormap(centile_values, colormap_name):
    """
    Construct a colormap based on the values in the centile array.
    
    Args:
    - centile_values: array containing centile averages.
    - colormap_name: name of the existing matplotlib colormap.
    
    Returns:
    - ListedColormap object.
    """
    base_colormap = plt.cm.get_cmap(colormap_name)
    
    # Get colors from the existing colormap based on the centile values
    colors = base_colormap(np.linspace(0, 1, len(centile_values)))
    
    return ListedColormap(colors)

def plot_surf(data, lh, rh=None, size=(800, 200), title=None, crange=None, cmap='turbo',
              save=None, show=True, cbar=True, interactive=False):
    """
    Plot surface data.
    
    Args:
    - data (array): Data array to be plotted.
    - lh (obj): Left hemisphere object.
    - rh (obj, optional): Right hemisphere object. If present, plots the right hemisphere. Defaults to None.
    - size (tuple, optional): Size of the plot. Defaults to (800, 200).
    - title (str, optional): Title for the colorbar. Defaults to None.
    - crange (tuple, optional): Color range. Defaults to None.
    - cmap (str, optional): Colormap for the data. Defaults to 'turbo'.
    - save (str, optional): Path to save the figure. If None, the figure is not saved. Defaults to None.
    - show (bool, optional): Whether to show the figure. Defaults to True.
    - cbar (bool, optional): Whether to show the colorbar. Defaults to True.
    - interactive (bool, optional): Whether the plot should be interactive. Defaults to False.

    Returns:
    - None
    """
    
    # Define common properties for the plot layers
    common_layer_props = {
        "color_range": crange,
        "cmap": cmap,
        "zero_transparent": False,
        "cbar_label": title,
        "cbar": cbar
    }

    if rh is not None:
        p = sp.Plot(lh, rh, size=size, zoom=1.2, layout='row')
        p.add_layer(data, **common_layer_props)
    else:
        # Adjust size for single hemisphere view
        size = (400, 200) if size == (800, 200) else size
        p = sp.Plot(lh, size=size, zoom=1.2, layout='row')
        p.add_layer(data, **common_layer_props)

    fig = p.build()

    # Display and/or save the figure
    if show:
        fig.show()
    if save:
        fig.savefig(save)

    return






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
        
        
def plot_fits_w_ci_one_axis(fitages,fitmetrics,metric_name,std_error,metric_labels=['SA','VS','MR'],annotate_max=True,annotate_offset = 5,cmap=None):
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
    if cmap is not None:
        colors = [cmap(i) for i in range (len(metric_labels))]
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

def plot_network_kdes_longitudinal(grads,net_names,net_labels,colormap,title='kde plots',x_min=None,x_max = None,sort_by_rank=True,timepoints = np.arange(50)*8,agemin=0, agemax=25,sorted_indices=None,return_indices=False):
    if x_min is None:
        x_min = np.min(grads)
    if x_max is None:
        x_max = np.max(grads)
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

#yeo7_colors_3d =np.array([
#    (0, 0, 0),  # medial wall
#    (200, 200, 200),  # cont
#    (255, 80, 80),  # default
#    (130, 100, 200),  # DorsAttn
#    (220, 128, 85),  # Limbic
#    (50, 100, 50),  # SalVenAttn
#    (0, 255, 0),  # SomMot
#    (0, 0, 255)   # Vis
#])/255

yeo7_colors_3d =np.array([
    (0, 0, 0),  # medial wall
    (139, 160, 164),  # cont
    (207, 53, 80),  # default
    (95, 68, 137),  # DorsAttn
    (207, 103, 71),  # Limbic
    (68, 137, 95),  # SalVenAttn
    (110, 164, 25),  # SomMot
    (25, 110, 164)   # Vis
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

sa_max_options = [False, True, True, False, True, False, False, False]

mr_max_options = [False, True, False, True, False, True, False, False]