#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:30:18 2023

@author: patricktaylor
"""
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
