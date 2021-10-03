# Library of Functions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def display_factorial_planes(X_projected, pca, axis_ranks, alpha=0.8,
                             illustrative_var=None, zoom=0.9):
    '''Display a scatter plot on a factorial plane, with points colored
    depending on values of illustrative_var.
    Illustrative_var types: cluster(array) or feature(Series)
    axis_rank: [(0,1)] for 1 graphe, or [(0,1),(1,2)] for 2 graphe'''

    # For each factorial plane
    for d1, d2 in axis_ranks:
        # Initialization
        fig, ax = plt.subplots(figsize=(10, 8))
        var = np.array(illustrative_var)
        # Display the points
        if var is None:
            ax.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
        # colored by cluster
        elif len(np.unique(var)) < 11:
            for value in np.unique(var):
                mask = np.where(var == value)
                ax.scatter(X_projected[mask, d1], X_projected[mask, d2],
                           alpha=alpha, label=value)
            ax.legend()
        # colored by feature values
        else:
            im = ax.scatter(X_projected[:, d1], X_projected[:, d2],
                            alpha=alpha, c=illustrative_var.values,
                            cmap='RdYlGn')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(illustrative_var.name, rotation=90)

        # Define the limits of the chart
        boundary = np.max(np.abs(X_projected[:, [d1, d2]])) / zoom
        ax.set(xlim=(-boundary, boundary), ylim=(-boundary, boundary))

        # Display grid lines
        ax.plot([-100, 100], [0, 0], color='grey', ls='--')
        ax.plot([0, 0], [-100, 100], color='grey', ls='--')
        
        if pca != None:
            # Label the axes, with the percentage of variance explained
            ax.set_xlabel(
                'PC{} ({:.2%})'.format(d1 + 1, pca.explained_variance_ratio_[d1]))
            ax.set_ylabel(
                'PC{} ({:.2%})'.format(d2 + 1, pca.explained_variance_ratio_[d2]))
            ax.set_title(
                "Projection of points (on PC{} and PC{})".format(d1 + 1, d2 + 1))
        else:
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('Projection of points via TSNE')

def snake_plot(X, cluster):
    '''This function gives snake plot of different cluster'''
    sns.set_theme()
    plt.figure(figsize=(15, 5))
    data_K_snake = X.assign(Cluster=cluster)
    data_K_snake = pd.melt(data_K_snake, id_vars=['Cluster'],
                           var_name='Metric')
    plt.title('Snake plot of normalized variables')
    plt.xlabel('Metric')
    plt.ylabel('Value')
    sns.lineplot(
        data=data_K_snake, x='Metric', y='value', hue='Cluster', ci=0,
        palette='RdYlGn')
    plt.show()


def gini(labels_):
    '''This function calculates the Gini coefficient of cluster.labels_,which
    is between 0 and 1. 0 indicates homogeneously distributed clusters.
    1 indicates most elements are in one cluster.'''
    _, array = np.unique(labels_, return_counts=True)
    array = np.sort(array)
    index = np.arange(1, array.shape[0]+1)
    n = array.shape[0]
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


def radar_plot(df_scaled, vars: list, ax, label=None, frac=1, color='m'):

    '''This function draw a radar plot. df_scaled is scaled dataframe with
    cluster labels. Vars is the list of varibles choosed to display.
    Labels decides which cluster to draw.'''

    # select variables to display
    df_radar = df_scaled.groupby('cluster')[vars].mean()
    # cluster name and size
    size = (df_scaled['cluster'] == label).sum()
    # axis setting of radar plot
    xticks = vars
    n_vars = len(vars)
    angles = np.linspace(0, 2 * np.pi, n_vars + 1)

    # plot a radar plot for label cluster
    values = df_radar.loc[label, :].values.tolist()
    # match dimension of angles
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label='test', c=color)
    ax.fill(angles, values, alpha=0.2, c=color)

    # axis setting of plot
    ax.set_title('Cluster_{}: {} customers'.format(label, size), pad=10,
                 c=color, fontdict={'fontsize': 14})
    # set position of first elements in angles at the top
    ax.set_theta_offset(np.pi / 2)
    # set clockwise plot
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticks)
    return ax

