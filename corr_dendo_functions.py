import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.cluster.hierarchy as sch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def correlation_pearson(data):
    df = pd.DataFrame(data)
    corr1 = df.corr(method='pearson')
    #plt.matshow(corr1, cmap='jet')
    #plt.colorbar()
    #plt.show()
    return corr1

def correlation_kendall(data):
    df = pd.DataFrame(data)
    corr1 = df.corr(method='kendall')
    #plt.matshow(corr1, cmap='jet')
    #plt.colorbar()
    #plt.show()
    return corr1

def correlation_spearman(data):
    df = pd.DataFrame(data)
    corr1 = df.corr(method='spearman')
    #plt.matshow(corr1, cmap='jet')
    #plt.colorbar()
    #plt.show()
    return corr1

def _plot_correlation_helper(df,size, root, canvas):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''
    #size = 10
    # Compute the correlation matrix for the received dataframe
    corr_func = df.corr()
    
    # Plot the correlation matrix
    fig, ax = plt.subplots(figsize=(size, size))
    # cax = ax.matshow(corr, cmap='RdYlGn')
    cax = ax.matshow(corr_func, cmap='jet')
    # plt.xticks(range(len(corr.columns)), corr.columns, rotation=90);
    # plt.yticks(range(len(corr.columns)), corr.columns);
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

def plot_correlation(data, corr, root, canvas):
    df = pd.DataFrame(data)
    X = corr.values
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    # ind = sch.fcluster(L, 0.2*d.max(), 'distance')
    ind = sch.fcluster(L, 50, criterion='maxclust')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    size = 5
    _plot_correlation_helper(df, size, root, canvas)

def _plot_dendrogram_helper(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs) # type: ignore

def plot_dendogram(data, root, canvas):
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data.transpose())
    clustering.labels_.shape
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    _plot_dendrogram_helper(clustering, truncate_mode="none", count_sort='none', show_contracted='true')
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    #plt.show()
    for i in range(20):
        plt.plot(np.array(range(len(data[:, i]))).reshape(-1, 1), data[:, i] + i)
    #ax = plt.gca()
    fig = plt.gcf()
    #plt.show()
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')