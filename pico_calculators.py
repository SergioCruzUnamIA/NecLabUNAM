from tkinter.filedialog import asksaveasfilename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, SGDRegressor, ElasticNet
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import linear_model
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import math

def loadData(data):
    data = np.load(data)
    rs = np.random.RandomState(0)
    data_ = data[:,1:] 
    print(data_.shape)
    return data_

def plotdata(data):
    plt.plot(np.array(range(len(data[:, 15]))).reshape(-1, 1), data[:, 15])
    ax = plt.gca()
    plt.show()

def normalizeData(data):
    norm_data = np.zeros(data.shape) # crea un arreglo con zeros en la forma de los datos
    for i in range(data.shape[1]): 
        reg = ElasticNet().fit(np.array(range(len(data[:, i]))).reshape(-1, 1), data[:, i])
        #reg = svm.SVR().fit(np.array(range(len(data_[:, i]))).reshape(-1, 1), data_[:, i])
        res = reg.predict(np.array(range(len(data[:, i]))).reshape(-1, 1))
        norm_data[:, i] = data[:, i] - res # resta la funcion
        min_data = min(norm_data[:, i])
        max_data = max(norm_data[:, i])
        #norm_data[:, i] = data_[:, i] 
        norm_data[:, i] = norm_data[:, i] - min_data # opcion para nomalizar los datos
        #norm_data[:, i] = norm_data[:, i] / min_data # opcion para nomalizar los datos
        #norm_data[:, i] = (norm_data[:, i] - min_data) / (max_data - min_data) # opcion para nomalizar los datos
    return norm_data

def plot_normalized_data(data):
    plt.plot(np.array(range(len(data[:, 15]))).reshape(-1, 1), data[:, 15])
    ax = plt.gca()
    plt.figure(figsize=(10,6))
    print(data.shape)
    plt.show()

def elliptic_envelope_peak(norm_data, root, canvas):
    pico_norm_data = norm_data[:,15]

    reg = ElasticNet().fit(np.array(range(len(pico_norm_data))).reshape(-1, 1), pico_norm_data)
    res = reg.predict(np.array(range(len(pico_norm_data))).reshape(-1, 1))

    new_data = pico_norm_data - res
    clf = EllipticEnvelope(random_state=0, contamination=0.01).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    #y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]

    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(pico_norm_data))).reshape(-1, 1), pico_norm_data - res)
    plt.plot(np.array(range(len(pico_norm_data))).reshape(-1, 1)[y_res], (pico_norm_data - res)[y_res], "o")
    ax = plt.gca()
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

def peak_caller(data, rise_percent, fall_percent, max_lookback, max_lookahead, root, canvas):
    peaks = []
    n = len(data)
    data = data[:, 15]
    
    for i in range(n):
        # Ajusta el rango de lookback si se excede el rango de datos
        lookback_start = max(0, i - max_lookback)
        
        # Ajusta el rango de lookback para acortar si encuentra un pico
        lookback_range = []
        for j in range(i - 1, lookback_start - 1, -1):
            if j in peaks:
                break
            lookback_range.insert(0, data[j])
        
        # Ajusta el rango de lookahead si se excede el rango de datos
        lookahead_end = min(n, i + max_lookahead + 1)
        
        # Ajusta el range de lookahead para acortar si encuentra un punto mas grande que el actual
        lookahead_range = []
        for j in range(i + 1, lookahead_end):
            if data[j] > data[i]:
                break
            lookahead_range.append(data[j])
        
        # Si no esta vacio el rango de lookback y lookahead se calcula si es un pico
        if len(lookback_range) > 0 and len(lookahead_range) > 0:
            rise = data[i] * (rise_percent / 100.0)
            fall = data[i] * (fall_percent / 100.0)
            
            # Checa si los datos incrementan y decrementan lo suficiente para ser pico
            # Compara el valor actual con el minimo de los datos en el rango de lookback
            # Compara el valor actual con el maximo de los datos en el rango de lookahead
            significant_rise = data[i] - np.min(lookback_range) >= rise
            significant_fall = data[i] - np.min(lookahead_range) >= fall
            
            if significant_rise and significant_fall:
                peaks.append(i)

    fig, ax = plt.subplots()
    plt.plot(data, label='Data')
    plt.scatter(peaks, data[peaks], color='darkorange', label='Peaks')
    plt.title('Peaks found with PeakCaller algorithm and noisy signal')
    ax = plt.gca()
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    return peaks

def local_outlier_factor_peak(data, root, canvas):
    data_sel = data[:, 15]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))

    #plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), data_sel - res)
    #plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), res - 350)
    #mean2 = (380 - 310) / 2
    #ax = plt.gca()
    #plt.show()

    new_data = data_sel - res
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    drawCanvas(root, canvas, data_sel, res, y_res)

def clf_peak(data, root, canvas): #hay dos elliptic envelope?
    data_sel = data[:, 15]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = EllipticEnvelope(random_state=0, contamination=0.01).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    drawCanvas(root, canvas, data_sel, res, y_res)

def isolation_forest_peak(data, root, canvas):
    data_sel = data[:, 15]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = IsolationForest(random_state=0, contamination=0.05).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    drawCanvas(root, canvas, data_sel, res, y_res)

def linear_model_peak(data,root, canvas):
    data_sel = data[:, 15]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = linear_model.SGDOneClassSVM(random_state=42, nu=0.131).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    drawCanvas(root, canvas, data_sel, res, y_res)

def lasso_peak(data, root, canvas): # hay dos local outlier factor
    data_sel = data[:, 15]
    reg = Lasso().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))

    new_data = data_sel - res
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    drawCanvas(root, canvas, data_sel, res, y_res)

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

def plot_corr(df,size, root, canvas):
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

def actually_plot_corr(data, corr, root, canvas):
    df = pd.DataFrame(data)
    X = corr.values
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    # ind = sch.fcluster(L, 0.2*d.max(), 'distance')
    ind = sch.fcluster(L, 50, criterion='maxclust')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    size = 5
    plot_corr(df, size, root, canvas)

def plot_dendrogram(model, **kwargs):
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

def actually_plot_dendo(data, root, canvas):
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data.transpose())
    clustering.labels_.shape
    plt.title("Hierarchical Clustering Dendrogram")
    # plot the top three levels of the dendrogram
    plot_dendrogram(clustering, truncate_mode="none", count_sort='none', show_contracted='true')
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

def programflow(data):
    data = loadData(data)
    normalizedData = normalizeData(data)
    #plot_normalized_data(normalizedData)
    return normalizedData

def drawCanvas(root, canvas, data_sel, res, y_res):
    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), data_sel - res)
    plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), res - 350)
    plt.plot(np.array(range(len(data_sel))).reshape(-1, 1)[y_res], (data_sel - res)[y_res], "o")
    mean2 = (380 - 310) / 2
    ax = plt.gca()
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
