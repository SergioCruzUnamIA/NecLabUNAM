import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.cluster.hierarchy as sch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
from tkinter.filedialog import asksaveasfilename

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
    
    # Add save button for correlation matrix
    def _save_correlation():
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension="Untitled.csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")],
            title="Save Correlation Matrix"
        )
        if filename:
            corr_func.to_csv(filename, header=False, index=False)
    
    def _save_image():
        filename = asksaveasfilename(
            initialfile = 'Untitled.png',
            defaultextension=".png",
            filetypes=[("All Files","*.*"),("Portable Graphics Format","*.png")]
        )
        plt.savefig(filename)
    
    button_frame = tk.Frame(root)
    button_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
    save_csv_button = tk.Button(
        button_frame,
        text="Save Correlation Matrix",
        command=_save_correlation
    )
    save_img_button = tk.Button(
        button_frame,
        text="Save Image",
        command=_save_image
    )
    save_csv_button.pack(side=tk.LEFT, padx=5)
    save_img_button.pack(side=tk.LEFT, padx=5)

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
    # plot the top three levels of the dendrogram
    _plot_dendrogram_helper(clustering, truncate_mode="none", count_sort='none', show_contracted='true')
    for i in range(20):
        plt.plot(np.array(range(len(data[:, i]))).reshape(-1, 1), data[:, i] + i)
    #ax = plt.gca()
    fig = plt.gcf()
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

def plot_time_series(norm_data):
    # New Tkinter window
    plot_window = tk.Toplevel()
    plot_window.title('Time Series Plot')
    plot_window.geometry('1000x1000')
    
    # Active series
    active_series = list(range(20))  # Initially all series are active
    
    # Main frame
    main_frame = tk.Frame(plot_window)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Left frame for listbox
    left_frame = tk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
    left_frame.pack_propagate(False)  # Maintain fixed width
    
    # Right frame for plot
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Listbox to left frame
    listbox_label = tk.Label(left_frame, text="Time Series")
    listbox_label.pack(pady=(0, 5))
    
    listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED)
    listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    # Populate listbox with series names
    for i in range(20): #TODO: check if only 20
        listbox.insert(tk.END, f"Series {i+1}")
    
    # Scrollbar for listbox
    scrollbar = tk.Scrollbar(left_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    
    # Initial plot
    fig, ax = plt.subplots(figsize=(8, 6))
    canvas_new = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_new.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def update_plot():
        """
            Update the plot with current active series
        """
        ax.clear()
        for i, series_idx in enumerate(active_series):
            ax.plot(np.array(range(len(norm_data[:, series_idx]))).reshape(-1, 1), 
                   norm_data[:, series_idx] + i)
        ax.set_title('Time Series Plot')
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal + Offset')
        canvas_new.draw()
    
    def delete_selected_series():
        """
            Deletes the selected series from listbox and replots
        """
        selection = listbox.curselection()
        if selection:
            selected_indices = sorted(selection, reverse=True)
            for selected_idx in selected_indices:
                if selected_idx < len(active_series):
                    active_series.pop(selected_idx)
                    listbox.delete(selected_idx)
            update_plot()
    
    update_plot()
    
    # Delete button for listbox
    delete_button = tk.Button(
        left_frame,
        text="Delete Series",
        command=delete_selected_series
    )
    delete_button.pack(fill=tk.X, pady=(5, 0))
    
    # Bottom frame
    button_frame = tk.Frame(plot_window)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)
    
    def _save_time_series_image():
        """
            Saves the time series as an image
        """
        filename = asksaveasfilename(
            initialfile='TimeSeries.png',
            defaultextension=".png",
            filetypes=[("Portable Graphics Format", "*.png"), ("All Files", "*.*")]
        )
        if filename:
            fig.savefig(filename)
    
    def _save_time_series_csv():
        """
            Saves the time series as a .csv file
        """
        filename = asksaveasfilename(
            initialfile='TimeSeries.csv',
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            # Create DataFrame with only active series
            active_data = norm_data[:, active_series]
            df = pd.DataFrame(active_data, columns=[f"Series_{i+1}" for i in active_series])
            df.to_csv(filename, index=False)
    
    # Save image button
    save_img_button = tk.Button(
        button_frame,
        text="Save Image",
        command=_save_time_series_image
    )
    save_img_button.pack(side=tk.LEFT, padx=5)
    
    # Save CSV button
    save_csv_button = tk.Button(
        button_frame,
        text="Save CSV",
        command=_save_time_series_csv
    )
    save_csv_button.pack(side=tk.LEFT, padx=5)
    
    # Close button
    close_button = tk.Button(button_frame, text="Close", command=plot_window.destroy)
    close_button.pack(side=tk.RIGHT, padx=5)