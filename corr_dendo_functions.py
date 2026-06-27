import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.cluster.hierarchy as sch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilename
from tkinter import messagebox

def get_main_plot_frame(main_window):
    """
    Find the main plot frame from the main window hierarchy.
    In the simplified layout, main_plot_frame is directly in the window at column=1.
    """
    if main_window is None:
        return None
    
    # Look for the frame at column 1 (the right side plot area)
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame):
            # Get grid info to check if it's at column 1
            try:
                grid_info = widget.grid_info()
                if grid_info.get('column') == 1:
                    return widget
            except:
                continue
    
    # Fallback to main_window if not found
    return main_window

def correlation_pearson(data):
    plt.close('all')
    df = pd.DataFrame(data)
    corr1 = df.corr(method='pearson')
    return corr1

def correlation_kendall(data):
    plt.close('all')
    df = pd.DataFrame(data)
    corr1 = df.corr(method='kendall')
    return corr1

def correlation_spearman(data):
    plt.close('all')
    df = pd.DataFrame(data)
    corr1 = df.corr(method='spearman')
    return corr1

def _plot_dendrogram_helper(model, **kwargs):
    """
    Helper function to create and plot a dendrogram from an AgglomerativeClustering model.
    
    Args:
        model: Fitted AgglomerativeClustering model
        **kwargs: Additional arguments passed to scipy.cluster.hierarchy.dendrogram
    """
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
    dendrogram(linkage_matrix, **kwargs)

def _plot_correlation_helper(df, size, root, canvas, is_precomputed_corr=False, corr_method=None, target_frame=None):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
        is_precomputed_corr: if True, df is already a correlation matrix
        corr_method: the correlation method used (e.g., 'pearson', 'kendall', 'spearman')
        target_frame: if provided, render into this frame directly instead of the main plot frame'''
    if target_frame is not None:
        plot_frame = target_frame
    else:
        plot_frame = get_main_plot_frame(root)
        for widget in list(plot_frame.winfo_children()):
            widget.destroy()
        plt.close('all')
    
    # Compute the correlation matrix for the received dataframe or use as-is
    if is_precomputed_corr:
        corr_func = df  # df is already a correlation matrix
    else:
        corr_func = df.corr()  # compute correlation matrix from data
    
    # For dendrogram, use correlation matrix values for clustering
    data_for_dendro = corr_func.values
    
    # Create figure with two subplots: dendrogram on top, correlation below (50/50 split)
    fig = plt.figure(figsize=(size, size * 2))
    
    # Dendrogram subplot (top, 50%)
    ax_dendro = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    
    # Compute dendrogram from correlation matrix
    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data_for_dendro)
    _plot_dendrogram_helper(clustering, truncate_mode="none", count_sort='none', show_contracted='true')
    ax_dendro.set_title('Dendrogram')
    
    # Correlation matrix subplot (bottom, 50%)
    ax_corr = plt.subplot2grid((2, 1), (1, 0), rowspan=1)
    cax = ax_corr.matshow(corr_func, cmap='jet', aspect='equal')
    
    # Set title based on correlation method
    title = f'{corr_method.capitalize()} Correlation Matrix' if corr_method else 'Correlation Matrix'
    ax_corr.set_title(title)
    
    # Add the colorbar legend
    cbar = fig.colorbar(cax, ax=ax_corr, ticks=[-1, 0, 1], aspect=40, shrink=.8)
    
    # Add spacing between subplots and adjust margins to center content
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05, left=0.1, right=0.9)
    
    # Get default filename from visualization_helpers if available
    def get_default_correlation_name(extension):
        try:
            from visualization_helpers import loaded_filename
            if loaded_filename:
                import os
                base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
                method_str = f"_{corr_method}" if corr_method else ""
                return f"{base_name}_correlation{method_str}{extension}"
        except:
            pass
        method_str = f"_{corr_method}" if corr_method else ""
        return f"correlation{method_str}{extension}"
    
    # Add save button for correlation matrix CSV
    def _save_correlation_csv():
        default_name = get_default_correlation_name('.csv')
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")],
            title="Save Correlation Matrix"
        )
        if filename:
            corr_func.to_csv(filename, header=True, index=True)
    
    def _save_correlation_image():
        default_name = get_default_correlation_name('.png')
        filename = asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("TIFF Image", "*.tiff"),
                ("SVG Vector", "*.svg"),
                ("EPS Vector", "*.eps"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            # Save only correlation matrix
            fig_corr = plt.figure(figsize=(size, size))
            ax_temp = fig_corr.add_subplot(111)
            cax_temp = ax_temp.matshow(corr_func, cmap='jet', aspect='equal')
            title_str = f'{corr_method.capitalize()} Correlation Matrix' if corr_method else 'Correlation Matrix'
            ax_temp.set_title(title_str)
            fig_corr.colorbar(cax_temp, ticks=[-1, 0, 1], aspect=40, shrink=.8)
            fig_corr.savefig(filename)
            plt.close(fig_corr)
    
    def _save_dendrogram_image():
        """Save dendrogram part of the plot"""
        default_name = get_default_correlation_name('_dendrogram.png')
        filename = asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("TIFF Image", "*.tiff"),
                ("SVG Vector", "*.svg"),
                ("EPS Vector", "*.eps"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            # Save only dendrogram
            fig_dendro = plt.figure(figsize=(size, size))
            ax_dendro_temp = fig_dendro.add_subplot(111)
            plt.sca(ax_dendro_temp)
            clustering_temp = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data_for_dendro)
            _plot_dendrogram_helper(clustering_temp, truncate_mode="none", count_sort='none', show_contracted='true')
            ax_dendro_temp.set_title('Dendrogram')
            fig_dendro.savefig(filename)
            plt.close(fig_dendro)
    
    def _save_all():
        default_name = get_default_correlation_name('_combined.png')
        filename = asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("TIFF Image", "*.tiff"),
                ("SVG Vector", "*.svg"),
                ("EPS Vector", "*.eps"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            fig.savefig(filename)
    
    # Create button frame first so it claims its space before the canvas expands
    button_frame = tk.Frame(plot_frame)
    button_frame.peak_button_frame = True  # Mark for cleanup
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # Display figure in the plot frame (packed after buttons so expand fills remaining space)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Create and pack buttons
    save_dendro_button = tk.Button(
        button_frame,
        text="Save Dendrogram",
        command=_save_dendrogram_image
    )
    save_csv_button = tk.Button(
        button_frame,
        text="Save Correlation Matrix",
        command=_save_correlation_csv
    )
    save_corr_img_button = tk.Button(
        button_frame,
        text="Save Correlation Matrix Image",
        command=_save_correlation_image
    )
    save_all_button = tk.Button(
        button_frame,
        text="Save All",
        command=_save_all
    )
    
    save_dendro_button.pack(side=tk.LEFT, padx=5)
    save_csv_button.pack(side=tk.LEFT, padx=5)
    save_corr_img_button.pack(side=tk.LEFT, padx=5)
    save_all_button.pack(side=tk.LEFT, padx=5)

    return canvas, fig

def plot_correlation(data, corr, root, canvas, corr_method=None):
    # Clean up by clearing the plot frame (don't rely on stale canvas reference)
    plot_frame = get_main_plot_frame(root)
    for widget in list(plot_frame.winfo_children()):
        widget.destroy()
    plt.close('all')

    # Also remove any existing button frames from root
    for widget in list(root.winfo_children()):
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()

    df = pd.DataFrame(data)
    X = corr.values
    d = sch.distance.pdist(X)   # vector of ('55' choose 2) pairwise distances
    L = sch.linkage(d, method='complete')
    ind = sch.fcluster(L, 50, criterion='maxclust')
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)
    size = 5
    canvas, _ = _plot_correlation_helper(df, size, root, canvas, corr_method=corr_method)
    return canvas

def plot_dendogram(data, root, canvas):
    # Remove any existing buttons
    for widget in list(root.winfo_children()):
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    # Get the main plot frame
    plot_frame = get_main_plot_frame(root)
    
    # Clear ALL widgets from the plot frame
    for widget in list(plot_frame.winfo_children()):
        widget.destroy()
    
    # Close all matplotlib figures
    plt.close('all')

    clustering = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(data.transpose())
    clustering.labels_.shape
    # plot the top three levels of the dendrogram
    _plot_dendrogram_helper(clustering, truncate_mode="none", count_sort='none', show_contracted='true')
    fig = plt.gcf()
    
    # Create canvas in the plot frame using pack
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Get default filename from visualization_helpers if available
    def get_default_dendogram_name(extension):
        try:
            from visualization_helpers import loaded_filename
            if loaded_filename:
                import os
                base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
                return f"{base_name}_dendrogram{extension}"
        except:
            pass
        return f"dendrogram{extension}"
    
    # Add save buttons for dendrogram
    def _save_dendrogram_image():
        default_name = get_default_dendogram_name('.png')
        filename = asksaveasfilename(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("TIFF Image", "*.tiff"),
                ("SVG Vector", "*.svg"),
                ("EPS Vector", "*.eps"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            fig.savefig(filename)
    
    def _save_dendrogram_csv():
        """Save dendrogram clustering data to CSV"""
        default_name = get_default_dendogram_name('.csv')
        filename = asksaveasfilename(
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")],
            title="Save Dendrogram Data"
        )
        
        if filename:
            # Create DataFrame with clustering information
            # Include cluster labels and linkage information
            df_data = {
                'Sample_Index': list(range(len(clustering.labels_))),
                'Cluster_Label': clustering.labels_
            }
            
            df = pd.DataFrame(df_data)
            
            # Add linkage matrix information as separate section
            linkage_data = []
            for i, (child1, child2) in enumerate(clustering.children_):
                linkage_data.append({
                    'Merge_Step': i,
                    'Child_1': int(child1),
                    'Child_2': int(child2),
                    'Distance': clustering.distances_[i]
                })
            
            df_linkage = pd.DataFrame(linkage_data)
            
            # Save both dataframes to CSV
            with open(filename, 'w', newline='') as f:
                f.write("# Cluster Labels\n")
                df.to_csv(f, index=False)
                f.write("\n# Linkage Matrix\n")
                df_linkage.to_csv(f, index=False)
            
            messagebox.showinfo("Success", f"Dendrogram data saved to {filename}")
    
    # Create button frame for dendrogram inside the plot frame using pack
    button_frame = tk.Frame(plot_frame)
    button_frame.peak_button_frame = True  # Mark for cleanup
    button_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Create save buttons
    save_img_button = tk.Button(
        button_frame,
        text="Save Image",
        command=_save_dendrogram_image
    )
    save_img_button.pack(side=tk.LEFT, padx=5)
    
    save_csv_button = tk.Button(
        button_frame,
        text="Save CSV",
        command=_save_dendrogram_csv
    )
    save_csv_button.pack(side=tk.LEFT, padx=5)
    
    # Configure button frame
    root.grid_rowconfigure(1, weight=0)  # Don't expand button row
    
    return canvas

def plot_time_series(norm_data, column_names=None, notebook=None):
    if notebook is not None:
        container = tk.Frame(notebook)
        notebook.add(container, text='Time Series')
        notebook.select(container)
        is_tab = True
    else:
        container = tk.Toplevel()
        container.title('Time Series Plot')
        screen_width = container.winfo_screenwidth()
        screen_height = container.winfo_screenheight()
        win_width = int(screen_width * 0.8)
        win_height = int(screen_height * 0.8)
        x = (screen_width - win_width) // 2
        y = (screen_height - win_height) // 2
        container.geometry(f"{win_width}x{win_height}+{x}+{y}")
        is_tab = False

    num_series = norm_data.shape[1]
    if column_names is None:
        column_names = [f"Column {i+1}" for i in range(num_series)]

    # Mutable state
    selection_indices = []
    ts_figs = {'signal': None, 'multi': None}

    def get_default_name(extension):
        try:
            from visualization_helpers import loaded_filename
            if loaded_filename:
                import os
                base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
                return f"{base_name}_time_series{extension}"
        except Exception:
            pass
        return f"time_series{extension}"

    # ── Bottom button bar (pack first to reserve space) ──
    button_frame = tk.Frame(container)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)

    # ── Content area ──
    content_frame = tk.Frame(container)
    content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 0))

    # Sidebar (left, fixed width)
    sidebar = tk.Frame(content_frame, relief=tk.RAISED, borderwidth=1, width=220)
    sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
    sidebar.pack_propagate(False)

    # Plot area (right, split top/bottom)
    plot_area = tk.Frame(content_frame, relief=tk.RAISED, borderwidth=1)
    plot_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    plot_area.rowconfigure(0, weight=1)
    plot_area.rowconfigure(1, weight=1)
    plot_area.columnconfigure(0, weight=1)

    top_plot = tk.Frame(plot_area)
    top_plot.grid(row=0, column=0, sticky='nsew')

    bottom_plot = tk.Frame(plot_area, relief=tk.GROOVE, borderwidth=1)
    bottom_plot.grid(row=1, column=0, sticky='nsew')

    # ── Sidebar: Data Columns ──
    tk.Label(sidebar, text="Data Columns", font=("Arial", 11, "bold")).pack(pady=(10, 5))

    col_lb_frame = tk.Frame(sidebar)
    col_lb_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    col_scrollbar = tk.Scrollbar(col_lb_frame)
    col_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    col_listbox = tk.Listbox(col_lb_frame, yscrollcommand=col_scrollbar.set,
                              selectmode=tk.EXTENDED, font=("Arial", 10))
    col_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    col_scrollbar.config(command=col_listbox.yview)

    for i in range(num_series):
        col_listbox.insert(tk.END, column_names[i] if i < len(column_names) else f"Column {i+1}")

    # ── Sidebar: Selection ──
    ttk.Separator(sidebar, orient='horizontal').pack(fill='x', padx=5, pady=5)

    tk.Label(sidebar, text="Selection", font=("Arial", 11, "bold")).pack(pady=(0, 5))

    sel_lb_frame = tk.Frame(sidebar)
    sel_lb_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    sel_scrollbar = tk.Scrollbar(sel_lb_frame)
    sel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    sel_listbox = tk.Listbox(sel_lb_frame, yscrollcommand=sel_scrollbar.set,
                              selectmode=tk.SINGLE, font=("Arial", 10))
    sel_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    sel_scrollbar.config(command=sel_listbox.yview)

    btn_add = tk.Button(sidebar, text="Add to Selection", command=lambda: add_to_selection())
    btn_add.pack(fill=tk.X, padx=5, pady=(2, 2))

    btn_remove = tk.Button(sidebar, text="Remove from Selection", command=lambda: remove_from_selection())
    btn_remove.pack(fill=tk.X, padx=5, pady=(0, 5))

    show_labels_var = tk.BooleanVar(value=True)
    ttk.Separator(sidebar, orient='horizontal').pack(fill='x', padx=5, pady=5)
    ttk.Checkbutton(
        sidebar, text="Show Labels",
        variable=show_labels_var,
        command=lambda: update_multi_series()
    ).pack(anchor='w', padx=5, pady=(0, 5))

    # ── Plot helpers ──
    def show_signal(event=None):
        sel = col_listbox.curselection()
        if not sel:
            return
        try:
            idx = col_listbox.index(tk.ACTIVE)
        except Exception:
            idx = sel[0]
        if ts_figs['signal'] is not None:
            plt.close(ts_figs['signal'])
            ts_figs['signal'] = None
        for w in list(top_plot.winfo_children()):
            w.destroy()
        col_name = column_names[idx] if idx < len(column_names) else f"Column {idx+1}"
        fig, ax = plt.subplots()
        ax.plot(np.array(range(len(norm_data[:, idx]))).reshape(-1, 1), norm_data[:, idx])
        ax.set_title(col_name)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        fig.tight_layout()
        ts_figs['signal'] = fig
        c = FigureCanvasTkAgg(fig, master=top_plot)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_multi_series():
        if ts_figs['multi'] is not None:
            plt.close(ts_figs['multi'])
            ts_figs['multi'] = None
        for w in list(bottom_plot.winfo_children()):
            w.destroy()
        if not selection_indices:
            tk.Label(bottom_plot,
                     text="Add columns to Selection to see time series",
                     font=("Arial", 12), fg="#666666").pack(fill=tk.BOTH, expand=True)
            return
        fig, ax = plt.subplots()
        for i, idx in enumerate(selection_indices):
            col_name = column_names[idx] if idx < len(column_names) else f"Column {idx+1}"
            ax.plot(np.array(range(len(norm_data[:, idx]))).reshape(-1, 1),
                    norm_data[:, idx] + i, label=col_name)
        ax.set_title('Time Series')
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal + Offset')
        if show_labels_var.get():
            ax.legend(fontsize=8, loc='upper right')
        fig.tight_layout()
        ts_figs['multi'] = fig
        c = FigureCanvasTkAgg(fig, master=bottom_plot)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def add_to_selection():
        sel = col_listbox.curselection()
        if not sel:
            return
        changed = False
        for idx in sel:
            if idx not in selection_indices:
                selection_indices.append(idx)
                changed = True
        if changed:
            selection_indices.sort()
            sel_listbox.delete(0, tk.END)
            for idx in selection_indices:
                col_name = column_names[idx] if idx < len(column_names) else f"Column {idx+1}"
                sel_listbox.insert(tk.END, col_name)
            update_multi_series()

    def remove_from_selection():
        sel = sel_listbox.curselection()
        if not sel:
            return
        list_idx = sel[0]
        sel_listbox.delete(list_idx)
        selection_indices.pop(list_idx)
        update_multi_series()

    col_listbox.bind('<<ListboxSelect>>', show_signal)

    # ── Initial placeholders ──
    tk.Label(top_plot, text="Click a column to view its signal",
             font=("Arial", 12), fg="#666666").pack(fill=tk.BOTH, expand=True)
    tk.Label(bottom_plot, text="Add columns to Selection to see time series",
             font=("Arial", 12), fg="#666666").pack(fill=tk.BOTH, expand=True)

    # ── Bottom buttons ──
    def save_image():
        if ts_figs['multi'] is None:
            messagebox.showwarning("No plot", "Add columns to selection first.")
            return
        filename = asksaveasfilename(
            initialfile=get_default_name('.png'),
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"),
                       ("TIFF Image", "*.tiff"), ("SVG Vector", "*.svg"), ("All Files", "*.*")]
        )
        if filename:
            ts_figs['multi'].savefig(filename)

    def save_csv():
        if not selection_indices:
            messagebox.showwarning("No selection", "Add columns to selection first.")
            return
        filename = asksaveasfilename(
            initialfile=get_default_name('.csv'),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            cols = [column_names[i] if i < len(column_names) else f"Column {i+1}"
                    for i in selection_indices]
            pd.DataFrame(norm_data[:, selection_indices], columns=cols).to_csv(filename, index=False)

    def close_window():
        for key in ('signal', 'multi'):
            if ts_figs[key] is not None:
                plt.close(ts_figs[key])
        if is_tab:
            try:
                notebook.forget(container)
            except Exception:
                pass
        else:
            container.destroy()

    tk.Button(button_frame, text="Save Image", command=save_image, width=12).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Save CSV", command=save_csv, width=12).pack(side=tk.LEFT, padx=5)
    close_label = "Close Tab" if is_tab else "Close"
    tk.Button(button_frame, text=close_label, command=close_window, width=12).pack(side=tk.RIGHT, padx=5)

def load_correlation_matrix(root, canvas):
    """
    Load a correlation matrix from a CSV file and display it
    """
    try:
        # Open file dialog to select correlation matrix file
        filename = askopenfilename(
            title="Load Correlation Matrix",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("All files", "*.*")
            ]
        )
        
        if not filename:
            return  # User cancelled
        
        # Load the correlation matrix based on file extension
        if filename.lower().endswith('.csv'):
            corr_matrix = pd.read_csv(filename, index_col=0)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            corr_matrix = pd.read_excel(filename, index_col=0)
        else:
            # Try CSV as default
            corr_matrix = pd.read_csv(filename, index_col=0)
        
        # Validate that it's a square matrix (correlation matrices should be square)
        if corr_matrix.shape[0] != corr_matrix.shape[1]:
            messagebox.showerror(
                "Invalid Matrix", 
                "The loaded matrix is not square. Correlation matrices must be square."
            )
            return
        
        # Check if values are in valid correlation range [-1, 1]
        if (corr_matrix.min().min() < -1.1) or (corr_matrix.max().max() > 1.1):
            response = messagebox.askyesno(
                "Warning", 
                "The matrix contains values outside the typical correlation range [-1, 1]. Continue anyway?"
            )
            if not response:
                return
        
        # Display the correlation matrix
        size = min(10, max(5, corr_matrix.shape[0] * 0.5))  # Dynamic size based on matrix dimensions
        canvas, _ = _plot_correlation_helper(corr_matrix, size, root, canvas, is_precomputed_corr=True, corr_method='Loaded Correlation')
        
        messagebox.showinfo(
            "Matrix Loaded", 
            f"Successfully loaded correlation matrix of size {corr_matrix.shape[0]}x{corr_matrix.shape[1]}"
        )
        
    except pd.errors.EmptyDataError:
        messagebox.showerror("Error", "The selected file is empty or invalid.")
    except pd.errors.ParserError:
        messagebox.showerror("Error", "Could not parse the file. Please ensure it's a valid CSV or Excel file.")
    except FileNotFoundError:
        messagebox.showerror("Error", "The selected file was not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the matrix:\n{str(e)}")
