import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.cluster.hierarchy as sch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
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
    
    # Get actual number of series from data
    num_series = norm_data.shape[1]
    
    # Active series - initially all series are active
    active_series = list(range(num_series))
    
    # Get default filename from visualization_helpers if available
    def get_default_time_series_name(extension):
        try:
            from visualization_helpers import loaded_filename
            if loaded_filename:
                import os
                base_name = os.path.splitext(os.path.basename(loaded_filename))[0]
                return f"{base_name}_time_series{extension}"
        except:
            pass
        return f"time_series{extension}"
    
    # Bottom frame for buttons (pack first so it doesn't overlap with plot)
    button_frame = tk.Frame(container)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)
    
    # Main frame (pack after bottom frame)
    main_frame = tk.Frame(container)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))
    
    # Left frame for listbox
    left_frame = tk.Frame(main_frame, width=200)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
    left_frame.pack_propagate(False)  # Maintain fixed width
    
    # Right frame for plots (will contain upper and lower frames)
    right_frame = tk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    # Configure grid for right frame to have two equal rows
    right_frame.grid_rowconfigure(0, weight=1)
    right_frame.grid_rowconfigure(1, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)
    
    # Upper frame for main graph (50% of right frame)
    upper_frame = tk.Frame(right_frame)
    upper_frame.grid(row=0, column=0, sticky='nsew', pady=(0, 5))
    
    # Lower frame for additional content (50% of right frame)
    lower_frame = tk.Frame(right_frame)
    lower_frame.grid(row=1, column=0, sticky='nsew', pady=(5, 0))
    
    # Listbox to left frame
    listbox_label = tk.Label(left_frame, text="Time Series")
    listbox_label.pack(pady=(0, 5))
    
    listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED)
    listbox.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
    
    # Populate listbox with series names based on actual number of columns
    for i in range(num_series):
        listbox.insert(tk.END, f"Series {i+1}")
    
    # Scrollbar for listbox
    scrollbar = tk.Scrollbar(left_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=listbox.yview)
    
    # Plot variables - created when Generate Series is clicked
    fig = None
    ax = None
    canvas_new = None
    
    # Lower frame plot variables - for individual series preview
    fig_lower = None
    ax_lower = None
    canvas_lower = None
    
    def update_plot():
        if ax is not None:
            ax.clear()
            for i, series_idx in enumerate(active_series):
                ax.plot(np.array(range(len(norm_data[:, series_idx]))).reshape(-1, 1), 
                       norm_data[:, series_idx] + i)
            ax.set_title('Time Series Plot')
            ax.set_xlabel('Time')
            ax.set_ylabel('Signal + Offset')
            canvas_new.draw()
    
    def generate_series():
        nonlocal fig, ax, canvas_new, active_series
        
        # Get selected items from listbox
        selection = listbox.curselection()
        
        if selection:
            # If series are selected, use only those
            active_series = [active_series[idx] for idx in selection]
            # Update listbox to show only selected series
            listbox.delete(0, tk.END)
            for idx in active_series:
                listbox.insert(tk.END, column_names[idx])
        else:
            # If no selection, use all series
            active_series = list(range(num_series))
        
        # Close previous figure if it exists
        if fig is not None:
            plt.close(fig)
        
        # Clear upper frame and create plot
        for widget in upper_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        canvas_new = FigureCanvasTkAgg(fig, master=upper_frame)
        canvas_new.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        update_plot()
    
    def display_single_series(event=None):
        nonlocal fig_lower, ax_lower, canvas_lower
        
        # Get the currently selected item (only show one at a time)
        selection = listbox.curselection()
        if not selection:
            return
        
        # Get the first selected item
        selected_idx = selection[0]
        if selected_idx >= len(active_series):
            return
        
        # Get the actual series index from active_series
        series_idx = active_series[selected_idx]
        
        # Close previous figure if it exists
        if fig_lower is not None:
            plt.close(fig_lower)
        
        # Clear lower frame and create plot
        for widget in lower_frame.winfo_children():
            widget.destroy()
        
        fig_lower, ax_lower = plt.subplots(figsize=(8, 4))
        canvas_lower = FigureCanvasTkAgg(fig_lower, master=lower_frame)
        canvas_lower.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Plot the single selected series
        ax_lower.plot(np.array(range(len(norm_data[:, series_idx]))).reshape(-1, 1), 
                     norm_data[:, series_idx])
        ax_lower.set_title(column_names[series_idx])
        ax_lower.set_xlabel('Time')
        ax_lower.set_ylabel('Signal')
        canvas_lower.draw()
    
    # Bind listbox selection to display single series in lower frame
    listbox.bind('<<ListboxSelect>>', display_single_series)
    
    def delete_selected_series():
        selection = listbox.curselection()
        if selection:
            selected_indices = sorted(selection, reverse=True)
            for selected_idx in selected_indices:
                if selected_idx < len(active_series):
                    active_series.pop(selected_idx)
                    listbox.delete(selected_idx)
            update_plot()
    
    def reset_series():
        nonlocal active_series
        
        # Reset active series to all series
        active_series = list(range(num_series))
        
        # Clear and repopulate listbox with all series
        listbox.delete(0, tk.END)
        for i in range(num_series):
            listbox.insert(tk.END, column_names[i])
    
    # Generate Series button
    generate_button = tk.Button(
        left_frame,
        text="Generate Series",
        command=generate_series
    )
    generate_button.pack(fill=tk.X, pady=(5, 0))
    
    # Delete button for listbox
    delete_button = tk.Button(
        left_frame,
        text="Delete Series",
        command=delete_selected_series
    )
    delete_button.pack(fill=tk.X, pady=(5, 0))
    
    # Reset button for listbox
    reset_button = tk.Button(
        left_frame,
        text="Reset Series",
        command=reset_series
    )
    reset_button.pack(fill=tk.X, pady=(5, 0))
    
    # Save functions for bottom frame buttons
    def _save_time_series_image():
        default_name = get_default_time_series_name('.png')
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
    
    def _save_time_series_csv():
        default_name = get_default_time_series_name('.csv')
        filename = asksaveasfilename(
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All Files", "*.*")]
        )
        if filename:
            # Create DataFrame with only active series
            active_data = norm_data[:, active_series]
            df = pd.DataFrame(active_data, columns=[column_names[i] for i in active_series])
            df.to_csv(filename, index=False)
    
    # Save image button
    save_img_button = tk.Button(
        button_frame,
        text="Save Image",
        command=_save_time_series_image,
        width=12
    )
    save_img_button.pack(side=tk.LEFT, padx=5)
    
    # Save CSV button
    save_csv_button = tk.Button(
        button_frame,
        text="Save CSV",
        command=_save_time_series_csv,
        width=12
    )
    save_csv_button.pack(side=tk.LEFT, padx=5)
    
    # Close button
    def close_window():
        if fig is not None:
            plt.close(fig)
        if is_tab:
            try:
                notebook.forget(container)
            except Exception:
                pass
        else:
            container.destroy()

    close_label = "Close Tab" if is_tab else "Close"
    close_button = tk.Button(button_frame, text=close_label, command=close_window, width=12)
    close_button.pack(side=tk.RIGHT, padx=5)

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
