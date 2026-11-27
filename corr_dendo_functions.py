import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename, askopenfilename
from tkinter import messagebox

from neclab_core import correlation as core_correlation
from neclab_core import stages as core_stages
from neclab_core.validation import ValidationError

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
    return core_correlation.correlation_matrix(data, 'pearson')

def correlation_kendall(data):
    plt.close('all')
    return core_correlation.correlation_matrix(data, 'kendall')

def correlation_spearman(data):
    plt.close('all')
    return core_correlation.correlation_matrix(data, 'spearman')

def _plot_dendrogram_helper(corr, ax=None):
    """Draw the dendrogram of a correlation matrix with the clustering
    shared by every view (neclab_core.correlation.cluster_linkage)."""
    labels = [str(c) for c in corr.columns] if hasattr(corr, "columns") else None
    dendrogram(core_correlation.cluster_linkage(corr), labels=labels, ax=ax, leaf_rotation=90)
    if ax is not None:
        ax.set_ylabel("Distance")


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
    
    # Create figure with two subplots: dendrogram on top, correlation below (50/50 split)
    fig = plt.figure(figsize=(size, size * 2))
    
    # Dendrogram subplot (top, 50%)
    ax_dendro = plt.subplot2grid((2, 1), (0, 0), rowspan=1)
    
    # Dendrogram of the correlation matrix (shared clustering)
    try:
        _plot_dendrogram_helper(corr_func, ax=ax_dendro)
        ax_dendro.set_title(f'Dendrogram ({core_correlation.LINKAGE_DESCRIPTION})')
    except ValueError as e:
        ax_dendro.text(0.5, 0.5, f"Can't build the dendrogram:\n{e}", ha='center', va='center',
                       transform=ax_dendro.transAxes, wrap=True)
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
    
    # Add save button for correlation matrix CSV/Excel
    def _save_correlation_csv():
        default_name = get_default_correlation_name('.csv')
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All Files", "*.*")],
            title="Save Correlation Matrix"
        )
        if filename:
            if filename.lower().endswith(('.xlsx', '.xls')):
                corr_func.to_excel(filename, header=True, index=True, engine='xlsxwriter')
            else:
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
            _plot_dendrogram_helper(corr_func, ax=ax_dendro_temp)
            ax_dendro_temp.set_title(f'Dendrogram ({core_correlation.LINKAGE_DESCRIPTION})')
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

def plot_time_series(norm_data, column_names=None, notebook=None, parent=None, after=None, select=True):
    """Time Series view of the columns of `norm_data` (samples x columns):
    a single-signal preview and an overlay of the selected columns. A tab
    of `notebook` (right after the tab `after`, if given), or a window
    without one. Returns its ui_tabs.Tab."""
    from ui_tabs import add_tab

    tab = add_tab(notebook, 'Time Series', parent=parent, after=after, select=select)
    container = tab.body

    num_series = norm_data.shape[1]
    if column_names is None:
        column_names = [f"Column {i+1}" for i in range(num_series)]

    # Mutable state
    selection_indices = []
    ts_figs = {'signal': None, 'multi': None}
    mouse_click = {'active': False}

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
    def draw_signal(idx):
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

    def on_column_click(event):
        """Set the displayed signal from the exact row under the mouse."""
        mouse_click['active'] = False
        idx = col_listbox.nearest(event.y)
        if idx < 0:
            return
        draw_signal(idx)

    def show_signal(event=None):
        """Keyboard navigation only; mouse clicks are handled by on_column_click."""
        if mouse_click['active']:
            return
        sel = col_listbox.curselection()
        if not sel:
            return
        draw_signal(sel[-1])

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

    col_listbox.bind('<ButtonPress-1>', lambda e: mouse_click.__setitem__('active', True))
    col_listbox.bind('<<ListboxSelect>>', show_signal)
    col_listbox.bind('<ButtonRelease-1>', on_column_click)

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
                       ("TIFF Image", "*.tiff"), ("SVG Vector", "*.svg"),
                       ("EPS Vector", "*.eps"), ("All Files", "*.*")]
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
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All Files", "*.*")]
        )
        if filename:
            cols = [column_names[i] if i < len(column_names) else f"Column {i+1}"
                    for i in selection_indices]
            df_out = pd.DataFrame(norm_data[:, selection_indices], columns=cols)
            if filename.lower().endswith(('.xlsx', '.xls')):
                df_out.to_excel(filename, index=False, engine='xlsxwriter')
            else:
                df_out.to_csv(filename, index=False)

    def release_figures(event):
        if event.widget is not tab.frame:
            return
        for key in ('signal', 'multi'):
            if ts_figs[key] is not None:
                plt.close(ts_figs[key])
                ts_figs[key] = None

    tab.frame.bind('<Destroy>', release_figures, add='+')
    tk.Button(button_frame, text="Save Image", command=save_image, width=12).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Save CSV", command=save_csv, width=12).pack(side=tk.LEFT, padx=5)
    return tab

def load_correlation_matrix(root, canvas, target_frame=None):
    """
    Load a correlation matrix from a CSV file and display it.
    'target_frame', if given, is drawn into directly instead of the frame
    _plot_correlation_helper falls back to finding inside 'root'.
    Returns (matrix DataFrame, filename), or None if nothing was loaded.
    """
    try:
        # Open file dialog to select correlation matrix file
        filename = askopenfilename(
            title="Load Correlation Matrix",
            filetypes=core_stages.open_filetypes("correlation")
        )

        if not filename:
            return  # User cancelled

        options = {}
        if filename.lower().endswith(('.xlsx', '.xls')):
            sheet_names = pd.ExcelFile(filename).sheet_names
            options['sheet_name'] = sheet_names[0]
            if len(sheet_names) > 1:
                from visualization_helpers import _show_sheet_selection_dialog
                chosen = _show_sheet_selection_dialog(root, sheet_names)
                if chosen is None:
                    return  # User cancelled sheet selection
                options['sheet_name'] = chosen

        # Read + validate (square, numeric, same labels) via the stage-7 format
        try:
            loaded = core_stages.load_item("correlation", filename, **options)
        except ValidationError as e:
            messagebox.showerror("Invalid Matrix", str(e))
            return
        corr_matrix = loaded.data
        if loaded.warnings and not messagebox.askyesno(
                "Warning", "\n\n".join(loaded.warnings) + "\n\nContinue anyway?"):
            return

        # Display the correlation matrix
        size = min(10, max(5, corr_matrix.shape[0] * 0.5))  # Dynamic size based on matrix dimensions
        canvas, _ = _plot_correlation_helper(corr_matrix, size, root, canvas, is_precomputed_corr=True,
                                              corr_method='Loaded Correlation', target_frame=target_frame)
        
        messagebox.showinfo(
            "Matrix Loaded",
            f"Successfully loaded correlation matrix of size {corr_matrix.shape[0]}x{corr_matrix.shape[1]}"
        )
        return corr_matrix, filename

    except pd.errors.EmptyDataError:
        messagebox.showerror("Error", "The selected file is empty or invalid.")
    except pd.errors.ParserError:
        messagebox.showerror("Error", "Could not parse the file. Please ensure it's a valid CSV or Excel file.")
    except FileNotFoundError:
        messagebox.showerror("Error", "The selected file was not found.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while loading the matrix:\n{str(e)}")
