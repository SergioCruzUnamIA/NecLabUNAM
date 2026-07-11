from tkinter import *
from tkinter import filedialog
from peak_functions import *
from corr_dendo_functions import *
import scipy.cluster.hierarchy as sch
import os
import pandas as pd
import tkinter as tk

# Global variables to store file and ROI information
selected_roi_index = None
loaded_filename = None
selected_roi_name = None
column_listbox = None  # Reference set by interface after layout creation
column_select_callback = None  # Callback for when column selection changes

def set_column_listbox(listbox):
    """Set reference to the column listbox widget from the main interface."""
    global column_listbox
    column_listbox = listbox

def initialize_visualization(window, menu_picos, canvas, listbox=None, on_column_select=None,
                              notebook=None, on_complete=None):
    """Opens a data file (.csv/.npy) and plots its first/selected column.

    Reading and normalizing the file can be slow for large files, so both
    steps run on a background thread with a progress window (see
    progress_utils.run_with_progress_window) instead of freezing the UI.
    Because of that this function no longer returns the resulting canvas
    directly -- pass on_complete(canvas) to be notified when the load
    finishes (canvas is None if the user cancelled the file or ROI
    dialog)."""
    global selected_roi_index, loaded_filename, selected_roi_name, column_listbox, column_select_callback

    if listbox is not None:
        column_listbox = listbox
    if on_column_select is not None:
        column_select_callback = on_column_select

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    filename = filedialog.askopenfilename(
        parent=window,
        title="Open File",
        initialdir=project_root,
        filetypes=[("Data files", "*.npy;*.csv"), ("Numpy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not filename:
        if on_complete is not None:
            on_complete(None)
        return

    from tkinter import messagebox
    from progress_utils import run_with_progress_window

    is_csv = filename.lower().endswith('.csv')

    def read_columns_worker(report_progress, report_error):
        report_progress(0, 1, f"Leyendo {os.path.basename(filename)}...")
        if is_csv:
            df = pd.read_csv(filename)
            column_names = df.columns.tolist()
            if 'TIME' in column_names:
                column_names.remove('TIME')
        else:
            # .npy: no ROI dialog needed -- use all data columns
            try:
                temp_data = np.load(filename, allow_pickle=True)
                num_columns = temp_data.shape[1] if temp_data.ndim == 2 else 1
            except Exception:
                num_columns = 1
            column_names = [f"Column {i + 1}" for i in range(num_columns)]
        report_progress(1, 1, "Listo")
        return column_names

    def after_columns_read(column_names):
        global selected_roi_index, loaded_filename, selected_roi_name

        loaded_filename = filename
        column_names_list = column_names

        if is_csv:
            selected_roi_index = _show_roi_selection_dialog(window, column_names)
            if selected_roi_index is None:
                if on_complete is not None:
                    on_complete(None)
                return
            selected_roi_name = column_names[selected_roi_index]
        else:
            selected_roi_index = 0
            selected_roi_name = column_names[0]

        set_file_info(loaded_filename, selected_roi_name)

        def normalize_worker(report_progress, report_error):
            report_progress(0, 1, "Procesando datos...")
            data = normalize_data(filename)
            report_progress(1, 1, "Listo")
            return data

        def after_normalize(data):
            result_canvas = _plot_data_with_menu(data, window, canvas, menu_picos, column_names_list, notebook)
            if on_complete is not None:
                on_complete(result_canvas)

        def on_normalize_error(exc):
            messagebox.showerror("Error", f"No se pudo procesar el archivo:\n{exc}")
            if on_complete is not None:
                on_complete(None)

        run_with_progress_window(
            window, title="Cargando Datos", message="Procesando y normalizando datos...",
            maximum=1, worker_fn=normalize_worker, on_complete=after_normalize,
            on_error=on_normalize_error)

    def on_read_error(exc):
        messagebox.showerror("Error", f"No se pudo leer el archivo:\n{exc}")
        if on_complete is not None:
            on_complete(None)

    run_with_progress_window(
        window, title="Abriendo Archivo", message="Leyendo archivo...",
        maximum=1, worker_fn=read_columns_worker, on_complete=after_columns_read,
        on_error=on_read_error)


def _show_roi_selection_dialog(parent, column_names):
    """
    Show a dialog for the user to select which ROI/column to visualize.
    Returns the selected column index or None if cancelled.
    """
    dialog = Toplevel(parent)
    dialog.title("Select ROI")
    dialog.geometry("400x500")
    dialog.transient(parent)
    dialog.grab_set()
    
    selected_index = None
    
    # Title label
    Label(dialog, text="Select which ROI to visualize:", font=('Arial', 12, 'bold')).pack(pady=10)
    
    # Create frame for listbox and scrollbar
    list_frame = Frame(dialog)
    list_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
    
    # Scrollbar
    scrollbar = Scrollbar(list_frame)
    scrollbar.pack(side=RIGHT, fill=Y)
    
    # Listbox
    listbox = Listbox(list_frame, yscrollcommand=scrollbar.set, font=('Arial', 10))
    listbox.pack(side=LEFT, fill=BOTH, expand=True)
    scrollbar.config(command=listbox.yview)
    
    # Populate listbox with column names
    for i, col_name in enumerate(column_names):
        listbox.insert(END, col_name)
    
    # Select first item by default
    if column_names:
        listbox.selection_set(0)
        listbox.activate(0)
    
    def on_ok():
        nonlocal selected_index
        selection = listbox.curselection()
        if selection:
            selected_index = selection[0]
            dialog.destroy()
        else:
            from tkinter import messagebox
            messagebox.showwarning("No Selection", "Please select an ROI")
    
    def on_cancel():
        nonlocal selected_index
        selected_index = None
        dialog.destroy()
    
    def on_double_click(event):
        on_ok()
    
    # Bind double-click to select
    listbox.bind('<Double-Button-1>', on_double_click)
    
    # Buttons frame
    button_frame = Frame(dialog)
    button_frame.pack(pady=10)
    
    Button(button_frame, text="OK", command=on_ok, width=10).pack(side=LEFT, padx=5)
    Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side=LEFT, padx=5)
    
    # Wait for dialog to close
    parent.wait_window(dialog)
    
    return selected_index


def _plot_data_with_menu(data, window, canvas, menu_picos, column_names_list=None, notebook=None):
    """
    Plot data and configure menu entries. Separated from _plot_data so _plot_data
    can be called independently for column switching.
    """
    global selected_roi_index, column_listbox

    # Find the main root window and store data
    main_window = window
    while main_window.master:
        main_window = main_window.master
    
    main_window.loaded_data = data
    main_window.current_column = 0

    # Populate the column listbox if available
    if column_listbox is not None:
        column_listbox.delete(0, END)
        for i in range(data.shape[1]):
            label = column_names_list[i] if column_names_list and i < len(column_names_list) else f"Column {i + 1}"
            column_listbox.insert(END, label)
        column_listbox.selection_set(0)

    # Display selected ROI column by default
    initial_col = selected_roi_index if selected_roi_index is not None else 0
    canvas = _plot_data(data, window, canvas, column_idx=initial_col)

    # Helper to get current column from main window attribute
    def get_current_column():
        return getattr(main_window, 'current_column', 0)

    # Enable Dendograma and Series de tiempo menu items after data is loaded
    menu_picos.entryconfig("Dendograma",
                           command=lambda: plot_dendogram(data, window, None),
                           state=NORMAL)
    menu_picos.entryconfig("Series de tiempo",
                           command=lambda: plot_time_series(data, column_names_list, notebook),
                           state=NORMAL)

    return canvas


def _plot_data(data, window, canvas, column_idx=0):
    """
    Plot a single data column into the main plot frame.
    Returns the new canvas.
    """
    # Clamp column index
    if column_idx >= data.shape[1]:
        column_idx = 0

    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data[:, column_idx]))).reshape(-1, 1),
             data[:, column_idx])
    plt.title(f'Column {column_idx + 1}')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Find the main plot frame (at column=1 in the data tab grid)
    main_plot_frame = None
    for widget in window.winfo_children():
        if isinstance(widget, tk.Frame):
            try:
                grid_info = widget.grid_info()
                if grid_info.get('column') == 1:
                    main_plot_frame = widget
                    break
            except:
                continue

    if main_plot_frame is None:
        main_plot_frame = window

    # Clear all existing widgets from plot frame
    for widget in list(main_plot_frame.winfo_children()):
        widget.destroy()

    # Close all matplotlib figures to free memory
    plt.close('all')

    # Create new canvas
    canvas = FigureCanvasTkAgg(fig, master=main_plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    return canvas


def _open_correlation_tab(data, method, notebook, column_names_list=None):
    """Open a new notebook tab displaying the correlation matrix + dendrogram."""
    if notebook is None:
        return

    from corr_dendo_functions import _plot_correlation_helper

    df = pd.DataFrame(data)
    if column_names_list and len(column_names_list) >= data.shape[1]:
        df.columns = column_names_list[:data.shape[1]]

    corr = df.corr(method=method)
    try:
        X = corr.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method='complete')
        ind = sch.fcluster(L, 50, criterion='maxclust')
        cols = [df.columns.tolist()[i] for i in list(np.argsort(ind))]
        df = df.reindex(cols, axis=1)
    except Exception:
        pass

    tab_frame = tk.Frame(notebook)
    notebook.add(tab_frame, text=f'{method.capitalize()} Corr.')
    notebook.select(tab_frame)

    header = tk.Frame(tab_frame)
    header.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

    content = tk.Frame(tab_frame)
    content.pack(fill=tk.BOTH, expand=True)

    canvas_tab, fig = _plot_correlation_helper(
        df, 5, None, None, corr_method=method, target_frame=content
    )

    tk.Button(
        header,
        text='Close Tab',
        command=lambda: [plt.close(fig), notebook.forget(tab_frame)]
    ).pack(side=tk.RIGHT, padx=5)
