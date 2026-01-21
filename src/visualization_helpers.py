from tkinter import *
from tkinter import filedialog
from peak_functions import *
from corr_dendo_functions import *
import os
import pandas as pd

# Global variables to store file and ROI information
selected_roi_index = None
loaded_filename = None
selected_roi_name = None

def initialize_visualization(window, menu_picos, canvas):
    global selected_roi_index, loaded_filename, selected_roi_name
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    filename = filedialog.askopenfilename(
        parent=window,
        title="Open File",
        initialdir=project_root,
        filetypes=[("Data files", "*.npy;*.csv"), ("Numpy files", "*.npy"), ("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not filename:
        return
    
    # Store the loaded filename
    loaded_filename = filename
    
    # Load raw data to get column names for CSV files
    if filename.lower().endswith('.csv'):
        df = pd.read_csv(filename)
        column_names = df.columns.tolist()
        
        # Remove 'TIME' column if it exists
        if 'TIME' in column_names:
            column_names.remove('TIME')
        
        # Show ROI selection dialog
        selected_roi_index = _show_roi_selection_dialog(window, column_names)
        
        if selected_roi_index is None:
            return  # User cancelled selection
        
        # Store the selected ROI name
        selected_roi_name = column_names[selected_roi_index]
    else:
        # For .npy files, we'll need to select by column index
        # First load to check how many columns
        temp_data = np.load(filename)
        num_columns = temp_data.shape[1]
        column_names = [f"Column {i}" for i in range(num_columns)]
        
        selected_roi_index = _show_roi_selection_dialog(window, column_names)
        
        if selected_roi_index is None:
            return
        
        # Store the selected ROI name
        selected_roi_name = column_names[selected_roi_index]
    
    # Pass filename and ROI info to peak functions
    set_file_info(loaded_filename, selected_roi_name)
    
    data = normalize_data(filename)
    canvas = _plot_data(data, window, canvas)

    menu_picos.entryconfig("Elliptic Envelope", command=lambda:elliptic_envelope_peak(data, selected_roi_index, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Peak Caller", command=lambda:actual_peak_caller(data, selected_roi_index, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Local Outlier Factor", command=lambda:local_outlier_factor_peak(data, selected_roi_index, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Peak Function 4", command=lambda:clf_peak(data, selected_roi_index, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Isolation Forest", command=lambda:isolation_forest_peak(data, selected_roi_index, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Linear Model", command=lambda:linear_model_peak(data, selected_roi_index, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Peak Function 7", command=lambda:lasso_peak(data, selected_roi_index, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Correlacion Pearson", command=lambda:plot_correlation(data,correlation_pearson(data),window,canvas,'pearson'), state=NORMAL)
    menu_picos.entryconfig("Correlacion Kendall", command=lambda:plot_correlation(data,correlation_kendall(data),window,canvas,'kendall'), state=NORMAL)
    menu_picos.entryconfig("Correlacion Spearman", command=lambda:plot_correlation(data,correlation_spearman(data),window,canvas,'spearman'), state=NORMAL)
    menu_picos.entryconfig("Dendograma", command=lambda:plot_dendogram(data, window, canvas), state=NORMAL)
    menu_picos.entryconfig("Series de tiempo", command=lambda:plot_time_series(data), state=NORMAL)

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

def _plot_data(data, window, canvas):
    global selected_roi_index
    
    # Use selected ROI index, default to 15 if not set (for backwards compatibility)
    roi_idx = selected_roi_index if selected_roi_index is not None else 15
    
    fig, ax = plt.subplots()
    plt.plot(np.array(range(len(data[:, roi_idx]))).reshape(-1, 1), data[:, roi_idx])
    
    show_scale_widget(window)
    
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    return canvas

def show_scale_widget(main_window):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame):
            for child in widget.winfo_children():
                if isinstance(child, tk.Scale):
                    child.grid()
                    break