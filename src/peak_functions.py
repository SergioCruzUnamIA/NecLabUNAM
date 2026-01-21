import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import linear_model
import tkinter as tk
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.filedialog import asksaveasfilename
import pandas as pd
from sklearn.impute import SimpleImputer
import os

# Global variables to store file and ROI information
current_filename = None
current_roi_name = None
current_peaks = None

def set_file_info(filename, roi_name):
    """Set the current file and ROI information for saving"""
    global current_filename, current_roi_name
    current_filename = filename
    current_roi_name = roi_name

def get_default_save_name(extension=".png"):
    """Generate default save name based on original filename and ROI"""
    global current_filename, current_roi_name
    
    if current_filename and current_roi_name:
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(current_filename))[0]
        # Clean ROI name (remove spaces and special characters for filename)
        roi_safe = current_roi_name.replace(' ', '_')
        return f"{base_name}_{roi_safe}{extension}"
    else:
        return f"Untitled{extension}"

def _is_npy_file(filename):
    return filename.lower().endswith('.npy')

def _is_csv_file(filename):
    return filename.lower().endswith('.csv')

def _load_data(data):
    if _is_npy_file(data):
        numpy_data = np.load(data)
        rs = np.random.RandomState(0)
        data_ = numpy_data[:,1:]
        return data_
    elif _is_csv_file(data):
        df = pd.read_csv(data)
        numpy_data = df.values
        
        imputer = SimpleImputer(strategy='mean')
        numpy_data = imputer.fit_transform(numpy_data)
        
        rs = np.random.RandomState(0)
        data_ = numpy_data
        return data_
    else:
        raise ValueError("Archivo no soportado. Por favor, use un archivo .npy o .csv")
    

def _normalize_data_helper(data):
    norm_data = np.zeros(data.shape) # crea un arreglo con zeros en la forma de los datos
    for i in range(data.shape[1]): 
        reg = ElasticNet().fit(np.array(range(len(data[:, i]))).reshape(-1, 1), data[:, i])
        #reg = svm.SVR().fit(np.array(range(len(data_[:, i]))).reshape(-1, 1), data_[:, i])
        res = reg.predict(np.array(range(len(data[:, i]))).reshape(-1, 1))
        norm_data[:, i] = data[:, i] - res
        min_data = min(norm_data[:, i])
        max_data = max(norm_data[:, i])
        #norm_data[:, i] = data_[:, i] 
        #norm_data[:, i] = norm_data[:, i] - min_data # opcion para nomalizar los datos
        #norm_data[:, i] = norm_data[:, i] / min_data # opcion para nomalizar los datos
        norm_data[:, i] = (norm_data[:, i] - min_data) / (max_data - min_data) # opcion para nomalizar los datos
    return norm_data

def normalize_data(data):
    data = _load_data(data)
    normalized_data = _normalize_data_helper(data)
    return normalized_data

def elliptic_envelope_peak(norm_data, roi_index, main_window=None, canvas=None):
    plot_mode = 0
    pico_norm_data = norm_data[:, roi_index]

    reg = ElasticNet().fit(np.array(range(len(pico_norm_data))).reshape(-1, 1), pico_norm_data)
    res = reg.predict(np.array(range(len(pico_norm_data))).reshape(-1, 1))

    new_data = pico_norm_data - res
    clf = EllipticEnvelope(random_state=0, contamination=0.01).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    #y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]

    draw_canvas(pico_norm_data, res, y_res, plot_mode, main_window, canvas)

def peak_caller(data, roi_index, rise_percent, fall_percent, max_lookback, max_lookahead, main_window=None, canvas=None):
    plot_mode = 1
    peaks = []
    n = len(data)
    data_sel = data[:, roi_index]
    
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
            if data_sel[j] > data_sel[i]:
                break
            lookahead_range.append(data_sel[j])
        
        # Si no esta vacio el rango de lookback y lookahead se calcula si es un pico
        if len(lookback_range) > 0 and len(lookahead_range) > 0:
            rise = data_sel[i] * (rise_percent / 100.0)
            fall = data_sel[i] * (fall_percent / 100.0)
            
            # Checa si los datos incrementan y decrementan lo suficiente para ser pico
            # Compara el valor actual con el minimo de los datos en el rango de lookback
            # Compara el valor actual con el maximo de los datos en el rango de lookahead
            significant_rise = data_sel[i] - np.min(lookback_range) >= rise
            significant_fall = data_sel[i] - np.min(lookahead_range) >= fall
            
            if significant_rise and significant_fall:
                peaks.append(i)

    res = np.zeros_like(data_sel)
    y_res = peaks
    draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas, data, roi_index, peaks, rise_percent, fall_percent, max_lookahead, max_lookback)
    return peaks

def actual_peak_caller(data, roi_index, main_window=None, canvas=None):
    return peak_caller(data, roi_index, rise_percent=5, fall_percent=5, max_lookback=10, max_lookahead=10, main_window=main_window, canvas=canvas)

def local_outlier_factor_peak(data, roi_index, main_window=None, canvas=None):
    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas)

def clf_peak(data, roi_index, main_window=None, canvas=None): #hay dos elliptic envelope?
    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = EllipticEnvelope(random_state=0, contamination=0.01).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas)

def isolation_forest_peak(data, roi_index, main_window=None, canvas=None):
    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = IsolationForest(random_state=0, contamination=0.05).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas)

def linear_model_peak(data, roi_index, main_window=None, canvas=None):
    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = linear_model.SGDOneClassSVM(random_state=42, nu=0.131).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas)

def lasso_peak(data, roi_index, main_window=None, canvas=None): # hay dos local outlier factor
    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = Lasso().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = LocalOutlierFactor(n_neighbors=20)
    y_pred = clf.fit_predict(new_data.reshape(-1, 1))
    # y_res = list(y_pred).index(-1)
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas)

def create_visualization_window():
    visualization_window = tk.Toplevel()
    visualization_window.title("Visualization")
    visualization_window.focus()
    button_close = tk.Button(
        visualization_window,
        text="Close window",
        command=visualization_window.destroy
    )
    button_save = tk.Button(
        visualization_window,
        text="Save",
        command=save
    )
    button_close.grid(row=1,column=2)
    button_save.grid(row=1,column=1)
    return visualization_window

#------------------------------------

def update_peak_caller(data_sel, *args):
    def update_graph():
        new_rise_percent = spinval_rise.get()
        new_fall_percent = spinval_fall.get()
        new_max_lookahead = spinval_lookahead.get()
        new_max_lookback = spinval_lookback.get()

    original_data, roi_index, peaks, rise_percent, fall_percent, max_lookback, max_lookahead = args
    fig, ax = plt.subplots()
    plt.plot(data_sel)
    plt.scatter(peaks, data_sel[peaks], color='darkorange')
    ax = plt.gca()

    window = create_visualization_window()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0,columnspan=3, sticky='nsew')

    peak_caller_frame = tk.Frame(window)
    spinval_rise = IntVar(value=rise_percent)
    spinbox_rise = ttk.Spinbox(peak_caller_frame, from_=1.0, to=100.0, textvariable=spinval_rise)
    label_rise = tk.Label(peak_caller_frame,text="Rise")
    spinval_fall = IntVar(value=fall_percent)
    spinbox_fall = ttk.Spinbox(peak_caller_frame, from_=1.0, to=100.0, textvariable=spinval_fall)
    label_fall = tk.Label(peak_caller_frame,text="Fall")
    spinval_lookahead = IntVar(value=max_lookahead)
    spinbox_lookahead = ttk.Spinbox(peak_caller_frame, from_=1.0, to=100.0, textvariable=spinval_lookahead)
    label_lookahead = tk.Label(peak_caller_frame,text="Max Lookahead")
    spinval_lookback = IntVar(value=max_lookback)
    spinbox_lookback = ttk.Spinbox(peak_caller_frame, from_=1.0, to=100.0, textvariable=spinval_lookback)
    label_lookback = tk.Label(peak_caller_frame,text="Max Lookback")
    update_button = tk.Button(
        peak_caller_frame, 
        text="Update Graph", 
        command=lambda:(peak_caller(original_data, roi_index, int(spinbox_rise.get()), int(spinbox_fall.get()), int(spinbox_lookback.get()), int(spinbox_lookahead.get())),window.destroy(), update_graph)
    )
    
    peak_caller_frame.grid(row=0, column=3, sticky='nsew')
    label_rise.grid(row=0,column=0)
    spinbox_rise.grid(row=0,column=1)
    label_fall.grid(row=1,column=0)
    spinbox_fall.grid(row=1,column=1)
    label_lookahead.grid(row=2,column=0)
    spinbox_lookahead.grid(row=2,column=1)
    label_lookback.grid(row=3,column=0)
    spinbox_lookback.grid(row=3,column=1)
    update_button.grid(row=4, column=1)

    return fig

def update_peak_caller_main(data_sel, main_window, canvas, *args):
    original_data, roi_index, peaks, rise_percent, fall_percent, max_lookback, max_lookahead = args
    fig, ax = plt.subplots()
    plt.plot(data_sel)
    plt.scatter(peaks, data_sel[peaks], color='darkorange')
    
    # Hide the scale widget when showing peaks
    hide_scale_widget(main_window)
    
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()
    canvas = FigureCanvasTkAgg(fig, master=main_window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    return canvas

def add_peak_buttons(main_window, data_sel, *args):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    button_frame = tk.Frame(main_window)
    button_frame.peak_button_frame = True
    button_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
    
    save_button = tk.Button(
        button_frame,
        text="Save Image",
        command=save
    )
    save_button.pack(side=tk.LEFT, padx=5)
    
    # Add Save Peaks CSV button if peaks are available
    if len(args) >= 3:
        peaks = args[2] if len(args) > 2 else []
        time_data = np.arange(len(data_sel))  # Use indices as time if no time column
        
        save_csv_button = tk.Button(
            button_frame,
            text="Save Peaks CSV",
            command=lambda: save_peaks_csv(peaks, time_data, data_sel)
        )
        save_csv_button.pack(side=tk.LEFT, padx=5)
    
    show_original_button = tk.Button(
        button_frame,
        text="Show Original",
        command=lambda: show_original_data(main_window, data_sel, args)
    )
    show_original_button.pack(side=tk.LEFT, padx=5)
    
    if len(args) >= 6:
        configure_button = tk.Button(
            button_frame,
            text="Configure Peaks",
            command=lambda: show_peak_config(main_window, data_sel, args)
        )
        configure_button.pack(side=tk.LEFT, padx=5)

def restore_normal_buttons(main_window, data_sel, args):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    add_peak_buttons(main_window, data_sel, *args)

def draw_canvas(data_sel, res, y_res, plot_mode, main_window=None, canvas=None, *args):
    match plot_mode:
        case 0:
            fig, ax = plt.subplots()
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), data_sel - res)
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1)[y_res], (data_sel - res)[y_res], "o")
            
            if main_window and canvas:
                hide_scale_widget(main_window)
                
        case 1:
            if main_window:
                canvas = update_peak_caller_main(data_sel, main_window, canvas, *args)
                add_peak_buttons(main_window, data_sel, *args)
                return canvas
            else:
                update_peak_caller(data_sel, *args)
                return
        case 2:
            fig, ax = plt.subplots()
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), data_sel - res)
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), res - 350)
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1)[y_res], (data_sel - res)[y_res], "o")
            mean2 = (380 - 310) / 2
            
            if main_window and canvas:
                hide_scale_widget(main_window)

    if main_window:
        if canvas is not None:
            canvas.get_tk_widget().grid_forget()
        canvas = FigureCanvasTkAgg(fig, master=main_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        
        if plot_mode in [0, 2]:
            # For plot modes 0 and 2, pass y_res as the peaks
            # Create a minimal args tuple: (None, None, peaks)
            peak_args = (None, None, y_res)
            add_peak_buttons(main_window, data_sel, *peak_args)
        
        return canvas
    else:
        window = create_visualization_window()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0,columnspan=3, sticky='nsew')
        return canvas

def show_original_data(main_window, data_sel, args):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    show_scale_widget(main_window)
    
    fig, ax = plt.subplots()
    plt.plot(data_sel)
    plt.legend()
    
    for widget in main_window.winfo_children():
        if hasattr(widget, 'get_tk_widget'):
            widget.get_tk_widget().grid_forget()
    
    canvas = FigureCanvasTkAgg(fig, master=main_window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    
    if args and len(args) >= 6:
        back_button_frame = tk.Frame(main_window)
        back_button_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        
        back_button = tk.Button(
            back_button_frame,
            text="Show Peaks",
            command=lambda: restore_peak_view(main_window, data_sel, args)
        )
        back_button.pack(side=tk.LEFT, padx=5)


def show_peak_config(main_window, data_sel, args):
    original_data, roi_index, peaks, rise_percent, fall_percent, max_lookback, max_lookahead = args[:7]
    
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    config_frame = tk.Frame(main_window)
    config_frame.peak_button_frame = True
    config_frame.grid(row=1, column=0, sticky='ew', padx=5, pady=5)
    
    tk.Label(config_frame, text="Rise %:").grid(row=0, column=0, padx=5)
    spinval_rise = tk.IntVar(value=rise_percent)
    spinbox_rise = ttk.Spinbox(config_frame, from_=1.0, to=100.0, textvariable=spinval_rise, width=10)
    spinbox_rise.grid(row=0, column=1, padx=5)
    
    tk.Label(config_frame, text="Fall %:").grid(row=0, column=2, padx=5)
    spinval_fall = tk.IntVar(value=fall_percent)
    spinbox_fall = ttk.Spinbox(config_frame, from_=1.0, to=100.0, textvariable=spinval_fall, width=10)
    spinbox_fall.grid(row=0, column=3, padx=5)
    
    tk.Label(config_frame, text="Lookback:").grid(row=1, column=0, padx=5)
    spinval_lookback = tk.IntVar(value=max_lookback)
    spinbox_lookback = ttk.Spinbox(config_frame, from_=1.0, to=100.0, textvariable=spinval_lookback, width=10)
    spinbox_lookback.grid(row=1, column=1, padx=5)
    
    tk.Label(config_frame, text="Lookahead:").grid(row=1, column=2, padx=5)
    spinval_lookahead = tk.IntVar(value=max_lookahead)
    spinbox_lookahead = ttk.Spinbox(config_frame, from_=1.0, to=100.0, textvariable=spinval_lookahead, width=10)
    spinbox_lookahead.grid(row=1, column=3, padx=5)
    
    button_row_frame = tk.Frame(config_frame)
    button_row_frame.grid(row=2, column=0, columnspan=4, pady=10)
    
    update_button = tk.Button(
        button_row_frame,
        text="Update Peaks",
        command=lambda: update_peaks_on_main(
            main_window, original_data, roi_index, data_sel,
            spinval_rise.get(), spinval_fall.get(),
            spinval_lookback.get(), spinval_lookahead.get()
        )
    )
    update_button.pack(side=tk.LEFT, padx=5)
    
    cancel_button = tk.Button(
        button_row_frame,
        text="Cancel",
        command=lambda: restore_normal_buttons(main_window, data_sel, args)
    )
    cancel_button.pack(side=tk.LEFT, padx=5)
    
    # Add both save buttons
    save_button = tk.Button(
        button_row_frame,
        text="Save Image",
        command=save
    )
    save_button.pack(side=tk.LEFT, padx=5)
    
    time_data = np.arange(len(data_sel))
    save_csv_button = tk.Button(
        button_row_frame,
        text="Save Peaks CSV",
        command=lambda: save_peaks_csv(peaks, time_data, data_sel)
    )
    save_csv_button.pack(side=tk.LEFT, padx=5)

def update_peaks_on_main(main_window, original_data, roi_index, data_sel, rise_percent, fall_percent, max_lookback, max_lookahead):
    canvas = None
    for widget in main_window.winfo_children():
        if hasattr(widget, 'get_tk_widget'):
            canvas = widget
            widget.get_tk_widget().grid_forget()
            break
    
    new_peaks = peak_caller(
        original_data, roi_index, rise_percent, fall_percent, 
        max_lookback, max_lookahead, 
        main_window=main_window, canvas=canvas
    )

def restore_peak_view(main_window, data_sel, args):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame):
            widget.destroy()
    
    hide_scale_widget(main_window)
    
    for widget in main_window.winfo_children():
        if hasattr(widget, 'get_tk_widget'):
            widget.get_tk_widget().grid_forget()
    
    if len(args) >= 7:
        original_data, roi_index, peaks, rise_percent, fall_percent, max_lookback, max_lookahead = args[:7]
        
        fig, ax = plt.subplots()
        plt.plot(data_sel)
        plt.scatter(peaks, data_sel[peaks], color='darkorange')
        plt.legend()
        
        canvas = FigureCanvasTkAgg(fig, master=main_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        
        add_peak_buttons(main_window, data_sel, *args)

#--------------------------------

def hide_scale_widget(main_window):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame):
            for child in widget.winfo_children():
                if isinstance(child, tk.Scale):
                    child.grid_remove()
                    break

def show_scale_widget(main_window):
    for widget in main_window.winfo_children():
        if isinstance(widget, tk.Frame):
            for child in widget.winfo_children():
                if isinstance(child, tk.Scale):
                    child.grid()
                    break

def save():
    """Save the current plot as an image"""
    default_name = get_default_save_name(".png")
    
    filename = asksaveasfilename(
        initialfile=default_name,
        defaultextension=".png",
        filetypes=[
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("JPEG files", "*.jpeg"), 
            ("TIFF files", "*.tiff"),
            ("TIFF files", "*.tif"),
            ("PDF files", "*.pdf"),
            ("All Files", "*.*")
        ]
    )
    if filename:  # Only save if user didn't cancel
        plt.savefig(filename, dpi=300, bbox_inches='tight')

def save_peaks_csv(peaks, time_data, signal_data):
    """Save peaks data to CSV file"""
    default_name = get_default_save_name("_peaks.csv")
    
    filename = asksaveasfilename(
        initialfile=default_name,
        defaultextension=".csv",
        filetypes=[
            ("CSV files", "*.csv"),
            ("All Files", "*.*")
        ],
        title="Save Peaks Data"
    )
    
    if filename:
        # Create DataFrame with peak information
        peak_indices = peaks
        peak_times = time_data[peak_indices] if time_data is not None else peak_indices
        peak_values = signal_data[peak_indices]
        
        df = pd.DataFrame({
            'Peak_Index': peak_indices,
            'Time': peak_times,
            'Signal_Value': peak_values
        })
        
        df.to_csv(filename, index=False)
        from tkinter import messagebox
        messagebox.showinfo("Success", f"Peaks saved to {os.path.basename(filename)}")