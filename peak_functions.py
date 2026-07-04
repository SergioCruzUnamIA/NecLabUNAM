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
        numpy_data = np.load(data, allow_pickle=True)
        rs = np.random.RandomState(0)
        data_ = numpy_data[:,1:]
        return data_
    
    elif _is_csv_file(data):
        df = pd.read_csv(data)
        # Skip first column (assumed to be time/index)
        numpy_data = df.iloc[:, 1:].values
        
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

def show_parameter_dialog(parent, title, params):
    """
    Show a modal dialog for editing function parameters before running.
    
    Args:
        parent: parent tkinter window (can be None)
        title: dialog window title
        params: list of dicts with keys:
            - 'name': display label
            - 'key': parameter key returned in result
            - 'default': default value
            - 'type': float or int
            - 'min': minimum value (optional)
            - 'max': maximum value (optional)
    
    Returns:
        dict of {key: value} or None if cancelled
    """
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(False, False)
    
    result = {}
    cancelled = [False]
    entries = {}
    
    # Title
    tk.Label(dialog, text=title, font=('Arial', 12, 'bold')).grid(
        row=0, column=0, columnspan=2, pady=(10, 15), padx=10
    )
    
    # Parameter fields
    for i, p in enumerate(params):
        tk.Label(dialog, text=f"{p['name']}:", font=('Arial', 10), anchor='w').grid(
            row=i + 1, column=0, sticky='w', padx=(15, 5), pady=3
        )
        var = tk.StringVar(value=str(p['default']))
        entry = tk.Entry(dialog, textvariable=var, width=15, font=('Arial', 10))
        entry.grid(row=i + 1, column=1, padx=(5, 15), pady=3)
        entries[p['key']] = (var, p)
    
    def on_ok():
        for key, (var, p) in entries.items():
            try:
                val = p['type'](var.get())
                result[key] = val
            except ValueError:
                from tkinter import messagebox
                messagebox.showerror("Invalid Value", f"'{p['name']}' must be a valid {p['type'].__name__}")
                return
        dialog.destroy()
    
    def on_cancel():
        cancelled[0] = True
        dialog.destroy()
    
    # Buttons
    btn_frame = tk.Frame(dialog)
    btn_frame.grid(row=len(params) + 1, column=0, columnspan=2, pady=10)
    tk.Button(btn_frame, text="Run", command=on_ok, width=10).pack(side=tk.LEFT, padx=5)
    tk.Button(btn_frame, text="Cancel", command=on_cancel, width=10).pack(side=tk.LEFT, padx=5)
    
    # Center on screen
    dialog.update_idletasks()
    w = dialog.winfo_width()
    h = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() - w) // 2
    y = (dialog.winfo_screenheight() - h) // 2
    dialog.geometry(f"+{x}+{y}")
    
    if parent:
        parent.wait_window(dialog)
    else:
        dialog.wait_window(dialog)
    
    if cancelled[0]:
        return None
    return result

def normalize_data(data):
    data = _load_data(data)
    normalized_data = _normalize_data_helper(data)
    return normalized_data

def elliptic_envelope_peak(norm_data, roi_index, main_window=None, canvas=None, target_frame=None, params=None):
    if params is None:
        params = show_parameter_dialog(main_window, "Elliptic Envelope Parameters", [
            {'name': 'Contamination', 'key': 'contamination', 'default': 0.01, 'type': float},
        ])
    if params is None:
        return None

    plot_mode = 0
    pico_norm_data = norm_data[:, roi_index]

    reg = ElasticNet().fit(np.array(range(len(pico_norm_data))).reshape(-1, 1), pico_norm_data)
    res = reg.predict(np.array(range(len(pico_norm_data))).reshape(-1, 1))

    new_data = pico_norm_data - res
    clf = EllipticEnvelope(random_state=0, contamination=params['contamination']).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]

    return draw_canvas(pico_norm_data, res, y_res, plot_mode, main_window, canvas, target_frame=target_frame)

def _detrend_signal(data_sel, smooth_window):
    """Detrend a 1-D signal by dividing by a smoothed background trend.

    Mirrors PeakCaller's approach: generate a smoothed version of the time
    series, then divide the original by it.  When smooth_window <= 1 the
    signal is divided by its mean (PeakCaller's "no trend" option, which
    scales the profile so the mean equals 1).
    """
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        smooth = np.convolve(data_sel, kernel, mode='same')
    else:
        smooth = np.full_like(data_sel, np.mean(data_sel))
    smooth = np.where(np.abs(smooth) < 1e-10, 1e-10, smooth)
    return data_sel / smooth


def _linear_detrend_signal(data_sel):
    """Remove a linear drift by subtracting the best-fit line (scipy.signal.detrend)."""
    from scipy.signal import detrend
    return detrend(data_sel, type='linear')


def _savgol_detrend_signal(data_sel, smooth_window):
    """Remove a smooth, possibly curved baseline estimated with a Savitzky-Golay filter."""
    from scipy.signal import savgol_filter
    window = min(smooth_window, len(data_sel) - 1 + (len(data_sel) % 2))
    window = max(window | 1, 5)  # force odd, at least 5
    polyorder = min(2, window - 1)
    baseline = savgol_filter(data_sel, window_length=window, polyorder=polyorder)
    return data_sel - baseline


def _rolling_mean_detrend_signal(data_sel, smooth_window):
    """Remove a rolling-mean baseline (subtractive, unlike _detrend_signal's ratio)."""
    window = max(1, smooth_window)
    baseline = pd.Series(data_sel).rolling(window=window, center=True, min_periods=1).mean().values
    return data_sel - baseline


def _butterworth_detrend_signal(data_sel, cutoff=0.01, order=3):
    """High-pass Butterworth filter: removes slow drift, keeps fast transients (peaks)."""
    from scipy.signal import butter, filtfilt
    b, a = butter(order, cutoff, btype='high', fs=1.0)
    return filtfilt(b, a, data_sel)


def _als_detrend_signal(data_sel, lam=1e5, p=0.01, niter=10):
    """Asymmetric Least Squares baseline correction (Eilers & Boelens)."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    y = np.asarray(data_sel, dtype=float)
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    z = y.copy()
    for _ in range(niter):
        W.setdiag(w)
        z = spsolve((W + D).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return y - z


def peak_caller(data, roi_index, rise_percent, fall_percent, max_lookback, max_lookahead,
                main_window=None, canvas=None, target_frame=None):
    """Detect peaks in data[:, roi_index] using look-back/look-ahead rise/fall criteria.

    Smoothing/detrending should be applied externally (via _detrend_signal) before
    calling this function — pass the already-detrended data array as ``data``.
    """
    plot_mode = 1
    peaks = []
    n = len(data)
    data_sel = data[:, roi_index]

    for i in range(n):
        # Shorten look-back if it reaches the start of data or a previous peak.
        lookback_start = max(0, i - max_lookback)
        lookback_range = []
        for j in range(i - 1, lookback_start - 1, -1):
            if j in peaks:
                break
            lookback_range.insert(0, data[j])

        # Shorten look-ahead if it reaches the end of data or a higher point.
        lookahead_end = min(n, i + max_lookahead + 1)
        lookahead_range = []
        for j in range(i + 1, lookahead_end):
            if data_sel[j] > data_sel[i]:
                break
            lookahead_range.append(data_sel[j])

        if lookback_range and lookahead_range:
            rise = data_sel[i] * (rise_percent / 100.0)
            fall = data_sel[i] * (fall_percent / 100.0)
            significant_rise = data_sel[i] - np.min(lookback_range) >= rise
            significant_fall = data_sel[i] - np.min(lookahead_range) >= fall
            if significant_rise and significant_fall:
                peaks.append(i)

    res = np.zeros_like(data_sel)
    y_res = peaks
    result = draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas,
                         data, roi_index, peaks, rise_percent, fall_percent,
                         max_lookahead, max_lookback,
                         target_frame=target_frame)
    if target_frame is not None:
        return result
    return peaks

def actual_peak_caller(data, roi_index, main_window=None, canvas=None, target_frame=None, params=None):
    if params is None:
        params = show_parameter_dialog(main_window, "Peak Caller Parameters", [
            {'name': 'Rise %', 'key': 'rise_percent', 'default': 5, 'type': int},
            {'name': 'Fall %', 'key': 'fall_percent', 'default': 5, 'type': int},
            {'name': 'Max Lookback (pts)', 'key': 'max_lookback', 'default': 10, 'type': int},
            {'name': 'Max Lookahead (pts)', 'key': 'max_lookahead', 'default': 10, 'type': int},
        ])
    if params is None:
        return None
    return peak_caller(data, roi_index,
                       rise_percent=params['rise_percent'],
                       fall_percent=params['fall_percent'],
                       max_lookback=params['max_lookback'],
                       max_lookahead=params['max_lookahead'],
                       main_window=main_window, canvas=canvas,
                       target_frame=target_frame)

def local_outlier_factor_peak(data, roi_index, main_window=None, canvas=None, target_frame=None, params=None):
    if params is None:
        params = show_parameter_dialog(main_window, "Local Outlier Factor Parameters", [
            {'name': 'N Neighbors', 'key': 'n_neighbors', 'default': 20, 'type': int},
        ])
    if params is None:
        return None

    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = LocalOutlierFactor(n_neighbors=params['n_neighbors'])
    y_pred = clf.fit_predict(new_data.reshape(-1, 1))
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    # Keep only upper peaks (above regression line)
    y_res = [i for i in y_res if new_data[i] > 0]
    return draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas, target_frame=target_frame)

def clf_peak(data, roi_index, main_window=None, canvas=None, target_frame=None, params=None):
    if params is None:
        params = show_parameter_dialog(main_window, "Peak Function 4 (Elliptic Envelope + SVR) Parameters", [
            {'name': 'Contamination', 'key': 'contamination', 'default': 0.01, 'type': float},
        ])
    if params is None:
        return None

    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = EllipticEnvelope(random_state=0, contamination=params['contamination']).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    # Keep only upper peaks (above regression line)
    y_res = [i for i in y_res if new_data[i] > 0]
    return draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas, target_frame=target_frame)

def isolation_forest_peak(data, roi_index, main_window=None, canvas=None, target_frame=None, params=None):
    if params is None:
        params = show_parameter_dialog(main_window, "Isolation Forest Parameters", [
            {'name': 'Contamination', 'key': 'contamination', 'default': 0.05, 'type': float},
        ])
    if params is None:
        return None

    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = IsolationForest(random_state=0, contamination=params['contamination']).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    # Keep only upper peaks (above regression line)
    y_res = [i for i in y_res if new_data[i] > 0]
    return draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas, target_frame=target_frame)

def linear_model_peak(data, roi_index, main_window=None, canvas=None, target_frame=None, params=None):
    if params is None:
        params = show_parameter_dialog(main_window, "Linear Model (SGDOneClassSVM) Parameters", [
            {'name': 'Nu', 'key': 'nu', 'default': 0.131, 'type': float},
        ])
    if params is None:
        return None

    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = linear_model.SGDOneClassSVM(random_state=42, nu=params['nu']).fit(new_data.reshape(-1, 1))
    y_pred = clf.predict(new_data.reshape(-1, 1))
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    # Keep only upper peaks (above regression line)
    y_res = [i for i in y_res if new_data[i] > 0]
    return draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas, target_frame=target_frame)

def lasso_peak(data, roi_index, main_window=None, canvas=None, target_frame=None, params=None):
    if params is None:
        params = show_parameter_dialog(main_window, "Peak Function 7 (Lasso + LOF) Parameters", [
            {'name': 'N Neighbors', 'key': 'n_neighbors', 'default': 20, 'type': int},
        ])
    if params is None:
        return None

    plot_mode = 2
    data_sel = data[:, roi_index]
    reg = Lasso().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
    res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
    new_data = data_sel - res
    clf = LocalOutlierFactor(n_neighbors=params['n_neighbors'])
    y_pred = clf.fit_predict(new_data.reshape(-1, 1))
    y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
    # Keep only upper peaks (above regression line)
    y_res = [i for i in y_res if new_data[i] > 0]
    return draw_canvas(data_sel, res, y_res, plot_mode, main_window, canvas, target_frame=target_frame)

def compute_peaks(data, col_idx, method_name, params):
    """Run peak detection for a single column and return peak time indices, no drawing."""
    data_sel = data[:, col_idx]

    if method_name == 'Elliptic Envelope':
        reg = ElasticNet().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
        res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
        new_data = data_sel - res
        clf = EllipticEnvelope(random_state=0, contamination=params['contamination']).fit(new_data.reshape(-1, 1))
        y_pred = clf.predict(new_data.reshape(-1, 1))
        return [i for i, x in enumerate(list(y_pred)) if x == -1]

    elif method_name == 'Peak Caller':
        n = len(data_sel)
        peaks = []
        rise_percent = params['rise_percent']
        fall_percent = params['fall_percent']
        max_lookback = params['max_lookback']
        max_lookahead = params['max_lookahead']
        for i in range(n):
            lookback_start = max(0, i - max_lookback)
            lookback_range = []
            for j in range(i - 1, lookback_start - 1, -1):
                if j in peaks:
                    break
                lookback_range.insert(0, data_sel[j])
            lookahead_end = min(n, i + max_lookahead + 1)
            lookahead_range = []
            for j in range(i + 1, lookahead_end):
                if data_sel[j] > data_sel[i]:
                    break
                lookahead_range.append(data_sel[j])
            if lookback_range and lookahead_range:
                rise = data_sel[i] * (rise_percent / 100.0)
                fall = data_sel[i] * (fall_percent / 100.0)
                if (data_sel[i] - np.min(lookback_range) >= rise and
                        data_sel[i] - np.min(lookahead_range) >= fall):
                    peaks.append(i)
        return peaks

    elif method_name == 'Local Outlier Factor':
        reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
        res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
        new_data = data_sel - res
        clf = LocalOutlierFactor(n_neighbors=params['n_neighbors'])
        y_pred = clf.fit_predict(new_data.reshape(-1, 1))
        y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
        return [i for i in y_res if new_data[i] > 0]

    elif method_name == 'Peak Function 4':
        reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
        res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
        new_data = data_sel - res
        clf = EllipticEnvelope(random_state=0, contamination=params['contamination']).fit(new_data.reshape(-1, 1))
        y_pred = clf.predict(new_data.reshape(-1, 1))
        y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
        return [i for i in y_res if new_data[i] > 0]

    elif method_name == 'Isolation Forest':
        reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
        res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
        new_data = data_sel - res
        clf = IsolationForest(random_state=0, contamination=params['contamination']).fit(new_data.reshape(-1, 1))
        y_pred = clf.predict(new_data.reshape(-1, 1))
        y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
        return [i for i in y_res if new_data[i] > 0]

    elif method_name == 'Linear Model':
        reg = svm.SVR().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
        res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
        new_data = data_sel - res
        clf = linear_model.SGDOneClassSVM(random_state=42, nu=params['nu']).fit(new_data.reshape(-1, 1))
        y_pred = clf.predict(new_data.reshape(-1, 1))
        y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
        return [i for i in y_res if new_data[i] > 0]

    elif method_name == 'Peak Function 7':
        reg = Lasso().fit(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
        res = reg.predict(np.array(range(len(data_sel))).reshape(-1, 1))
        new_data = data_sel - res
        clf = LocalOutlierFactor(n_neighbors=params['n_neighbors'])
        y_pred = clf.fit_predict(new_data.reshape(-1, 1))
        y_res = [i for i, x in enumerate(list(y_pred)) if x == -1]
        return [i for i in y_res if new_data[i] > 0]

    return []

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

    original_data, roi_index, peaks, rise_percent, fall_percent, max_lookahead, max_lookback = args
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
    original_data, roi_index, peaks, rise_percent, fall_percent, max_lookahead, max_lookback = args
    fig, ax = plt.subplots()
    plt.plot(data_sel)
    plt.scatter(peaks, data_sel[peaks], color='darkorange')
    
    # Get the plot frame and clear it completely
    plot_frame = get_main_plot_frame(main_window)
    
    # Clear ALL widgets from the plot frame
    for widget in list(plot_frame.winfo_children()):
        widget.destroy()
    
    # Close all matplotlib figures
    plt.close('all')
    
    # Create canvas in the plot frame
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas

def add_peak_buttons(main_window, data_sel, *args):
    # Remove any existing peak button frames
    for widget in list(main_window.winfo_children()):
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    # Create button frame and add it at row 1 (below the main plot area)
    button_frame = tk.Frame(main_window)
    button_frame.peak_button_frame = True
    button_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
    
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
    for widget in list(main_window.winfo_children()):
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    add_peak_buttons(main_window, data_sel, *args)

def draw_canvas(data_sel, res, y_res, plot_mode, main_window=None, canvas=None, *args, target_frame=None):
    # Remove any existing peak button frames first (before creating new plot)
    if main_window and not target_frame:
        for widget in list(main_window.winfo_children()):
            if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
                widget.destroy()

    fig = None

    match plot_mode:
        case 0:
            fig, ax = plt.subplots()
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), data_sel - res)
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1)[y_res], (data_sel - res)[y_res], "o")

        case 1:
            if target_frame is not None:
                # Tab integration: render result directly without the interactive Update UI
                peaks = args[2] if len(args) > 2 else []
                fig, ax = plt.subplots()
                ax.plot(np.array(range(len(data_sel))).reshape(-1, 1), data_sel)
                if peaks:
                    ax.scatter(peaks, data_sel[peaks], color='darkorange', label='Peaks')
                ax.set_title('Peak Caller')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
            elif main_window:
                canvas = update_peak_caller_main(data_sel, main_window, canvas, *args)
                add_peak_buttons(main_window, data_sel, *args)
                return canvas
            else:
                update_peak_caller(data_sel, *args)
                return

        case 2:
            fig, ax = plt.subplots()
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), data_sel, label='Signal')
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1), res, label='Regression fit', linestyle='--')
            plt.plot(np.array(range(len(data_sel))).reshape(-1, 1)[y_res], data_sel[y_res], "o", label='Detected peaks')
            plt.legend()

    if target_frame is not None:
        for w in list(target_frame.winfo_children()):
            w.destroy()
        plt.close('all')
        c = FigureCanvasTkAgg(fig, master=target_frame)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return c, fig
    elif main_window:
        # Get the main plot frame instead of plotting to entire window
        plot_frame = get_main_plot_frame(main_window)

        # Clear ALL widgets from the plot frame
        for widget in list(plot_frame.winfo_children()):
            widget.destroy()

        # Close all matplotlib figures
        plt.close('all')

        # Create canvas in the plot frame using pack (not grid)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if plot_mode in [0, 2]:
            peak_args = (None, None, y_res)
            add_peak_buttons(main_window, data_sel, *peak_args)

        return canvas
    else:
        window = create_visualization_window()
        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, sticky='nsew')
        return canvas

def show_original_data(main_window, data_sel, args):
    # Remove button frames
    for widget in list(main_window.winfo_children()):
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    fig, ax = plt.subplots()
    plt.plot(data_sel)
    plt.legend()
    
    # Get the plot frame and clear it completely
    plot_frame = get_main_plot_frame(main_window)
    
    # Clear ALL widgets from the plot frame
    for widget in list(plot_frame.winfo_children()):
        widget.destroy()
    
    # Close all matplotlib figures
    plt.close('all')
    
    # Create canvas in the plot frame
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
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
    original_data, roi_index, peaks, rise_percent, fall_percent, max_lookahead, max_lookback = args[:7]
    
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
    # Remove button frames
    for widget in list(main_window.winfo_children()):
        if isinstance(widget, tk.Frame) and hasattr(widget, 'peak_button_frame'):
            widget.destroy()
    
    # Get the plot frame and clear it completely
    plot_frame = get_main_plot_frame(main_window)
    
    # Clear ALL widgets from the plot frame
    for widget in list(plot_frame.winfo_children()):
        widget.destroy()
    
    # Close all matplotlib figures
    plt.close('all')
    
    if len(args) >= 7:
        original_data, roi_index, peaks, rise_percent, fall_percent, max_lookahead, max_lookback = args[:7]
        
        fig, ax = plt.subplots()
        plt.plot(data_sel)
        plt.scatter(peaks, data_sel[peaks], color='darkorange')
        plt.legend()
        
        # Create canvas in the plot frame
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
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
            ("SVG files", "*.svg"),
            ("EPS files", "*.eps"),
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
