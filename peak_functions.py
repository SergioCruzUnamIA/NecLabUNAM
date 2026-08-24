import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
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

def _is_excel_file(filename):
    return filename.lower().endswith(('.xlsx', '.xls'))

def _load_data(data, sheet_name=None):
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

    elif _is_excel_file(data):
        df = pd.read_excel(data, sheet_name=sheet_name if sheet_name is not None else 0)
        # Skip first column (assumed to be time/index), same convention as CSV
        numpy_data = df.iloc[:, 1:].values

        imputer = SimpleImputer(strategy='mean')
        numpy_data = imputer.fit_transform(numpy_data)

        return numpy_data

    else:
        raise ValueError("Unsupported file. Please use a .npy, .csv, or .xlsx/.xls file")

def _normalize_data_helper(data):
    norm_data = np.zeros(data.shape) # creates an array of zeros with the shape of the data
    for i in range(data.shape[1]):
        reg = ElasticNet().fit(np.array(range(len(data[:, i]))).reshape(-1, 1), data[:, i])
        #reg = svm.SVR().fit(np.array(range(len(data_[:, i]))).reshape(-1, 1), data_[:, i])
        res = reg.predict(np.array(range(len(data[:, i]))).reshape(-1, 1))
        norm_data[:, i] = data[:, i] - res
        min_data = min(norm_data[:, i])
        max_data = max(norm_data[:, i])
        #norm_data[:, i] = data_[:, i]
        #norm_data[:, i] = norm_data[:, i] - min_data # option to normalize the data
        #norm_data[:, i] = norm_data[:, i] / min_data # option to normalize the data
        norm_data[:, i] = (norm_data[:, i] - min_data) / (max_data - min_data) # option to normalize the data
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

def normalize_data(data, sheet_name=None):
    data = _load_data(data, sheet_name=sheet_name)
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

def _lower_convex_hull(x, y):
    """Vertices of the lower convex hull of points (x, y) via a monotone chain."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull = []
    for p in zip(x, y):
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) <= 0:
            hull.pop()
        hull.append(p)
    return np.array([p[0] for p in hull]), np.array([p[1] for p in hull])


def _collect_genuine_lowest_points(y, window_frac=0.25, overlap=0.5):
    """Pool the lower-convex-hull vertices found in overlapping windows.

    A hull vertex is only ever a point that is not exceeded by the line
    between its neighbors, so a point from inside a peak can never qualify --
    this is what makes the pooled set trustworthy "lowest points" to build a
    baseline from.
    """
    n = len(y)
    x = np.arange(n)
    window = max(20, int(n * window_frac))
    stride = max(1, int(window * (1 - overlap)))

    pts = {}
    start = 0
    while start < n:
        end = min(start + window, n)
        hull_x, hull_y = _lower_convex_hull(x[start:end], y[start:end])
        for xi, yi in zip(hull_x, hull_y):
            if xi not in pts or yi < pts[xi]:
                pts[xi] = yi
        if end == n:
            break
        start += stride

    xs = np.array(sorted(pts.keys()))
    ys = np.array([pts[k] for k in xs])
    return xs, ys


def _pick_k_lowest_points(cand_x, cand_y, n, k):
    """Keep the lowest candidate point in each of k equal time-bins (the
    nearest candidate if a bin is empty), so the chosen points stay spread
    across the whole recording instead of clustering in one region."""
    edges = np.linspace(0, n, k + 1)
    chosen_x, chosen_y = [], []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        mask = (cand_x >= lo) & (cand_x < hi if i < k - 1 else cand_x <= hi)
        if mask.any():
            sub_x, sub_y = cand_x[mask], cand_y[mask]
            j = np.argmin(sub_y)
            chosen_x.append(sub_x[j])
            chosen_y.append(sub_y[j])
        else:
            center = (lo + hi) / 2
            j = np.argmin(np.abs(cand_x - center))
            chosen_x.append(cand_x[j])
            chosen_y.append(cand_y[j])

    # Without an anchor near the very start/end, np.interp flat-extrapolates
    # the baseline from the nearest chosen point, silently absorbing any real
    # trend at the edges into the "after" signal instead of the baseline.
    # Anchor each edge with the lowest genuine candidate within a small
    # boundary window -- not simply the raw edge sample -- so a single noisy
    # point at x=0 or x=n-1 can't single-handedly set the baseline.
    boundary = max(1, n // 10)
    if min(chosen_x) > boundary:
        left_mask = cand_x < boundary
        if left_mask.any():
            j = np.argmin(cand_y[left_mask])
            chosen_x.append(cand_x[left_mask][j])
            chosen_y.append(cand_y[left_mask][j])
    if max(chosen_x) < (n - 1) - boundary:
        right_mask = cand_x > (n - 1) - boundary
        if right_mask.any():
            j = np.argmin(cand_y[right_mask])
            chosen_x.append(cand_x[right_mask][j])
            chosen_y.append(cand_y[right_mask][j])

    order = np.argsort(chosen_x)
    cx = np.array(chosen_x)[order]
    cy = np.array(chosen_y)[order]
    cx, uniq_idx = np.unique(cx, return_index=True)
    return cx, cy[uniq_idx]


def convex_envelope_lowest_points(data_sel, n_points=6):
    """Return the (x, y) genuine lowest points used to build the Convex
    Envelope baseline for a 1-D signal -- exposed separately so callers can
    plot them alongside the signal."""
    y = np.asarray(data_sel, dtype=float)
    n = len(y)
    cand_x, cand_y = _collect_genuine_lowest_points(y)
    k = max(2, min(n_points, len(cand_x)))
    return _pick_k_lowest_points(cand_x, cand_y, n, k)


def _convex_envelope_detrend_signal(data_sel, n_points=6):
    """De-trend by subtracting a baseline built from a handful of the
    signal's genuine lowest points (found via local convexity, so none of
    them can fall inside a peak), connected piecewise-linearly.

    Using few, well-spread key points -- rather than one straight line --
    lets each flat, non-peak segment settle near its own level instead of
    being bridged by a single tilted line, while leaving peaks untouched.
    Subtraction (rather than dividing by the baseline) keeps this numerically
    stable even when the signal's lowest points are close to zero.

    The signal's own lowest point (the minimum of the baseline points used)
    is added back after subtraction, so the result keeps sitting near the
    original floor level instead of collapsing to zero.
    """
    y = np.asarray(data_sel, dtype=float)
    x = np.arange(len(y))

    px, py = convex_envelope_lowest_points(y, n_points=n_points)

    baseline = np.interp(x, px, py)
    return y - baseline + py.min()


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
    """Save peaks data to a CSV or Excel file"""
    default_name = get_default_save_name("_peaks.csv")

    filename = asksaveasfilename(
        initialfile=default_name,
        defaultextension=".csv",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx"),
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

        if filename.lower().endswith(('.xlsx', '.xls')):
            df.to_excel(filename, index=False, engine='xlsxwriter')
        else:
            df.to_csv(filename, index=False)
        from tkinter import messagebox
        messagebox.showinfo("Success", f"Peaks saved to {os.path.basename(filename)}")


def compute_sccd_metrics(values, peaks, decay_fraction=0.6):
    """
    Compute the single-cell calcium dynamics (SCCD) metrics of Patel et al.
    2015 (J. Neurosci. Methods 243:26-38), Fig. 3A, from an already-detected
    list of peak indices:
        a) amplitude          - peak value minus the value at event onset
        b) inter-event-interval (IEI) - spacing between consecutive onsets
        c) resting fluorescence - mean of the signal outside every event
        d) rise time           - onset -> first crossing of half amplitude
        e) fall time            - tau of an exponential decay fit to the
                                   decay phase; falls back to the
                                   interpolated time-to-half-amplitude if the
                                   fit is poor (R^2 < 0.9)

    There is no explicit sampling rate/time axis for the generic column data
    this operates on, so all durations (rise time, fall time, IEI) are
    reported in samples, matching the sample-index x-axis already used
    elsewhere in the app (e.g. the peak scatter on the Multiple Files plot).

    Args:
        values: 1-D array-like signal.
        peaks: iterable of peak sample indices (as returned by compute_peaks).
        decay_fraction: fraction of the gap to the next onset (or to the end
            of the series, for the last event) treated as "still decaying"
            and excluded, together with the event itself, when estimating
            resting fluorescence (c).

    Returns:
        dict with 'resting_mean', 'resting_std', 'iei' (np.ndarray) and
        'events' (list of dicts with onset_i, peak_i, onset_val, peak_val,
        amplitude, rise_time, fall_time, r2), or None if no peaks were given.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    peaks = sorted(set(int(p) for p in peaks if 0 <= int(p) < n))
    if not peaks:
        return None

    # Onset = the local trough immediately before the rise: walking
    # backward from the peak, the signal descends (bar noise) until the
    # base of the rise, then fluctuates around baseline. That trough is
    # found without needing to know the recording's sampling rate or scan
    # the whole preceding gap (which would just find the deepest noise dip
    # anywhere in it). The walk is done on a lightly smoothed copy and
    # tracks a running minimum, tolerating small backward up-ticks (within
    # the segment's own noise level) so a single noisy sample on the rising
    # edge doesn't stop it prematurely; it only stops once the smoothed
    # signal has clearly turned back up. The reported onset index/value
    # still come from the raw signal.
    def _find_onset(seg_start, p_i):
        seg = values[seg_start:p_i + 1]
        m = len(seg)
        if m < 3:
            return seg_start
        win = min(9, m)
        if win > 1 and win % 2 == 0:
            win -= 1
        if win > 1:
            # Edge-padded (not zero-padded) so the smoothed value right at
            # the peak isn't pulled down by an implicit zero outside the
            # window - that would falsely look like a downturn and stop
            # the backward walk immediately, at the peak itself.
            pad = win // 2
            padded = np.pad(seg, (pad, pad), mode='edge')
            kernel = np.ones(win) / win
            smoothed = np.convolve(padded, kernel, mode='valid')
        else:
            smoothed = seg
        diffs = np.diff(smoothed)
        tol = 1.5 * float(np.std(diffs)) if len(diffs) > 1 else 0.0
        i = m - 1
        running_min = smoothed[i]
        best_i = i
        while i > 0:
            i -= 1
            if smoothed[i] <= running_min:
                running_min = smoothed[i]
                best_i = i
            elif smoothed[i] > running_min + tol:
                break
        return seg_start + best_i

    events = []
    prev_boundary = 0
    for p_i in peaks:
        seg_start = prev_boundary if prev_boundary < p_i else max(0, p_i - 1)
        onset_i = _find_onset(seg_start, p_i)
        events.append({'onset_i': onset_i, 'peak_i': p_i})
        prev_boundary = p_i

    # c) Resting fluorescence: mean of the signal outside every event window.
    quiet_mask = np.ones(n, dtype=bool)
    for k, e in enumerate(events):
        onset_i = e['onset_i']
        next_onset_i = events[k + 1]['onset_i'] if k + 1 < len(events) else n
        excl_end = min(n, onset_i + max(1, int(decay_fraction * max(1, next_onset_i - onset_i))))
        quiet_mask[onset_i:excl_end] = False
    resting_vals = values[quiet_mask] if quiet_mask.any() else values
    resting_mean = float(resting_vals.mean())
    resting_std = float(resting_vals.std())

    def exp_decay(tt, amp, tau, c):
        return amp * np.exp(-tt / tau) + c

    for k, e in enumerate(events):
        onset_i, p_i = e['onset_i'], e['peak_i']
        onset_val, peak_val = float(values[onset_i]), float(values[p_i])
        amplitude = peak_val - onset_val
        half_val = onset_val + amplitude / 2.0

        # d) rise time
        rise_seg = values[onset_i:p_i + 1]
        cross = np.where(rise_seg >= half_val)[0]
        rise_time = float(cross[0]) if len(cross) else float(p_i - onset_i)

        # e) fall time: exponential fit on the decay phase, peak -> next
        # onset (or end of series), with an R^2 >= 0.9 acceptance threshold.
        next_onset_i = events[k + 1]['onset_i'] if k + 1 < len(events) else n
        decay_end = min(n, max(p_i + 2, next_onset_i))
        tt_decay = np.arange(decay_end - p_i, dtype=float)
        y_decay = values[p_i:decay_end]

        fall_time, r2 = None, None
        if len(tt_decay) >= 4 and amplitude > 0:
            try:
                popt, _ = curve_fit(
                    exp_decay, tt_decay, y_decay,
                    p0=[amplitude, max(1.0, len(tt_decay) / 4), resting_mean],
                    bounds=([0, 0.1, -np.inf], [np.inf, len(tt_decay) * 5, np.inf]),
                    maxfev=5000)
                y_fit = exp_decay(tt_decay, *popt)
                ss_res = np.sum((y_decay - y_fit) ** 2)
                ss_tot = np.sum((y_decay - y_decay.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                if r2 >= 0.9:
                    fall_time = abs(popt[1])
            except Exception:
                pass
        if fall_time is None:
            below = np.where(y_decay <= half_val)[0]
            fall_time = float(below[0]) if len(below) else float(decay_end - p_i)

        e.update(onset_val=onset_val, peak_val=peak_val, amplitude=amplitude,
                 rise_time=rise_time, fall_time=float(fall_time), r2=r2)

    onset_indices = np.array([e['onset_i'] for e in events], dtype=float)
    iei = np.diff(onset_indices)

    return {'resting_mean': resting_mean, 'resting_std': resting_std,
            'events': events, 'iei': iei}


def _annotate_sccd_metrics(ax, values, metrics):
    """Draw amplitude (a), inter-event-interval (b), resting fluorescence
    (c), rise time (d) and fall time (e) directly on an already-plotted
    trace, all on the one axes -- the layout approved for Fig. 3A-style
    review before this was wired into the app."""
    events = metrics['events']
    resting_mean, resting_std = metrics['resting_mean'], metrics['resting_std']
    n = len(values)

    ax.axhspan(resting_mean - resting_std, resting_mean + resting_std,
               color='violet', alpha=0.22, zorder=0,
               label=f'c: resting fluorescence (mean±SD) = {resting_mean:.3g}±{resting_std:.3g}')
    ax.axhline(resting_mean, color='purple', linewidth=1, linestyle=':', zorder=1)

    y_top = float(np.max(values))
    y_bottom = float(np.min(values))
    span = max(y_top - y_bottom, 1e-9)
    y_iei = y_bottom - 0.12 * span
    ax.set_ylim(y_iei - 0.08 * span, y_top + 0.08 * span)

    for i, e in enumerate(events):
        onset_i, p_i = e['onset_i'], e['peak_i']
        ax.plot(p_i, e['peak_val'], 'o', color='crimson', markersize=7, zorder=4,
                 label='a: amplitude (point)' if i == 0 else None)
        ax.plot([p_i, p_i], [e['onset_val'], e['peak_val']],
                 color='crimson', linewidth=1, alpha=0.5, zorder=2)
        ax.plot(onset_i, e['onset_val'], '|', color='magenta', markersize=14,
                 markeredgewidth=2, zorder=4, label='Event onset' if i == 0 else None)

        label = f"a={e['amplitude']:.3g}\nd={e['rise_time']:.0f}  e={e['fall_time']:.0f}"
        ax.annotate(label, (p_i, e['peak_val']), textcoords="offset points",
                     xytext=(6, 8), fontsize=8, color='black', ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='crimson', alpha=0.85))

    onset_positions = [e['onset_i'] for e in events]
    for i in range(len(onset_positions) - 1):
        x0, x1 = onset_positions[i], onset_positions[i + 1]
        ax.annotate('', xy=(x1, y_iei), xytext=(x0, y_iei),
                     arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
        ax.text((x0 + x1) / 2, y_iei - 0.05 * span, f"b={x1 - x0:.0f}",
                 ha='center', fontsize=8, fontweight='bold')

    ax.set_xlim(0, n)
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9, ncol=2)


def show_sccd_metrics_window(parent, series, peak_method, peak_params, column_label):
    """Open a scrollable Toplevel with one annotated SCCD-metrics plot
    (amplitude/IEI/resting fluorescence/rise time/fall time, all on the
    same graph) per loaded sheet, for the peaks the currently selected
    Peak Finder method/parameters find in each sheet's series of the given
    column.

    Args:
        parent: parent Tk window.
        series: list of (label, values) tuples, one per loaded sheet, as
            returned by NecLabApp._compute_multi_xls_series.
        peak_method: name of the peak finder method (must not be 'None').
        peak_params: params dict for that method (as produced by
            show_parameter_dialog).
        column_label: display name of the column, used in the window title.
    """
    win = tk.Toplevel(parent)
    win.title(f"SCCD Metrics — {column_label}")
    win.geometry("1100x750")

    outer = tk.Frame(win)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas_scroll = tk.Canvas(outer, highlightthickness=0)
    vbar = ttk.Scrollbar(outer, orient='vertical', command=canvas_scroll.yview)
    inner = tk.Frame(canvas_scroll)
    inner.bind('<Configure>', lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox('all')))
    canvas_scroll.create_window((0, 0), window=inner, anchor='nw')
    canvas_scroll.configure(yscrollcommand=vbar.set)
    canvas_scroll.pack(side='left', fill='both', expand=True)
    vbar.pack(side='right', fill='y')

    any_events = False
    for label, values in series:
        peaks = compute_peaks(np.asarray(values).reshape(-1, 1), 0, peak_method, peak_params)
        metrics = compute_sccd_metrics(values, peaks) if peaks else None

        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(values, color='black', linewidth=0.8, zorder=1, label='Signal')

        if metrics and metrics['events']:
            any_events = True
            _annotate_sccd_metrics(ax, values, metrics)
            ax.set_title(f"{label} — {len(metrics['events'])} event(s) detected")
        else:
            ax.set_title(f"{label} — no events detected with current peak finder settings")

        ax.set_xlabel('Sample index')
        ax.set_ylabel('Value')
        fig.tight_layout()

        chart_frame = tk.Frame(inner, highlightbackground='#cccccc', highlightthickness=1)
        chart_frame.pack(fill='x', padx=10, pady=10)
        fig_canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    if not any_events:
        tk.Label(inner, fg='firebrick', font=('Arial', 10, 'bold'), wraplength=1000,
                 text="No calcium-transient events were detected in any sheet with the "
                      "current Peak Finder method/parameters.").pack(pady=20)

    return win
