"""
Event (peak) detection - part of stage 5.

The same 7 Peak Finder methods as the Multiple Files tab, without any
drawing: `compute_peaks` finds the peaks of one signal, `detect_events`
runs a method on every signal of a signals table and returns an events
table (one row per peak), the stage 5 "events" item.
"""

import numpy as np
import pandas as pd
from sklearn import linear_model, svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.neighbors import LocalOutlierFactor

from .signals import parse_cell_column

# Each method's parameters: {method: (dialog title, [{'name', 'key', 'default', 'type'}, ...])}.
PEAK_METHODS = {
    'Elliptic Envelope': ('Elliptic Envelope Parameters', [
        {'name': 'Contamination', 'key': 'contamination', 'default': 0.01, 'type': float},
    ]),
    'Peak Caller': ('Peak Caller Parameters', [
        {'name': 'Rise %', 'key': 'rise_percent', 'default': 5, 'type': int},
        {'name': 'Fall %', 'key': 'fall_percent', 'default': 5, 'type': int},
        {'name': 'Max Lookback', 'key': 'max_lookback', 'default': 10, 'type': int},
        {'name': 'Max Lookahead', 'key': 'max_lookahead', 'default': 10, 'type': int},
    ]),
    'Local Outlier Factor': ('Local Outlier Factor Parameters', [
        {'name': 'N Neighbors', 'key': 'n_neighbors', 'default': 20, 'type': int},
    ]),
    'Peak Function 4': ('Peak Function 4 (Elliptic Envelope + SVR) Parameters', [
        {'name': 'Contamination', 'key': 'contamination', 'default': 0.01, 'type': float},
    ]),
    'Isolation Forest': ('Isolation Forest Parameters', [
        {'name': 'Contamination', 'key': 'contamination', 'default': 0.05, 'type': float},
    ]),
    'Linear Model': ('Linear Model (SGDOneClassSVM) Parameters', [
        {'name': 'Nu', 'key': 'nu', 'default': 0.131, 'type': float},
    ]),
    'Peak Function 7': ('Peak Function 7 (Lasso + LOF) Parameters', [
        {'name': 'N Neighbors', 'key': 'n_neighbors', 'default': 20, 'type': int},
    ]),
}

NO_PEAKS = 'None'


def default_params(method):
    """{key: default} for a method's parameters."""
    return {p['key']: p['default'] for p in PEAK_METHODS[method][1]}


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


def detect_events(signals_df, method, params=None):
    """Peaks of every signal in a signals table.

    Args:
        signals_df: frames x signals ("cell_<id>" columns for cells).
        method: one of PEAK_METHODS.
        params: the method's parameters (default: its defaults).

    Returns:
        Events table, one row per peak, sorted by cell and frame:
        cell_id (or signal, for columns that aren't cells), frame, value
        (the signal's value at the peak).
    """
    if method not in PEAK_METHODS:
        raise ValueError(f"Unknown peak method {method!r}")
    params = {**default_params(method), **(params or {})}
    rows = []
    cells = all(parse_cell_column(c) is not None for c in signals_df.columns)
    for col in signals_df.columns:
        values = signals_df[col].to_numpy(dtype=float)
        if np.isfinite(values).sum() < 3:
            continue
        values = np.nan_to_num(values, nan=float(np.nanmean(values)))
        if np.ptp(values) == 0:
            continue    # a flat signal has no peaks (and some methods can't fit it)
        for frame in sorted(int(p) for p in compute_peaks(values.reshape(-1, 1), 0, method, params)):
            who = {"cell_id": parse_cell_column(col)} if cells else {"signal": str(col)}
            rows.append({**who, "frame": frame, "value": float(values[frame])})
    columns = (["cell_id"] if cells else ["signal"]) + ["frame", "value"]
    df = pd.DataFrame(rows, columns=columns)
    if cells:
        df["cell_id"] = df["cell_id"].astype(int)
    df["frame"] = df["frame"].astype(int)
    return df


def events_by_column(events_df, columns):
    """{column: array of peak frames} for the given signal columns."""
    out = {}
    for col in columns:
        cid = parse_cell_column(col)
        if cid is not None and "cell_id" in events_df.columns:
            sel = events_df["cell_id"] == cid
        elif "signal" in events_df.columns:
            sel = events_df["signal"].astype(str) == str(col)
        else:
            continue
        out[col] = events_df.loc[sel, "frame"].to_numpy(dtype=int)
    return out
