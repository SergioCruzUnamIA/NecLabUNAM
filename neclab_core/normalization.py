"""
Signal normalization and smoothing (pipeline stage 6), used by every view:
the cell detection window's correlation analysis, the Multiple Files tab
(line plot, heatmap and "Save Smoothed Data") and peak detection.

Two entry points:

normalize_signals(df, method, smoothing_points=None)
    One signals table (frames x cells): each column normalized on its own.
    Methods (NORMALIZATION_METHODS):
        "none"             values as they are
        "divide_by_min"    value / minimum of that signal
        "delta_f_over_f0"  (F - F0) / F0, with F0 = a low percentile of the
                           signal (10th by default): the usual ΔF/F0
        "detrend_minmax"   subtract a linear (ElasticNet) trend, then scale
                           to 0..1 (the original notebook normalization)
        "lowest_points"    divide by the Convex Envelope baseline (the line
                           through the signal's genuine lowest points), so
                           the lowest points lie on a horizontal line at 1
    then, optionally, one of:
        subtract_baseline  subtract the lowest-points baseline of the
                           normalized signal, so its lowest points lie on a
                           horizontal line at 0 (no division: works with
                           any method and any signal level)
        smoothing_points   Convex Envelope smoothing: subtract that baseline
                           and add the signal's lowest level back

divide_by_min_tables(tables, mode)
    Several tables at once (the Multiple Files sheets), dividing by a
    minimum chosen by `mode` (MIN_MODES): per column, per sheet, per column
    across all sheets, or one global minimum ("none" = no division).

Missing values (NaN) are kept by the normalizations; fill_missing() fills
them by linear interpolation where a complete series is needed.
"""

import numpy as np
import pandas as pd

NORMALIZATION_METHODS = {
    "none": "None (raw values)",
    "divide_by_min": "Divide by minimum",
    "delta_f_over_f0": "ΔF/F0 (baseline = 10th percentile)",
    "detrend_minmax": "Detrend + scale to 0–1",
    "lowest_points": "Divide by lowest-points baseline (lowest points → 1)",
}

# Methods that use the lowest-points baseline themselves (the "Lowest points"
# setting applies to them, and smoothing with that same baseline is redundant).
LOWEST_POINTS_METHODS = ("lowest_points",)

# Method name used briefly for "subtract the baseline" before it became the
# subtract_baseline option; read as method "none" + subtract_baseline=True.
LEGACY_SUBTRACT_METHOD = "subtract_lowest_points"

DEFAULT_LOWEST_POINTS = 6

MIN_MODES = {
    "local": "minimum of that column in each sheet",
    "sheet": "minimum of the entire sheet",
    "column_global": "minimum of that column across all sheets",
    "all_global": "minimum of all columns and sheets",
    "none": "no normalization (data already normalized)",
}


# ------------------------------------------------------------------ helpers

def fill_missing(values):
    """Fill NaNs by linear interpolation (edges take the nearest value)."""
    values = np.asarray(values, dtype=float)
    if not np.isnan(values).any():
        return values
    return pd.Series(values).interpolate(limit_direction="both").to_numpy()


def _finite_min(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.min()) if finite.size else None


def _usable_divisor(d):
    """None and 0 mean "don't divide" (a zero minimum would give inf)."""
    return d if d is not None and d != 0 else None


# ----------------------------------------------------- single-signal methods

def divide_by_min(values):
    values = np.asarray(values, dtype=float)
    d = _usable_divisor(_finite_min(values))
    return values / d if d is not None else values.copy()


def delta_f_over_f0(values, percentile=10):
    """ΔF/F0 with F0 = the given percentile of the signal (ignoring NaNs).
    If F0 is 0, returns F - F0 (dividing would give inf)."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values.copy()
    f0 = float(np.percentile(finite, percentile))
    return (values - f0) / f0 if f0 != 0 else values - f0


def detrend_minmax(values):
    """Subtract a linear trend fitted with ElasticNet, then scale to 0..1
    (from the signal-processing notebook). A flat signal gives zeros."""
    from sklearn.linear_model import ElasticNet

    values = fill_missing(values)
    x = np.arange(len(values)).reshape(-1, 1)
    trend = ElasticNet().fit(x, values).predict(x)
    detrended = values - trend
    lo, hi = detrended.min(), detrended.max()
    return (detrended - lo) / (hi - lo) if hi > lo else np.zeros_like(detrended)


_METHOD_FUNCTIONS = {
    "none": lambda v: np.asarray(v, dtype=float).copy(),
    "divide_by_min": divide_by_min,
    "delta_f_over_f0": delta_f_over_f0,
    "detrend_minmax": detrend_minmax,
}


# --------------------------------------------- Convex Envelope smoothing
# Baseline built from a handful of the signal's genuine lowest points,
# found via local convexity so none can fall inside a peak.

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


def smooth_convex_envelope(data_sel, n_points=6):
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


def lowest_points_baseline(values, n_points=DEFAULT_LOWEST_POINTS):
    """The Convex Envelope baseline (piecewise-linear line through the
    signal's genuine lowest points) and the points used: (baseline, px)."""
    y = fill_missing(values)
    px, py = convex_envelope_lowest_points(y, n_points=n_points)
    return np.interp(np.arange(len(y)), px, py), np.asarray(px, dtype=int)


def divide_by_lowest_points(values, n_points=DEFAULT_LOWEST_POINTS):
    """Divide a signal by its lowest-points baseline, so the lowest points
    lie on a horizontal line at 1 (the original NecLab Convex Envelope
    normalization).

    Raises ValueError when the baseline gets close to zero or below it
    (e.g. data already normalized to 0..1): dividing would blow up. Use
    the subtraction (smoothing) or ΔF/F0 for such signals.
    """
    values = np.asarray(values, dtype=float)
    missing = ~np.isfinite(values)
    if missing.all():
        return values.copy()
    baseline, _px = lowest_points_baseline(values, n_points)
    scale = float(np.nanmax(np.abs(values)))
    if baseline.min() <= max(1e-6 * scale, 1e-12):
        raise ValueError(
            f"its lowest-points baseline reaches {baseline.min():.3g}, so dividing by it would blow up "
            "(this happens with signals near or below zero, e.g. already normalized). "
            "Use smoothing (subtraction) or ΔF/F0 for these signals.")
    out = fill_missing(values) / baseline
    out[missing] = np.nan
    return out


def subtract_lowest_points(values, n_points=DEFAULT_LOWEST_POINTS):
    """Subtract a signal's lowest-points baseline, so the lowest points lie
    on a horizontal line at 0. Unlike divide_by_lowest_points there is no
    division, so it works whatever the signal's level (even near or below
    zero), and unlike Convex Envelope smoothing the signal's minimum isn't
    added back."""
    values = np.asarray(values, dtype=float)
    missing = ~np.isfinite(values)
    if missing.all():
        return values.copy()
    baseline, _px = lowest_points_baseline(values, n_points)
    out = fill_missing(values) - baseline
    out[missing] = np.nan
    return out


def smooth_keeping_gaps(values, n_points):
    """Convex Envelope smoothing of a series with missing values: NaNs are
    filled for the computation and put back afterwards."""
    values = np.asarray(values, dtype=float)
    missing = ~np.isfinite(values)
    if missing.all():
        return values.copy()
    out = smooth_convex_envelope(fill_missing(values) if missing.any() else values, n_points=n_points)
    out[missing] = np.nan
    return out


# ------------------------------------------------------------ tables

def normalize_signals(df, method="none", smoothing_points=None, lowest_points=DEFAULT_LOWEST_POINTS,
                      subtract_baseline=False):
    """Normalize every column of a signals table on its own, then
    optionally subtract its lowest-points baseline or smooth it.

    Args:
        df: signals table (frames x cells).
        method: a key of NORMALIZATION_METHODS.
        smoothing_points: Convex Envelope key points (None/0 = no smoothing).
            Ignored when subtract_baseline is on (it would remove the same
            baseline a second time).
        lowest_points: number of lowest points for the lowest-points
            baseline (the "lowest_points" method and subtract_baseline).
        subtract_baseline: after normalizing, subtract the lowest-points
            baseline of the normalized signal, so its lowest points lie on
            a horizontal line at 0 (in the method's units).

    Returns a new DataFrame with the same index and columns. Raises
    ValueError (naming the signal) if a signal can't be normalized.
    """
    if method == LEGACY_SUBTRACT_METHOD:
        method, subtract_baseline = "none", True
    if method not in NORMALIZATION_METHODS:
        raise ValueError(f"method must be one of {tuple(NORMALIZATION_METHODS)}, got {method!r}")
    out = {}
    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        try:
            if method == "lowest_points":
                values = divide_by_lowest_points(values, int(lowest_points))
            else:
                values = _METHOD_FUNCTIONS[method](values)
        except ValueError as e:
            raise ValueError(f"{col}: {e}") from None
        if subtract_baseline:
            values = subtract_lowest_points(values, int(lowest_points))
        elif smoothing_points:
            values = smooth_keeping_gaps(values, int(smoothing_points))
        out[col] = values
    return pd.DataFrame(out, index=df.index)


def is_identity(method, smoothing_points=None, subtract_baseline=False):
    """True if normalize_signals(method, smoothing_points, subtract_baseline=...)
    returns the data unchanged."""
    return method == "none" and not smoothing_points and not subtract_baseline


def min_divisors(tables, mode):
    """Divisor of every column of every table for divide_by_min_tables.

    Returns a list (one per table) of {column: divisor or None}; None means
    "don't divide" (no finite values, a zero minimum, or mode "none").
    Columns are matched across tables by name.
    """
    if mode not in MIN_MODES:
        raise ValueError(f"mode must be one of {tuple(MIN_MODES)}, got {mode!r}")
    if mode == "none":
        return [{c: None for c in t.columns} for t in tables]
    if mode == "local":
        return [{c: _usable_divisor(_finite_min(t[c])) for c in t.columns} for t in tables]
    if mode == "sheet":
        return [{c: d for c in t.columns}
                for t, d in ((t, _usable_divisor(_finite_min(t.to_numpy(dtype=float).ravel())))
                             for t in tables)]
    if mode == "column_global":
        pooled = {}
        for t in tables:
            for c in t.columns:
                m = _finite_min(t[c])
                if m is not None:
                    pooled[c] = m if c not in pooled else min(pooled[c], m)
        return [{c: _usable_divisor(pooled.get(c)) for c in t.columns} for t in tables]
    # all_global
    mins = [m for m in (_finite_min(t.to_numpy(dtype=float).ravel()) for t in tables) if m is not None]
    d = _usable_divisor(min(mins)) if mins else None
    return [{c: d for c in t.columns} for t in tables]


def divide_by_min_tables(tables, mode):
    """Divide several tables (e.g. Multiple Files sheets) by the minimum
    chosen by `mode` (see MIN_MODES). NaNs are kept. Returns new tables."""
    out = []
    for t, divisors in zip(tables, min_divisors(tables, mode)):
        cols = {}
        for c in t.columns:
            v = t[c].to_numpy(dtype=float)
            cols[c] = v / divisors[c] if divisors[c] is not None else v.copy()
        out.append(pd.DataFrame(cols, index=t.index))
    return out
