"""
Stage 6 for a set of signals: normalization + event (peak) detection.

`Analyzer.run` takes the raw signals (stage 5) and the stage 6 settings
and returns the normalized signals, the events table and what the stage
records. Every signal is normalized and searched for peaks on its own, so
the peaks of each signal are cached: when one cell is added to the
selection only that cell's peaks are detected (the slow part of some
methods), as long as the settings don't change.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import events, normalization

DEFAULT_SETTINGS = {
    "method": "none",                                  # a key of normalization.NORMALIZATION_METHODS
    "lowest_points": normalization.DEFAULT_LOWEST_POINTS,
    "smooth": False,                                   # "Smoothing (subtract lowest-points baseline)"
    "subtract": False,                                 # "Subtract lowest-points baseline"
    "peak_method": events.NO_PEAKS,                    # a key of events.PEAK_METHODS, or NO_PEAKS
    "peak_params": {},
}


def settings_from_params(params):
    """Stage 6 settings from the params a stage 6 record was saved with
    (also those of earlier versions); defaults for anything missing."""
    settings = {**DEFAULT_SETTINGS, "peak_params": {}}
    params = params or {}
    method = params.get("method")
    subtract = False
    if method == normalization.LEGACY_SUBTRACT_METHOD:      # saved by an earlier version
        method, subtract = "none", True
    if method in normalization.NORMALIZATION_METHODS:
        smoothing_points = int(params.get("smoothing_points") or 0)
        settings.update(
            method=method, smooth=smoothing_points > 0,
            subtract=subtract or bool(params.get("subtract_baseline")),
            lowest_points=int(params.get("lowest_points") or smoothing_points
                              or normalization.DEFAULT_LOWEST_POINTS))
    peak_method = params.get("peak_method")
    if peak_method in events.PEAK_METHODS:
        settings["peak_method"] = peak_method
        settings["peak_params"] = dict(params.get("peak_params") or {})
    return settings


def smoothing_points(settings):
    """Points used for smoothing, or None. Smoothing is skipped when the
    baseline is subtracted, or with "Divide by lowest-points baseline":
    either already removes that same baseline."""
    if (settings["smooth"] and not settings["subtract"]
            and settings["method"] not in normalization.LOWEST_POINTS_METHODS):
        return int(settings["lowest_points"])
    return None


def peak_params(settings):
    """The peak method's parameters (its defaults + the chosen ones), or {}."""
    method = settings["peak_method"]
    if method not in events.PEAK_METHODS:
        return {}
    return {**events.default_params(method), **(settings.get("peak_params") or {})}


def describe(settings):
    """Short description of the normalization ("raw" when there is none)."""
    method, points, subtract = settings["method"], int(settings["lowest_points"]), settings["subtract"]
    smoothing = smoothing_points(settings)
    parts = []
    if method == "lowest_points":
        parts.append(f"divided by lowest-points baseline ({points} points)")
    elif method != "none":
        parts.append(normalization.NORMALIZATION_METHODS[method])
    if subtract:
        parts.append(f"lowest-points baseline subtracted ({points} points)")
    if smoothing:
        parts.append(f"smoothed ({smoothing} lowest points)")
    return ", ".join(parts) or "raw"


def is_identity(settings):
    return normalization.is_identity(settings["method"], smoothing_points(settings), settings["subtract"])


@dataclass
class AnalysisResult:
    norm_df: pd.DataFrame            # the signals as normalized (= the raw ones when there is no normalization)
    events_df: object                # events table, or None when no peak method is chosen (or it failed)
    description: str                 # e.g. "raw", "ΔF/F0, smoothed (6 lowest points)"
    identity: bool                   # True if norm_df is the raw signals
    settings: dict
    lowest_points_used: dict = field(default_factory=dict)  # {column: frames of the lowest points used}
    events_error: str = ""           # why peak detection failed, if it did

    @property
    def params(self):
        """Settings recorded with stage 6."""
        s = self.settings
        params = {"method": s["method"], "smoothing_points": smoothing_points(s) or 0,
                  "lowest_points": int(s["lowest_points"]), "subtract_baseline": bool(s["subtract"])}
        if self.events_df is not None:
            params["peak_method"] = s["peak_method"]
            params["peak_params"] = peak_params(s)
            params["peaks_on"] = "raw signals" if self.identity else "normalized signals"
        return params

    @property
    def items(self):
        """Stage 6 items: the normalized signals (always, so the stage is
        usable even with no normalization) and the events, if any."""
        items = {"normalized_signals": self.norm_df}
        if self.events_df is not None:
            items["events"] = self.events_df
        return items


def _settings_key(settings):
    return (settings["method"], int(settings["lowest_points"]), smoothing_points(settings),
            bool(settings["subtract"]), settings["peak_method"],
            tuple(sorted(peak_params(settings).items())))


class Analyzer:
    """Runs stage 6, caching each signal's peaks for the current settings."""

    def __init__(self):
        self._key = None
        self._events = {}       # {column: (raw values, events rows of that column)}

    def clear(self):
        self._key = None
        self._events = {}

    def run(self, signals_df, settings):
        """Normalize `signals_df` and detect its events with `settings`
        (see DEFAULT_SETTINGS). Raises ValueError (naming the signal) if a
        signal can't be normalized; a failing peak method is reported in
        the result's events_error instead."""
        settings = {**DEFAULT_SETTINGS, **settings}
        smoothing = smoothing_points(settings)
        method, points, subtract = settings["method"], int(settings["lowest_points"]), settings["subtract"]
        identity = is_identity(settings)
        norm_df = (signals_df if identity else
                   normalization.normalize_signals(signals_df, method, smoothing, points,
                                                   subtract_baseline=subtract))
        result = AnalysisResult(norm_df, None, describe(settings), identity, settings)

        # Positions of the lowest points used, to mark them on a plot: on
        # the normalized (before subtraction/smoothing) signal when the
        # baseline is subtracted or smoothed, on the raw signal when it only
        # divides.
        lowest_method = method in normalization.LOWEST_POINTS_METHODS
        if lowest_method or smoothing or subtract:
            base = (signals_df if lowest_method and not subtract
                    else normalization.normalize_signals(signals_df, method, None, points))
            for col in base.columns:
                if np.isfinite(base[col].to_numpy(dtype=float)).sum() >= 2:
                    result.lowest_points_used[col] = normalization.lowest_points_baseline(base[col], points)[1]

        if settings["peak_method"] in events.PEAK_METHODS:
            try:
                result.events_df = self._detect(signals_df, norm_df, settings)
            except Exception as e:      # the methods' own errors (e.g. a fit that fails)
                result.events_error = f"{settings['peak_method']} failed: {e}"
        return result

    def _detect(self, signals_df, norm_df, settings):
        key = _settings_key(settings)
        if key != self._key:
            self._key, self._events = key, {}
        method, params = settings["peak_method"], peak_params(settings)
        parts = []
        for col in norm_df.columns:
            raw = signals_df[col].to_numpy(dtype=float)
            cached = self._events.get(col)
            if cached is None or not np.array_equal(cached[0], raw, equal_nan=True):
                cached = (raw.copy(), events.detect_events(norm_df[[col]], method, params))
                self._events[col] = cached
            parts.append(cached[1])
        if not parts:
            return events.detect_events(norm_df, method, params)
        df = pd.concat(parts, ignore_index=True)
        sort_by = [c for c in ("cell_id", "signal") if c in df.columns] + ["frame"]
        return df.sort_values(sort_by, kind="stable").reset_index(drop=True)
