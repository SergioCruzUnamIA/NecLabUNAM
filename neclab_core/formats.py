"""
File formats shared by every part of NecLab.

Signals table (one file per recording)
    .csv / .xlsx  header row "frame, cell_<id>, cell_<id>, ..."; one row per
                  frame. The first column is the frame number (0..T-1).
    .npy          2D float array (frames, 1 + cells); column 0 is the frame
                  number, the rest are the cells in ID order (a .npy can't
                  store the IDs themselves - use .csv/.xlsx to keep them).
    Every NecLab loader recognises the frame column and does not treat it
    as a signal.

Cells
    <name>.tif or <name>.npy  label image (int32, 0 = background, n = cell n)
    <name>.csv                cell table written next to it
                              (cell_id, n_pixels, centroid_row, centroid_col)
    Legacy "Export Coordinates" .txt files can also be loaded.
"""

import os
import re

import numpy as np
import pandas as pd

from .segmentation import cell_table, pixel_lists_to_label_image
from .signals import parse_cell_column

TIME_COLUMN_NAMES = ("frame", "time_s")
SIGNAL_EXTENSIONS = (".csv", ".xlsx", ".npy")
LABEL_IMAGE_EXTENSIONS = (".tif", ".tiff", ".npy")


def _ext(path):
    return os.path.splitext(str(path))[1].lower()


# ------------------------------------------------------------ raw data

def load_ometiff(path):
    """Read an OME-TIFF file. Returns (array, metadata dict)."""
    from pyometiff import OMETIFFReader

    img, metadata, _xml = OMETIFFReader(fpath=str(path)).read()
    return img, metadata


# ------------------------------------------------------------ signals

def is_frame_index(values):
    """True if `values` is exactly 0, 1, 2, ..., n-1 (a frame-number column)."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size < 2:
        return False
    try:
        values = values.astype(float)
    except (TypeError, ValueError):
        return False
    return bool(np.array_equal(values, np.arange(values.size, dtype=float)))


def is_time_column_name(name):
    """True if a header names the time axis ("frame" or "time_s")."""
    return str(name).strip().lower() in TIME_COLUMN_NAMES


def write_signals(df, path):
    """Write a signals table (index = frame, columns = cell_<id>) to
    .csv, .xlsx or .npy (see the module docstring for the layout)."""
    ext = _ext(path)
    out = df.copy()
    out.index = pd.RangeIndex(len(out), name="frame")
    if ext == ".csv":
        out.to_csv(path, index=True, index_label="frame")
    elif ext in (".xlsx", ".xls"):
        out.to_excel(path, index=True, index_label="frame", engine="xlsxwriter")
    elif ext == ".npy":
        frame_col = np.arange(len(out), dtype=float).reshape(-1, 1)
        np.save(path, np.hstack([frame_col, out.to_numpy(dtype=float)]))
    else:
        raise ValueError(f"Unsupported signals file type: {ext!r} (use .csv, .xlsx or .npy)")


def read_signals(path):
    """Read a signals table written by write_signals.

    Returns a DataFrame with index "frame" and one column per cell. Columns
    named "cell_<id>" keep their IDs; .npy columns are named
    "column_1".."column_N" because the file has no IDs.
    """
    ext = _ext(path)
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] > 1 and is_frame_index(arr[:, 0]):
            arr = arr[:, 1:]
        cols = [f"column_{i}" for i in range(1, arr.shape[1] + 1)]
        return pd.DataFrame(arr.astype(float), columns=cols,
                            index=pd.RangeIndex(arr.shape[0], name="frame"))

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported signals file type: {ext!r} (use .csv, .xlsx or .npy)")

    if df.shape[1] and is_time_column_name(df.columns[0]):
        df = df.iloc[:, 1:]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = pd.RangeIndex(len(df), name="frame")
    return df


def signal_cell_ids(df):
    """Cell IDs from a signals table's "cell_<id>" columns (None for other columns)."""
    return [parse_cell_column(c) for c in df.columns]


# ------------------------------------------------------------ cells

def cells_table_path(label_path):
    """Path of the cell table written next to a label image."""
    return os.path.splitext(str(label_path))[0] + ".csv"


def save_cells(labels, path):
    """Save a label image (.tif or .npy) and its cell table (.csv, same name).

    Returns (label_image_path, table_path).
    """
    ext = _ext(path)
    labels = np.asarray(labels, dtype=np.int32)
    if ext in (".tif", ".tiff"):
        import tifffile
        tifffile.imwrite(path, labels)
    elif ext == ".npy":
        np.save(path, labels)
    else:
        raise ValueError(f"Unsupported label image type: {ext!r} (use .tif or .npy)")
    table_path = cells_table_path(path)
    cell_table(labels).to_csv(table_path, index=False)
    return path, table_path


def load_cells(path, shape=None):
    """Load cells as a label image.

    Accepts a label image (.tif/.tiff/.npy, IDs kept as saved) or a legacy
    "Export Coordinates" .txt file (IDs renumbered 1..N in raster order;
    needs `shape`). If `shape` is given, the result must match it.
    """
    ext = _ext(path)
    if ext in (".tif", ".tiff"):
        import tifffile
        labels = tifffile.imread(path)
    elif ext == ".npy":
        labels = np.load(path, allow_pickle=False)
    elif ext == ".txt":
        if shape is None:
            raise ValueError("Loading a coordinates .txt file needs the image shape")
        labels = pixel_lists_to_label_image(_read_coordinates_txt(path), shape)
    else:
        raise ValueError(f"Unsupported cells file type: {ext!r} (use .tif, .npy or .txt)")

    labels = np.squeeze(np.asarray(labels))
    if labels.ndim != 2:
        raise ValueError(f"A label image must be 2D, got shape {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.all(np.equal(np.mod(labels, 1), 0)):
            raise ValueError("A label image must contain whole numbers (cell IDs)")
    if labels.min() < 0:
        raise ValueError("A label image can't contain negative values")
    if shape is not None and tuple(labels.shape) != tuple(shape):
        raise ValueError(f"Label image shape {labels.shape} does not match the image {tuple(shape)}")
    return labels.astype(np.int32)


_CLUSTER_HEADER = re.compile(r"^Cluster\s+(\d+)\b")


def _read_coordinates_txt(path):
    """Parse a legacy "Export Coordinates" file into pixel lists.

    Format: "Cluster <n> (<k> points):" followed by one "row,col" line per
    pixel; lines starting with "#" are comments.
    """
    clusters = []
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if _CLUSTER_HEADER.match(line):
                current = []
                clusters.append(current)
                continue
            if current is None:
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                current.append((int(float(parts[0])), int(float(parts[1]))))
    return clusters
