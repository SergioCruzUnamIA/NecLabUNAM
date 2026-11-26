"""
File formats shared by every part of NecLab, one section per pipeline
stage. Every reader validates what it loads (see validation.py) and raises
validation.ValidationError with a readable message when a file can't be
used.

Stage 0/1 - recording (image stack)
    .ome.tif / .tif / .tiff / .npy   array (frames, height, width)

Stage 2 - variability maps
    .npz         one or more named 2D maps ({"Population Variance": map, ...})
    .npy / .tif  a single 2D map (named after the file)

Stage 3 - candidate-pixel mask
    .tif         uint8, 0 = background, 255 = candidate pixel
    .npy         bool array
    (also accepted: any 2D image containing only 0/1 or 0/255)

Stage 4 - cells
    <name>.tif or <name>.npy  label image (int32, 0 = background, n = cell n)
    <name>.csv                cell table written next to it
                              (cell_id, n_pixels, centroid_row, centroid_col)
    Also accepted: ImageJ ROIs (.roi, or a RoiSet .zip from the ROI
    Manager; cell n = n-th ROI in the file) and legacy "Export
    Coordinates" .txt files.

Stage 5/6 - signals table (one file per recording)
    .csv / .xlsx  header row "frame, cell_<id>, cell_<id>, ..."; one row per
                  frame. The first column is the frame number (0..T-1).
    .npy          2D float array (frames, 1 + cells); column 0 is the frame
                  number, the rest are the cells in ID order (a .npy can't
                  store the IDs themselves - use .csv/.xlsx to keep them).
    Every NecLab loader recognises the frame column and does not treat it
    as a signal. Tables from other programs are accepted too: a header row
    is optional, and the first column is treated as time when it is named
    "frame"/"time"/"time_s" or holds 0, 1, 2, ... (or as told).

Stage 6 - events (detected peaks)
    .csv / .xlsx  one row per event: frame, cell_id (or signal), plus any
                  extra columns. The Multiple Files "Save Peaks CSV" export
                  (Column, Sheet, Sample_Index, Peak_Flag) is accepted too.

Stage 7 - correlation matrix
    .csv / .xlsx  square table with the signal names as both the first
                  column and the header row.
"""

import os
import re

import numpy as np
import pandas as pd

from . import validation
from .segmentation import cell_table, pixel_lists_to_label_image
from .signals import parse_cell_column

TIME_COLUMN_NAMES = ("frame", "time_s", "time", "t")
SIGNAL_EXTENSIONS = (".csv", ".xlsx", ".npy")
LABEL_IMAGE_EXTENSIONS = (".tif", ".tiff", ".npy")


def _ext(path):
    return os.path.splitext(str(path))[1].lower()


def _stem(path):
    name = os.path.basename(str(path))
    for suffix in (".ome.tiff", ".ome.tif"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _is_ome(path):
    return str(path).lower().endswith((".ome.tif", ".ome.tiff"))


def _read_tiff(path):
    import tifffile
    return tifffile.imread(str(path))


def _write_tiff(path, array):
    import tifffile
    tifffile.imwrite(str(path), array)


def _read_table(path, header, sheet_name=0):
    ext = _ext(path)
    if ext == ".csv":
        return pd.read_csv(path, header=header)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, header=header, sheet_name=sheet_name)
    raise validation.ValidationError(f"Unsupported table type {ext!r} (use .csv or .xlsx).")


# ======================================================= stage 0/1: stack

def load_ometiff(path):
    """Read an OME-TIFF file. Returns (array, metadata dict)."""
    from pyometiff import OMETIFFReader

    img, metadata, _xml = OMETIFFReader(fpath=str(path)).read()
    return img, metadata


def load_stack(path):
    """Read a recording (.ome.tif, .tif, .tiff or .npy) and validate it.

    Returns (stack, metadata); metadata is the OME metadata dict for
    OME-TIFF files and {} otherwise.
    """
    ext = _ext(path)
    if _is_ome(path):
        stack, metadata = load_ometiff(path)
    elif ext in (".tif", ".tiff"):
        stack, metadata = _read_tiff(path), {}
    elif ext == ".npy":
        stack, metadata = np.load(path, allow_pickle=False), {}
    else:
        raise validation.ValidationError(f"Unsupported recording type {ext!r} (use .tif, .ome.tif or .npy).")
    stack = np.asarray(stack)
    # tifffile/pyometiff can keep singleton channel/Z axes: drop them.
    if stack.ndim > 3:
        squeezed = np.squeeze(stack)
        if squeezed.ndim == 3:
            stack = squeezed
    validation.check_stack(stack)
    return stack, metadata or {}


def save_stack(stack, path):
    """Write a (processed) recording as .tif or .npy."""
    validation.check_stack(stack)
    ext = _ext(path)
    if ext in (".tif", ".tiff"):
        _write_tiff(path, np.asarray(stack))
    elif ext == ".npy":
        np.save(path, np.asarray(stack))
    else:
        raise validation.ValidationError(f"Unsupported recording type {ext!r} (use .tif or .npy).")


# ======================================================= stage 2: maps

def save_maps(maps, path):
    """Write variability maps: several named maps to .npz, or a single map
    to .npy/.tif. `maps` is {name: 2D array}."""
    validation.check_maps(maps)
    ext = _ext(path)
    if ext == ".npz":
        np.savez_compressed(path, **{str(k): np.asarray(v) for k, v in maps.items()})
    elif ext in (".npy", ".tif", ".tiff"):
        if len(maps) != 1:
            raise validation.ValidationError("A .npy/.tif file holds a single map; use .npz to save several.")
        (only,) = maps.values()
        if ext == ".npy":
            np.save(path, np.asarray(only))
        else:
            _write_tiff(path, np.asarray(only, dtype=np.float32))
    else:
        raise validation.ValidationError(f"Unsupported map type {ext!r} (use .npz, .npy or .tif).")


def load_maps(path, frame_shape=None):
    """Read variability maps. Returns ({name: 2D float array}, warnings)."""
    ext = _ext(path)
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            maps = {name: np.asarray(data[name]) for name in data.files}
    elif ext == ".npy":
        maps = {_stem(path): np.load(path, allow_pickle=False)}
    elif ext in (".tif", ".tiff"):
        maps = {_stem(path): np.squeeze(_read_tiff(path))}
    else:
        raise validation.ValidationError(f"Unsupported map type {ext!r} (use .npz, .npy or .tif).")
    warnings = validation.check_maps(maps, frame_shape)
    return {k: np.asarray(v, dtype=np.float64) for k, v in maps.items()}, warnings


# ======================================================= stage 3: mask

def save_mask(mask, path):
    """Write a candidate-pixel mask (.tif as 0/255 uint8, .npy as bool)."""
    mask = np.asarray(mask, dtype=bool)
    validation.check_mask(mask)
    ext = _ext(path)
    if ext in (".tif", ".tiff"):
        _write_tiff(path, mask.astype(np.uint8) * 255)
    elif ext == ".npy":
        np.save(path, mask)
    else:
        raise validation.ValidationError(f"Unsupported mask type {ext!r} (use .tif or .npy).")


def load_mask(path, frame_shape=None):
    """Read a candidate-pixel mask. Returns (bool array, warnings)."""
    ext = _ext(path)
    if ext in (".tif", ".tiff"):
        mask = np.squeeze(_read_tiff(path))
    elif ext == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        raise validation.ValidationError(f"Unsupported mask type {ext!r} (use .tif or .npy).")
    warnings = validation.check_mask(mask, frame_shape)
    return np.asarray(mask).astype(bool), warnings


# ======================================================= stage 4: cells

def cells_table_path(label_path):
    """Path of the cell table written next to a label image."""
    return os.path.splitext(str(label_path))[0] + ".csv"


def save_cells(labels, path):
    """Save a label image (.tif or .npy) and its cell table (.csv, same name).

    Returns (label_image_path, table_path).
    """
    ext = _ext(path)
    labels = np.asarray(labels, dtype=np.int32)
    validation.check_labels(labels)
    if ext in (".tif", ".tiff"):
        _write_tiff(path, labels)
    elif ext == ".npy":
        np.save(path, labels)
    else:
        raise validation.ValidationError(f"Unsupported label image type {ext!r} (use .tif or .npy).")
    table_path = cells_table_path(path)
    cell_table(labels).to_csv(table_path, index=False)
    return path, table_path


def load_cells(path, shape=None):
    """Load cells as a label image.

    Accepts a label image (.tif/.tiff/.npy, IDs kept as saved), ImageJ ROIs
    (.roi or .zip; cell n = n-th ROI) or a legacy "Export Coordinates" .txt
    file (IDs renumbered 1..N in raster order). ROI and .txt files need
    `shape`; if `shape` is given the result must match it.
    """
    ext = _ext(path)
    if ext in (".tif", ".tiff"):
        labels = _read_tiff(path)
    elif ext == ".npy":
        labels = np.load(path, allow_pickle=False)
    elif ext in (".roi", ".zip"):
        if shape is None:
            raise validation.ValidationError("Loading ImageJ ROIs needs the image size: load the recording first.")
        labels, _skipped = rois_to_labels(path, shape)
    elif ext == ".txt":
        if shape is None:
            raise validation.ValidationError("Loading a coordinates .txt file needs the image size: "
                                             "load the recording first.")
        labels = pixel_lists_to_label_image(_read_coordinates_txt(path), shape)
    else:
        raise validation.ValidationError(f"Unsupported cells file type {ext!r} (use .tif, .npy, .roi, .zip or .txt).")

    labels = np.squeeze(np.asarray(labels))
    validation.check_labels(labels, shape)
    return labels.astype(np.int32)


def rois_to_labels(path, shape):
    """Rasterise ImageJ area ROIs into a label image.

    Cell n is the n-th ROI in the file (the ROI Manager's order). Where
    ROIs overlap, the later ROI wins. Returns (labels, skipped) where
    `skipped` lists the names of non-area ROIs (lines, points, ...) that
    were ignored.
    """
    import roifile
    from skimage.draw import polygon

    rois = roifile.roiread(str(path))
    if not isinstance(rois, list):
        rois = [rois]
    area_types = {roifile.ROI_TYPE.POLYGON, roifile.ROI_TYPE.RECT, roifile.ROI_TYPE.OVAL,
                  roifile.ROI_TYPE.FREEHAND, roifile.ROI_TYPE.TRACED}
    labels = np.zeros(shape, dtype=np.int32)
    skipped = []
    next_id = 1
    for i, roi in enumerate(rois, start=1):
        name = roi.name or f"ROI {i}"
        if roi.roitype not in area_types or roi.subtype not in (roifile.ROI_SUBTYPE.UNDEFINED,):
            skipped.append(name)
            continue
        coords = np.asarray(roi.coordinates(), dtype=float)
        if coords.ndim != 2 or len(coords) < 3:
            skipped.append(name)
            continue
        # ImageJ pixel (x, y) covers [x, x+1) x [y, y+1); test pixel centres.
        rr, cc = polygon(coords[:, 1] - 0.5, coords[:, 0] - 0.5, shape=shape)
        if rr.size == 0:
            skipped.append(name)
            continue
        labels[rr, cc] = next_id
        next_id += 1
    if next_id == 1:
        raise validation.ValidationError("The ROI file has no area ROIs (polygons, rectangles, ovals) "
                                         "inside the image.")
    return labels, skipped


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
    if not clusters:
        raise validation.ValidationError("No 'Cluster <n>' sections found in the coordinates file.")
    return clusters


# ======================================================= stage 5/6: signals

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
    """True if a header names the time axis ("frame", "time", "time_s", "t")."""
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
        raise validation.ValidationError(f"Unsupported signals file type {ext!r} (use .csv, .xlsx or .npy).")


def _row_is_numeric(row):
    return pd.to_numeric(pd.Series(list(row)), errors="coerce").notna().all()


def read_signals(path, time_column="auto", sheet_name=0, n_frames=None, cell_ids=None):
    """Read a signals table (NecLab's own or from another program).

    Args:
        path: .csv, .xlsx or .npy file.
        time_column: "auto" (drop the first column when it is named like a
            time axis or holds 0, 1, 2, ...), True (always drop it) or
            False (it is a signal).
        sheet_name: Excel sheet to read.
        n_frames, cell_ids: what is already loaded, for validation.

    Returns:
        (DataFrame with index "frame" and one numeric column per signal,
         warnings). Columns keep their header names ("cell_<id>" for NecLab
        files); files without a header get "column_1".."column_N".
    """
    ext = _ext(path)
    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise validation.ValidationError(f"A signals .npy must be 2D (frames x signals), got shape {arr.shape}.")
        raw = pd.DataFrame(arr)
        has_header = False
    else:
        raw = _read_table(path, header=None, sheet_name=sheet_name)
        has_header = raw.shape[0] > 0 and not _row_is_numeric(raw.iloc[0])
        if has_header:
            header = [str(h).strip() for h in raw.iloc[0]]
            raw = raw.iloc[1:].reset_index(drop=True)
            raw.columns = header

    if raw.shape[1] > 1 and time_column is not False:
        first_name = raw.columns[0]
        first_values = pd.to_numeric(raw.iloc[:, 0], errors="coerce").to_numpy()
        drop = time_column is True or (
            (has_header and is_time_column_name(first_name)) or is_frame_index(first_values))
        if drop:
            raw = raw.iloc[:, 1:]

    if not has_header:
        raw.columns = [f"column_{i}" for i in range(1, raw.shape[1] + 1)]
    df = raw.apply(pd.to_numeric, errors="coerce")
    df.index = pd.RangeIndex(len(df), name="frame")
    warnings = validation.check_signals(df, n_frames=n_frames, cell_ids=cell_ids)
    return df.astype(float), warnings


def signal_cell_ids(df):
    """Cell IDs from a signals table's "cell_<id>" columns (None for other columns)."""
    return [parse_cell_column(c) for c in df.columns]


# ======================================================= stage 6: events

EVENT_COLUMNS = ("cell_id", "frame")


def write_events(df, path):
    """Write an events table (one row per event; frame + cell_id/signal)."""
    validation.check_events(df)
    ext = _ext(path)
    if ext == ".csv":
        df.to_csv(path, index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=False, engine="xlsxwriter")
    else:
        raise validation.ValidationError(f"Unsupported events file type {ext!r} (use .csv or .xlsx).")


def read_events(path, n_frames=None, cell_ids=None):
    """Read an events table. Also accepts the Multiple Files "Save Peaks
    CSV" export (Column, Sheet, Sample_Index, Peak_Flag): its flagged rows
    become events with columns signal, sheet, frame.

    Returns (DataFrame, warnings).
    """
    df = _read_table(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    if {"Column", "Sheet", "Sample_Index", "Peak_Flag"} <= set(df.columns):
        flagged = df[pd.to_numeric(df["Peak_Flag"], errors="coerce") == 1]
        df = pd.DataFrame({"signal": flagged["Column"].astype(str).to_numpy(),
                           "sheet": flagged["Sheet"].astype(str).to_numpy(),
                           "frame": pd.to_numeric(flagged["Sample_Index"]).astype(int).to_numpy()})
    warnings = validation.check_events(df, n_frames=n_frames, cell_ids=cell_ids)
    df = df.copy()
    df["frame"] = pd.to_numeric(df["frame"]).astype(int)
    if "cell_id" in df.columns:
        df["cell_id"] = pd.to_numeric(df["cell_id"]).astype(int)
    return df.reset_index(drop=True), warnings


# ======================================================= stage 7: correlation

def write_correlation(df, path):
    """Write a correlation matrix (labels as first column and header)."""
    validation.check_correlation(df)
    ext = _ext(path)
    if ext == ".csv":
        df.to_csv(path, index=True)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(path, index=True, sheet_name="Correlation", engine="xlsxwriter")
    else:
        raise validation.ValidationError(f"Unsupported correlation file type {ext!r} (use .csv or .xlsx).")


def read_correlation(path, sheet_name=0):
    """Read a correlation matrix. Returns (square DataFrame, warnings)."""
    ext = _ext(path)
    if ext == ".csv":
        df = pd.read_csv(path, index_col=0)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, index_col=0, sheet_name=sheet_name)
    else:
        raise validation.ValidationError(f"Unsupported correlation file type {ext!r} (use .csv or .xlsx).")
    if df.empty:
        raise validation.ValidationError("The correlation file is empty.")
    warnings = validation.check_correlation(df)
    df = df.apply(pd.to_numeric, errors="coerce").astype(float)
    df.index = list(df.columns)
    return df, warnings
