"""
Cell detection: candidate-pixel mask -> cells.

The result of every detection method is a *label image*: a 2D integer array
the size of one frame, where 0 is background and every other value is the
permanent ID of a cell. IDs are numbered 1..N in raster order of each
cell's first pixel (top to bottom, then left to right), so the same
detection settings on the same data always give the same IDs, and a saved
label image keeps them across sessions.

Internally the splitting algorithms work on "pixel lists" (one list of
(row, col) tuples per region), which is what the original notebook code
used; label images are converted to and from them at the edges.
"""

import logging
import math

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

# 8-connectivity: diagonal neighbours belong to the same region.
_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)

SPLIT_METHODS = ("none", "peaks", "kmeans")


# ---------------------------------------------------------------- basics

def binarize(image, threshold):
    """Boolean mask of the pixels strictly above `threshold`."""
    return np.asarray(image) > threshold


def connected_components(mask):
    """Label 8-connected regions of a boolean mask (0 = background)."""
    labels, _ = ndimage.label(np.asarray(mask, dtype=bool), structure=_EIGHT_CONNECTED)
    return labels.astype(np.int32)


def label_image_to_pixel_lists(labels):
    """Label image -> list of pixel lists, one per non-zero label, in label
    order; the pixels of each region are in raster order."""
    labels = np.asarray(labels)
    coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return []
    values = labels[coords[:, 0], coords[:, 1]]
    order = np.argsort(values, kind="stable")
    coords, values = coords[order], values[order]
    splits = np.flatnonzero(np.diff(values)) + 1
    return [[(int(r), int(c)) for r, c in group] for group in np.split(coords, splits)]


def pixel_lists_to_label_image(pixel_lists, shape):
    """Pixel lists -> label image with IDs renumbered in raster order.

    Pixels claimed by more than one list keep the later list's ID; empty
    lists are skipped.
    """
    labels = np.zeros(shape, dtype=np.int32)
    for i, pixels in enumerate(pixel_lists, start=1):
        if len(pixels) == 0:
            continue
        arr = np.asarray(pixels, dtype=int).reshape(-1, 2)
        labels[arr[:, 0], arr[:, 1]] = i
    return relabel_raster_order(labels)


def relabel_raster_order(labels):
    """Renumber the labels 1..N in raster order of each label's first pixel."""
    labels = np.asarray(labels)
    flat = labels.ravel()
    present = flat > 0
    if not present.any():
        return np.zeros(labels.shape, dtype=np.int32)
    values, first_index = np.unique(flat[present], return_index=True)
    order = np.argsort(np.flatnonzero(present)[first_index])
    mapping = np.zeros(int(values.max()) + 1, dtype=np.int32)
    mapping[values[order]] = np.arange(1, len(values) + 1, dtype=np.int32)
    return mapping[labels].astype(np.int32)


def cell_ids(labels):
    """Sorted list of the cell IDs present in a label image."""
    ids = np.unique(np.asarray(labels))
    return [int(i) for i in ids if i > 0]


def cell_pixels(labels):
    """{cell_id: (N, 2) array of (row, col)} for every cell, pixels in raster order."""
    labels = np.asarray(labels)
    coords = np.argwhere(labels > 0)
    if coords.size == 0:
        return {}
    values = labels[coords[:, 0], coords[:, 1]]
    order = np.argsort(values, kind="stable")
    coords, values = coords[order], values[order]
    splits = np.flatnonzero(np.diff(values)) + 1
    groups = np.split(coords, splits)
    ids = values[np.concatenate(([0], splits))] if len(values) else []
    return {int(i): g for i, g in zip(ids, groups)}


def cell_table(labels):
    """One row per cell: cell_id, n_pixels, centroid_row, centroid_col."""
    import pandas as pd

    pixels = cell_pixels(labels)
    rows = []
    for cid, coords in pixels.items():
        rows.append({
            "cell_id": cid,
            "n_pixels": int(len(coords)),
            "centroid_row": float(coords[:, 0].mean()),
            "centroid_col": float(coords[:, 1].mean()),
        })
    return pd.DataFrame(rows, columns=["cell_id", "n_pixels", "centroid_row", "centroid_col"])


def subset_labels(labels, keep_ids):
    """Copy of a label image keeping only `keep_ids` (other cells -> 0); IDs unchanged."""
    labels = np.asarray(labels)
    return np.where(np.isin(labels, list(keep_ids)), labels, 0).astype(np.int32)


def filter_by_size(pixel_lists, min_size, max_size):
    """Keep the regions with min_size <= number of pixels <= max_size."""
    return [cl for cl in pixel_lists if min_size <= len(cl) <= max_size]


# ------------------------------------------------- split by peaks ("advanced")

def find_peaks(Z, dz_dx, dz_dy, selected_points):
    """Local maxima of Z, from sign changes in the derivatives.
    Original code by Jose."""
    peaks = []
    for i, j in selected_points:
        if 1 <= i < Z.shape[0] - 1 and 1 <= j < Z.shape[1] - 1:
            if dz_dx[i, j - 1] > 0 and dz_dx[i, j] <= 0 and dz_dy[i - 1, j] > 0 and dz_dy[i, j] <= 0:
                peaks.append((i, j))
    return peaks


def assign_points_to_peaks(Z, peaks, selected_points):
    """Assign every point to the nearest peak reached by following the
    gradient uphill. Returns {peak_index: [points]}. Original code by Jose."""
    peak_sets = {i: [] for i in range(len(peaks))}

    peak_map = np.zeros_like(Z, dtype=int) - 1
    gx = np.gradient(Z, axis=1)
    gy = np.gradient(Z, axis=0)

    for label_id, (pi, pj) in enumerate(peaks):
        if 0 <= pi < Z.shape[0] and 0 <= pj < Z.shape[1]:
            peak_map[pi, pj] = label_id

    for i, j in selected_points:
        if not (0 <= i < Z.shape[0] and 0 <= j < Z.shape[1]):
            continue

        x, y = i, j
        seen = []
        while (x, y) not in seen and (x, y) not in peaks:
            if len(seen) > 100:  # avoid infinite loops
                break
            if not (0 <= x < Z.shape[0] and 0 <= y < Z.shape[1]):
                break
            if abs(gx[x, y]) > 1e-10 or abs(gy[x, y]) > 1e-10:
                dx, dy = int(np.sign(gx[x, y])), int(np.sign(gy[x, y]))
            else:
                break
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < Z.shape[0]) and (0 <= new_y < Z.shape[1]):
                seen.append((x, y))
                x, y = new_x, new_y
            else:
                break

        if peaks:
            min_dist = float("inf")
            sel_peak = None
            for p in peaks:
                dst = math.dist(p, (x, y))
                if dst < min_dist:
                    min_dist = dst
                    sel_peak = p
            if sel_peak is not None:
                peak_id = peak_map[sel_peak[0], sel_peak[1]]
                if peak_id >= 0 and peak_id in peak_sets:
                    peak_sets[peak_id].append((i, j))

    return peak_sets


def _split_one_by_peaks(var_im, cl, dz_dx, dz_dy):
    """Split one oversized region around the local maxima of var_im.
    Returns a list of pixel lists (the region itself if it can't be split)."""
    if len(cl) < 3:
        return [cl]

    xs = np.array([p[0] for p in cl])
    ys = np.array([p[1] for p in cl])
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    if not (max_x > min_x and max_y > min_y):
        return [cl]

    Z = np.zeros((max_x - min_x + 1, max_y - min_y + 1))
    for i, j in cl:
        Z[i - min_x, j - min_y] = var_im[i, j]

    relative_cl = [(i - min_x, j - min_y) for i, j in cl]
    peaks = find_peaks(Z, dz_dx[min_x:max_x + 1, min_y:max_y + 1],
                       dz_dy[min_x:max_x + 1, min_y:max_y + 1], relative_cl)
    if not peaks:
        return [cl]

    peak_sets = assign_points_to_peaks(Z, peaks, relative_cl)
    return [[(p[0] + min_x, p[1] + min_y) for p in points]
            for points in peak_sets.values() if points]


def split_by_peaks(var_im, pixel_lists, min_size=20, max_size=200):
    """"Advanced" processing: regions smaller than min_size are discarded,
    regions within [min_size, max_size] are kept, and larger regions are
    split around the local maxima of the variability map.

    Returns the list of resulting pixel lists.
    """
    var_im = np.asarray(var_im)
    dz_dx = np.gradient(var_im, axis=1)
    dz_dy = np.gradient(var_im, axis=0)

    result = []
    for cl in pixel_lists:
        if len(cl) < min_size:
            continue
        if len(cl) <= max_size:
            result.append(cl)
            continue
        try:
            result.extend(_split_one_by_peaks(var_im, cl, dz_dx, dz_dy))
        except Exception:
            logger.warning("Peak split failed for a region of %d pixels; kept it whole",
                           len(cl), exc_info=True)
            result.append(cl)
    return result


# ------------------------------------------------------ split by KMeans

def split_large_kmeans(pixel_lists, var_im, min_size=20, max_size=200):
    """Split oversized regions with KMeans on (row*10, col*10, scaled
    variability) features. Regions smaller than min_size are discarded,
    regions within [min_size, max_size] are kept, and each region larger
    than max_size is split into (size // max_size) + 1 parts; parts smaller
    than min_size are discarded. Based on the cluster-Copy4.ipynb notebook.
    """
    var_im = np.asarray(var_im)
    max_ax = np.max(var_im.shape)
    max_var = var_im.max()
    if max_var == 0:
        max_var = 1
    norm_var_im = (var_im / max_var) * max_ax

    result = []
    for cl in pixel_lists:
        cl_size = len(cl)
        if cl_size < min_size:
            continue
        if cl_size <= max_size:
            result.append(cl)
            continue
        num_sub = (cl_size // max_size) + 1
        try:
            features = [(p[0] * 10, p[1] * 10, norm_var_im[p[0], p[1]]) for p in cl]
            km_labels = KMeans(n_clusters=num_sub, random_state=0, n_init="auto").fit(features).labels_
            cl_arr = np.asarray(cl)
            for value in np.unique(km_labels):
                sub = [tuple(int(v) for v in p) for p in cl_arr[km_labels == value]]
                if len(sub) >= min_size:
                    result.append(sub)
        except Exception:
            logger.warning("KMeans split failed for a region of %d pixels; kept it whole",
                           cl_size, exc_info=True)
            result.append(cl)
    return result


# ------------------------------------------------------------ full run

def detect_cells(var_im, processed_im, threshold, min_size=20, max_size=200, split="none"):
    """Full cell detection.

    1. Threshold `processed_im` (the sharpened variability map).
    2. Find 8-connected regions.
    3. Filter/split by size with `split`:
         "none"   keep regions with min_size <= size <= max_size
         "peaks"  split oversized regions around variability peaks
         "kmeans" split oversized regions with KMeans
       Peaks and KMeans use the unsharpened `var_im`.

    Returns:
        (labels, mask): the label image (int32, IDs 1..N in raster order)
        and the boolean candidate-pixel mask.
    """
    if split not in SPLIT_METHODS:
        raise ValueError(f"split must be one of {SPLIT_METHODS}, got {split!r}")
    mask = binarize(processed_im, threshold)
    return detect_cells_from_mask(mask, var_im, min_size, max_size, split), mask


def detect_cells_from_mask(mask, var_im=None, min_size=20, max_size=200, split="none"):
    """Cell detection starting from a candidate-pixel mask (e.g. one loaded
    from a file): steps 2-3 of detect_cells.

    "peaks" and "kmeans" need the variability map `var_im`; "none" doesn't.
    Returns the label image.
    """
    if split not in SPLIT_METHODS:
        raise ValueError(f"split must be one of {SPLIT_METHODS}, got {split!r}")
    mask = np.asarray(mask, dtype=bool)
    if split != "none":
        if var_im is None:
            raise ValueError(f"Splitting with {split!r} needs the variability map")
        if np.asarray(var_im).shape != mask.shape:
            raise ValueError("The variability map and the mask must be the same size")
    regions = label_image_to_pixel_lists(connected_components(mask))

    if split == "none":
        cells = filter_by_size(regions, min_size, max_size)
    elif split == "peaks":
        cells = split_by_peaks(var_im, regions, min_size, max_size)
    else:
        cells = split_large_kmeans(regions, var_im, min_size, max_size)

    return pixel_lists_to_label_image(cells, mask.shape)
