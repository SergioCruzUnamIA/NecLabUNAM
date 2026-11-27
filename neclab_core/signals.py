"""
Per-cell signal extraction.

Cells don't move during a recording, so each cell is a fixed set of pixels
(its pixels in the label image) and its signal is the mean of those pixels
in every frame.
"""

import numpy as np
import pandas as pd

from .segmentation import cell_ids as _cell_ids, cell_pixels

CELL_COLUMN_PREFIX = "cell_"


def cell_column(cell_id):
    """Column name used for a cell's signal in tables and files ("cell_<id>")."""
    return f"{CELL_COLUMN_PREFIX}{int(cell_id)}"


def parse_cell_column(name):
    """"cell_<id>" -> int id, or None if the name isn't a cell column."""
    name = str(name)
    if name.startswith(CELL_COLUMN_PREFIX):
        try:
            return int(name[len(CELL_COLUMN_PREFIX):])
        except ValueError:
            return None
    return None


def signal_label(name):
    """How a signal column is shown: "Cell 12" for "cell_12", else its name."""
    cid = parse_cell_column(name)
    return f"Cell {cid}" if cid is not None else str(name)


def extract_signals(stack, labels, cell_ids=None):
    """Mean intensity of each cell in every frame.

    Args:
        stack: raw data, shape (frames, height, width).
        labels: label image, shape (height, width).
        cell_ids: which cells to extract (default: all, in ID order).

    Returns:
        DataFrame with one row per frame (index named "frame", 0..T-1) and
        one float column per cell, named "cell_<id>".
    """
    stack = np.asarray(stack)
    labels = np.asarray(labels)
    if stack.ndim != 3:
        raise ValueError(f"stack must be (frames, height, width), got shape {stack.shape}")
    if labels.shape != stack.shape[1:]:
        raise ValueError(f"label image shape {labels.shape} does not match frame shape {stack.shape[1:]}")

    ids = _cell_ids(labels) if cell_ids is None else [int(i) for i in cell_ids]
    pixels = cell_pixels(labels)
    missing = [i for i in ids if i not in pixels]
    if missing:
        raise ValueError(f"cell IDs not in the label image: {missing}")

    columns = {}
    for cid in ids:
        coords = pixels[cid]
        columns[cell_column(cid)] = stack[:, coords[:, 0], coords[:, 1]].mean(axis=1, dtype=np.float64)

    df = pd.DataFrame(columns, index=pd.RangeIndex(stack.shape[0], name="frame"))
    return df
