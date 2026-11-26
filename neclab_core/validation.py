"""
Checks run on every piece of data that enters the pipeline, whether it was
computed in this session or loaded from a file.

Each check either returns normally (optionally with a list of warning
strings for things that are unusual but usable) or raises ValidationError
with a message written for the person using the app, so a bad file is
rejected at the door instead of failing halfway through a later stage.

Cross-stage checks take what is already loaded as keyword arguments
(frame_shape, n_frames, cell_ids): pass whatever is known, leave the rest
as None, and only the checks that can be made are made.
"""

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Data that can't be used by the pipeline (message is user-facing)."""


def _shape_text(shape):
    return " x ".join(str(int(s)) for s in shape)


def check_frame_shape(shape, frame_shape, what):
    """`shape` (height, width) must equal `frame_shape` when it is known."""
    if frame_shape is not None and tuple(shape) != tuple(frame_shape):
        raise ValidationError(
            f"The {what} is {_shape_text(shape)} pixels but the loaded image frames are "
            f"{_shape_text(frame_shape)}. They must be the same size.")


# --------------------------------------------------------------- images

def check_stack(stack):
    """Recording: numeric array (frames, height, width) with at least 2 frames."""
    stack = np.asarray(stack)
    if stack.ndim == 2:
        raise ValidationError(
            "This file holds a single 2D image, not a recording (frames x height x width). "
            "If it is a variability map, open it as stage 2 instead.")
    if stack.ndim != 3:
        raise ValidationError(
            f"Expected a recording with shape (frames, height, width), got {stack.shape}. "
            "Files with several channels or Z-planes aren't supported yet.")
    if not (np.issubdtype(stack.dtype, np.integer) or np.issubdtype(stack.dtype, np.floating)):
        raise ValidationError(f"The recording must contain numbers, got data type {stack.dtype}.")
    if stack.shape[0] < 2:
        raise ValidationError("The recording has a single frame; at least 2 are needed.")
    if min(stack.shape[1:]) < 2:
        raise ValidationError(f"The frames are too small ({_shape_text(stack.shape[1:])} pixels).")
    return []


def check_map(image, frame_shape=None, name="variability map"):
    """Variability/feature map: 2D numeric array with some finite values."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValidationError(f"The {name} must be a 2D image, got shape {image.shape}.")
    if not (np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.floating)
            or image.dtype == bool):
        raise ValidationError(f"The {name} must contain numbers, got data type {image.dtype}.")
    check_frame_shape(image.shape, frame_shape, name)
    finite = np.isfinite(image) if np.issubdtype(image.dtype, np.floating) else np.ones(image.shape, bool)
    if not finite.any():
        raise ValidationError(f"The {name} has no finite values.")
    warnings = []
    if not finite.all():
        warnings.append(f"The {name} has {int((~finite).sum())} non-finite pixels (NaN/inf); "
                        "they are never selected as cell pixels.")
    return warnings


def check_maps(maps, frame_shape=None):
    """{name: 2D map}: at least one map, all the same size."""
    if not maps:
        raise ValidationError("No variability maps found in the file.")
    warnings = []
    shapes = {name: np.asarray(m).shape for name, m in maps.items()}
    if len(set(shapes.values())) > 1:
        detail = ", ".join(f"{n}: {_shape_text(s)}" for n, s in shapes.items())
        raise ValidationError(f"The maps in the file have different sizes ({detail}).")
    for name, m in maps.items():
        warnings += check_map(m, frame_shape, name=f"map '{name}'")
    return warnings


def check_mask(mask, frame_shape=None):
    """Candidate-pixel mask: 2D, only two values (bool, 0/1 or 0/255), not empty."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValidationError(f"A mask must be a 2D image, got shape {mask.shape}.")
    check_frame_shape(mask.shape, frame_shape, "mask")
    if mask.dtype != bool:
        values = np.unique(mask)
        if not (set(values.tolist()) <= {0, 1} or set(values.tolist()) <= {0, 255}):
            shown = ", ".join(str(v) for v in values[:6]) + (" ..." if len(values) > 6 else "")
            raise ValidationError(
                "A mask must contain only two values (0/1 or 0/255), "
                f"but this image contains {len(values)} different values ({shown}). "
                "If it is a variability map, open it as stage 2 instead.")
    warnings = []
    if not np.asarray(mask, dtype=bool).any():
        warnings.append("The mask is empty (no pixels selected).")
    return warnings


def check_labels(labels, frame_shape=None):
    """Label image: 2D, whole non-negative numbers, at least one cell."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValidationError(f"A label image must be 2D, got shape {labels.shape}.")
    if labels.dtype == bool:
        raise ValidationError("This is a mask (two values), not a label image. "
                              "Open it as stage 3 (candidate mask) instead.")
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.issubdtype(labels.dtype, np.floating) or \
                not np.all(np.equal(np.mod(labels[np.isfinite(labels)], 1), 0)) or \
                not np.isfinite(labels).all():
            raise ValidationError("A label image must contain whole numbers (cell IDs).")
    if labels.min() < 0:
        raise ValidationError("A label image can't contain negative values.")
    check_frame_shape(labels.shape, frame_shape, "label image")
    warnings = []
    if labels.max() == 0:
        warnings.append("The label image contains no cells.")
    return warnings


# --------------------------------------------------------------- tables

def check_signals(df, n_frames=None, cell_ids=None, name="signals table"):
    """Signals table: one row per frame, one numeric column per cell.

    n_frames: frame count of the loaded recording, if any.
    cell_ids: IDs of the loaded cells, if any; every "cell_<id>" column must
        be one of them.
    """
    from .signals import parse_cell_column

    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"The {name} must be a table.")
    if df.shape[1] == 0:
        raise ValidationError(f"The {name} has no data columns.")
    if df.shape[0] < 2:
        raise ValidationError(f"The {name} has {df.shape[0]} rows; at least 2 frames are needed.")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    empty = [str(c) for c in df.columns if numeric[c].isna().all()]
    if empty:
        raise ValidationError(f"These columns of the {name} contain no numbers: {', '.join(empty[:10])}.")
    if n_frames is not None and df.shape[0] != n_frames:
        raise ValidationError(
            f"The {name} has {df.shape[0]} rows but the loaded recording has {n_frames} frames.")
    if cell_ids is not None:
        known = set(int(i) for i in cell_ids)
        ids = [parse_cell_column(c) for c in df.columns]
        unknown = [f"cell_{i}" for i in ids if i is not None and i not in known]
        if unknown:
            raise ValidationError(
                f"The {name} has cells that aren't in the loaded cells: {', '.join(unknown[:10])}.")
    warnings = []
    n_missing = int(numeric.isna().sum().sum())
    if n_missing:
        warnings.append(f"The {name} has {n_missing} missing/non-numeric values; "
                        "they are filled by interpolation where the signals are used.")
    if len(set(map(str, df.columns))) != df.shape[1]:
        raise ValidationError(f"The {name} has repeated column names.")
    return warnings


def check_events(df, n_frames=None, cell_ids=None):
    """Events table: a "frame" column plus "cell_id" or "signal"."""
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("The events table must be a table.")
    if "frame" not in df.columns:
        raise ValidationError("The events table needs a 'frame' column.")
    if "cell_id" not in df.columns and "signal" not in df.columns:
        raise ValidationError("The events table needs a 'cell_id' (or 'signal') column.")
    frames = pd.to_numeric(df["frame"], errors="coerce")
    if frames.isna().any() or (frames < 0).any() or not np.all(np.mod(frames, 1) == 0):
        raise ValidationError("The 'frame' column must contain whole, non-negative frame numbers.")
    if n_frames is not None and len(frames) and frames.max() >= n_frames:
        raise ValidationError(
            f"The events table refers to frame {int(frames.max())} but the recording has {n_frames} frames.")
    if cell_ids is not None and "cell_id" in df.columns:
        unknown = sorted(set(pd.to_numeric(df["cell_id"], errors="coerce").dropna().astype(int))
                         - set(int(i) for i in cell_ids))
        if unknown:
            raise ValidationError(f"The events table has cells that aren't loaded: {unknown[:10]}.")
    return []


def check_correlation(df, tolerance=1e-6):
    """Correlation matrix: square, same labels on rows and columns, numeric.

    Values outside [-1, 1] and asymmetry are reported as warnings (the
    matrix may come from another measure of similarity).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError("The correlation matrix must be a table.")
    if df.shape[0] != df.shape[1]:
        raise ValidationError(
            f"The matrix has {df.shape[0]} rows and {df.shape[1]} columns. A correlation matrix must be square.")
    if df.shape[0] < 2:
        raise ValidationError("The correlation matrix must have at least 2 signals.")
    values = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValidationError("The correlation matrix contains empty or non-numeric values.")
    warnings = []
    if [str(i) for i in df.index] != [str(c) for c in df.columns]:
        warnings.append("The row labels don't match the column labels; the column labels are used.")
    if values.min() < -1 - tolerance or values.max() > 1 + tolerance:
        warnings.append("The matrix contains values outside the correlation range [-1, 1].")
    if not np.allclose(values, values.T, atol=1e-6):
        warnings.append("The matrix isn't symmetric.")
    return warnings


# --------------------------------------------------------- cross-stage

def check_compatibility(stack=None, maps=None, mask=None, labels=None, signals=None):
    """Check that the pieces loaded so far fit together (same frame size,
    same frame count, cell IDs that exist). Raises ValidationError on the
    first mismatch; returns the list of warnings otherwise."""
    frame_shape = None
    n_frames = None
    warnings = []
    if stack is not None:
        warnings += check_stack(stack)
        frame_shape = tuple(np.asarray(stack).shape[1:])
        n_frames = int(np.asarray(stack).shape[0])
    if maps is not None:
        warnings += check_maps(maps, frame_shape)
        frame_shape = frame_shape or tuple(np.asarray(next(iter(maps.values()))).shape)
    if mask is not None:
        warnings += check_mask(mask, frame_shape)
        frame_shape = frame_shape or tuple(np.asarray(mask).shape)
    cell_ids = None
    if labels is not None:
        warnings += check_labels(labels, frame_shape)
        cell_ids = [int(i) for i in np.unique(labels) if i > 0]
    if signals is not None:
        warnings += check_signals(signals, n_frames=n_frames, cell_ids=cell_ids)
    return warnings
