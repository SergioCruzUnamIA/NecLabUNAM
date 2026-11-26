"""
Per-pixel variability maps over time, used to find the pixels that belong
to active cells.

Every function takes the raw stack with shape (frames, height, width) and
returns a 2D map with shape (height, width).
"""

import numpy as np
import cv2 as cv

# (name, default threshold) for each method, in menu order.
VARIABILITY_METHODS = [
    ("Range", 100),
    ("Population Variance", 120),
    ("Sample Variance", 200),
    ("Population Standard Deviation", 12),
    ("Sample Standard Deviation", 5),
    ("Coefficient of Variation", 5),
    ("Interquartile Range (IQR)", 20),
]

# 3x3 sharpening kernel applied to the variability map before thresholding.
SHARPEN_KERNEL = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def method_names():
    """Names of the variability methods, in menu order."""
    return [name for name, _ in VARIABILITY_METHODS]


def calculate_variability(stack, method=1):
    """Variability of every pixel over time.

    Args:
        stack: array (frames, height, width) - the raw data, never a
            display-adjusted copy.
        method: index into VARIABILITY_METHODS (0-6).

    Returns:
        (map, default_threshold, method_name)
    """
    if not 0 <= method < len(VARIABILITY_METHODS):
        raise ValueError("Method must be between 0 and 6")
    stack = np.asarray(stack)
    name, default_th = VARIABILITY_METHODS[method]

    if method == 0:
        var_im = np.max(stack, axis=0) - np.min(stack, axis=0)
    elif method == 1:
        var_im = np.var(stack, axis=0)
    elif method == 2:
        var_im = np.var(stack, axis=0, ddof=1)
    elif method == 3:
        var_im = np.std(stack, axis=0)
    elif method == 4:
        var_im = np.std(stack, axis=0, ddof=1)
    elif method == 5:
        with np.errstate(divide="ignore", invalid="ignore"):
            var_im = (np.std(stack, axis=0, ddof=1) / np.mean(stack, axis=0)) * 100
    else:
        var_im = np.percentile(stack, 75, axis=0) - np.percentile(stack, 25, axis=0)

    return var_im, default_th, name


def sharpen(var_im):
    """Sharpen a variability map with SHARPEN_KERNEL (same type as the input)."""
    return cv.filter2D(var_im, -1, SHARPEN_KERNEL)
