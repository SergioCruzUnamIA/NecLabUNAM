# image_processing.py
"""
PIL/matplotlib helpers for showing images. The calculations are in
neclab_core.display (display-only adjustments) and neclab_core.variability.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from neclab_core import display


def auto_contrast(img_array, cutoff=2, ignore=2):
    """
    Returns a display COPY of the stack with auto contrast applied to each
    frame separately. Only for viewing: the per-frame rescaling changes
    intensities over time, so the result must never be used for analysis.

    Args:
        img_array (numpy.ndarray): Image stack with shape (num_slices, height, width).
        cutoff (int): Percentage of pixels to clip from each end of the histogram.
        ignore (int): Background value left out of the histogram.

    Returns:
        numpy.ndarray: 8-bit copy of the stack with auto contrast applied.
    """
    value_range = display.display_range(img_array)
    return np.stack([display.auto_contrast(display.to_display_uint8(frame, value_range), cutoff, ignore)
                     for frame in img_array])


def show_histogram(img_array):
    """
    Shows the histogram of the per-pixel variance of the image stack.

    Args:
        img_array (numpy.ndarray): Image stack with shape (num_slices, height, width).
    """
    var_im = np.var(img_array, axis=0)
    plt.hist(var_im.ravel(), bins=50)
    plt.title("Histogram of Variance")
    plt.xlabel("Pixel variance (flattened)")
    plt.ylabel("Frequency")
    plt.show()


def mask_to_pil(mask):
    """Boolean mask -> black/white RGB PIL image."""
    return Image.fromarray(display.mask_to_uint8(mask)).convert('RGB')


def binarize_variance(img_array, threshold=150):
    """
    Binary RGB PIL image of the pixels whose variance over the stack is
    above `threshold`.
    """
    return mask_to_pil(display.threshold_mask(np.var(img_array, axis=0), threshold))


def threshold_image_pil(image_2d, threshold=200):
    """
    Binary RGB PIL image of the pixels of a 2D image above `threshold`
    (255 = above, 0 = at or below).
    """
    return mask_to_pil(display.threshold_mask(image_2d, threshold))
