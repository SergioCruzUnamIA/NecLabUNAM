"""
Display-only image adjustments.

Nothing in this module modifies the analysis data: every function works on
a single frame and returns a new 8-bit array meant only for the screen.
Brightness, contrast, auto contrast and the threshold preview all live
here, so they can never leak into the variability analysis.
"""

import numpy as np
from PIL import Image, ImageOps


def display_range(stack):
    """Value range used to map a stack to 8 bits for display.

    8-bit data is shown as is (0-255). Anything else (e.g. 16-bit or float)
    is scaled linearly from the stack's own minimum to its maximum, so the
    whole stack shares one mapping and frames stay comparable.
    """
    stack = np.asarray(stack)
    if stack.dtype == np.uint8:
        return 0.0, 255.0
    finite = stack[np.isfinite(stack)] if np.issubdtype(stack.dtype, np.floating) else stack
    if finite.size == 0:
        return 0.0, 1.0
    return float(finite.min()), float(finite.max())


def to_display_uint8(frame, value_range=None):
    """Return an 8-bit copy of `frame` for display.

    8-bit frames are returned unchanged (as a copy). Other types are scaled
    from `value_range` (lo, hi) to 0-255; pass the range from
    display_range(stack) so every frame uses the same mapping.
    """
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame.copy()
    lo, hi = value_range if value_range is not None else display_range(frame)
    if hi <= lo:
        return np.zeros(frame.shape, dtype=np.uint8)
    scaled = (np.nan_to_num(frame.astype(np.float64), nan=lo) - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def auto_contrast(frame_u8, cutoff=2, ignore=2):
    """Auto contrast of one 8-bit frame (PIL's ImageOps.autocontrast).

    `cutoff` is the percentage of pixels clipped at each end of the
    histogram; `ignore` is a background value left out of the histogram.
    """
    pil_img = Image.fromarray(np.asarray(frame_u8, dtype=np.uint8))
    return np.asarray(ImageOps.autocontrast(pil_img, cutoff=cutoff, ignore=ignore))


def apply_brightness_contrast(frame_u8, brightness=0, contrast=0):
    """Brightness (-100..100 offset) and contrast (-100..100) on an 8-bit frame."""
    img = np.asarray(frame_u8).astype(np.float32)
    img = img + brightness
    if contrast != 0:
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        img = factor * (img - 128) + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def threshold_mask(image, threshold):
    """Boolean mask of pixels strictly above `threshold`."""
    return np.asarray(image) > threshold


def mask_to_uint8(mask):
    """Boolean mask -> 8-bit image (0 / 255), for display."""
    return np.asarray(mask, dtype=bool).astype(np.uint8) * 255
