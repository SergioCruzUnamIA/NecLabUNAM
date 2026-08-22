# image_processing.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

def auto_contrast(img_array, cutoff=2, ignore=2):
    """
    Applies auto-contrast to each slice of the image stack.
    Works on a COPY so the original data isn't modified,
    since the contrast is only for display, not processing.

    Args:
        img_array (numpy.ndarray): Image stack with shape (num_slices, height, width).
        cutoff (int): Percentage of pixels to clip from each end of the histogram.
        ignore (int): Number of pixels to ignore at the ends of the histogram.

    Returns:
        numpy.ndarray: Copy of the array with auto-contrast applied.
    """
    result = img_array.copy()
    for i in range(result.shape[0]):
        im_pil = Image.fromarray(result[i, :, :])
        if im_pil.mode != 'RGB':
            im_pil = im_pil.convert('RGB')
        im2 = ImageOps.autocontrast(im_pil, cutoff=cutoff, ignore=ignore).convert('L')
        result[i, :, :] = np.array(im2)
    return result


def show_histogram(img_array):
    """
    Shows the histogram of the variance of the image stack.
    Computes the variance along axis 0 (stack) and then plots the 2D histogram.

    Args:
        img_array (numpy.ndarray): Image stack with shape (num_slices, height, width).
    """
    var_im = np.var(img_array, axis=0)
    plt.hist(var_im.ravel(), bins=50)
    plt.title("Histogram of Variance")
    plt.xlabel("Pixel variance (flattened)")
    plt.ylabel("Frequency")
    plt.show()


def binarize_variance(img_array, threshold=150):
    """
    Computes the variance along the stack (axis 0) and creates a 2D binary
    image using the given threshold.

    Args:
        img_array (numpy.ndarray): Image stack with shape (num_slices, height, width).
        threshold (int): Threshold for binarizing the variance.

    Returns:
        PIL.Image: Resulting binary (RGB) image.
    """
    # Compute the variance per pixel along axis 0
    var_im = np.var(img_array, axis=0)
    # Convert to a 1D vector
    var_im_flat = var_im.ravel()
    # Generate the binary mask according to the threshold
    bin_mask = (var_im_flat > threshold).astype(np.uint8) * 255
    # Reshape back to 2D
    bin_mask_2d = bin_mask.reshape(var_im.shape)
    # Create the PIL image
    pil_img = Image.fromarray(bin_mask_2d)
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return pil_img

def threshold_image_pil(image_2d, threshold=200):
    """
    Applies a simple threshold to a 2D image (one slice) and returns the binary image as a PIL.Image.

    Args:
        image_2d (numpy.ndarray): 2D image (height x width).
        threshold (int): Threshold value. Pixels with value > threshold become 255 (white)
                         and the rest become 0 (black).

    Returns:
        PIL.Image: Binary image in RGB mode.
    """
    # Flatten the image to a 1D vector
    arr_flat = image_2d.ravel()
    # Apply the threshold and multiply by 255 to get values of 0 or 255
    bin_mask = (arr_flat > threshold).astype(np.uint8) * 255
    # Reconstruct the image in its original shape (2D)
    bin_mask_2d = bin_mask.reshape(image_2d.shape)
    # Create the PIL image
    pil_img = Image.fromarray(bin_mask_2d)
    # The image must be in RGB mode for compatibility with Tkinter
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    return pil_img
