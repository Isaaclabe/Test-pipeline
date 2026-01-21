"""
Utility functions for the image processing pipeline.
"""

import os
import cv2
import glob
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger("Pipeline")


def load_images_from_folder(
    folder_path: str,
    mask_h_percent: float = 0.0,
    mask_w_percent: float = 0.0
) -> List[np.ndarray]:
    """
    Load all images from a folder.
    
    Args:
        folder_path: Path to the folder containing images
        mask_h_percent: Percentage of height to mask from bottom-right
        mask_w_percent: Percentage of width to mask from bottom-right
        
    Returns:
        List of loaded images
    """
    if not os.path.exists(folder_path):
        return []

    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
        image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    image_paths = sorted(set(image_paths))
    images = []

    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # Apply corner masking if specified
            if mask_h_percent > 0 or mask_w_percent > 0:
                img = apply_corner_mask(img, mask_h_percent, mask_w_percent)
            images.append(img)

    return images


def apply_corner_mask(
    img: np.ndarray,
    h_percent: float,
    w_percent: float
) -> np.ndarray:
    """
    Apply a black mask to the bottom-right corner of an image.
    Useful for hiding watermarks or timestamps.
    """
    h, w = img.shape[:2]
    mask_h = int(h * h_percent)
    mask_w = int(w * w_percent)
    
    result = img.copy()
    result[h - mask_h:, w - mask_w:] = 0
    
    return result


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_image(img: np.ndarray, path: str) -> bool:
    """Save image to disk."""
    try:
        ensure_dir(os.path.dirname(path))
        cv2.imwrite(path, img)
        return True
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False


def resize_for_display(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Resize image for display purposes."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img
