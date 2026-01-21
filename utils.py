import os
import cv2
import numpy as np
import glob
import shutil
from typing import List, Tuple, Optional

class ImageUtils:
    """Utilities for image I/O, directory management, and geometry."""
    
    @staticmethod
    def crop_bottom_percent(image: np.ndarray, percent: float = 15.0) -> np.ndarray:
        """Crops the bottom percentage of an image.
        
        Args:
            image: Input image as numpy array
            percent: Percentage of the bottom to crop (default 15%)
            
        Returns:
            Cropped image with bottom portion removed
        """
        if image is None:
            return None
        height = image.shape[0]
        crop_height = int(height * (1 - percent / 100.0))
        return image[:crop_height, :]
    
    @staticmethod
    def load_images_from_folder(folder_path: str, crop_bottom: bool = True, crop_percent: float = 15.0) -> List[np.ndarray]:
        """Loads all images from a folder.
        
        Args:
            folder_path: Path to the folder containing images
            crop_bottom: Whether to crop the bottom of images (default True)
            crop_percent: Percentage of bottom to crop (default 15%)
        """
        images = []
        if not os.path.exists(folder_path):
            return images
            
        exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in exts:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
        image_files = sorted(list(set(image_files))) 

        for f in image_files:
            img = cv2.imread(f)
            if img is not None:
                if crop_bottom:
                    img = ImageUtils.crop_bottom_percent(img, crop_percent)
                images.append(img)
        return images

    @staticmethod
    def clear_and_create_dir(path: str):
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: Could not clear directory {path}: {e}")
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def save_crop(image: np.ndarray, mask: np.ndarray, save_path: str):
        """Saves a crop of the image defined by the mask."""
        if image is None or mask is None:
            return

        try:
            mask = mask.astype(np.uint8)
            if mask.max() > 1:
                mask = (mask > 0).astype(np.uint8)
                
            coords = cv2.findNonZero(mask)
            if coords is None:
                return
                
            x, y, w, h = cv2.boundingRect(coords)
            
            crop_img = image[y:y+h, x:x+w]
            crop_mask = mask[y:y+h, x:x+w]
            
            b, g, r = cv2.split(crop_img)
            alpha = crop_mask * 255
            
            out = cv2.merge([b, g, r, alpha])
            cv2.imwrite(save_path, out)
        except Exception as e:
            print(f"Failed to save crop {save_path}: {e}")

    @staticmethod
    def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculates Intersection over Union."""
        if mask1 is None or mask2 is None: return 0.0
        
        # Ensure dimensions match
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Convert to boolean logic to handle 0/255 or 0/1 masks
        m1 = mask1 > 0
        m2 = mask2 > 0
        
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        
        if union == 0:
            return 0.0
        return intersection / union
