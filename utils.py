import os
import cv2
import numpy as np
import glob
import shutil
from typing import List, Tuple, Optional

class ImageUtils:
    """Utilities for image I/O, directory management, and geometry."""
    
    @staticmethod
    def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
        """Loads all images from a folder."""
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
