import cv2
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ImageStitcher:
    """
    Handles the stitching of multiple images.
    Strictly uses 'Canvas Mode' (horizontal concatenation) to ensure no data is lost.
    """
    
    def __init__(self):
        pass # No complex stitcher initialization needed

    def stitch(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        if not images:
            return None
        if len(images) == 1:
            return images[0]

        logger.info(f"Concatenating {len(images)} images (Canvas Mode)...")
        return self._concat_images(images)

    def _concat_images(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Resizes images to the same height and appends them horizontally.
        This creates a 'canvas' style strip.
        """
        try:
            # Target height = average height of images to keep scale consistent
            target_h = int(sum(img.shape[0] for img in images) / len(images))
            
            resized_imgs = []
            for img in images:
                h, w = img.shape[:2]
                # Maintain aspect ratio
                if h == 0: continue
                aspect = w / h
                new_w = int(target_h * aspect)
                
                # Resize
                resized = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
                resized_imgs.append(resized)
                
            if not resized_imgs:
                return images[0]

            # Concatenate horizontally
            canvas = np.hstack(resized_imgs)
            return canvas
            
        except Exception as e:
            logger.error(f"Concatenation failed: {e}")
            # Fallback to the first image if something goes critically wrong
            return images[0]
