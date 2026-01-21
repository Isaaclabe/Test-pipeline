"""
Image Stitcher Module
Handles stitching multiple images into a single panorama.
"""

import cv2
import numpy as np
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("Pipeline")


class ImageStitcher:
    """
    Image stitcher for combining multiple overlapping images.
    Uses translation-only constraint for linear scans.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )

    def stitch(
        self,
        images: List[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], List[Optional[np.ndarray]]]:
        """
        Stitch a list of ordered images into a panorama.
        
        Args:
            images: List of images to stitch (in order)
            
        Returns:
            Tuple of (stitched_result, list_of_step_visualizations)
        """
        if not images:
            return None, []

        # Resize for stability
        processed = self._preprocess_images(images, max_h=1500)
        n = len(processed)

        if n == 1:
            return processed[0], []

        logger.info(f"      Stitching {n} images...")

        # Iterative stitching
        panorama = processed[0]
        step_visuals = []

        for i in range(1, n):
            next_img = processed[i]
            result, step_vis = self._stitch_pair(panorama, next_img, i)

            if result is not None:
                panorama = result
                if self.debug:
                    logger.info(f"        Step {i}: Image {i+1} joined successfully")
            else:
                # Fallback: simple concatenation
                panorama = self._concat_horizontal(panorama, next_img)
                if self.debug:
                    logger.warning(f"        Step {i}: Fallback to concatenation")

            if step_vis is not None:
                step_visuals.append(step_vis)

        return panorama, step_visuals

    def _stitch_pair(
        self,
        pano: np.ndarray,
        curr: np.ndarray,
        step: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Stitch current image to the right of the panorama.
        """
        h_pano, w_pano = pano.shape[:2]
        h_curr, w_curr = curr.shape[:2]

        # Define ROI (overlapping regions)
        overlap_ratio = 0.5
        roi_pano_x = int(w_pano * (1.0 - overlap_ratio))
        roi_pano = pano[:, roi_pano_x:]
        roi_curr_w = int(w_curr * overlap_ratio)
        roi_curr = curr[:, :roi_curr_w]

        # Feature matching
        kp1, des1 = self.sift.detectAndCompute(roi_pano, None)
        kp2, des2 = self.sift.detectAndCompute(roi_curr, None)

        if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
            return None, None

        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2 and m_n[0].distance < 0.70 * m_n[1].distance:
                good.append(m_n[0])

        if len(good) < 5:
            return None, None

        # Compute shift
        pts_curr = np.float32([kp2[m.trainIdx].pt for m in good])
        pts_pano = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_pano[:, 0] += roi_pano_x  # Adjust to global coords

        deltas = pts_pano - pts_curr
        dx = np.median(deltas[:, 0])
        dy = np.median(deltas[:, 1])

        idx, idy = int(round(dx)), int(round(dy))

        # Sanity checks
        if idx < -w_curr * 0.1:
            return None, None
        if abs(idy) > h_pano * 0.3:
            return None, None

        # Create canvas
        min_x, min_y = min(0, idx), min(0, idy)
        max_x = max(w_pano, idx + w_curr)
        max_y = max(h_pano, idy + h_curr)

        final_w = max_x - min_x
        final_h = max_y - min_y

        if final_w > 25000:
            return None, None

        output = np.zeros((final_h, final_w, 3), dtype=np.uint8)
        off_x, off_y = -min_x, -min_y

        # Paste: current image first (background), then panorama (foreground)
        output[off_y + idy:off_y + idy + h_curr, off_x + idx:off_x + idx + w_curr] = curr
        output[off_y:off_y + h_pano, off_x:off_x + w_pano] = pano

        # Create visualization
        step_vis = None
        if self.debug:
            step_vis = cv2.drawMatches(
                roi_pano, kp1, roi_curr, kp2, good[:50],
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

        return output, step_vis

    def _preprocess_images(
        self,
        images: List[np.ndarray],
        max_h: int = 1500
    ) -> List[np.ndarray]:
        """Resize images to consistent height."""
        processed = []
        for img in images:
            if img is None:
                continue
            h, w = img.shape[:2]
            if h > max_h:
                scale = max_h / h
                new_w = int(w * scale)
                img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
            processed.append(img)
        return processed

    def _concat_horizontal(
        self,
        img1: np.ndarray,
        img2: np.ndarray
    ) -> np.ndarray:
        """Simple horizontal concatenation."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_h = max(h1, h2)

        output = np.zeros((max_h, w1 + w2, 3), dtype=np.uint8)
        output[:h1, :w1] = img1
        output[:h2, w1:w1 + w2] = img2

        return output
