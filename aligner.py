"""
Image Aligner Module
Handles alignment/warping of signface images to match face images.
Uses multiple methods with fallback cascade for robustness.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List, Dict

logger = logging.getLogger("Pipeline")


class ImageAligner:
    """
    Robust Image Aligner with multi-method fallback.
    
    Methods cascade (in order):
    1. SIFT with multi-scale
    2. AKAZE
    3. ORB
    4. Template matching (last resort)
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        
        # Initialize feature detectors
        self.sift = cv2.SIFT_create(
            nfeatures=8000,
            contrastThreshold=0.02,
            edgeThreshold=15
        )
        self.akaze = cv2.AKAZE_create(threshold=0.001)
        self.orb = cv2.ORB_create(nfeatures=5000)
        
        # Matchers
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=100)
        )
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def align(
        self,
        reference: np.ndarray,
        source: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Align source image to reference image.
        
        Args:
            reference: The target/reference image (face)
            source: The image to warp (signface)
            
        Returns:
            Tuple of (warped_image, homography_matrix, match_visualization)
        """
        if reference is None or source is None:
            return None, None, None

        methods = [
            ("SIFT", self._align_sift),
            ("SIFT_MultiScale", self._align_sift_multiscale),
            ("AKAZE", self._align_akaze),
            ("ORB", self._align_orb),
            ("Template", self._align_template),
        ]

        for method_name, method_func in methods:
            try:
                warped, H, match_vis = method_func(reference, source)
                
                if H is not None and self._is_valid_homography(H, reference.shape, source.shape):
                    if self._check_warped_content(warped):
                        if self.debug:
                            logger.info(f"      Alignment succeeded with {method_name}")
                        return warped, H, match_vis
            except Exception as e:
                logger.debug(f"      {method_name} failed: {e}")
                continue

        return None, None, None

    def _align_sift(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Standard SIFT alignment."""
        kp1, des1 = self.sift.detectAndCompute(reference, None)
        kp2, des2 = self.sift.detectAndCompute(source, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, None

        matches = self.flann.knnMatch(des1, des2, k=2)
        good = self._ratio_test(matches, 0.7)

        if len(good) < 10:
            return None, None, None

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_src = np.float32([kp2[m.trainIdx].pt for m in good])

        match_vis = self._draw_matches(reference, kp1, source, kp2, good, "SIFT")
        
        return self._compute_homography(reference, source, pts_src, pts_ref, match_vis)

    def _align_sift_multiscale(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Multi-scale SIFT for handling zoom differences."""
        best_result = (None, None, None)
        best_inliers = 0
        
        scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
        
        for scale in scales:
            h, w = source.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            
            if new_w < 50 or new_h < 50 or new_w > 4000 or new_h > 4000:
                continue
                
            if scale != 1.0:
                interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                scaled_source = cv2.resize(source, (new_w, new_h), interpolation=interp)
            else:
                scaled_source = source

            kp1, des1 = self.sift.detectAndCompute(reference, None)
            kp2, des2 = self.sift.detectAndCompute(scaled_source, None)

            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                continue

            matches = self.flann.knnMatch(des1, des2, k=2)
            good = self._ratio_test(matches, 0.7)

            if len(good) < 10:
                continue

            pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
            pts_src = np.float32([kp2[m.trainIdx].pt for m in good])
            
            # Scale points back to original source coordinates
            if scale != 1.0:
                pts_src = pts_src / scale

            H, mask = cv2.findHomography(
                pts_src.reshape(-1, 1, 2),
                pts_ref.reshape(-1, 1, 2),
                cv2.USAC_MAGSAC, 5.0, maxIters=5000
            )

            if H is None or mask is None:
                continue

            inliers = np.sum(mask)
            if inliers > best_inliers and self._is_valid_homography(H, reference.shape, source.shape):
                h_ref, w_ref = reference.shape[:2]
                warped = cv2.warpPerspective(source, H, (w_ref, h_ref))
                if self._check_warped_content(warped):
                    best_inliers = inliers
                    match_vis = self._draw_matches(reference, kp1, scaled_source, kp2, good, f"SIFT@{scale}x")
                    best_result = (warped, H, match_vis)

        return best_result

    def _align_akaze(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """AKAZE alignment - good for scale invariance."""
        kp1, des1 = self.akaze.detectAndCompute(reference, None)
        kp2, des2 = self.akaze.detectAndCompute(source, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, None

        matches = self.bf_hamming.knnMatch(des1, des2, k=2)
        good = self._ratio_test(matches, 0.75)

        if len(good) < 10:
            return None, None, None

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_src = np.float32([kp2[m.trainIdx].pt for m in good])

        match_vis = self._draw_matches(reference, kp1, source, kp2, good, "AKAZE")
        
        return self._compute_homography(reference, source, pts_src, pts_ref, match_vis)

    def _align_orb(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """ORB alignment."""
        kp1, des1 = self.orb.detectAndCompute(reference, None)
        kp2, des2 = self.orb.detectAndCompute(source, None)

        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None, None, None

        matches = self.bf_hamming.knnMatch(des1, des2, k=2)
        good = self._ratio_test(matches, 0.75)

        if len(good) < 10:
            return None, None, None

        pts_ref = np.float32([kp1[m.queryIdx].pt for m in good])
        pts_src = np.float32([kp2[m.trainIdx].pt for m in good])

        match_vis = self._draw_matches(reference, kp1, source, kp2, good, "ORB")
        
        return self._compute_homography(reference, source, pts_src, pts_ref, match_vis)

    def _align_template(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Template matching fallback for difficult cases."""
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
        src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if len(source.shape) == 3 else source

        h_ref, w_ref = ref_gray.shape[:2]
        h_src, w_src = src_gray.shape[:2]

        best_val = -1
        best_scale = 1.0
        best_loc = None

        scales = np.linspace(0.1, 2.0, 30)

        for scale in scales:
            new_w = int(w_src * scale)
            new_h = int(h_src * scale)

            if new_w >= w_ref or new_h >= h_ref or new_w < 20 or new_h < 20:
                continue

            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            scaled = cv2.resize(src_gray, (new_w, new_h), interpolation=interp)

            result = cv2.matchTemplate(ref_gray, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                best_scale = scale
                best_loc = max_loc

        if best_val < 0.4:
            return None, None, None

        # Compute homography from match location
        new_w = int(w_src * best_scale)
        new_h = int(h_src * best_scale)
        x, y = best_loc

        src_pts = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]])
        dst_pts = np.float32([
            [x, y],
            [x + new_w, y],
            [x + new_w, y + new_h],
            [x, y + new_h]
        ])

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)

        if not self._is_valid_homography(H, reference.shape, source.shape):
            return None, None, None

        warped = cv2.warpPerspective(source, H, (w_ref, h_ref))
        
        # Create visualization for template matching
        match_vis = reference.copy()
        cv2.rectangle(match_vis, (x, y), (x + new_w, y + new_h), (0, 255, 0), 3)
        cv2.putText(match_vis, f"Template Match: {best_val:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return warped, H, match_vis

    def _ratio_test(self, matches: List, ratio: float = 0.75) -> List:
        """Apply Lowe's ratio test."""
        good = []
        for m_n in matches:
            if len(m_n) == 2 and m_n[0].distance < ratio * m_n[1].distance:
                good.append(m_n[0])
        return good

    def _compute_homography(
        self,
        reference: np.ndarray,
        source: np.ndarray,
        pts_src: np.ndarray,
        pts_ref: np.ndarray,
        match_vis: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute homography and warp image."""
        if len(pts_src) < 4:
            return None, None, None

        H, mask = cv2.findHomography(
            pts_src.reshape(-1, 1, 2),
            pts_ref.reshape(-1, 1, 2),
            cv2.USAC_MAGSAC, 5.0, maxIters=5000
        )

        if H is None:
            return None, None, None

        # Check inlier ratio
        if mask is not None:
            inlier_ratio = np.sum(mask) / len(mask)
            if inlier_ratio < 0.15 or np.sum(mask) < 8:
                return None, None, None

        if not self._is_valid_homography(H, reference.shape, source.shape):
            return None, None, None

        h, w = reference.shape[:2]
        warped = cv2.warpPerspective(source, H, (w, h))

        if not self._check_warped_content(warped):
            return None, None, None

        return warped, H, match_vis

    def _is_valid_homography(
        self,
        H: np.ndarray,
        ref_shape: Tuple,
        src_shape: Tuple
    ) -> bool:
        """Validate homography matrix."""
        if H is None:
            return False

        # Check determinant
        det = np.linalg.det(H[:2, :2])
        if det < 0.01 or det > 100.0:
            return False

        # Check condition number
        try:
            cond = np.linalg.cond(H)
            if cond > 1e6:
                return False
        except:
            return False

        # Check for flipping
        h_src, w_src = src_shape[:2]
        h_ref, w_ref = ref_shape[:2]

        src_corners = np.float32([
            [0, 0], [w_src, 0], [w_src, h_src], [0, h_src]
        ]).reshape(-1, 1, 2)

        try:
            dst_corners = cv2.perspectiveTransform(src_corners, H).reshape(-1, 2)
        except:
            return False

        # Check orientation (detect flips)
        def cross_sign(p1, p2, p3):
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        orig_sign = cross_sign([0, 0], [w_src, 0], [w_src, h_src])
        trans_sign = cross_sign(dst_corners[0], dst_corners[1], dst_corners[2])

        if orig_sign * trans_sign < 0:
            return False

        # Check area ratio
        def polygon_area(corners):
            n = len(corners)
            area = 0
            for i in range(n):
                j = (i + 1) % n
                area += corners[i][0] * corners[j][1]
                area -= corners[j][0] * corners[i][1]
            return abs(area) / 2

        trans_area = polygon_area(dst_corners)
        orig_area = w_src * h_src
        area_ratio = trans_area / orig_area

        if area_ratio < 0.005 or area_ratio > 100:
            return False

        return True

    def _check_warped_content(self, warped: np.ndarray) -> bool:
        """Check if warped image has enough visible content."""
        if warped is None:
            return False

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        non_black = np.count_nonzero(gray > 10)
        total = gray.shape[0] * gray.shape[1]

        return non_black / total > 0.05

    def _draw_matches(
        self,
        img1: np.ndarray,
        kp1: List,
        img2: np.ndarray,
        kp2: List,
        matches: List,
        method_name: str
    ) -> np.ndarray:
        """Draw feature matches for visualization."""
        try:
            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2, matches[:100],
                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            cv2.putText(match_img, f"{method_name}: {len(matches)} matches",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return match_img
        except:
            return None
