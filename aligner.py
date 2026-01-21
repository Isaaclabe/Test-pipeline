import cv2
import numpy as np
import torch
import logging
import math
from typing import Tuple, Optional

# Try imports
try:
    from kornia.feature import LoFTR
    HAS_LOFTR = True
except ImportError:
    HAS_LOFTR = False

try:
    # LightGlue requires kornia >= 0.7.0
    from kornia.feature import LightGlue, DiskFeatures
    HAS_LIGHTGLUE = True
except ImportError:
    HAS_LIGHTGLUE = False

logger = logging.getLogger(__name__)

class ImageAligner:
    """
    Advanced Aligner using Tiled Matching to handle small objects in large images.
    Supports: LightGlue (Best), LoFTR (Good), SIFT (Fallback).
    """

    def __init__(self, method: str = "loftr"):
        self.method = method.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loftr = None
        self.lightglue = None
        self.disk = None
        
        # SIFT (Always available fallback)
        self.sift = cv2.SIFT_create(nfeatures=10000)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # ORB (Final fallback)
        self.orb = cv2.ORB_create(nfeatures=10000)
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Init Deep Models
        if self.device == 'cuda':
            if HAS_LIGHTGLUE and (self.method == "lightglue" or self.method == "superglue"):
                try:
                    logger.info("Initializing LightGlue + Disk...")
                    self.lightglue = LightGlue(features="disk").to(self.device).eval()
                    self.disk = DiskFeatures(n_features=2048).to(self.device).eval()
                    self.method = "lightglue"
                except Exception as e:
                    logger.warning(f"LightGlue init failed: {e}")

            if HAS_LOFTR and (self.method == "loftr" or self.lightglue is None):
                if self.method == "loftr": # Only init if specifically asked or fallback
                    try:
                        logger.info("Initializing LoFTR (Outdoor)...")
                        self.loftr = LoFTR(pretrained="outdoor").to(self.device).eval()
                        self.method = "loftr"
                    except Exception as e:
                        logger.warning(f"LoFTR init failed: {e}")

    def align_image(self, target_img: np.ndarray, source_img: np.ndarray, debug_path: str = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Main entry point. Uses Tiled Matching if image aspect ratio suggests a panorama.
        """
        if target_img is None or source_img is None:
            return None, None

        # Heuristic: If target (Face) is much wider than source (Signface), use Tiling.
        h_t, w_t = target_img.shape[:2]
        
        # Tiling is useful for wide images where simple resize kills detail
        is_wide = w_t > 1.5 * h_t and w_t > 1500
        
        # Initialize vars
        warped, H = None, None

        if is_wide:
            warped, H = self._align_tiled(target_img, source_img, debug_path)
        else:
            warped, H = self._run_matcher(target_img, source_img, debug_path, offset=(0,0))

        # Cascade Fallback: LoFTR -> SIFT -> ORB
        if not self._is_valid_homography(H):
            logger.info(f"Primary method ({self.method}) failed, trying SIFT fallback...")
            warped, H = self._align_sift(target_img, source_img, debug_path, offset=(0,0))
            
            if not self._is_valid_homography(H):
                logger.info("SIFT fallback failed, trying ORB fallback...")
                warped, H = self._align_orb(target_img, source_img, debug_path, offset=(0,0))
                
                if not self._is_valid_homography(H):
                    logger.warning("All alignment methods failed (LoFTR -> SIFT -> ORB)")
                    return None, None

        return warped, H

    def _is_valid_homography(self, H):
        if H is None: return False
        det = np.linalg.det(H[:2, :2])
        if det < 0: return False # Flipped
        if abs(det) < 0.01 or abs(det) > 50: return False
        return True

    def _run_matcher(self, target, source, debug_path, offset=(0,0)):
        """Routes to the selected matching algorithm."""
        if self.method == "lightglue" and self.lightglue:
            return self._align_lightglue(target, source, debug_path, offset)
        elif self.method == "loftr" and self.loftr:
            return self._align_loftr(target, source, debug_path, offset)
        else:
            return self._align_sift(target, source, debug_path, offset)

    # --- TILING LOGIC ---
    def _align_tiled(self, target, source, debug_path):
        h, w = target.shape[:2]
        tile_size = int(h * 1.2) 
        stride = int(tile_size * 0.7) 
        
        tiles_checked = 0
        
        for x_start in range(0, w, stride):
            x_end = min(x_start + tile_size, w)
            if x_end - x_start < tile_size // 2: break 
            
            tile = target[:, x_start:x_end]
            
            # Use 'None' for debug_path inside tiles to avoid spamming plots for every tile
            # or pass modified path if strictly needed.
            warped, H_tile = self._run_matcher(tile, source, None, offset=(x_start, 0))
            
            if H_tile is not None and self._is_valid_homography(H_tile):
                # Translation matrix for the tile offset
                T = np.array([[1, 0, x_start], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                H_global = T @ H_tile
                
                try:
                    warped_global = cv2.warpPerspective(source, H_global, (w, h))
                    return warped_global, H_global
                except:
                    pass

            tiles_checked += 1
            if tiles_checked > 5: break 

        return None, None

    # --- MATCHERS ---

    def _smart_resize(self, img, max_dim=1024):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img, scale

    def _plot_matches(self, img1, kpts1, img2, kpts2, mask, save_path):
        """Debug helper to visualize matches."""
        if not save_path: return
        try:
            import matplotlib.pyplot as plt
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            canvas[:h1, :w1] = img1
            canvas[:h2, w1:w1+w2] = img2
            
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            
            # mask is shape (N,1) usually from RANSAC
            inliers = mask.ravel() == 1
            p1 = kpts1[inliers]
            p2 = kpts2[inliers]
            for (x1, y1), (x2, y2) in zip(p1, p2):
                ax.plot([x1, x2 + w1], [y1, y2], c='lime', alpha=0.6, linewidth=1)
                
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            pass

    def _align_lightglue(self, target, source, debug_path, offset):
        # LightGlue works best on ~1024
        t_small, scale_t = self._smart_resize(target, 1024)
        s_small, scale_s = self._smart_resize(source, 1024)

        t_ten = torch.from_numpy(cv2.cvtColor(t_small, cv2.COLOR_BGR2RGB)).permute(2,0,1)[None].float().to(self.device) / 255.0
        s_ten = torch.from_numpy(cv2.cvtColor(s_small, cv2.COLOR_BGR2RGB)).permute(2,0,1)[None].float().to(self.device) / 255.0

        with torch.no_grad():
            feat_t = self.disk(t_ten)
            feat_s = self.disk(s_ten)
            matches = self.lightglue({"image0": feat_s, "image1": feat_t}) # source->target

        kpts_s = feat_s[0].keypoints[matches['matches'][0][:, 0]].cpu().numpy()
        kpts_t = feat_t[0].keypoints[matches['matches'][0][:, 1]].cpu().numpy()

        if len(kpts_s) < 8: return None, None

        # Scale back
        kpts_s /= scale_s
        kpts_t /= scale_t

        H, mask = cv2.findHomography(kpts_s, kpts_t, cv2.USAC_MAGSAC, 5.0)
        
        if debug_path and H is not None:
            self._plot_matches(target, kpts_t, source, kpts_s, mask, debug_path)

        if H is None: return None, None
        
        h, w = target.shape[:2]
        warped = cv2.warpPerspective(source, H, (w, h))
        return warped, H

    def _align_loftr(self, target, source, debug_path, offset):
        t_small, scale_t = self._smart_resize(target, 840)
        s_small, scale_s = self._smart_resize(source, 840)

        img0 = torch.from_numpy(cv2.cvtColor(t_small, cv2.COLOR_BGR2GRAY)).float()[None, None].to(self.device) / 255.0
        img1 = torch.from_numpy(cv2.cvtColor(s_small, cv2.COLOR_BGR2GRAY)).float()[None, None].to(self.device) / 255.0

        with torch.no_grad():
            matches = self.loftr({"image0": img0, "image1": img1})

        conf = matches['confidence']
        valid = conf > 0.65
        mkpts0 = matches['keypoints0'][valid].cpu().numpy() # Target (image0)
        mkpts1 = matches['keypoints1'][valid].cpu().numpy() # Source (image1)

        if len(mkpts0) < 8: return None, None

        mkpts0 /= scale_t
        mkpts1 /= scale_s

        H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.USAC_MAGSAC, 5.0)

        if debug_path and H is not None:
            self._plot_matches(target, mkpts0, source, mkpts1, mask, debug_path)

        if H is None: return None, None
        
        h, w = target.shape[:2]
        warped = cv2.warpPerspective(source, H, (w, h))
        return warped, H

    def _align_sift(self, target, source, debug_path, offset):
        g_t = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        g_s = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(g_t, None)
        kp2, des2 = self.sift.detectAndCompute(g_s, None)

        if des1 is None or des2 is None: return None, None

        matches = self.flann.knnMatch(des1, des2, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance:
                good.append(m_n[0])

        if len(good) < 8: return None, None

        pts_dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_src, pts_dst, cv2.USAC_MAGSAC, 5.0)

        if debug_path and H is not None:
            # SIFT matches are indexed via keypoints
            # pts_dst correspond to target image
            p_dst = pts_dst.reshape(-1, 2)
            # pts_src correspond to source image
            p_src = pts_src.reshape(-1, 2)
            self._plot_matches(target, p_dst, source, p_src, mask, debug_path)

        if H is None: return None, None
        h, w = target.shape[:2]
        warped = cv2.warpPerspective(source, H, (w, h))
        return warped, H

    def _align_orb(self, target, source, debug_path, offset):
        """ORB-based alignment as final fallback."""
        g_t = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        g_s = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.orb.detectAndCompute(g_t, None)
        kp2, des2 = self.orb.detectAndCompute(g_s, None)

        if des1 is None or des2 is None: return None, None

        # Use knnMatch with k=2 for ratio test
        matches = self.bf_matcher.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test
        good = []
        for m_n in matches:
            if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance:
                good.append(m_n[0])

        if len(good) < 8: return None, None

        pts_dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts_src, pts_dst, cv2.USAC_MAGSAC, 5.0)

        if debug_path and H is not None:
            p_dst = pts_dst.reshape(-1, 2)
            p_src = pts_src.reshape(-1, 2)
            self._plot_matches(target, p_dst, source, p_src, mask, debug_path)

        if H is None: return None, None
        h, w = target.shape[:2]
        warped = cv2.warpPerspective(source, H, (w, h))
        return warped, H

    def warp_mask(self, mask: np.ndarray, h_matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if mask is None or h_matrix is None: return None
        try:
            return cv2.warpPerspective(mask, h_matrix, (target_shape[1], target_shape[0]), flags=cv2.INTER_NEAREST)
        except: return None
