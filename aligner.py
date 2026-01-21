"""
Image Aligner Module - State of the Art Feature Matching

Implements multiple SoTA methods with automatic fallback:
1. LoFTR (Local Feature TRansformer) - Semi-dense matching
2. LightGlue + DISK - Learned sparse matching
3. LightGlue + SuperPoint - Alternative learned features
4. KeyNet + HardNet - Learned detector + descriptor
5. SIFT + Multi-scale - Classical robust baseline
6. AKAZE / ORB - Fast classical methods
7. Template Matching - Last resort
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger("Pipeline")

# Try to import deep learning libraries
try:
    import torch
    import kornia
    import kornia.feature as KF
    HAS_KORNIA = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Kornia available. Using device: {DEVICE}")
except ImportError:
    HAS_KORNIA = False
    DEVICE = None
    logger.warning("Kornia not available. Using classical methods only.")


class ImageAligner:
    """
    State-of-the-Art Image Aligner with deep learning and classical fallbacks.
    
    Methods cascade (in order of preference):
    1. LoFTR (Transformer-based, semi-dense)
    2. LightGlue + DISK (Learned sparse)
    3. LightGlue + SuperPoint (Alternative learned)
    4. KeyNet + HardNet (Learned detector/descriptor)
    5. SIFT Multi-scale (Robust classical)
    6. AKAZE (Scale-invariant binary)
    7. ORB (Fast binary)
    8. Template Matching (Last resort)
    """

    def __init__(self, debug: bool = False, method: str = "auto"):
        self.debug = debug
        self.device = DEVICE
        self.method = method.lower()
        
        # Deep learning models (lazy initialization)
        self._loftr = None
        self._lightglue_disk = None
        self._lightglue_sp = None
        self._disk = None
        self._superpoint = None
        self._keynet_hardnet = None
        
        # Classical feature detectors
        self.sift = cv2.SIFT_create(
            nfeatures=8000,
            contrastThreshold=0.02,
            edgeThreshold=15
        )
        self.akaze = cv2.AKAZE_create(threshold=0.0008)
        self.orb = cv2.ORB_create(nfeatures=5000)
        
        # Matchers
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=100)
        )
        self.bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # =========================================================================
    # LAZY MODEL INITIALIZATION
    # =========================================================================
    
    @property
    def loftr(self):
        """Lazy load LoFTR model."""
        if self._loftr is None and HAS_KORNIA:
            try:
                logger.info("      Loading LoFTR model...")
                self._loftr = KF.LoFTR(pretrained="outdoor").to(self.device).eval()
            except Exception as e:
                logger.warning(f"      Failed to load LoFTR: {e}")
        return self._loftr
    
    @property
    def disk(self):
        """Lazy load DISK model."""
        if self._disk is None and HAS_KORNIA:
            try:
                logger.info("      Loading DISK model...")
                self._disk = KF.DISK.from_pretrained("depth").to(self.device).eval()
            except Exception as e:
                logger.warning(f"      Failed to load DISK: {e}")
        return self._disk
    
    @property
    def lightglue_disk(self):
        """Lazy load LightGlue for DISK."""
        if self._lightglue_disk is None and HAS_KORNIA and self.disk is not None:
            try:
                logger.info("      Loading LightGlue (DISK)...")
                self._lightglue_disk = KF.LightGlue(features="disk").to(self.device).eval()
            except Exception as e:
                logger.warning(f"      Failed to load LightGlue+DISK: {e}")
        return self._lightglue_disk
    
    @property
    def superpoint(self):
        """Lazy load SuperPoint model."""
        if self._superpoint is None and HAS_KORNIA:
            try:
                logger.info("      Loading SuperPoint model...")
                self._superpoint = KF.SuperPoint(max_num_keypoints=2048).to(self.device).eval()
            except Exception as e:
                logger.warning(f"      Failed to load SuperPoint: {e}")
        return self._superpoint
    
    @property  
    def lightglue_superpoint(self):
        """Lazy load LightGlue for SuperPoint."""
        if self._lightglue_sp is None and HAS_KORNIA and self.superpoint is not None:
            try:
                logger.info("      Loading LightGlue (SuperPoint)...")
                self._lightglue_sp = KF.LightGlue(features="superpoint").to(self.device).eval()
            except Exception as e:
                logger.warning(f"      Failed to load LightGlue+SuperPoint: {e}")
        return self._lightglue_sp

    @property
    def keynet_hardnet(self):
        """Lazy load KeyNet + HardNet pipeline."""
        if self._keynet_hardnet is None and HAS_KORNIA:
            try:
                logger.info("      Loading KeyNet + HardNet...")
                self._keynet_hardnet = KF.KeyNetHardNet(num_features=5000).to(self.device).eval()
            except Exception as e:
                logger.warning(f"      Failed to load KeyNet+HardNet: {e}")
        return self._keynet_hardnet

    # =========================================================================
    # MAIN ALIGNMENT INTERFACE
    # =========================================================================

    def align(
        self,
        reference: np.ndarray,
        source: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Align source image to reference image using best available method.
        
        Args:
            reference: The target/reference image (face)
            source: The image to warp (signface)
            
        Returns:
            Tuple of (warped_image, homography_matrix, match_visualization)
        """
        if reference is None or source is None:
            return None, None, None

        # Define all available methods
        all_methods = {}
        
        if HAS_KORNIA:
            all_methods.update({
                "loftr": ("LoFTR", self._align_loftr),
                "lightglue_disk": ("LightGlue+DISK", self._align_lightglue_disk),
                "lightglue_superpoint": ("LightGlue+SuperPoint", self._align_lightglue_superpoint),
                "keynet_hardnet": ("KeyNet+HardNet", self._align_keynet_hardnet),
            })
        
        # Classical methods (always available)
        all_methods.update({
            "sift": ("SIFT", self._align_sift),
            "sift_multiscale": ("SIFT_MultiScale", self._align_sift_multiscale),
            "akaze": ("AKAZE", self._align_akaze),
            "orb": ("ORB", self._align_orb),
            "template": ("Template", self._align_template),
        })

        # Build method cascade based on selection
        methods = []
        if self.method == "auto":
            # Use all methods in cascade order
            if HAS_KORNIA:
                methods.extend([
                    all_methods["loftr"],
                    all_methods["lightglue_disk"],
                    all_methods["lightglue_superpoint"],
                    all_methods["keynet_hardnet"],
                ])
            methods.extend([
                all_methods["sift"],
                all_methods["sift_multiscale"],
                all_methods["akaze"],
                all_methods["orb"],
                all_methods["template"],
            ])
        elif self.method in all_methods:
            # Use only the selected method
            methods = [all_methods[self.method]]
        else:
            logger.warning(f"Unknown method '{self.method}', using auto cascade")
            return self._align_auto_cascade(reference, source, all_methods)

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

    def _align_auto_cascade(
        self,
        reference: np.ndarray,
        source: np.ndarray,
        all_methods: dict
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Fallback auto cascade when method is unknown."""
        if HAS_KORNIA:
            method_order = ["loftr", "lightglue_disk", "lightglue_superpoint", "keynet_hardnet"]
        else:
            method_order = []
        method_order.extend(["sift", "sift_multiscale", "akaze", "orb", "template"])
        
        for method_key in method_order:
            if method_key not in all_methods:
                continue
            method_name, method_func = all_methods[method_key]
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

    # =========================================================================
    # DEEP LEARNING METHODS
    # =========================================================================

    def _align_loftr(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """LoFTR: Local Feature TRansformer - Semi-dense matching."""
        if self.loftr is None:
            return None, None, None
        
        # Convert to grayscale tensors
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        
        # Try multiple resolutions
        resolutions = [840, 640, 512]
        confidence_thresholds = [0.5, 0.4, 0.3]
        
        best_result = (None, None, None)
        best_inliers = 0
        
        for max_dim in resolutions:
            for conf_thresh in confidence_thresholds:
                try:
                    ref_tensor, scale_ref = self._prepare_tensor_gray(ref_gray, max_dim=max_dim)
                    src_tensor, scale_src = self._prepare_tensor_gray(src_gray, max_dim=max_dim)
                    
                    with torch.no_grad():
                        input_dict = {"image0": ref_tensor, "image1": src_tensor}
                        result = self.loftr(input_dict)
                    
                    mkpts0 = result["keypoints0"].cpu().numpy()
                    mkpts1 = result["keypoints1"].cpu().numpy()
                    confidence = result["confidence"].cpu().numpy()
                    
                    mask = confidence > conf_thresh
                    pts_ref = mkpts0[mask] / scale_ref
                    pts_src = mkpts1[mask] / scale_src
                    
                    if len(pts_ref) < 10:
                        continue
                    
                    H, inlier_mask = cv2.findHomography(
                        pts_src.reshape(-1, 1, 2),
                        pts_ref.reshape(-1, 1, 2),
                        cv2.USAC_MAGSAC, 5.0, maxIters=5000
                    )
                    
                    if H is None or inlier_mask is None:
                        continue
                    
                    inliers = np.sum(inlier_mask)
                    if inliers > best_inliers and self._is_valid_homography(H, reference.shape, source.shape):
                        h, w = reference.shape[:2]
                        warped = cv2.warpPerspective(source, H, (w, h))
                        if self._check_warped_content(warped):
                            best_inliers = inliers
                            match_vis = self._draw_matches_pts(reference, pts_ref, source, pts_src, inlier_mask, "LoFTR")
                            best_result = (warped, H, match_vis)
                except:
                    continue
        
        return best_result

    def _align_lightglue_disk(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """LightGlue with DISK features."""
        if self.disk is None or self.lightglue_disk is None:
            return None, None, None
        
        # Prepare tensors (RGB)
        ref_tensor, scale_ref = self._prepare_tensor_rgb(reference, max_dim=1024)
        src_tensor, scale_src = self._prepare_tensor_rgb(source, max_dim=1024)
        
        with torch.no_grad():
            # Extract DISK features
            feats0 = self._disk(ref_tensor, n=2048, pad_if_not_divisible=True)
            feats1 = self._disk(src_tensor, n=2048, pad_if_not_divisible=True)
            
            kpts0, descs0 = feats0.keypoints, feats0.descriptors
            kpts1, descs1 = feats1.keypoints, feats1.descriptors
            
            # LightGlue matching
            input_dict = {
                "image0": {"keypoints": kpts0, "descriptors": descs0},
                "image1": {"keypoints": kpts1, "descriptors": descs1},
            }
            result = self._lightglue_disk(input_dict)
        
        matches = result["matches"][0].cpu().numpy()
        if len(matches) < 10:
            return None, None, None
        
        pts_ref = kpts0[0, matches[:, 0]].cpu().numpy() / scale_ref
        pts_src = kpts1[0, matches[:, 1]].cpu().numpy() / scale_src
        
        return self._finalize_alignment(reference, source, pts_ref, pts_src, "LightGlue+DISK")

    def _align_lightglue_superpoint(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """LightGlue with SuperPoint features."""
        if self.superpoint is None or self.lightglue_superpoint is None:
            return None, None, None
        
        # Prepare grayscale tensors
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        
        ref_tensor, scale_ref = self._prepare_tensor_gray(ref_gray, max_dim=1024)
        src_tensor, scale_src = self._prepare_tensor_gray(src_gray, max_dim=1024)
        
        with torch.no_grad():
            # Extract SuperPoint features
            feats0 = self._superpoint(ref_tensor)
            feats1 = self._superpoint(src_tensor)
            
            # LightGlue matching
            input_dict = {
                "image0": feats0,
                "image1": feats1,
            }
            result = self._lightglue_sp(input_dict)
        
        matches = result["matches"][0].cpu().numpy()
        if len(matches) < 10:
            return None, None, None
        
        kpts0 = feats0["keypoints"][0].cpu().numpy()
        kpts1 = feats1["keypoints"][0].cpu().numpy()
        
        pts_ref = kpts0[matches[:, 0]] / scale_ref
        pts_src = kpts1[matches[:, 1]] / scale_src
        
        return self._finalize_alignment(reference, source, pts_ref, pts_src, "LightGlue+SP")

    def _align_keynet_hardnet(
        self,
        reference: np.ndarray,
        source: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """KeyNet detector + HardNet descriptor."""
        if self.keynet_hardnet is None:
            return None, None, None
        
        # Prepare grayscale tensors
        ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        
        ref_tensor, scale_ref = self._prepare_tensor_gray(ref_gray, max_dim=1024)
        src_tensor, scale_src = self._prepare_tensor_gray(src_gray, max_dim=1024)
        
        with torch.no_grad():
            # Detect and describe
            lafs0, resps0, descs0 = self._keynet_hardnet(ref_tensor)
            lafs1, resps1, descs1 = self._keynet_hardnet(src_tensor)
            
            # Match with SMNN
            dists, matches = KF.match_smnn(descs0.squeeze(0), descs1.squeeze(0), th=0.95)
        
        if len(matches) < 10:
            return None, None, None
        
        # Get keypoint centers from LAFs
        kpts0 = KF.get_laf_center(lafs0).squeeze(0).cpu().numpy()
        kpts1 = KF.get_laf_center(lafs1).squeeze(0).cpu().numpy()
        
        pts_ref = kpts0[matches[:, 0]] / scale_ref
        pts_src = kpts1[matches[:, 1]] / scale_src
        
        return self._finalize_alignment(reference, source, pts_ref, pts_src, "KeyNet+HardNet")

    # =========================================================================
    # CLASSICAL METHODS
    # =========================================================================

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
        """AKAZE alignment."""
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
        """Template matching fallback."""
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
        
        match_vis = reference.copy()
        cv2.rectangle(match_vis, (x, y), (x + new_w, y + new_h), (0, 255, 0), 3)
        cv2.putText(match_vis, f"Template: {best_val:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return warped, H, match_vis

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _prepare_tensor_gray(
        self,
        img: np.ndarray,
        max_dim: int = 1024
    ) -> Tuple[torch.Tensor, float]:
        """Prepare grayscale image as tensor."""
        h, w = img.shape[:2]
        scale = 1.0
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Make dimensions divisible by 8 (for LoFTR)
        h, w = img.shape[:2]
        new_h, new_w = (h // 8) * 8, (w // 8) * 8
        img = img[:new_h, :new_w]
        
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
        
        return tensor, scale

    def _prepare_tensor_rgb(
        self,
        img: np.ndarray,
        max_dim: int = 1024
    ) -> Tuple[torch.Tensor, float]:
        """Prepare RGB image as tensor."""
        h, w = img.shape[:2]
        scale = 1.0
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(img_rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return tensor, scale

    def _finalize_alignment(
        self,
        reference: np.ndarray,
        source: np.ndarray,
        pts_ref: np.ndarray,
        pts_src: np.ndarray,
        method_name: str
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Finalize alignment from matched points."""
        if len(pts_ref) < 10:
            return None, None, None
        
        H, mask = cv2.findHomography(
            pts_src.reshape(-1, 1, 2),
            pts_ref.reshape(-1, 1, 2),
            cv2.USAC_MAGSAC, 5.0, maxIters=5000
        )
        
        if H is None:
            return None, None, None
        
        if not self._is_valid_homography(H, reference.shape, source.shape):
            return None, None, None
        
        h, w = reference.shape[:2]
        warped = cv2.warpPerspective(source, H, (w, h))
        
        if not self._check_warped_content(warped):
            return None, None, None
        
        match_vis = self._draw_matches_pts(reference, pts_ref, source, pts_src, mask, method_name)
        
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

    def _draw_matches_pts(
        self,
        img1: np.ndarray,
        pts1: np.ndarray,
        img2: np.ndarray,
        pts2: np.ndarray,
        mask: Optional[np.ndarray],
        method_name: str
    ) -> np.ndarray:
        """Draw point matches for visualization."""
        try:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            
            canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            canvas[:h1, :w1] = img1
            canvas[:h2, w1:] = img2
            
            n_draw = min(100, len(pts1))
            indices = np.random.choice(len(pts1), n_draw, replace=False) if len(pts1) > n_draw else range(len(pts1))
            
            for i in indices:
                if mask is not None and len(mask) > i and mask[i] == 0:
                    continue
                p1 = (int(pts1[i][0]), int(pts1[i][1]))
                p2 = (int(pts2[i][0]) + w1, int(pts2[i][1]))
                color = (0, 255, 0)
                cv2.line(canvas, p1, p2, color, 1)
                cv2.circle(canvas, p1, 3, color, -1)
                cv2.circle(canvas, p2, 3, color, -1)
            
            cv2.putText(canvas, f"{method_name}: {len(pts1)} matches",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return canvas
        except:
            return None
