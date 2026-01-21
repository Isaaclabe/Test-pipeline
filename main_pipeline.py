"""
Main Pipeline for Image Stitching and Alignment

This pipeline processes store image data:
1. Stitches multiple face images into panoramas
2. Warps signface images to align with their corresponding face images

Directory Structure:
    data-image/
    ├── store1/
    │   ├── data1/
    │   │   ├── face1/
    │   │   │   ├── image1.jpg
    │   │   │   └── ...
    │   │   ├── signface1/
    │   │   │   ├── image1.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   └── data2/
    │       └── ...
    └── store2/
        └── ...

Output Naming:
    - Stitched face: store{i}_year{j}_face{k}_s.jpg
    - Normal face: store{i}_year{j}_face{k}_n.jpg  
    - Warped signface: store{i}_year{j}_signface{k}_w_{l}.jpg
"""

import os
import glob
import logging
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

from utils import load_images_from_folder, ensure_dir, save_image
from stitcher import ImageStitcher
from aligner import ImageAligner


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_ROOT_DIR = "./data-image"
DEFAULT_OUTPUT_DIR = "./store_process"

# Corner masking (to hide watermarks/timestamps)
MASK_H_PERCENT = 0.05  # 5% from bottom
MASK_W_PERCENT = 0.15  # 15% from right


# =============================================================================
# LOGGING SETUP
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """Colored log formatter for terminal output."""
    COLORS = {
        logging.DEBUG: "\x1b[38;20m",
        logging.INFO: "\x1b[36;20m",
        logging.WARNING: "\x1b[33;20m",
        logging.ERROR: "\x1b[31;20m",
        logging.CRITICAL: "\x1b[31;1m",
    }
    RESET = "\x1b[0m"
    FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        formatter = logging.Formatter(
            color + self.FORMAT + self.RESET,
            datefmt='%H:%M:%S'
        )
        return formatter.format(record)


def setup_logger():
    """Setup the pipeline logger."""
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter())
    logger.addHandler(handler)
    
    return logger


logger = setup_logger()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class StorePipeline:
    """
    Main pipeline for processing store images.
    """

    def __init__(
        self,
        root_dir: str,
        output_dir: str,
        debug: bool = False,
        mask_h: float = MASK_H_PERCENT,
        mask_w: float = MASK_W_PERCENT
    ):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.debug = debug
        self.mask_h = mask_h
        self.mask_w = mask_w
        
        # Initialize components
        self.stitcher = ImageStitcher(debug=debug)
        self.aligner = ImageAligner(debug=debug)
        
        # Create output directory
        ensure_dir(output_dir)

    def run(self):
        """Run the full pipeline."""
        logger.info("=" * 60)
        logger.info("STORE IMAGE PROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Root Directory: {self.root_dir}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"Debug Mode: {self.debug}")
        logger.info(f"Corner Mask: H={self.mask_h*100:.1f}%, W={self.mask_w*100:.1f}%")
        logger.info("=" * 60)

        if not os.path.exists(self.root_dir):
            logger.error(f"Root directory not found: {self.root_dir}")
            return

        # Find all stores
        store_paths = sorted(glob.glob(os.path.join(self.root_dir, "store*")))
        
        if not store_paths:
            logger.warning("No store directories found.")
            return

        for store_path in store_paths:
            self._process_store(store_path)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

    def _process_store(self, store_path: str):
        """Process a single store directory."""
        store_name = os.path.basename(store_path)
        store_id = store_name.replace("store", "")
        
        logger.info("-" * 50)
        logger.info(f"Processing: {store_name}")
        
        # Find all data (year) folders
        data_paths = sorted(glob.glob(os.path.join(store_path, "data*")))
        
        for data_path in data_paths:
            self._process_data(data_path, store_id)

    def _process_data(self, data_path: str, store_id: str):
        """Process a single data/year directory."""
        data_name = os.path.basename(data_path)
        year_id = data_name.replace("data", "")
        
        logger.info(f"  > Data Folder: {data_name} (Year {year_id})")
        
        # Create output directory for this data folder
        data_output_dir = ensure_dir(
            os.path.join(self.output_dir, f"{data_name}_process")
        )
        
        # Debug output directory
        debug_dir = None
        if self.debug:
            debug_dir = ensure_dir(os.path.join(data_output_dir, "debug"))
        
        # Find all face folders
        face_folders = sorted(glob.glob(os.path.join(data_path, "face*")))
        
        for face_folder in face_folders:
            face_name = os.path.basename(face_folder)
            
            # Skip signface folders
            if "signface" in face_name:
                continue
                
            face_id = face_name.replace("face", "")
            
            # Process this face and its corresponding signface
            self._process_face_pair(
                data_path=data_path,
                data_output_dir=data_output_dir,
                debug_dir=debug_dir,
                store_id=store_id,
                year_id=year_id,
                face_id=face_id
            )

    def _process_face_pair(
        self,
        data_path: str,
        data_output_dir: str,
        debug_dir: Optional[str],
        store_id: str,
        year_id: str,
        face_id: str
    ):
        """Process a face folder and its corresponding signface folder."""
        face_folder = os.path.join(data_path, f"face{face_id}")
        signface_folder = os.path.join(data_path, f"signface{face_id}")
        
        logger.info(f"    Processing face{face_id} / signface{face_id}")
        
        # Create output directories
        face_output_dir = ensure_dir(
            os.path.join(data_output_dir, f"face{face_id}_process")
        )
        signface_output_dir = ensure_dir(
            os.path.join(data_output_dir, f"signface{face_id}_process")
        )
        
        # =====================================================================
        # STEP 1: Load and stitch face images
        # =====================================================================
        face_images = load_images_from_folder(
            face_folder, self.mask_h, self.mask_w
        )
        
        if not face_images:
            logger.warning(f"      No images found in face{face_id}")
            return
        
        # Stitch or use single image
        if len(face_images) > 1:
            logger.info(f"      Stitching {len(face_images)} face images...")
            final_face, stitch_visuals = self.stitcher.stitch(face_images)
            
            if final_face is None:
                logger.warning(f"      Stitching failed, using first image")
                final_face = face_images[0]
                face_suffix = "n"  # Normal (fallback)
            else:
                face_suffix = "s"  # Stitched
                
            # Save stitch debug visualizations
            if self.debug and debug_dir and stitch_visuals:
                for i, vis in enumerate(stitch_visuals):
                    if vis is not None:
                        save_image(
                            vis,
                            os.path.join(
                                debug_dir,
                                f"store{store_id}_year{year_id}_face{face_id}_stitch_step{i+1}.jpg"
                            )
                        )
        else:
            final_face = face_images[0]
            face_suffix = "n"  # Normal (single image)
            logger.info(f"      Single face image (no stitching needed)")
        
        # Save final face image
        face_filename = f"store{store_id}_year{year_id}_face{face_id}_{face_suffix}.jpg"
        save_image(final_face, os.path.join(face_output_dir, face_filename))
        logger.info(f"      Saved: {face_filename}")
        
        # =====================================================================
        # STEP 2: Warp signface images
        # =====================================================================
        if not os.path.exists(signface_folder):
            logger.info(f"      No signface{face_id} folder found")
            return
        
        signface_images = load_images_from_folder(
            signface_folder, self.mask_h, self.mask_w
        )
        
        if not signface_images:
            logger.info(f"      No images in signface{face_id}")
            return
        
        logger.info(f"      Warping {len(signface_images)} signface images...")
        
        success_count = 0
        fail_count = 0
        
        for idx, signface_img in enumerate(signface_images):
            # Align/warp the signface to the face
            warped, H, match_vis = self.aligner.align(final_face, signface_img)
            
            # Output filename
            sf_filename = f"store{store_id}_year{year_id}_signface{face_id}_w_{idx}.jpg"
            
            if warped is not None:
                save_image(warped, os.path.join(signface_output_dir, sf_filename))
                success_count += 1
                
                # Save debug visualizations
                if self.debug and debug_dir:
                    # Save match visualization
                    if match_vis is not None:
                        save_image(
                            match_vis,
                            os.path.join(
                                debug_dir,
                                f"store{store_id}_year{year_id}_signface{face_id}_matches_{idx}.jpg"
                            )
                        )
                    
                    # Save comparison plot
                    self._save_comparison_plot(
                        final_face, signface_img, warped,
                        os.path.join(
                            debug_dir,
                            f"store{store_id}_year{year_id}_signface{face_id}_compare_{idx}.jpg"
                        ),
                        f"store{store_id}_year{year_id}_signface{face_id}_{idx}"
                    )
            else:
                fail_count += 1
                logger.warning(f"        Failed to warp: {sf_filename}")
                
                # Save failure debug
                if self.debug and debug_dir:
                    self._save_comparison_plot(
                        final_face, signface_img, None,
                        os.path.join(
                            debug_dir,
                            f"store{store_id}_year{year_id}_signface{face_id}_FAILED_{idx}.jpg"
                        ),
                        f"FAILED: store{store_id}_year{year_id}_signface{face_id}_{idx}"
                    )
        
        logger.info(f"      Results: {success_count} success, {fail_count} failed")

    def _save_comparison_plot(
        self,
        reference: np.ndarray,
        source: np.ndarray,
        warped: Optional[np.ndarray],
        save_path: str,
        title: str
    ):
        """Save a side-by-side comparison plot."""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(title, fontsize=14)
            
            # Reference (Face)
            axes[0].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Reference (Face)")
            axes[0].axis('off')
            
            # Source (Signface)
            axes[1].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Source (Signface)")
            axes[1].axis('off')
            
            # Warped Result
            if warped is not None:
                axes[2].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
                axes[2].set_title("Warped Result")
            else:
                axes[2].text(0.5, 0.5, "FAILED", ha='center', va='center',
                            fontsize=24, color='red', transform=axes[2].transAxes)
                axes[2].set_title("Warped Result (FAILED)")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.debug(f"Failed to save comparison plot: {e}")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Store Image Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "-i", "--input",
        default=DEFAULT_ROOT_DIR,
        help=f"Input data directory (default: {DEFAULT_ROOT_DIR})"
    )
    parser.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode (saves visualizations)"
    )
    parser.add_argument(
        "--mask-h",
        type=float,
        default=MASK_H_PERCENT,
        help=f"Height mask percentage (default: {MASK_H_PERCENT})"
    )
    parser.add_argument(
        "--mask-w",
        type=float,
        default=MASK_W_PERCENT,
        help=f"Width mask percentage (default: {MASK_W_PERCENT})"
    )
    
    args = parser.parse_args()
    
    pipeline = StorePipeline(
        root_dir=args.input,
        output_dir=args.output,
        debug=args.debug,
        mask_h=args.mask_h,
        mask_w=args.mask_w
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
