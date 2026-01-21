import os
import cv2
import glob
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

# Local imports
from utils import ImageUtils
from stitcher import ImageStitcher
from aligner import ImageAligner
from sam3_detector import SAM3SignDetector

# --- CONFIGURATION ---
ROOT_DIR = "/content/Test-pipeline/data-image"
OUTPUT_DIR = "/content/Test-pipeline/store_process"
MASK_OUTPUT_DIR = "/content/Test-pipeline/mask_output"
DEBUG_MODE = True
ALIGNMENT_METHOD = "sift"  # Options: "sift", "orb", "loftr" (if GPU)
HF_TOKEN = ""  # Set your HuggingFace token here for SAM3
# ---------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Step1Pipeline:
    def __init__(self, root_dir, output_dir, debug=False):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.debug = debug
        self.stitcher = ImageStitcher()
        self.aligner = ImageAligner(method=ALIGNMENT_METHOD)
        
        # Clean/Create Output Directory
        # if os.path.exists(self.output_dir):
        #     shutil.rmtree(self.output_dir)
        # os.makedirs(self.output_dir, exist_ok=True)

    def plot_debug(self, face_img, signface_img, warped_signface, title, save_path):
        """Plots Face, Signface, and Result side-by-side."""
        if not self.debug: return
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(title, fontsize=12)
            
            # 1. Face
            axes[0].imshow(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Face (Reference)")
            axes[0].axis('off')
            
            # 2. Signface (Original)
            axes[1].imshow(cv2.cvtColor(signface_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Signface (Source)")
            axes[1].axis('off')
            
            # 3. Warped Result
            if warped_signface is not None:
                axes[2].imshow(cv2.cvtColor(warped_signface, cv2.COLOR_BGR2RGB))
                axes[2].set_title("Warped Signface")
            else:
                # Black image if failed
                axes[2].imshow(np.zeros_like(face_img))
                axes[2].set_title("Warp Failed")
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
        except Exception as e:
            logger.error(f"Failed to plot debug image: {e}")

    def process_store(self, store_path):
        store_name = os.path.basename(store_path)
        logger.info(f"Processing Store: {store_name}")

        # Find data folders (data1, data2...)
        data_paths = sorted(glob.glob(os.path.join(store_path, "data*")))
        
        for data_path in data_paths:
            data_name = os.path.basename(data_path) # e.g. "data1" -> used as "year" ID
            logger.info(f"  > Processing Data: {data_name}")

            # Create Base Process Directory: store_process/dataX_process
            data_process_dir = os.path.join(self.output_dir, f"{data_name}_process")
            os.makedirs(data_process_dir, exist_ok=True)

            # Debug folder for this data
            debug_dir = os.path.join(data_process_dir, "debug_plots")
            if self.debug: os.makedirs(debug_dir, exist_ok=True)

            # Iterate through faces (assuming 1 to 10 to be safe, or scan folders)
            # Scanning folders starting with 'face'
            face_folders = sorted(glob.glob(os.path.join(data_path, "face*")))
            
            for face_folder in face_folders:
                face_folder_name = os.path.basename(face_folder) # e.g. "face1"
                # Extract index '1' from 'face1'
                try:
                    face_idx = face_folder_name.replace("face", "")
                except:
                    continue

                # Corresponding signface folder
                signface_folder = os.path.join(data_path, f"signface{face_idx}")
                
                # --- Create Output Folders ---
                # store_process/dataX_process/faceY_process
                face_proc_dir = os.path.join(data_process_dir, f"{face_folder_name}_process")
                os.makedirs(face_proc_dir, exist_ok=True)
                
                # store_process/dataX_process/signfaceY_process
                signface_proc_dir = os.path.join(data_process_dir, f"signface{face_idx}_process")
                os.makedirs(signface_proc_dir, exist_ok=True)

                # --- 1. Process Face Images (Stitching) ---
                face_images = ImageUtils.load_images_from_folder(face_folder)
                if not face_images:
                    logger.warning(f"No images in {face_folder_name}")
                    continue

                main_face_img = None
                face_filename = ""

                if len(face_images) > 1:
                    # Stitch
                    main_face_img = self.stitcher.stitch(face_images)
                    if main_face_img is None:
                        # Fallback
                        main_face_img = face_images[0]
                        face_filename = f"{store_name}_{data_name}_{face_folder_name}_n.jpg" # Fallback to normal?
                    else:
                        # Suffix _s for stitched
                        face_filename = f"{store_name}_{data_name}_{face_folder_name}_s.jpg"
                else:
                    # Single image
                    main_face_img = face_images[0]
                    # Suffix _n for normal
                    face_filename = f"{store_name}_{data_name}_{face_folder_name}_n.jpg"

                # Save Face Image
                face_save_path = os.path.join(face_proc_dir, face_filename)
                cv2.imwrite(face_save_path, main_face_img)
                
                # --- 2. Process Signface Images (Warping) ---
                if os.path.exists(signface_folder):
                    signface_images = ImageUtils.load_images_from_folder(signface_folder)
                    
                    for l, sf_img in enumerate(signface_images):
                        # Warp signface to face perspective
                        warped_sf, _ = self.aligner.align_image(main_face_img, sf_img)
                        
                        # Naming: store_year_signface_w_id
                        sf_filename = f"{store_name}_{data_name}_signface{face_idx}_w_{l}.jpg"
                        sf_save_path = os.path.join(signface_proc_dir, sf_filename)
                        
                        if warped_sf is not None:
                            cv2.imwrite(sf_save_path, warped_sf)
                        else:
                            logger.warning(f"Failed to warp {sf_filename}")

                        # Debug Plot
                        if self.debug:
                            debug_name = f"debug_{store_name}_{data_name}_face{face_idx}_sf{l}.jpg"
                            self.plot_debug(
                                main_face_img, 
                                sf_img, 
                                warped_sf, 
                                f"Warping: {sf_filename}", 
                                os.path.join(debug_dir, debug_name)
                            )

    def run(self):
        logger.info("Starting Pipeline Step 1...")
        
        if not os.path.exists(self.root_dir):
            logger.error("Root directory not found.")
            return

        store_paths = sorted(glob.glob(os.path.join(self.root_dir, "store*")))
        if not store_paths:
            logger.warning("No stores found.")
            return

        for store in store_paths:
            self.process_store(store)
            
        logger.info("Pipeline Step 1 Completed.")


class Step2Pipeline:
    """
    Step 2: SAM3 Segmentation and Mask Matching
    
    - Runs SAM3 on face images (keeps all masks temporarily)
    - Runs SAM3 on warped signface images (keeps only largest mask)
    - Compares signface masks with face masks per store/data/face
    - Keeps only overlapping masks
    - If signface mask has no overlap, uses it directly as face mask
    """
    
    def __init__(self, input_dir: str, output_dir: str, hf_token: str, debug: bool = False):
        self.input_dir = input_dir  # store_process directory from Step 1
        self.output_dir = output_dir  # mask_output directory
        self.debug = debug
        self.detector = SAM3SignDetector(hf_token=hf_token, text_prompt="sign")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IoU (Intersection over Union) between two masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        
        # Ensure same dimensions
        if mask1.shape != mask2.shape:
            mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        m1 = mask1 > 0
        m2 = mask2 > 0
        
        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        
        if union == 0:
            return 0.0
        return intersection / union
    
    def masks_overlap(self, mask1: np.ndarray, mask2: np.ndarray, threshold: float = 0.1) -> bool:
        """Check if two masks have significant overlap."""
        return self.calculate_overlap(mask1, mask2) > threshold
    
    def process_data_folder(self, data_process_dir: str, store_name: str, data_name: str):
        """Process a single data folder (e.g., data1_process)."""
        logger.info(f"    Processing {data_name}...")
        
        # Find all face_process folders
        face_proc_folders = sorted(glob.glob(os.path.join(data_process_dir, "face*_process")))
        
        for face_proc_folder in face_proc_folders:
            face_folder_name = os.path.basename(face_proc_folder).replace("_process", "")
            face_idx = face_folder_name.replace("face", "")
            
            logger.info(f"      Processing {face_folder_name}...")
            
            # Corresponding signface_process folder
            signface_proc_folder = os.path.join(data_process_dir, f"signface{face_idx}_process")
            
            # --- 1. Load and segment face images ---
            face_images = glob.glob(os.path.join(face_proc_folder, "*.jpg"))
            face_masks_all = []  # Store all face masks temporarily
            
            for face_img_path in face_images:
                face_img = cv2.imread(face_img_path)
                if face_img is None:
                    continue
                    
                masks = self.detector.detect_segmentation(face_img)
                logger.info(f"        Face image: found {len(masks)} masks")
                face_masks_all.extend(masks)
            
            # --- 2. Load and segment signface images (keep only largest) ---
            signface_largest_masks = []  # Store largest mask from each signface
            
            if os.path.exists(signface_proc_folder):
                signface_images = glob.glob(os.path.join(signface_proc_folder, "*.jpg"))
                
                for sf_img_path in signface_images:
                    sf_img = cv2.imread(sf_img_path)
                    if sf_img is None:
                        continue
                    
                    masks = self.detector.detect_segmentation(sf_img)
                    logger.info(f"        Signface image: found {len(masks)} masks")
                    
                    # Keep only the largest mask
                    largest_mask = SAM3SignDetector.get_largest_mask(masks)
                    if largest_mask is not None:
                        signface_largest_masks.append(largest_mask)
            
            # --- 3. Compare and select overlapping masks ---
            selected_masks = []
            
            for sf_mask in signface_largest_masks:
                has_overlap = False
                
                for face_mask in face_masks_all:
                    if self.masks_overlap(sf_mask, face_mask):
                        # Found overlap - add the face mask to selected
                        selected_masks.append(face_mask)
                        has_overlap = True
                
                # If signface mask has no overlap with any face mask, use it directly
                if not has_overlap:
                    logger.info(f"        Signface mask has no overlap, using as face mask")
                    selected_masks.append(sf_mask)
            
            # Also check if any face masks weren't matched but should be included
            # (optional: you might want to keep all face masks that overlap with ANY signface mask)
            
            # --- 4. Remove duplicate masks ---
            unique_masks = []
            for mask in selected_masks:
                is_duplicate = False
                for existing in unique_masks:
                    if self.calculate_overlap(mask, existing) > 0.9:  # 90% overlap = duplicate
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique_masks.append(mask)
            
            # --- 5. Save selected masks ---
            # Extract year from data_name (e.g., "data1" -> "1")
            year_idx = data_name.replace("data", "")
            store_idx = store_name.replace("store", "")
            
            for mask_id, mask in enumerate(unique_masks):
                # Naming: store<i>_year<j>_face<k>_mask_<l>.png
                mask_filename = f"store{store_idx}_year{year_idx}_face{face_idx}_mask_{mask_id}.png"
                mask_save_path = os.path.join(self.output_dir, mask_filename)
                
                # Convert mask to 0-255 for saving
                mask_to_save = (mask * 255).astype(np.uint8)
                cv2.imwrite(mask_save_path, mask_to_save)
                logger.info(f"        Saved: {mask_filename}")
            
            logger.info(f"      {face_folder_name}: saved {len(unique_masks)} masks")
    
    def run(self):
        logger.info("Starting Pipeline Step 2 (SAM3 Segmentation)...")
        
        if not os.path.exists(self.input_dir):
            logger.error(f"Input directory not found: {self.input_dir}")
            return
        
        # Find all data_process folders in the input directory
        # Structure: store_process/data1_process, data2_process, etc.
        data_process_folders = sorted(glob.glob(os.path.join(self.input_dir, "data*_process")))
        
        if not data_process_folders:
            logger.warning("No data_process folders found.")
            return
        
        # We need to determine store name from the processed images
        # For now, we'll process each data folder and extract store info from filenames
        for data_proc_folder in data_process_folders:
            data_name = os.path.basename(data_proc_folder).replace("_process", "")
            
            # Get store name from image filenames in face folders
            face_folders = glob.glob(os.path.join(data_proc_folder, "face*_process"))
            if face_folders:
                sample_images = glob.glob(os.path.join(face_folders[0], "*.jpg"))
                if sample_images:
                    # Extract store name from filename like "store1_data1_face1_s.jpg"
                    sample_filename = os.path.basename(sample_images[0])
                    store_name = sample_filename.split("_")[0]
                else:
                    store_name = "store1"  # Default fallback
            else:
                store_name = "store1"
            
            self.process_data_folder(data_proc_folder, store_name, data_name)
        
        logger.info("Pipeline Step 2 Completed.")


if __name__ == "__main__":
    # Run Step 1
    pipeline1 = Step1Pipeline(ROOT_DIR, OUTPUT_DIR, debug=DEBUG_MODE)
    pipeline1.run()
    
    # Run Step 2
    pipeline2 = Step2Pipeline(OUTPUT_DIR, MASK_OUTPUT_DIR, hf_token=HF_TOKEN, debug=DEBUG_MODE)
    pipeline2.run()
