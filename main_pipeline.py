import os
import cv2
import glob
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Local imports
from utils import ImageUtils
from stitcher import ImageStitcher
from aligner import ImageAligner

# --- CONFIGURATION ---
ROOT_DIR = "/content/Test-pipeline/data-image"
OUTPUT_DIR = "/content/Test-pipeline/store_process"
DEBUG_MODE = True
ALIGNMENT_METHOD = "sift" # Options: "sift", "orb", "loftr" (if GPU)
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

if __name__ == "__main__":
    pipeline = Step1Pipeline(ROOT_DIR, OUTPUT_DIR, debug=DEBUG_MODE)
    pipeline.run()
