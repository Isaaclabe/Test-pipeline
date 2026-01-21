import os
import cv2
import torch
import numpy as np
import threading
import requests
import shutil
from PIL import Image
from typing import List, Optional
from huggingface_hub import login

# SAM3 Imports
try:
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    HAS_SAM3 = True
except ImportError:
    print("WARNING: SAM3 not found. Please install via: pip install 'git+https://github.com/facebookresearch/sam3.git'")
    HAS_SAM3 = False


class SAM3SignDetector:
    """
    Real AI Model using Facebook's SAM3 for sign detection.
    """
    _instance = None
    _lock = threading.Lock()  # Global lock for GPU access

    def __init__(self, hf_token: str, text_prompt: str = "sign"):
        self.hf_token = hf_token
        self.text_prompt = text_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        
        self._initialize_model()

    def _download_bpe_file(self, dest_path: str):
        """Downloads the missing BPE file from the SAM3 repository."""
        url = "https://github.com/facebookresearch/sam3/raw/main/assets/bpe_simple_vocab_16e6.txt.gz"
        print(f"Downloading BPE file from {url} to {dest_path}...")
        
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download BPE file: {e}")
            raise RuntimeError("Could not acquire BPE file needed for SAM3 text prompts.")

    def _initialize_model(self):
        """Initializes the SAM3 model."""
        if not HAS_SAM3:
            print("SAM3 not available. Detector will not function.")
            return
            
        print(f"Initializing SAM3 on {self.device}...")
        
        # 1. AUTHENTICATION
        try:
            login(token=self.hf_token)
        except Exception as e:
            print("Auth warning: Check HF_TOKEN if model fails to load.")

        # 2. LOCATE/DOWNLOAD BPE ASSETS
        sam3_root = os.path.dirname(sam3.__file__)
        possible_paths = [
            os.path.join(sam3_root, "..", "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
            os.path.join(os.getcwd(), "assets", "bpe_simple_vocab_16e6.txt.gz")
        ]
        
        bpe_path = None
        for p in possible_paths:
            if os.path.exists(p):
                bpe_path = p
                break
        
        if bpe_path is None:
            print("BPE asset not found. Downloading...")
            local_asset_path = os.path.join(os.getcwd(), "assets", "bpe_simple_vocab_16e6.txt.gz")
            self._download_bpe_file(local_asset_path)
            bpe_path = local_asset_path

        # 3. BUILD MODEL
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        try:
            self.model = build_sam3_image_model(bpe_path=bpe_path)
            self.model.to(self.device)
            self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
            print("SAM3 Initialization Complete.")
        except Exception as e:
            print(f"Error building SAM3 model: {e}")
            raise

    def detect_segmentation(self, image: np.ndarray) -> List[np.ndarray]:
        """Runs SAM3 inference and returns list of binary masks."""
        if self.model is None:
            return []

        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        dtype = torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32

        with SAM3SignDetector._lock:
            with torch.autocast(self.device, dtype=dtype):
                try:
                    inference_state = self.processor.set_image(pil_image)
                    self.processor.reset_all_prompts(inference_state)
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state, 
                        prompt=self.text_prompt
                    )
                    
                    masks_tensor = None
                    if hasattr(inference_state, 'pred_masks'):
                        masks_tensor = inference_state.pred_masks
                    elif isinstance(inference_state, dict) and 'pred_masks' in inference_state:
                        masks_tensor = inference_state['pred_masks']
                    else:
                        if 'masks' in inference_state:
                            masks_tensor = inference_state['masks']

                    if masks_tensor is None:
                        return []

                    output_masks = []
                    if isinstance(masks_tensor, torch.Tensor):
                        masks_np = masks_tensor.detach().cpu().numpy()
                    else:
                        masks_np = masks_tensor

                    for m in masks_np:
                        if m.ndim == 3 and m.shape[0] == 1:
                            m = m.squeeze(0)
                        
                        if m.dtype != bool and m.max() > 1.0:
                            binary_mask = (m > 0).astype(np.uint8)
                        else:
                            binary_mask = m.astype(np.uint8)

                        h_orig, w_orig = image.shape[:2]
                        if binary_mask.shape != (h_orig, w_orig):
                            binary_mask = cv2.resize(binary_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

                        output_masks.append(binary_mask)
                        
                    return output_masks

                except Exception as e:
                    print(f"Error during SAM3 inference: {e}")
                    return []

    def is_fragmented(self, mask: np.ndarray) -> bool:
        """Check if mask has multiple disconnected components."""
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        return num_labels > 2
    
    @staticmethod
    def get_largest_mask(masks: List[np.ndarray]) -> Optional[np.ndarray]:
        """Returns the mask with the largest area from a list of masks."""
        if not masks:
            return None
        
        largest_mask = None
        largest_area = 0
        
        for mask in masks:
            area = np.sum(mask > 0)
            if area > largest_area:
                largest_area = area
                largest_mask = mask
                
        return largest_mask
