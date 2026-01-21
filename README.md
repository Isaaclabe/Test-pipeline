# Store Image Processing Pipeline

A Python pipeline for stitching and aligning store/shop images.

## Overview

This pipeline processes store image data to:
1. **Stitch** multiple face images into panoramas
2. **Warp/Align** signface (close-up sign) images to match their corresponding face images

## Data Structure

```
data-image/
├── store1/
│   ├── data1/                    # Year 1
│   │   ├── face1/                # Main store front images
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── image3.jpg
│   │   ├── signface1/            # Close-up sign images
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   ├── face2/
│   │   ├── signface2/
│   │   └── ...
│   └── data2/                    # Year 2
│       └── ...
├── store2/
└── ...
```

## Output Structure

```
store_process/
├── data1_process/
│   ├── face1_process/
│   │   └── store1_year1_face1_s.jpg    # Stitched face
│   ├── signface1_process/
│   │   ├── store1_year1_signface1_w_0.jpg   # Warped signface
│   │   └── store1_year1_signface1_w_1.jpg
│   └── debug/                          # Debug visualizations (if enabled)
│       ├── store1_year1_face1_stitch_step1.jpg
│       ├── store1_year1_signface1_matches_0.jpg
│       └── store1_year1_signface1_compare_0.jpg
└── data2_process/
    └── ...
```

## Naming Convention

- **Stitched face**: `store{i}_year{j}_face{k}_s.jpg` (s = stitched)
- **Normal face**: `store{i}_year{j}_face{k}_n.jpg` (n = normal, single image)
- **Warped signface**: `store{i}_year{j}_signface{k}_w_{l}.jpg` (w = warped, l = index)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main_pipeline.py -i ./data-image -o ./store_process
```

### With Debug Mode

```bash
python main_pipeline.py -i ./data-image -o ./store_process --debug
```

Debug mode saves:
- Feature matching visualizations
- Stitching step visualizations  
- Side-by-side comparison plots (reference, source, warped)

### All Options

```bash
python main_pipeline.py --help

Options:
  -i, --input     Input data directory (default: ./data-image)
  -o, --output    Output directory (default: ./store_process)
  -d, --debug     Enable debug mode (saves visualizations)
  --mask-h        Height mask percentage for bottom-right corner (default: 0.05)
  --mask-w        Width mask percentage for bottom-right corner (default: 0.15)
```

## Pipeline Components

### 1. Image Stitcher (`stitcher.py`)
- Combines multiple overlapping face images into a single panorama
- Uses SIFT features with translation-only constraint
- Handles linear scans (left-to-right) of store fronts

### 2. Image Aligner (`aligner.py`)
- Warps signface images to align with the face reference
- Multi-method cascade for robustness:
  1. SIFT (standard)
  2. SIFT Multi-scale (handles zoom differences)
  3. AKAZE (scale-invariant)
  4. ORB (fast binary features)
  5. Template Matching (last resort)
- Robust homography validation (detects flips, extreme distortions)

### 3. Utilities (`utils.py`)
- Image loading with optional corner masking (hides watermarks/timestamps)
- File/directory management

## Requirements

- Python 3.8+
- OpenCV (opencv-python)
- NumPy
- Matplotlib

## Troubleshooting

**Alignment fails:**
- The pipeline tries 5 different methods automatically
- Check debug output for feature matching visualization
- Ensure signface images have enough overlap with face images

**Stitching fails:**
- Verify images are in correct left-to-right order
- Check for sufficient overlap between consecutive images
- Debug mode shows per-step stitching results
