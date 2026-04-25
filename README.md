# SLAC Crystal Segmentation Pipeline

Automated crystal detection and size assessment for fixed-target serial crystallography experiments at SLAC National Accelerator Laboratory (LCLS / MFX instrument).

Developed as part of the Multi-user High-throughput SSPX project under Dr. Elyse Schriber.

---

## Background

Fixed-target serial crystallography requires depositing protein crystals uniformly across chip arrays — each hole in the chip ideally holds one crystal. This pipeline was built to:

1. Isolate the chip region from raw microscope images
2. Detect individual crystals, including overlapping ones
3. Measure crystal size distribution to assess uniformity

Two parallel approaches were developed: a classical ImageJ macro pipeline and a deep learning U-Net segmentation model.

---

## Repo Structure

```
slac-crystal-pipeline/
├── preprocess.py            # Step 1: circular crop + resize to 1024x1024
├── imagej/
│   └── crystal_detect.ijm   # ImageJ/Fiji macro: CLAHE + Watershed + particle analysis
└── unet/
    ├── make_dataset.py      # Convert LabelMe annotations to image/mask pairs
    ├── unet_train.py        # Train U-Net (PyTorch)
    └── infer.py             # Run inference on new images
```

---

## Workflow

### Option A — ImageJ pipeline (no ML required)

Best for quick batch processing without a labeled dataset.

```
raw images → preprocess.py → imagej/crystal_detect.ijm
```

1. Run `preprocess.py` on your raw microscope images
2. Open `crystal_detect.ijm` in Fiji, select input/output folders when prompted

### Option B — U-Net pipeline

Better for overlapping crystals or when you have labeled training data.

```
raw images → preprocess.py → annotate in LabelMe → make_dataset.py → unet_train.py → infer.py
```

1. `python preprocess.py --input raw/ --output processed/`
2. Annotate processed images using [LabelMe](https://github.com/labelmeai/labelme) (polygon tool)
3. `python unet/make_dataset.py` (run from folder containing .json files)
4. `python unet/unet_train.py`
5. `python unet/infer.py --input processed/ --model unet_model.pth --output predictions/`

---

## Requirements

**Python**
```
pip install opencv-python numpy torch albumentations tqdm matplotlib
```

**ImageJ**
- [Fiji](https://fiji.sc/) (includes ImageJ2 + required plugins)

---

## Limitations

- U-Net was trained on a small dataset — predictions are noisy on unseen chip types
- Watershed separation works well for lightly overlapping crystals but struggles with dense clusters
- Inference threshold (0.2) was tuned for LaB6 crystals at 20x magnification — may need adjustment for other samples

---

## Sample Data

`c2w4_LaB6_20xmag.json` — example LabelMe annotation file for a LaB6 crystal chip image (20x magnification), included as a format reference for building your own training dataset.
