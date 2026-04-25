"""
preprocess.py
-------------
Prepares raw microscope chip images for crystal detection.

Pipeline (run in order):
  1. crop_chip()   - isolates the circular chip region, blacks out background
  2. resize_imgs() - center-crops to square and resizes to 1024x1024

Usage:
  python preprocess.py --input raw/ --output processed/
"""

import os
import argparse
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

TARGET_SIZE = 1024  # px — must match U-Net input size


def crop_chip(input_dir, output_dir):
    """
    Isolates the circular chip area from each image.
    Uses a circular mask at 90% of the shorter image dimension,
    centered on the image. Background is set to black.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".png", ".tif"))]

    if not files:
        print(f"No images found in {input_dir}")
        return

    print(f"Step 1/2 — Circular crop: {len(files)} images")
    for filename in tqdm(files):
        img_path = os.path.join(input_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            print(f"  Skipped (unreadable): {filename}")
            continue

        h, w = image.shape[:2]
        r = int(min(h, w) * 0.9) // 2
        cx, cy = w // 2, h // 2

        # Circular mask
        Y, X = np.ogrid[:h, :w]
        mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

        masked = np.zeros_like(image)
        masked[mask] = image[mask]

        # Crop to bounding square around circle
        crop = masked[cy - r: cy + r, cx - r: cx + r]
        cv2.imwrite(os.path.join(output_dir, filename), crop)

    print("  Done.\n")


def resize_imgs(input_dir, output_dir):
    """
    Center-crops each image to a square (based on height),
    then resizes to TARGET_SIZE x TARGET_SIZE.
    Overwrites files in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob(os.path.join(input_dir, "*.jpg")) +
                   glob(os.path.join(input_dir, "*.png")) +
                   glob(os.path.join(input_dir, "*.tif")))

    if not paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Step 2/2 — Resize to {TARGET_SIZE}x{TARGET_SIZE}: {len(paths)} images")
    for path in tqdm(paths):
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        cx = w // 2
        half = h // 2
        cropped = img[0:h, cx - half: cx + half]
        resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        out_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(out_path, resized)

    print("  Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess chip images for crystal detection.")
    parser.add_argument("--input",  required=True, help="Folder of raw microscope images")
    parser.add_argument("--output", required=True, help="Folder to save processed images")
    args = parser.parse_args()

    tmp_dir = args.output + "_tmp"
    crop_chip(args.input, tmp_dir)
    resize_imgs(tmp_dir, args.output)

    # Clean up temp folder
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Preprocessed images saved to: {args.output}")
