"""
make_dataset.py
---------------
Converts LabelMe polygon annotations (.json) into image/mask pairs
for U-Net training.

Expects:
  - .json files from LabelMe in the current directory
  - Corresponding image files referenced inside each .json

Output:
  dataset/
    images/   <- PNG copies of source images
    masks/    <- Binary masks (white = crystal, black = background)

Usage:
  1. Annotate chip images in LabelMe (polygon tool, one shape per crystal)
  2. Save .json files alongside images
  3. Run: python make_dataset.py
"""

import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks",  exist_ok=True)

json_paths = sorted(glob("./*.json"))
print(f"Found {len(json_paths)} annotation files")

skipped = 0
for json_path in tqdm(json_paths):
    with open(json_path, "r") as f:
        data = json.load(f)

    image_path = data["imagePath"]
    img = cv2.imread(image_path)

    if img is None:
        print(f"  Could not read image: {image_path} — skipping")
        skipped += 1
        continue

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        points = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

    base = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(f"dataset/images/{base}.png", img)
    cv2.imwrite(f"dataset/masks/{base}_mask.png", mask)

print(f"\nDone. {len(json_paths) - skipped} pairs saved to dataset/")
if skipped:
    print(f"  {skipped} files skipped (missing images)")
