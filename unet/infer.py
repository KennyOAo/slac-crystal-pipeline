"""
infer.py
--------
Runs trained U-Net on preprocessed chip images and saves prediction masks.

Input:
  A folder of preprocessed images (1024x1024, from preprocess.py)
  A trained model weights file (.pth, from unet_train.py)

Output:
  For each input image, saves a grayscale prediction mask:
    <filename>_pred.png  — raw sigmoid output (0–255)

Usage:
  python infer.py --input processed/ --model unet_model.pth --output predictions/

Notes:
  - Threshold of 0.2 was chosen empirically on this dataset.
    Increase if you're getting too many false positives.
  - Model runs on GPU if available, otherwise CPU.
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

THRESHOLD = 0.2  # pixel classified as crystal if prediction > this value


# ── Model (must match unet_train.py) ─────────────────────────────────────────

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        def cbr(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(inplace=True),
            )

        self.enc1, self.pool1 = cbr(3,   32),  nn.MaxPool2d(2)
        self.enc2, self.pool2 = cbr(32,  64),  nn.MaxPool2d(2)
        self.enc3, self.pool3 = cbr(64,  128), nn.MaxPool2d(2)
        self.enc4, self.pool4 = cbr(128, 256), nn.MaxPool2d(2)
        self.bottleneck       = cbr(256, 512)

        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, stride=2), cbr(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, stride=2), cbr(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128,  64, 2, stride=2), cbr(128,  64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64,   32, 2, stride=2), cbr(64,   32)
        self.out_conv        = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x);            e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(input_dir, model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded: {model_path}")

    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith((".png", ".jpg", ".tif"))]
    print(f"Running inference on {len(files)} images...")

    for fname in tqdm(files):
        img = cv2.imread(os.path.join(input_dir, fname))
        if img is None:
            print(f"  Skipped (unreadable): {fname}")
            continue

        img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (1024, 1024))
        img_norm    = img_resized / 255.0
        tensor      = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(tensor).cpu().squeeze().numpy()

        # Save raw prediction (grayscale 0-255)
        pred_img  = (pred * 255).astype(np.uint8)
        base      = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(output_dir, f"{base}_pred.png"), pred_img)

        # Also save thresholded binary mask
        binary = ((pred > THRESHOLD) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"{base}_mask.png"), binary)

    print(f"Done. Predictions saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run U-Net crystal segmentation inference.")
    parser.add_argument("--input",  required=True, help="Folder of preprocessed images")
    parser.add_argument("--model",  required=True, help="Path to .pth model weights")
    parser.add_argument("--output", required=True, help="Folder to save prediction masks")
    args = parser.parse_args()

    run_inference(args.input, args.model, args.output)
