"""
unet_train.py
-------------
Trains a U-Net to segment crystals in preprocessed chip images.

Architecture: lightweight U-Net (32→64→128→256→512 channels)
Loss:         Binary Cross Entropy
Optimizer:    Adam (lr=1e-4)
Augmentation: horizontal/vertical flip, random 90° rotation (albumentations)

Input:
  dataset/images/  <- PNG images (1024x1024, from make_dataset.py)
  dataset/masks/   <- Binary masks (1024x1024, white = crystal)

Output:
  unet_model.pth              <- final model weights
  unet_model_epoch{N}.pth     <- checkpoint every 10 epochs
  training_loss.png           <- loss curve plot

Usage:
  python unet_train.py

Notes:
  - Trained on a small dataset (~few dozen images) — results will be noisy.
    More annotated data improves performance significantly.
  - Inference threshold of 0.2 (in infer.py) was chosen empirically for this dataset.
"""

import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
NUM_EPOCHS = 50
IMG_SIZE   = 1024

print(f"Device: {DEVICE}")


# ── Dataset ──────────────────────────────────────────────────────────────────

class CrystalDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.transform  = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        aug  = self.transform(image=img, mask=mask)
        return aug["image"], aug["mask"].permute(2, 0, 1).float()


transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

img_paths  = sorted(glob("./dataset/images/*.png"))
mask_paths = sorted(glob("./dataset/masks/*_mask.png"))
print(f"Training on {len(img_paths)} image/mask pairs")

dataset    = CrystalDataset(img_paths, mask_paths, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ── Model ─────────────────────────────────────────────────────────────────────

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
        e1 = self.enc1(x);           e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2)); e4 = self.enc4(self.pool3(e3))
        b  = self.bottleneck(self.pool4(e4))
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out_conv(d1))


model     = UNet().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# ── Training loop ─────────────────────────────────────────────────────────────

losses = []
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0

    for imgs, masks in tqdm(dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), masks)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg = epoch_loss / len(dataloader)
    losses.append(avg)
    print(f"  Loss: {avg:.6f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f"unet_model_epoch{epoch}.pth")
        print(f"  Checkpoint saved: unet_model_epoch{epoch}.pth")

torch.save(model.state_dict(), "unet_model.pth")
print("Final model saved: unet_model.pth")

# Loss curve
plt.figure()
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("Training Loss")
plt.savefig("training_loss.png")
print("Loss curve saved: training_loss.png")
