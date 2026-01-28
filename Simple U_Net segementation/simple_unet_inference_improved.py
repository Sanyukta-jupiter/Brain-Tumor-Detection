# -------------------------------
# pc_brats_inference_combined_window.py
# -------------------------------

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder containing 4-modality PNGs
base_folder = "test_patient"  # e.g., "C:/BRATS2020/test_patient"
output_dir = os.path.join(base_folder, "output")
os.makedirs(output_dir, exist_ok=True)

# Path to checkpoint
checkpoint_path = "checkpoint_epoch_39.pth"

# -------------------------------
# Define Simple U-Net Model
# -------------------------------
class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv_block = UNetDownBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

class SimpleUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.enc1 = UNetDownBlock(4, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetDownBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetDownBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = UNetDownBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = UNetDownBlock(512, 1024)
        self.up4 = UNetUpBlock(1024, 512)
        self.up3 = UNetUpBlock(512, 256)
        self.up2 = UNetUpBlock(256, 128)
        self.up1 = UNetUpBlock(128, 64)
        self.final_conv = nn.Conv2d(64, num_classes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))
        dec4 = self.up4(bottleneck, enc4)
        dec3 = self.up3(dec4, enc3)
        dec2 = self.up2(dec3, enc2)
        dec1 = self.up1(dec2, enc1)
        output = self.final_conv(dec1)
        return self.sigmoid(output)

# -------------------------------
# Load Model
# -------------------------------
model = SimpleUNet().to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("✅ Model loaded successfully.")

# -------------------------------
# Utility Functions
# -------------------------------
def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.resize(img, (240, 240))
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() + 1e-8)
    return img

def load_4channel_images(folder):
    modalities = ['flair', 't1', 't1ce', 't2']
    imgs = []
    flair_vis = None
    for m in modalities:
        path = os.path.join(folder, f"{m}.png")
        img = load_gray(path)
        imgs.append(img)
        if m == 'flair':
            flair_vis = img
    tensor = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0).float().to(device)
    return tensor, flair_vis

def analyze_prediction(prob):
    """Calculate confidence and tumor presence"""
    mask = (prob > 0.5).astype(np.uint8)
    tumor_detected = np.any(mask)
    confidence = float(prob[mask==1].mean()) if tumor_detected else float(prob.mean())
    return mask, tumor_detected, confidence

def show_and_save_results(flair, prob_map, mask, confidence, output_path):
    """
    Display FLAIR, probability map, mask, overlay in one figure
    with info box showing Tumor detected, Confidence
    and save the figure to output_path
    """
    tumor_detected = np.any(mask)
    overlay = cv2.cvtColor((flair*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    overlay[:,:,2] = np.maximum(overlay[:,:,2], mask*255)

    fig, axes = plt.subplots(1,4, figsize=(20,5))
    
    axes[0].imshow(flair, cmap='gray')
    axes[0].set_title("FLAIR Input")
    axes[0].axis('off')
    
    axes[1].imshow(prob_map, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Probability Map")
    axes[1].axis('off')
    
    axes[2].imshow(mask*255, cmap='gray')
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].axis('off')
    
    info_text = f"Tumor Detected: {'YES' if tumor_detected else 'NO'}\n"
    info_text += f"Confidence: {confidence:.4f}"
    
    axes[3].text(0.05, 0.95, info_text, color='white', fontsize=12,
                 verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)

# -------------------------------
# Run Inference
# -------------------------------
tensor, flair_vis = load_4channel_images(base_folder)

with torch.no_grad():
    pred = model(tensor)
    prob = pred.squeeze().cpu().numpy()

mask, tumor_detected, confidence = analyze_prediction(prob)

# Save and show combined window
result_filename = "results_tumor.png" if tumor_detected else "results_no_tumor.png"
result_path = os.path.join(output_dir, result_filename)
show_and_save_results(flair_vis, prob, mask, confidence, result_path)

print(f"✅ Inference completed. Tumor Detected: {'YES' if tumor_detected else 'NO'}, Confidence: {confidence:.4f}")
print(f"✅ Results saved to: {result_path}")
