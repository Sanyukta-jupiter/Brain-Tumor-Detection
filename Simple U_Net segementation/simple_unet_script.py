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
import time

# -------------------------------
# CONFIG (FIXED IMAGE FOLDER)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIXED FOLDER WHERE YOUR IMAGES ARE LOCATED
base_folder = "."

# Output folder
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
        # Note: conv_block expects (in_channels, out_channels) where in_channels = concat channels
        self.conv_block = UNetDownBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if necessary
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
# checkpoint must contain "model_state_dict"
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # fallback: assume checkpoint is state_dict
    model.load_state_dict(checkpoint)
model.eval()
print("✅ Model loaded successfully.")

# -------------------------------
# Utility Functions
# -------------------------------
def load_gray(path, target_size=(240,240)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"File not found: {path}")
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)
    # normalize 0..1
    mn = img.min()
    mx = img.max()
    if mx - mn > 1e-8:
        img = (img - mn) / (mx - mn)
    else:
        img = img - mn
    return img

def load_4channel_images(folder):
    modalities = ['flair', 't1', 't1ce', 't2']
    imgs = []
    flair_vis = None
    for m in modalities:
        # try multiple extensions if needed
        found = False
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            path = os.path.join(folder, f"{m}{ext}")
            if os.path.exists(path):
                img = load_gray(path)
                imgs.append(img)
                if m == 'flair':
                    flair_vis = img
                found = True
                break
        if not found:
            # try case-insensitive search
            for fname in os.listdir(folder):
                if m.lower() in fname.lower() and fname.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff')):
                    path = os.path.join(folder, fname)
                    img = load_gray(path)
                    imgs.append(img)
                    if m == 'flair':
                        flair_vis = img
                    found = True
                    break
        if not found:
            raise FileNotFoundError(f"Could not find modality '{m}' image in folder: {folder}")
    tensor = torch.from_numpy(np.stack(imgs, axis=0)).unsqueeze(0).float().to(device)
    return tensor, flair_vis

def analyze_prediction(prob, threshold=0.5):
    """Calculate confidence and tumor presence"""
    binary_mask = (prob > threshold).astype(np.float32)
    tumor_pixels = np.sum(binary_mask)
    total_pixels = binary_mask.size
    tumor_ratio = tumor_pixels / total_pixels if total_pixels>0 else 0.0
    tumor_detected = tumor_pixels > 0
    avg_confidence = np.mean(prob[binary_mask == 1]) if tumor_detected else 0.0
    max_confidence = float(np.max(prob)) if prob.size>0 else 0.0
    confidence_score = float(avg_confidence) if tumor_detected else (1.0 - max_confidence)
    return {
        'binary_mask': binary_mask,
        'tumor_detected': tumor_detected,
        'tumor_pixels': float(tumor_pixels),
        'tumor_ratio': tumor_ratio,
        'avg_confidence': float(avg_confidence),
        'max_confidence': float(max_confidence),
        'confidence_score': float(confidence_score)
    }

def visualize_results(flair_img, prediction, analysis_results, output_path=None):
    """Create visualization of results (2x3 layout like screenshot)"""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], wspace=0.25, hspace=0.3)

    # 1. FLAIR
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(flair_img, cmap='gray')
    ax1.set_title('')
    ax1.axis('off')

    # 2. Probability heatmap (HOT colormap)
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(prediction, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('Probability Heatmap')
    ax2.axis('off')
    cbar = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    # 3. Binary mask
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(analysis_results['binary_mask'], cmap='gray')
    ax3.set_title('Binary Mask')
    ax3.axis('off')

    # 4. Overlay
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(flair_img, cmap='gray')
    ax4.imshow(analysis_results['binary_mask'], cmap='Reds', alpha=0.45)
    ax4.set_title('Tumor Region Highlighted')
    ax4.axis('off')

    # 5. Tumor-area intensity distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Tumor-Area Intensity Distribution')
    ax5.set_xlabel('Intensity (0–1)')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3)

    tumor_mask = analysis_results['binary_mask'] > 0
    tumor_pixels = flair_img[tumor_mask]
    if tumor_pixels.size > 0:
        counts, bins, patches = ax5.hist(tumor_pixels, bins=40, alpha=0.75)
        # optional smoothing line (if scipy available)
        try:
            from scipy.ndimage import gaussian_filter1d
            smooth = gaussian_filter1d(counts, sigma=1.0)
            centers = (bins[:-1] + bins[1:]) / 2
            ax5.plot(centers, smooth, linewidth=2)
        except Exception:
            pass
    else:
        ax5.text(0.5, 0.5, "No tumor pixels found", fontsize=12, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6))

    # 6. Results box
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    results_text = (
        f"RESULTS:\n\n"
        f"Tumor Detected: {'YES' if analysis_results['tumor_detected'] else 'NO'}\n"
        f"Confidence Score: {analysis_results['confidence_score']:.4f}\n\n"
        f"Tumor Pixels: {analysis_results['tumor_pixels']:.1f}\n"
        f"Tumor Ratio: {analysis_results['tumor_ratio']*100:.4f}%\n\n"
        f"Avg Confidence: {analysis_results['avg_confidence']:.4f}\n"
        f"Max Confidence: {analysis_results['max_confidence']:.4f}\n"
    )

    ax6.text(0.02, 0.55, results_text, fontsize=12, va='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='gray', alpha=0.85))

    # Title and layout
    fig.suptitle('Brain Tumor Segmentation Results', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

    plt.show()
    return fig

# -------------------------------
# Run Inference
# -------------------------------
print(f"\n📂 Loading images from: {base_folder}")
tensor, flair_vis = load_4channel_images(base_folder)

start_time = time.time()
with torch.no_grad():
    pred = model(tensor)
    prob = pred.squeeze().cpu().numpy()
inference_time = time.time() - start_time
print(f"⏱ Inference time: {inference_time:.2f}s")

analysis = analyze_prediction(prob, threshold=0.5)

# Save binary mask and prediction images as before
pred_img = (prob * 255).astype(np.uint8)
mask_img = (analysis['binary_mask'] * 255).astype(np.uint8)

ts = time.strftime("%Y%m%d_%H%M%S")
pred_path = os.path.join(output_dir, f'prediction_{ts}.png')
mask_path = os.path.join(output_dir, f'mask_{ts}.png')
cv2.imwrite(pred_path, pred_img)
cv2.imwrite(mask_path, mask_img)

# Save visualization (2x3 layout)
vis_path = os.path.join(output_dir, f'visualization_{ts}.png')
visualize_results(flair_vis, prob, analysis, vis_path)

# Save a small text summary
txt_path = os.path.join(output_dir, f'results_{ts}.txt')
with open(txt_path, 'w') as f:
    f.write("Brain Tumor Segmentation Results\n")
    f.write("="*40 + "\n")
    f.write(f"Folder: {base_folder}\n")
    f.write(f"Model: {checkpoint_path}\n")
    f.write(f"Inference time: {inference_time:.2f}s\n\n")
    f.write(f"Tumor Detected: {'YES' if analysis['tumor_detected'] else 'NO'}\n")
    f.write(f"Confidence Score: {analysis['confidence_score']:.4f}\n")
    f.write(f"Tumor Pixels: {analysis['tumor_pixels']}\n")
    f.write(f"Tumor Ratio: {analysis['tumor_ratio']*100:.4f}%\n")
    f.write(f"Avg Confidence: {analysis['avg_confidence']:.4f}\n")
    f.write(f"Max Confidence: {analysis['max_confidence']:.4f}\n")

print(f"\n✅ Inference completed. Tumor Detected: {'YES' if analysis['tumor_detected'] else 'NO'}")
print(f"✅ Confidence Score: {analysis['confidence_score']:.4f}")
print(f"✅ Results saved to: {output_dir}")
