"""
Segmentation Validation / Test Script
Evaluates a trained segmentation head on validation/test data and saves predictions.

Updated to match the new SegmentationHead architecture from train_segmentation.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0,        # Background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

color_palette = np.array([
    [0,   0,   0  ],  # Background     - black
    [34,  139, 34 ],  # Trees          - forest green
    [0,   255, 0  ],  # Lush Bushes    - lime
    [210, 180, 140],  # Dry Grass      - tan
    [139, 90,  43 ],  # Dry Bushes     - brown
    [128, 128, 0  ],  # Ground Clutter - olive
    [139, 69,  19 ],  # Logs           - saddle brown
    [128, 128, 128],  # Rocks          - gray
    [160, 82,  45 ],  # Landscape      - sienna
    [135, 206, 235],  # Sky            - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir      = os.path.join(data_dir, 'Color_Images')
        self.masks_dir      = os.path.join(data_dir, 'Segmentation')
        self.transform      = transform
        self.mask_transform = mask_transform
        self.data_ids       = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id  = self.data_ids[idx]
        image    = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask     = Image.open(os.path.join(self.masks_dir, data_id))
        mask     = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask  = self.mask_transform(mask) * 255

        return image, mask, data_id


# ============================================================================
# Model — MUST match train_segmentation.py exactly
# ============================================================================

class SegmentationHead(nn.Module):
    """
    Deeper ConvNeXt-style head with BatchNorm + residual connections.
    This exactly matches the architecture saved in segmentation_head_best.pth.
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        # Stem: project from embedding space to feature space
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Block 1: depthwise separable conv
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Block 2: refinement
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        self.dropout    = nn.Dropout2d(p=0.1)
        self.classifier = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x) + x     # residual connection
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    iou_per_class = []
    for c in range(num_classes):
        pred_c   = pred == c
        target_c = target == c
        inter    = (pred_c & target_c).sum().float()
        union    = (pred_c | target_c).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((inter / union).item())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)

    dice_per_class = []
    for c in range(num_classes):
        pred_c   = pred == c
        target_c = target == c
        inter    = (pred_c & target_c).sum().float()
        denom    = pred_c.sum().float() + target_c.sum().float()
        dice_per_class.append(((2. * inter + smooth) / (denom + smooth)).item())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


# ============================================================================
# Visualization
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save side-by-side: input | ground truth | prediction."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1) * std + mean
    img  = np.clip(img, 0, 1)

    gt_color   = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);       axes[0].set_title('Input Image');   axes[0].axis('off')
    axes[1].imshow(gt_color);  axes[1].set_title('Ground Truth');  axes[1].axis('off')
    axes[2].imshow(pred_color);axes[2].set_title('Prediction');    axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics text file and per-class IoU bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    # Text file
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write(f"Mean Dice:         {results['mean_dice']:.4f}\n")
        f.write(f"Pixel Accuracy:    {results['pixel_acc']:.4f}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(class_names, results['class_iou']):
            f.write(f"  {name:<20}: {'N/A' if np.isnan(iou) else f'{iou:.4f}'}\n")

    print(f"Saved evaluation metrics to {filepath}")

    # Bar chart
    valid_iou = [v if not np.isnan(v) else 0 for v in results['class_iou']]
    mean_iou  = results['mean_iou']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(n_classes), valid_iou,
           color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU  (Mean IoU: {mean_iou:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.4f}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Value labels on bars
    for i, val in enumerate(valid_iou):
        ax.text(i, val + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation test/inference script')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'segmentation_head.pth'),
                        help='Path to trained model weights (.pth)')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, 'Offroad_Segmentation_testImages'),
                        help='Path to test dataset folder')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(script_dir, 'predictions'),
                        help='Directory to save prediction outputs')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of side-by-side comparison images to save')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -------------------------------------------------------------------------
    # Image size — must match training
    # -------------------------------------------------------------------------
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    # -------------------------------------------------------------------------
    # Transforms — no augmentation for testing
    # -------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    print(f"Loading dataset from {args.data_dir}...")
    valset     = MaskDataset(data_dir=args.data_dir, transform=transform,
                             mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Loaded {len(valset)} samples")

    # -------------------------------------------------------------------------
    # DINOv2 Backbone
    # -------------------------------------------------------------------------
    print("Loading DINOv2 backbone...")
    backbone_model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14"
    )
    backbone_model.eval()
    backbone_model.to(device)
    for param in backbone_model.parameters():
        param.requires_grad = False
    print("Backbone loaded successfully!")

    # -------------------------------------------------------------------------
    # Get embedding dimension
    # -------------------------------------------------------------------------
    sample_img, _, _ = valset[0]
    with torch.no_grad():
        sample_out = backbone_model.forward_features(
            sample_img.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_embedding = sample_out.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # -------------------------------------------------------------------------
    # Load Segmentation Head — new architecture
    # -------------------------------------------------------------------------
    print(f"Loading model from {args.model_path}...")
    classifier = SegmentationHead(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully!")

    # -------------------------------------------------------------------------
    # Output subdirectories
    # -------------------------------------------------------------------------
    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(masks_color_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------------------------
    print(f"\nRunning evaluation on {len(valset)} images...")

    iou_scores      = []
    dice_scores     = []
    pixel_accs      = []
    all_class_iou   = []
    all_class_dice  = []
    sample_count    = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for imgs, labels, data_ids in pbar:
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            # Forward pass
            feat    = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits  = classifier(feat)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)

            pred_masks = torch.argmax(outputs, dim=1)

            # Metrics
            iou,  class_iou  = compute_iou(outputs,  labels, n_classes)
            dice, class_dice = compute_dice(outputs,  labels, n_classes)
            acc              = compute_pixel_accuracy(outputs, labels)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accs.append(acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            # Save per-image outputs
            for i in range(imgs.shape[0]):
                data_id   = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Raw class-ID mask
                pred_np = pred_masks[i].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_np).save(
                    os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Colored mask
                pred_color = mask_to_color(pred_np)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Side-by-side comparison (first N samples only)
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels[i], pred_masks[i],
                        os.path.join(comparisons_dir,
                                     f'sample_{sample_count}_comparison.png'),
                        data_id
                    )

                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.4f}", acc=f"{acc:.4f}")

    # -------------------------------------------------------------------------
    # Aggregate and print results
    # -------------------------------------------------------------------------
    mean_iou       = float(np.nanmean(iou_scores))
    mean_dice      = float(np.nanmean(dice_scores))
    mean_pixel_acc = float(np.mean(pixel_accs))
    avg_class_iou  = np.nanmean(all_class_iou, axis=0)
    avg_class_dice = np.nanmean(all_class_dice, axis=0)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Mean IoU:       {mean_iou:.4f}")
    print(f"  Mean Dice:      {mean_dice:.4f}")
    print(f"  Pixel Accuracy: {mean_pixel_acc:.4f}")
    print("=" * 50)
    print("\nPer-Class IoU:")
    print("-" * 40)
    for name, iou in zip(class_names, avg_class_iou):
        print(f"  {name:<20}: {'N/A' if np.isnan(iou) else f'{iou:.4f}'}")

    results = {
        'mean_iou':  mean_iou,
        'mean_dice': mean_dice,
        'pixel_acc': mean_pixel_acc,
        'class_iou': avg_class_iou,
        'class_dice': avg_class_dice,
    }
    save_metrics_summary(results, args.output_dir)

    print(f"\nDone! Processed {len(valset)} images.")
    print(f"Outputs saved to: {args.output_dir}/")
    print(f"  masks/         → raw class-ID masks (0-9)")
    print(f"  masks_color/   → colored RGB masks")
    print(f"  comparisons/   → side-by-side comparisons ({args.num_samples} samples)")
    print(f"  evaluation_metrics.txt")
    print(f"  per_class_metrics.png")


if __name__ == "__main__":
    main()
