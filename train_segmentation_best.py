"""
Segmentation Training Script - Maximum IoU Version
Trains a segmentation head on top of DINOv2 backbone

Key improvements for maximum IoU:
- Joint image+mask augmentation (fixes sync bug from previous version)
- Deeper segmentation head with residual connections + BatchNorm
- AdamW optimizer (better convergence than SGD)
- CosineAnnealingWarmRestarts scheduler
- Combined CrossEntropy + Dice loss (directly optimizes IoU-like metric)
- Label smoothing to prevent overconfidence
- Gradient clipping for stability
- Best model checkpointing
- FP16 mixed precision on GPU
- Per-class IoU tracking + confusion matrix
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.switch_backend('Agg')


# ============================================================================
# Class Definitions
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

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass',
    'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)


# ============================================================================
# Mask Conversion
# ============================================================================

def convert_mask(mask):
    """Convert raw mask pixel values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Joint Transform — image and mask transformed together (fixes sync bug)
# ============================================================================

class JointTransform:
    """
    Applies augmentations to BOTH image and mask simultaneously.
    This ensures they always stay in sync — critical for correct IoU.
    """
    def __init__(self, size, augment=True):
        self.size    = size       # (H, W)
        self.augment = augment

        self.img_color = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05
        )
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, mask):
        # 1. Resize both (NEAREST for mask to preserve class IDs)
        image = TF.resize(image, self.size)
        mask  = TF.resize(mask,  self.size,
                          interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment:
            # 2. Random horizontal flip — same for both
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # 3. Random vertical flip — same for both
            if random.random() > 0.8:
                image = TF.vflip(image)
                mask  = TF.vflip(mask)

            # 4. Random rotation — same angle for both
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask, angle,
                               interpolation=transforms.InterpolationMode.NEAREST)

            # 5. Color jitter — image ONLY (mask is class IDs, not colors)
            image = self.img_color(image)

        # 6. Convert to tensors
        image = self.normalize(image)
        mask  = TF.to_tensor(mask) * 255    # class IDs 0-9

        return image, mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, joint_transform=None):
        self.image_dir       = os.path.join(data_dir, 'Color_Images')
        self.masks_dir       = os.path.join(data_dir, 'Segmentation')
        self.joint_transform = joint_transform
        self.data_ids        = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image   = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask    = Image.open(os.path.join(self.masks_dir, data_id))
        mask    = convert_mask(mask)

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        return image, mask


# ============================================================================
# Model — Deeper Segmentation Head with residual connections
# ============================================================================

class SegmentationHead(nn.Module):
    """
    Deeper ConvNeXt-style head with:
    - BatchNorm for stable training
    - Residual connection to prevent vanishing gradients
    - Dropout for regularization
    - Proper weight initialization
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

        # Block 1: depthwise separable conv (efficient + large receptive field)
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # Block 2: refinement with smaller kernel
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

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x) + x     # residual connection
        x = self.block2(x)
        x = self.dropout(x)
        return self.classifier(x)


# ============================================================================
# Combined Loss: CrossEntropy + Dice (directly optimizes IoU)
# ============================================================================

class CombinedLoss(nn.Module):
    """
    CrossEntropy gives stable per-pixel gradients.
    Dice loss directly optimizes the F1/IoU-like metric.
    Together they converge to higher IoU than CE alone.
    """
    def __init__(self, class_weights=None, ce_weight=0.6, dice_weight=0.4,
                 num_classes=10, smooth=1e-6, label_smoothing=0.05):
        super().__init__()
        self.ce        = nn.CrossEntropyLoss(weight=class_weights,
                                              label_smoothing=label_smoothing)
        self.ce_w      = ce_weight
        self.dice_w    = dice_weight
        self.n_classes = num_classes
        self.smooth    = smooth

    def dice_loss(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, self.n_classes).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(0, 2, 3))
        denom = pred_soft.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))
        dice  = (2. * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, pred, target):
        return self.ce_w * self.ce(pred, target) + \
               self.dice_w * self.dice_loss(pred, target)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_list = []
    for c in range(num_classes):
        pred_c   = pred == c
        target_c = target == c
        inter    = (pred_c & target_c).sum().float()
        union    = (pred_c | target_c).sum().float()
        if union == 0:
            iou_list.append(float('nan'))
        else:
            iou_list.append((inter / union).item())
    return np.nanmean(iou_list)


def compute_per_class_iou(pred, target, num_classes=10):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    result = []
    for c in range(num_classes):
        pred_c   = pred == c
        target_c = target == c
        inter    = (pred_c & target_c).sum().float()
        union    = (pred_c | target_c).sum().float()
        result.append(float('nan') if union == 0 else (inter / union).item())
    return result


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    scores = []
    for c in range(num_classes):
        pred_c   = pred == c
        target_c = target == c
        inter    = (pred_c & target_c).sum().float()
        denom    = pred_c.sum().float() + target_c.sum().float()
        scores.append(((2. * inter + smooth) / (denom + smooth)).item())
    return np.mean(scores)


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10):
    iou_scores, dice_scores, pixel_accs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.squeeze(1).long().to(device)
            feat    = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = model(feat)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            iou_scores.append(compute_iou(outputs, labels, num_classes))
            dice_scores.append(compute_dice(outputs, labels, num_classes))
            pixel_accs.append(compute_pixel_accuracy(outputs, labels))
    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accs)


def evaluate_per_class(model, backbone, data_loader, device, num_classes=10):
    per_class_accum          = [[] for _ in range(num_classes)]
    all_preds, all_labels_np = [], []

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Per-class eval", leave=False):
            imgs, labels = imgs.to(device), labels.squeeze(1).long().to(device)
            feat    = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = model(feat)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            for c, val in enumerate(compute_per_class_iou(outputs, labels, num_classes)):
                if not np.isnan(val):
                    per_class_accum[c].append(val)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels_np.extend(labels.cpu().numpy().flatten())

    model.train()
    mean_iou = [np.mean(v) if v else float('nan') for v in per_class_accum]
    return mean_iou, np.array(all_preds), np.array(all_labels_np)


# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'],   label='Val')
    axes[0, 0].set_title('Loss vs Epoch'); axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_iou'], label='Train')
    axes[0, 1].plot(history['val_iou'],   label='Val')
    axes[0, 1].set_title('IoU vs Epoch'); axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU'); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(history['train_dice'], label='Train')
    axes[1, 0].plot(history['val_dice'],   label='Val')
    axes[1, 0].set_title('Dice Score vs Epoch'); axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice'); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(history['train_pixel_acc'], label='Train')
    axes[1, 1].plot(history['val_pixel_acc'],   label='Val')
    axes[1, 1].set_title('Pixel Accuracy vs Epoch'); axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy'); axes[1, 1].legend(); axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150)
    plt.close()

    # LR schedule
    if history.get('lr'):
        plt.figure(figsize=(8, 4))
        plt.plot(history['lr'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch'); plt.ylabel('LR'); plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lr_schedule.png'), dpi=150)
        plt.close()

    print(f"Saved training plots to '{output_dir}'")


def save_confusion_matrix(all_preds, all_labels, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cm      = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Saved confusion matrix to '{output_dir}/confusion_matrix.png'")


def save_per_class_iou_plot(per_class_iou, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    values   = [v if not np.isnan(v) else 0.0 for v in per_class_iou]
    mean_iou = np.nanmean(per_class_iou)

    plt.figure(figsize=(12, 5))
    bars = plt.bar(CLASS_NAMES, values, color='steelblue', edgecolor='white')
    plt.axhline(y=mean_iou, color='red', linestyle='--',
                label=f'Mean IoU: {mean_iou:.3f}')
    plt.title('Per-Class IoU — Validation Set')
    plt.xlabel('Class'); plt.ylabel('IoU')
    plt.xticks(rotation=45, ha='right'); plt.ylim(0, 1.05); plt.legend()
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150)
    plt.close()
    print(f"Saved per-class IoU chart to '{output_dir}/per_class_iou.png'")


def save_history_to_file(history, per_class_iou, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')

    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f}  (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f}  (Epoch {np.argmax(history['val_dice']) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f}  (Epoch {np.argmax(history['val_pixel_acc']) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f}  (Epoch {np.argmin(history['val_loss']) + 1})\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Class IoU (Final Validation — Best Model):\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(CLASS_NAMES, per_class_iou):
            f.write(f"  {name:<20}: {'N/A' if np.isnan(iou) else f'{iou:.4f}'}\n")
        f.write(f"\n  Mean IoU: {np.nanmean(per_class_iou):.4f}\n")
        f.write("=" * 60 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 110 + "\n")
        hdrs = ['Epoch', 'TrLoss', 'ValLoss', 'TrIoU', 'ValIoU',
                'TrDice', 'ValDice', 'TrAcc', 'ValAcc', 'LR']
        f.write("{:<7}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<12}\n".format(*hdrs))
        f.write("-" * 110 + "\n")
        for i in range(len(history['train_loss'])):
            lr_val = history['lr'][i] if history.get('lr') else 0.0
            f.write("{:<7}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<10.4f}{:<12.6f}\n".format(
                i + 1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i],  history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
                lr_val
            ))

    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    # -------------------------------------------------------------------------
    # Device setup
    # -------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # -------------------------------------------------------------------------
    # Hyperparameters
    # -------------------------------------------------------------------------
    batch_size = 2
    w          = int(((960 / 2) // 14) * 14)
    h          = int(((540 / 2) // 14) * 14)
    lr         = 3e-4       # AdamW works best with this range
    n_epochs   = 30
    USE_FP16   = device.type == 'cuda'

    print(f"\nConfig: {w}x{h} | Epochs: {n_epochs} | LR: {lr} | "
          f"Batch: {batch_size} | FP16: {USE_FP16}")

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, 'Offroad_Segmentation_Training_Dataset', 'val')

    # -------------------------------------------------------------------------
    # Joint Transforms
    # -------------------------------------------------------------------------
    train_transform = JointTransform(size=(h, w), augment=True)
    val_transform   = JointTransform(size=(h, w), augment=False)   # no augmentation for val

    # -------------------------------------------------------------------------
    # Datasets & Loaders
    # -------------------------------------------------------------------------
    trainset = MaskDataset(data_dir=data_dir, joint_transform=train_transform)
    valset   = MaskDataset(data_dir=val_dir,  joint_transform=val_transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    print(f"\nTraining samples:   {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # -------------------------------------------------------------------------
    # DINOv2 Backbone — frozen
    # -------------------------------------------------------------------------
    print("\nLoading DINOv2 backbone...")
    backbone_model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14"
    )
    backbone_model.eval()
    backbone_model.to(device)
    for param in backbone_model.parameters():
        param.requires_grad = False
    print("Backbone loaded and frozen.")

    # -------------------------------------------------------------------------
    # Get embedding dimension
    # -------------------------------------------------------------------------
    with torch.no_grad():
        sample_imgs, _ = next(iter(train_loader))
        sample_out = backbone_model.forward_features(
            sample_imgs.to(device))["x_norm_patchtokens"]
    n_embedding = sample_out.shape[2]
    print(f"Embedding dim: {n_embedding}  |  Token shape: {sample_out.shape}")

    # -------------------------------------------------------------------------
    # Segmentation Head
    # -------------------------------------------------------------------------
    classifier = SegmentationHead(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    ).to(device)

    total_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # -------------------------------------------------------------------------
    # Loss: Combined CE + Dice
    # -------------------------------------------------------------------------
    class_weights = torch.tensor(
        [0.8, 1.2, 1.2, 1.0, 1.2, 1.5, 2.0, 2.0, 0.8, 0.8],
        dtype=torch.float32
    ).to(device)

    loss_fct = CombinedLoss(
        class_weights=class_weights,
        ce_weight=0.6,
        dice_weight=0.4,
        num_classes=n_classes,
        label_smoothing=0.05
    )

    # -------------------------------------------------------------------------
    # Optimizer: AdamW
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)

    # -------------------------------------------------------------------------
    # LR Scheduler: Cosine with warm restarts
    # -------------------------------------------------------------------------
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6
    )

    # -------------------------------------------------------------------------
    # FP16
    # -------------------------------------------------------------------------
    scaler = torch.cuda.amp.GradScaler() if USE_FP16 else None

    # -------------------------------------------------------------------------
    # History
    # -------------------------------------------------------------------------
    history = {
        'train_loss': [], 'val_loss':        [],
        'train_iou':  [], 'val_iou':         [],
        'train_dice': [], 'val_dice':        [],
        'train_pixel_acc': [], 'val_pixel_acc': [],
        'lr': []
    }

    best_val_iou    = 0.0
    best_model_path = os.path.join(script_dir, "segmentation_head_best.pth")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    for epoch in range(n_epochs):

        # ---- Train ----
        classifier.train()
        train_losses = []

        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch+1:02d}/{n_epochs} [Train]",
                                 leave=False):
            imgs   = imgs.to(device)
            labels = labels.squeeze(1).long().to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                feat = backbone_model.forward_features(imgs)["x_norm_patchtokens"]

            if USE_FP16:
                with torch.cuda.amp.autocast():
                    logits  = classifier(feat)
                    outputs = F.interpolate(logits, size=imgs.shape[2:],
                                            mode="bilinear", align_corners=False)
                    loss = loss_fct(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits  = classifier(feat)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                loss = loss_fct(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()

            train_losses.append(loss.item())

        # ---- Validate ----
        classifier.eval()
        val_losses = []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"Epoch {epoch+1:02d}/{n_epochs} [Val]  ",
                                     leave=False):
                imgs   = imgs.to(device)
                labels = labels.squeeze(1).long().to(device)
                feat    = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(feat)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                val_losses.append(loss_fct(outputs, labels).item())

        # ---- Compute metrics ----
        train_iou, train_dice, train_acc = evaluate_metrics(
            classifier, backbone_model, train_loader, device, n_classes)
        val_iou, val_dice, val_acc = evaluate_metrics(
            classifier, backbone_model, val_loader, device, n_classes)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss   = np.mean(val_losses)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(float(train_iou))
        history['val_iou'].append(float(val_iou))
        history['train_dice'].append(float(train_dice))
        history['val_dice'].append(float(val_dice))
        history['train_pixel_acc'].append(float(train_acc))
        history['val_pixel_acc'].append(float(val_acc))
        history['lr'].append(current_lr)

        # Save best model
        best_marker = ""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(classifier.state_dict(), best_model_path)
            best_marker = "  ← BEST"

        print(
            f"Epoch {epoch+1:02d}/{n_epochs} | "
            f"Loss {epoch_train_loss:.4f}/{epoch_val_loss:.4f} | "
            f"IoU {train_iou:.4f}/{val_iou:.4f} | "
            f"Acc {train_acc:.4f}/{val_acc:.4f} | "
            f"LR {current_lr:.6f}{best_marker}"
        )

    # -------------------------------------------------------------------------
    # Final per-class evaluation using best saved model
    # -------------------------------------------------------------------------
    print("\nLoading best model for final evaluation...")
    classifier.load_state_dict(torch.load(best_model_path))

    per_class_iou, all_preds, all_labels = evaluate_per_class(
        classifier, backbone_model, val_loader, device, n_classes
    )

    print("\nPer-Class IoU (Validation — Best Model):")
    print("-" * 40)
    for name, iou in zip(CLASS_NAMES, per_class_iou):
        print(f"  {name:<20}: {'N/A' if np.isnan(iou) else f'{iou:.4f}'}")
    print(f"\n  Mean IoU: {np.nanmean(per_class_iou):.4f}")

    # -------------------------------------------------------------------------
    # Save all outputs
    # -------------------------------------------------------------------------
    print("\nSaving plots and results...")
    save_training_plots(history, output_dir)
    save_confusion_matrix(all_preds, all_labels, output_dir)
    save_per_class_iou_plot(per_class_iou, output_dir)
    save_history_to_file(history, per_class_iou, output_dir)

    final_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), final_path)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best Val IoU:       {best_val_iou:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    print(f"  Best model:         {best_model_path}")
    print(f"  Final model:        {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
