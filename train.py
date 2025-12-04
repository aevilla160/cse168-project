import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import tifffile as tiff
from sklearn.model_selection import StratifiedShuffleSplit

# ===== CONFIG =====
DATA_ROOT = "./EuroSAT_MS"  # <- change this
BATCH_SIZE = 64
NUM_WORKERS = 32
NUM_CLASSES = 10
RANDOM_SEED = 42

# ImageNet normalization (for ResNet-50 pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
class EuroSATMSDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.class_names = sorted(os.listdir(root))  # folder names = labels
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples = []  # list of (path, label)
        for cls_name in self.class_names:
            cls_dir = os.path.join(root, cls_name)
            for p in glob(os.path.join(cls_dir, "*.tif")):
                self.samples.append((p, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # Read 13-band tif: shape (H, W, 13)
        img = tiff.imread(path).astype(np.float32)

        # Scale from original range (~0-10000) to [0,1]
        img = img / 10000.0

        # Build RGB from B04/B03/B02 (indices 3,2,1 in 0-based)
        # adjust indices if your band order differs
        r = img[:, :, 3]
        g = img[:, :, 2]
        b = img[:, :, 1]
        rgb = np.stack([r, g, b], axis=2)  # (H, W, 3)

        rgb = np.clip(rgb, 0.0, 1.0)

        # To tensor: (C, H, W)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        if self.transform is not None:
            rgb = self.transform(rgb)

        return rgb, label

train_transform = transforms.Compose([
    # Expect tensor in [0,1], (C,H,W); so use tensor-level transforms only
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


full_dataset = EuroSATMSDataset(DATA_ROOT, transform=None)

# Build label vector for stratified split
labels = np.array([lbl for _, lbl in full_dataset.samples])

splitter = StratifiedShuffleSplit(
    n_splits=1, test_size=0.2, random_state=RANDOM_SEED
)
train_idx, val_idx = next(splitter.split(np.zeros_like(labels), labels))

train_dataset = EuroSATMSDataset(DATA_ROOT, transform=train_transform)
val_dataset   = EuroSATMSDataset(DATA_ROOT, transform=val_transform)

train_dataset = Subset(train_dataset, train_idx)
val_dataset   = Subset(val_dataset,   val_idx)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)


import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained on ImageNet
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Replace final FC layer with 10-class classifier
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)

model = model.to(device)


import torch
from torch.optim import SGD
from torch.nn.functional import cross_entropy

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = cross_entropy(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


# Freeze all params except final layer
for name, param in model.named_parameters():
    param.requires_grad = (name.startswith("fc."))

stage1_params = [p for p in model.parameters() if p.requires_grad]

optimizer = SGD(stage1_params, lr=0.01, momentum=0.9, weight_decay=1e-4)

EPOCHS_STAGE1 = 5  # you can adjust

for epoch in range(EPOCHS_STAGE1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"[Stage 1][Epoch {epoch+1}/{EPOCHS_STAGE1}] "
          f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"Val loss={val_loss:.4f} acc={val_acc:.4f}")

    # Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True

optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
EPOCHS_STAGE2 = 15  # adjust as needed

best_val_acc = 0.0
best_state_dict = None

for epoch in range(EPOCHS_STAGE2):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
    val_loss, val_acc = evaluate(model, val_loader)

    # Simple step LR schedule: drop LR halfway
    if epoch == EPOCHS_STAGE2 // 2:
        for g in optimizer.param_groups:
            g["lr"] = 1e-4

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = model.state_dict()

    print(f"[Stage 2][Epoch {epoch+1}/{EPOCHS_STAGE2}] "
          f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
          f"Val loss={val_loss:.4f} acc={val_acc:.4f}")

print(f"Best val accuracy: {best_val_acc:.4f}")
if best_state_dict is not None:
    torch.save(best_state_dict, "resnet50_eurosat_ms_rgb_best.pth")


best_model = resnet50(weights=None)
in_features = best_model.fc.in_features
best_model.fc = nn.Linear(in_features, NUM_CLASSES)
best_model.load_state_dict(torch.load("resnet50_eurosat_ms_rgb_best.pth"))
best_model = best_model.to(device)

val_loss, val_acc = evaluate(best_model, val_loader)
print(f"Final 80/20 split accuracy (val set): {val_acc:.4f}")

