"""
LymphAware Training Script
Author: Teerapong Panboonyuen
IEEE Access 2026

Domain-Aware Bias Disruption for Reliable Lymphoma Diagnosis
"""

import os
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import timm
from sklearn.metrics import roc_auc_score


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class LymphomaDataset(Dataset):

    def __init__(self, root, transform=None):

        self.samples = []
        self.transform = transform

        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            class_dir = os.path.join(root, c)

            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                self.samples.append((path, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# Artifact Shift Module (Shortcut Perturbation)
# ============================================================

class ArtifactShift:

    def __init__(self):

        self.color = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )

        self.blur = transforms.GaussianBlur(5)

    def __call__(self, img):

        if random.random() < 0.5:
            img = self.color(img)

        if random.random() < 0.3:
            img = self.blur(img)

        return img


# ============================================================
# Model
# ============================================================

class MorphologyEncoder(nn.Module):

    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg"
        )

        self.out_dim = self.model.num_features

    def forward(self, x):
        return self.model(x)


class ShortcutBranch(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, f):
        return self.net(f)


class LymphAware(nn.Module):

    def __init__(self, backbone="resnet50", num_classes=3):
        super().__init__()

        self.encoder = MorphologyEncoder(backbone)

        dim = self.encoder.out_dim

        self.shortcut = ShortcutBranch(dim)

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x):

        feat = self.encoder(x)

        shortcut_feat = self.shortcut(feat)

        logits = self.classifier(feat)

        return {
            "logits": logits,
            "feat": feat,
            "shortcut": shortcut_feat
        }


# ============================================================
# Losses
# ============================================================

def orthogonality_loss(f1, f2):

    f1 = F.normalize(f1, dim=1)
    f2 = F.normalize(f2, dim=1)

    return torch.mean(torch.abs((f1 * f2).sum(dim=1)))


def compute_loss(outputs, labels):

    logits = outputs["logits"]
    feat = outputs["feat"]
    shortcut = outputs["shortcut"]

    cls = F.cross_entropy(logits, labels)

    ortho = orthogonality_loss(feat, shortcut)

    loss = cls + 0.1 * ortho

    return loss, cls, ortho


# ============================================================
# Metrics
# ============================================================

def compute_metrics(model, loader, device):

    model.eval()

    probs = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            img = batch["image"].to(device)
            label = batch["label"].to(device)

            out = model(img)

            p = torch.softmax(out["logits"], dim=1)

            probs.append(p.cpu())
            labels.append(label.cpu())

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()

    auc = roc_auc_score(labels, probs, multi_class="ovr")

    preds = np.argmax(probs, axis=1)

    fpr = np.mean(preds != labels)

    return auc, fpr


# ============================================================
# Training
# ============================================================

def train(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    transform_train = transforms.Compose([
        ArtifactShift(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = LymphomaDataset(args.train_dir, transform_train)
    val_ds = LymphomaDataset(args.val_dir, transform_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    model = LymphAware(
        backbone=args.backbone,
        num_classes=args.num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    best_auc = 0

    for epoch in range(args.epochs):

        model.train()

        loop = tqdm(train_loader)

        for batch in loop:

            img = batch["image"].to(device)
            label = batch["label"].to(device)

            outputs = model(img)

            loss, cls, ortho = compute_loss(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(
                f"Epoch [{epoch}/{args.epochs}] "
                f"Loss {loss:.4f}"
            )

        auc, fpr = compute_metrics(model, val_loader, device)

        print(f"\nValidation AUC: {auc:.4f} | FPR: {fpr:.4f}")

        if auc > best_auc:
            best_auc = auc

            torch.save(
                model.state_dict(),
                os.path.join(args.output, "best_model.pth")
            )

            print("✅ Best model saved")


# ============================================================
# Main
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)

    parser.add_argument("--output", type=str, default="./output")

    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--num_classes", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    train(args)