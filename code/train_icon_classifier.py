import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    image_size: int = 64
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_frac: float = 0.2
    seed: int = 1337


def make_transforms(image_size: int):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)],
                p=0.7,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_transforms, val_transforms


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def main() -> int:
    ap = argparse.ArgumentParser(description="Train hero icon classifier (ImageFolder).")
    ap.add_argument("--data-dir", default="data/icon_dataset", help="ImageFolder root.")
    ap.add_argument("--out-dir", default="results/icon_classifier", help="Output folder.")
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_frac=args.val_frac,
    )

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(cfg.seed)

    train_transforms, val_transforms = make_transforms(cfg.image_size)

    base_ds = datasets.ImageFolder(cfg.data_dir)
    n = len(base_ds)
    if n < 10:
        print(f"Dataset too small: {n} images in {cfg.data_dir}")
        return 1

    val_n = max(1, int(n * cfg.val_frac))
    train_n = n - val_n
    train_ds, val_ds = random_split(
        base_ds, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.seed)
    )

    # Assign transforms after split (ImageFolder stores them on the dataset object).
    train_ds.dataset.transform = train_transforms
    val_ds.dataset.transform = val_transforms

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    num_classes = len(base_ds.classes)
    print(f"Classes: {num_classes}")
    print(f"Train/val: {train_n}/{val_n}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        steps = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            train_loss += loss.item()
            train_acc += accuracy(logits.detach(), yb)
            steps += 1

        train_loss /= max(1, steps)
        train_acc /= max(1, steps)

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        vsteps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                val_loss += loss.item()
                val_acc += accuracy(logits, yb)
                vsteps += 1

        val_loss /= max(1, vsteps)
        val_acc /= max(1, vsteps)

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": base_ds.classes, "cfg": asdict(cfg)}, best_path)

    # Save metadata
    with open(os.path.join(cfg.out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(base_ds.classes, f, indent=2)
    with open(os.path.join(cfg.out_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\nBest val acc: {best_val_acc:.3f}")
    print(f"Saved: {best_path}")
    return 0


if __name__ == "__main__":
    main()

