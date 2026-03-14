import json, time, random, shutil
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

CFG = {
    "data_root"       : "./pixelart-classification",
    "output_dir"      : "./train_output",
    "num_classes"     : 2,
    "img_size"        : 224,
    "epochs"          : 40,
    "batch_size"      : 32,
    "lr"              : 1e-4,
    "weight_decay"    : 1e-4,
    "patience"        : 8,
    "seed"            : 42,
    "val_split"       : 0.10,
    "test_split"      : 0.10,
    "use_amp"         : False,
    "use_class_weight": True,
}

CLASSES = ["non-pixelart", "pixelart"]


def set_seed(seed):
    # make runs reproducible across restarts
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def split_dataset(data_root, output_dir, val_split, test_split, seed):
    split_root = output_dir / "split_data"
    if split_root.exists():
        # reuse existing split so val/test sets never leak into training
        return split_root

    rng = random.Random(seed)
    for cls in CLASSES:
        cls_dir = data_root / cls
        imgs = sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(imgs)

        n       = len(imgs)
        n_test  = int(n * test_split)
        n_val   = int(n * val_split)
        splits  = {
            "train": imgs[:n - n_test - n_val],
            "val"  : imgs[n - n_test - n_val : n - n_test],
            "test" : imgs[n - n_test:],
        }
        for split_name, paths in splits.items():
            dest = split_root / split_name / cls
            dest.mkdir(parents=True, exist_ok=True)
            for p in paths:
                d = dest / p.name
                if not d.exists():
                    shutil.copy2(p, d)  # copy, not move — keeps original intact

        print(f"    [{cls}] train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")

    return split_root


def get_transforms(img_size):
    # ImageNet stats — safe to reuse even for pixel art since we're fine-tuning
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    NR   = transforms.InterpolationMode.NEAREST  # no blurring on pixel art edges

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32), interpolation=NR),
        transforms.RandomCrop(img_size),          # cheap spatial augmentation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10, interpolation=NR),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        # val/test: deterministic resize only, no augmentation
        transforms.Resize((img_size, img_size), interpolation=NR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def build_model(num_classes, device):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # freeze backbone initially — only train the new head for the first few epochs
    for p in model.features.parameters():
        p.requires_grad = False

    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_f, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return model.to(device)


def unfreeze_backbone(model, lr, optimizer):
    # called after warmup — backbone gets a much lower lr to avoid destroying pretrained weights
    print("\n  Unfreeze backbone")
    for p in model.features.parameters():
        p.requires_grad = True
    optimizer.add_param_group({
        "params": list(model.features.parameters()),
        "lr"    : lr * 0.1,  # 10x smaller than head lr
    })


def make_sampler(dataset):
    # oversample minority class so each batch sees roughly equal class counts
    targets = [s[1] for s in dataset.samples]
    counts  = np.bincount(targets).astype(np.float64)
    w       = 1.0 / counts
    return WeightedRandomSampler([w[t] for t in targets], len(targets), replacement=True)


def compute_class_weights(dataset, device):
    # inverse-frequency weights — rarer class gets higher loss penalty
    targets = [s[1] for s in dataset.samples]
    counts  = np.bincount(targets).astype(np.float64)
    total   = counts.sum()
    w = torch.tensor(
        [total / (len(counts) * c) for c in counts],
        dtype=torch.float32
    ).to(device)
    print(f"  Class counts : {dict(zip(CLASSES, counts.astype(int)))}")
    print(f"  Class weights: {w.cpu().numpy().round(4)}")
    return w


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # skip corrupted batches rather than crashing the whole run
        if torch.isnan(imgs).any() or torch.isinf(imgs).any():
            nan_batches += 1
            continue

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            nan_batches += 1
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # prevent gradient explosion
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    if nan_batches:
        print(f"    {nan_batches} batch(es) skipped due to NaN/Inf")
    if total == 0:
        return float("nan"), 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)

        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

        # collect for classification_report / confusion matrix later
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def plot_history(history, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train", color="#2196F3")
    ax1.plot(history["val_loss"],   label="Val",   color="#F44336")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(history["train_acc"], label="Train", color="#2196F3")
    ax2.plot(history["val_acc"],   label="Val",   color="#F44336")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=120)
    plt.close()


def plot_confusion(cm, class_names, output_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # white text on dark cells, black on light ones
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontweight="bold")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close()


def main():
    set_seed(CFG["seed"])
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"    GPU : {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    split_root = split_dataset(
        Path(CFG["data_root"]), output_dir,
        CFG["val_split"], CFG["test_split"], CFG["seed"]
    )

    train_tf, val_tf = get_transforms(CFG["img_size"])
    train_ds = datasets.ImageFolder(split_root / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(split_root / "val",   transform=val_tf)
    test_ds  = datasets.ImageFolder(split_root / "test",  transform=val_tf)
    print(f"\n  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    # train uses WeightedRandomSampler — val/test stay ordered and unshuffled
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        sampler=make_sampler(train_ds),
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"],
        shuffle=False, num_workers=2, pin_memory=True
    )

    model = build_model(CFG["num_classes"], device)
    print(f"\n  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    class_w   = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_w)

    # only pass trainable params — frozen backbone params are excluded at this point
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    # cosine decay to lr=1e-6 — smooth cooldown avoids abrupt lr drops
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=1e-6
    )

    history       = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc  = 0.0
    patience_cnt  = 0
    unfreeze_done = False
    WARMUP        = 5  # epochs to train head only before unlocking the backbone

    print(f"\n{'=' * 60}")
    print(f"TRAIN  ({CFG['epochs']} epochs, lr={CFG['lr']}, bs={CFG['batch_size']})")
    print(f"{'=' * 60}")

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        # unfreeze backbone once warmup is done
        if epoch == WARMUP + 1 and not unfreeze_done:
            unfreeze_backbone(model, CFG["lr"], optimizer)
            unfreeze_done = True

        train_loss, train_acc         = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc, _, _     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        vram = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"  Ep {epoch:02d}/{CFG['epochs']}  "
              f"loss {train_loss:.4f}/{val_loss:.4f}  "
              f"acc {train_acc:.4f}/{val_acc:.4f}  "
              f"VRAM {vram:.2f}GB  {time.time() - t0:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            # save full checkpoint so we can resume or inspect later
            torch.save({
                "epoch"        : epoch,
                "model_state"  : model.state_dict(),
                "val_acc"      : val_acc,
                "class_to_idx" : train_ds.class_to_idx,
                "cfg"          : CFG,
            }, output_dir / "best_model.pth")
            print(f"    saved best model (val_acc={val_acc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"\n  EarlyStopping at epoch {epoch}")
                break

    print(f"\n{'=' * 60}\nTEST EVALUATION\n{'=' * 60}")

    # reload best checkpoint — not the last epoch weights
    ckpt = torch.load(output_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n  Test Acc : {test_acc:.4f}  ({test_acc * 100:.1f}%)")
    print(f"\n{classification_report(labels, preds, target_names=CLASSES)}")

    cm = confusion_matrix(labels, preds)
    print(f"  Confusion Matrix:\n{cm}")

    plot_history(history, output_dir)
    plot_confusion(cm, CLASSES, output_dir)
    print(f"  Plots saved to: {output_dir}")

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "best_val_acc": float(best_val_acc),
            "test_acc"    : float(test_acc),
            "epochs_run"  : len(history["train_loss"]),
            "timestamp"   : datetime.now().isoformat(),
            "cfg"         : CFG,
        }, f, indent=2)

    print(f"\n  best_val={best_val_acc:.4f}  test={test_acc:.4f}")


if __name__ == "__main__":
    main()