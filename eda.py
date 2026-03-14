"""
EDA - Pixel Art Classification Dataset
Run this before training to understand your data.
Tested on GTX 1650 (4GB VRAM) setup (My PC config. You can adjust training params if needed.).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

DATASET_ROOT = Path(".")
CLASSES      = ["pixelart", "non-pixelart"]
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
OUTPUT_DIR   = Path("eda_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def find_dataset_root(base: Path) -> Path:
    # try common dataset folder names before giving up
    for candidate in [base, base / "pixelart-classification",
                      base / "dataset", base / "data"]:
        if candidate.exists():
            subdirs = [d.name for d in candidate.iterdir() if d.is_dir()]
            # only accept if it contains at least one of our class folders
            if any(c in subdirs for c in CLASSES):
                return candidate
    return base


def collect_images(root: Path) -> dict[str, list[Path]]:
    data: dict[str, list[Path]] = {}
    for cls in CLASSES:
        cls_dir = root / cls
        # rglob handles nested sub-folders inside a class directory
        imgs = [p for p in cls_dir.rglob("*") if p.suffix.lower() in IMG_EXTS] if cls_dir.exists() else []
        data[cls] = imgs
    return data


def safe_open(path: Path):
    # force RGB so all images share the same channel count (some PNGs are RGBA)
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None  # skip corrupted files silently


# --- Section 1: Class distribution ---

def analyze_distribution(data: dict) -> dict:
    stats = {cls: len(imgs) for cls, imgs in data.items()}
    total = sum(stats.values())

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)
    for cls, n in stats.items():
        pct = n / total * 100 if total else 0
        bar = "█" * int(pct / 2)  # ascii bar scaled to 50 chars max
        print(f"  {cls:<20} {n:>6} images  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':<20} {total:>6} images")

    if stats and total:
        # ratio > 1 means the majority class has more samples
        ratio = max(stats.values()) / max(min(stats.values()), 1)
        label = "Imbalanced" if ratio > 3 else "Slightly imbalanced" if ratio > 1.5 else "Balanced"
        print(f"\n  {label} (ratio = {ratio:.1f}x)")

    return stats


# --- Section 2: Image size and format ---

def analyze_sizes(data: dict, sample_n: int = 500) -> dict:
    print("\n" + "=" * 60)
    print("IMAGE SIZE & FORMAT ANALYSIS")
    print("=" * 60)

    size_report = {}
    for cls, paths in data.items():
        # cap at sample_n — opening every image is slow
        sample = paths[:sample_n]
        widths, heights, broken = [], [], 0
        fmt_counter: Counter = Counter()

        for p in sample:
            img = safe_open(p)
            if img is None:
                broken += 1  # track how many files failed to open
                continue
            w, h = img.size
            widths.append(w)
            heights.append(h)
            fmt_counter[p.suffix.lower()] += 1

        if widths:
            w_arr, h_arr = np.array(widths), np.array(heights)
            print(f"\n  [{cls}]")
            print(f"  Width  : min={w_arr.min()}, max={w_arr.max()}, mean={w_arr.mean():.0f}, median={np.median(w_arr):.0f}")
            print(f"  Height : min={h_arr.min()}, max={h_arr.max()}, mean={h_arr.mean():.0f}, median={np.median(h_arr):.0f}")
            print(f"  Formats: {dict(fmt_counter)}")
            if broken:
                print(f"  Corrupted: {broken} images")

            size_report[cls] = {"widths": widths, "heights": heights, "broken": broken}

    return size_report


# --- Section 3: Pixel/color statistics ---

def analyze_pixel_stats(data: dict, sample_n: int = 200) -> dict:
    print("\n" + "=" * 60)
    print("PIXEL / COLOR STATISTICS")
    print("=" * 60)

    report = {}
    for cls, paths in data.items():
        means_r, means_g, means_b, stds, unique_counts = [], [], [], [], []

        for p in paths[:sample_n]:
            img = safe_open(p)
            if img is None:
                continue
            arr = np.array(img).astype(np.float32)

            # per-channel means reveal colour bias in the dataset
            means_r.append(arr[:, :, 0].mean())
            means_g.append(arr[:, :, 1].mean())
            means_b.append(arr[:, :, 2].mean())

            # global std — low value means low-contrast image
            stds.append(arr.std())

            # count distinct RGB tuples — pixel art typically has very few
            unique_counts.append(
                len(set(map(tuple, arr.reshape(-1, 3).astype(np.uint8))))
            )

        if means_r:
            uc_med = int(np.median(unique_counts))
            print(f"\n  [{cls}]  (sample={len(means_r)})")
            print(f"  Mean RGB : R={np.mean(means_r):.1f}, G={np.mean(means_g):.1f}, B={np.mean(means_b):.1f}")
            print(f"  Std pixel: {np.mean(stds):.2f}")
            print(f"  Unique colors/image: median={uc_med}, max={int(np.max(unique_counts))}")

            if cls == "pixelart":
                # <256 unique colors is a strong indicator of real pixel art
                note = "Few colors — clear pixel art signature" if uc_med < 256 else "High color count — possible mislabels"
                print(f"  {note}")

            report[cls] = {
                "mean_r": float(np.mean(means_r)),
                "mean_g": float(np.mean(means_g)),
                "mean_b": float(np.mean(means_b)),
                "mean_std": float(np.mean(stds)),
                "median_unique_colors": uc_med,
            }

    return report


# --- Section 4: Duplicate detection ---

def detect_duplicates(data: dict) -> int:
    print("\n" + "=" * 60)
    print("DUPLICATE DETECTION")
    print("=" * 60)

    from hashlib import md5
    # map file hash -> list of (class, path) to spot exact byte-for-byte duplicates
    hash_map: dict[str, list] = defaultdict(list)

    for cls, paths in data.items():
        for p in paths:
            try:
                h = md5(p.read_bytes()).hexdigest()
                hash_map[h].append((cls, p))
            except Exception:
                pass  # unreadable file — skip

    # any hash bucket with more than one entry is a duplicate group
    dup_groups = {h: v for h, v in hash_map.items() if len(v) > 1}
    total_dup  = sum(len(v) - 1 for v in dup_groups.values())

    for _, group in list(dup_groups.items())[:10]:  # print first 10 only
        print(f"  DUP: {', '.join(f'{c}/{p.name}' for c, p in group)}")

    if total_dup == 0:
        print("  No duplicates found.")
    else:
        print(f"\n  {total_dup} duplicate images — remove before training.")

    return total_dup


# --- Section 5: Visualizations ---

def plot_samples(data: dict, n_each: int = 8):
    fig, axes = plt.subplots(2, n_each, figsize=(n_each * 2, 5))
    fig.suptitle("Sample Images", fontsize=14, fontweight="bold")

    for row, cls in enumerate(CLASSES):
        paths = data.get(cls, [])[:]
        np.random.shuffle(paths)  # randomise so we don't always see the same files
        for col in range(n_each):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(paths):
                img = safe_open(paths[col])
                if img:
                    ax.imshow(img)
                    if col == 0:
                        ax.set_ylabel(cls, fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "samples.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/samples.png")


def plot_distribution(stats: dict):
    labels, values = list(stats.keys()), list(stats.values())
    # green/red for the standard two-class case, blue fallback otherwise
    colors = ["#4CAF50", "#F44336"] if "pixelart" in labels else ["#2196F3"] * len(labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)

    # label each bar with its exact count
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontweight="bold")

    ax.set_title("Class Distribution", fontweight="bold")
    ax.set_ylabel("Number of images")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution.png", dpi=120)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/distribution.png")


def plot_size_scatter(size_report: dict):
    colors_map = {"pixelart": "#4CAF50", "non-pixelart": "#F44336"}
    fig, ax = plt.subplots(figsize=(7, 5))

    for cls, rep in size_report.items():
        ax.scatter(rep["widths"], rep["heights"], alpha=0.3, s=10,
                   label=cls, color=colors_map.get(cls, "blue"))

    # 224px reference lines — standard input size for most CNN backbones
    ax.axvline(224, color="gray", linestyle="--", linewidth=1, label="224px")
    ax.axhline(224, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Dimensions Scatter", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "size_scatter.png", dpi=120)
    plt.close()
    print(f"  Saved: {OUTPUT_DIR}/size_scatter.png")


# --- Section 6: Training recommendations for GTX 1650 ---

def print_recommendations(stats: dict, pixel_report: dict):
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS — GTX 1650 (4GB VRAM)")
    print("=" * 60)

    total = sum(stats.values())

    # batch size scales with dataset size; too large risks OOM on 4GB
    bs = 32 if total < 1000 else 64 if total < 5000 else 128
    bs_note = (
        "small dataset — use heavy augmentation" if total < 1000
        else "stable"                             if total < 5000
        else "can increase if VRAM allows"
    )

    print("""
  Architecture options:
    MobileNetV3-Small  (recommended) — ~1.2GB VRAM, fast inference
    EfficientNet-B0                  — ~1.5GB VRAM, good accuracy/speed
    ResNet-18                        — simpler, weaker than MobileNet
    ResNet-50+, ViT                  — likely OOM on 4GB with large batches
""")
    print(f"  Training params:")
    print(f"    Input size : 224x224 (or 128x128 for faster iteration)")
    print(f"    Batch size : {bs}  ({bs_note})")
    print(f"    Optimizer  : AdamW, lr=3e-4, weight_decay=1e-4")
    print(f"    Scheduler  : CosineAnnealingLR or OneCycleLR")
    print(f"    Epochs     : 20-50 with EarlyStopping (patience=7)")
    print(f"    AMP        : torch.cuda.amp.autocast() — saves ~30% VRAM")
    print(f"    Loss       : CrossEntropyLoss")

    if stats:
        ratio = max(stats.values()) / max(min(stats.values()), 1)
        if ratio > 2:
            # imbalanced data pushes the model to predict the majority class too often
            print(f"\n  Class imbalance detected (ratio {ratio:.1f}x):")
            print("    Use class_weight or WeightedRandomSampler")

    print("""
  Augmentation:
    RandomHorizontalFlip, RandomRotation(15),
    ColorJitter(brightness=0.3, contrast=0.3),
    RandomResizedCrop(224, scale=(0.8, 1.0))
    Note: avoid heavy blur/interpolation on pixel art
          use interpolation=NEAREST when resizing

  Evaluation:
    Metrics : Accuracy, F1-score, Confusion Matrix
    Split   : 80/10/10 or K-Fold if data is limited
""")


# --- Section 7: Save JSON report ---

def save_json_report(dist_stats: dict, pixel_report: dict, dup_count: int):
    report = {
        "class_counts":     dist_stats,
        "total_images":     sum(dist_stats.values()),
        # ratio > 1.0 signals imbalance; helps decide whether to use weighted loss
        "imbalance_ratio":  max(dist_stats.values()) / max(min(dist_stats.values()), 1) if dist_stats else None,
        "pixel_stats":      pixel_report,
        "duplicates_found": dup_count,
    }
    out = OUTPUT_DIR / "eda_report.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out}")
    return report


# --- Main ---

def main():
    print("PIXEL ART CLASSIFICATION — EDA")
    print("Run from the folder containing your dataset\n")

    root = find_dataset_root(DATASET_ROOT)
    print(f"  Dataset root: {root.resolve()}")

    data = collect_images(root)

    missing = [c for c in CLASSES if not data.get(c)]
    if missing:
        print(f"\n  Missing class folders: {missing}")
        print("  Make sure you're running from the right directory, or update DATASET_ROOT.")
        return

    dist_stats   = analyze_distribution(data)
    size_report  = analyze_sizes(data)
    pixel_report = analyze_pixel_stats(data)
    dup_count    = detect_duplicates(data)

    print("\n" + "=" * 60)
    print("SAVING VISUALIZATIONS")
    print("=" * 60)
    plot_samples(data)
    plot_distribution(dist_stats)
    if size_report:
        plot_size_scatter(size_report)

    print_recommendations(dist_stats, pixel_report)

    print("=" * 60)
    print("SAVING REPORT")
    print("=" * 60)
    save_json_report(dist_stats, pixel_report, dup_count)

    print(f"\n  EDA complete. Check folder: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()