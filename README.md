# 🎮 Pixel Art Classifier

A deep learning image classifier that detects whether an image is **pixel art** or **non-pixel art**, served as a web app via **FastAPI**.

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/minh1billion/pixelart-classification)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.0%25-brightgreen)

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **98.03%** |
| Best Val Accuracy | **98.88%** |
| Epochs Trained | 24 (early stopped) |
| Model | EfficientNet-B0 (~20 MB) |

**Confusion Matrix (test set):**
```
              Predicted
              non-PA   PA
True  non-PA    92      1
      PA         6    257
```

---

## Quickstart

### Run from source

```bash
# 1. Clone
git clone https://github.com/minh1billion/pixelart-classifier
cd pixelart-classifier

# 2. Install
pip install -r requirements.txt

# 3. Download model weights
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='minh1billion/pixelart-classification',
    filename='best_model.pth',
    local_dir='./train_output'
)
"

# 4. Start
uvicorn server:app --reload
# open http://localhost:8000
```

---

## Dataset

The training dataset is hosted on Google Drive (~114 MB).

📦 **[Download data.zip](https://drive.google.com/drive/folders/1i-w3_in-O93gQvBj1ktRxrBdIdFhGw11)**

After downloading, extract into the project root so the structure looks like:

```
pixelart-classifier/
├── data/
│   ├── pixelart/
│   └── non-pixelart/
```

**Dataset stats:**

| Class | Images |
|-------|--------|
| pixelart | 2 631 |
| non-pixelart | 938 |
| **Total** | **3 569** |

---

## Train From Scratch

> Make sure the dataset is downloaded and extracted first (see above).

**Step 1 — Run EDA** *(optional but recommended)*

Analyses class distribution, image sizes, colour stats, and duplicate detection. Outputs charts and a JSON report to `eda_output/`.

```bash
python eda.py
```

**Step 2 — Train**

Splits the data, trains EfficientNet-B0, and saves the best checkpoint to `train_output/best_model.pth`.

```bash
python train.py
```

Training config (editable at the top of `train.py`):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `img_size` | 224 | Input resolution |
| `epochs` | 40 | Early stopping at patience=8 |
| `batch_size` | 32 | Safe for 4 GB VRAM |
| `lr` | 1e-4 | Head LR; backbone uses 1e-5 after warmup |

Tested on **GTX 1650 (4 GB VRAM)**, ~24 epochs, ~30 min total.

---

## Web Interface

Open **http://localhost:8000** — features include:

- Drag & drop, click to browse, or `Ctrl+V` to paste from clipboard
- `Enter` to classify, `Esc` to clear
- Confidence bars per class
- Click image to zoom fullscreen
- Classification history (last 20, clickable)
- Copy result as JSON or download
- Batch test up to 100 images with per-file breakdown and JSON export
- Auto-generated API docs at `/docs`

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/status` | Model status + device info |
| `POST` | `/api/predict` | Classify a single image |
| `POST` | `/api/predict-batch` | Classify up to 100 images |

```bash
curl -X POST http://localhost:8000/api/predict -F "file=@sprite.png"
```

```json
{
  "label": "pixelart",
  "confidence": 0.9923,
  "probs": { "pixelart": 0.9923, "non-pixelart": 0.0077 },
  "filename": "sprite.png",
  "dimensions": { "width": 64, "height": 64 }
}
```

---

## Project Structure

```
pixelart-classifier/
├── data/                      # Dataset (download separately)
├── eda_output/                # EDA charts and report (generated)
├── train_output/              # Weights and training artifacts (generated)
│   └── best_model.pth         # Also available on HuggingFace
├── static/
│   └── index.html             # Web UI (no build step needed)
├── server.py                  # FastAPI app
├── train.py                   # Training script
├── eda.py                     # EDA script
└── requirements.txt
```

---

## Model Architecture

EfficientNet-B0 backbone with a custom head:

```
Dropout(0.3) → Linear(1280→128) → ReLU → Dropout(0.2) → Linear(128→2)
```

Training uses a 5-epoch warmup (head only), then full fine-tuning at 10× lower backbone LR with CosineAnnealingLR decay.

---

## License

MIT