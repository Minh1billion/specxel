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
pip install -e ".[train]"

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

## Web Interface

Open **http://localhost:8000** — features include:
- Drag & drop or click to load an image
- Instant classification with confidence bars
- Batch test up to 100 images at once
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
├── server.py                  # FastAPI app
├── static/index.html          # Web UI (no build step needed)
├── train.py                   # Training script
├── eda.py                     # EDA script
├── pyproject.toml             # Packaging config
└── train_output/              # Weights live on HuggingFace
```

---

## Train From Scratch

```bash
# Prepare dataset folders, then:
python eda.py    # optional analysis
python train.py  # saves best_model.pth to train_output/
```

Tested on GTX 1650 (4 GB VRAM), batch size 32, ~24 epochs.

---

## Model Architecture

EfficientNet-B0 backbone with a custom head:
`Dropout(0.3) → Linear(1280→128) → ReLU → Dropout(0.2) → Linear(128→2)`

Training uses a 5-epoch warmup (head only), then full fine-tuning at 10× lower backbone LR with CosineAnnealingLR decay.

---

## License

MIT