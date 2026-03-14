"""
server.py — Pixel Art Classifier · FastAPI Backend
Run: uvicorn server:app --reload
  or: pixelart-classifier (if installed via pip)
"""

from __future__ import annotations

import io
import base64
import zipfile
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


DEFAULT_CKPT = Path(__file__).parent / "train_output" / "best_model.pth"
STATIC_DIR   = Path(__file__).parent / "static"
CLASSES      = ["non-pixelart", "pixelart"]


app = FastAPI(
    title       = "Pixel Art Classifier",
    description = "Classify images as pixel art or non-pixel art using EfficientNet-B0",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


class ModelState:
    model        : Optional[nn.Module] = None
    idx_to_class : Optional[dict]      = None
    tf           : Optional[object]    = None
    device       : torch.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_acc      : Optional[float]     = None
    loaded       : bool                = False
    error        : Optional[str]       = None

state = ModelState()


def _build_model(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_f, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes),
    )
    return m


def _get_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(
            (img_size, img_size),
            interpolation=transforms.InterpolationMode.NEAREST,
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_model(ckpt_path: Path = DEFAULT_CKPT):
    """Load checkpoint into global state. Call once at startup."""
    try:
        ckpt = torch.load(ckpt_path, map_location=state.device, weights_only=False)
        cfg  = ckpt.get("cfg", {})

        model = _build_model(cfg.get("num_classes", 2))
        model.load_state_dict(ckpt["model_state"])
        model = model.to(state.device).eval()

        state.model        = model
        state.idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
        state.tf           = _get_transform(cfg.get("img_size", 224))
        state.val_acc      = ckpt.get("val_acc")
        state.loaded       = True
        state.error        = None
        print(f"[OK] Model loaded from {ckpt_path}  (device={state.device})")
    except FileNotFoundError:
        state.error = f"Checkpoint not found: {ckpt_path}"
        print(f"[WARN] {state.error}")
    except Exception as e:
        state.error = str(e)
        print(f"[ERROR] {e}")


@torch.no_grad()
def _predict(img: Image.Image) -> dict:
    tensor = state.tf(img.convert("RGB")).unsqueeze(0).to(state.device)
    logits = state.model(tensor)
    probs  = torch.softmax(logits, dim=1)[0]
    idx    = int(probs.argmax())
    label  = state.idx_to_class[idx]
    return {
        "label"     : label,
        "confidence": round(float(probs[idx]), 4),
        "probs"     : {
            state.idx_to_class[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        },
    }


@app.on_event("startup")
async def startup():
    load_model()


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found in static/")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def status():
    return {
        "loaded"  : state.loaded,
        "error"   : state.error,
        "device"  : str(state.device),
        "val_acc" : state.val_acc,
        "classes" : CLASSES,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "Model not loaded")

    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp", "image/gif"}
    if file.content_type and file.content_type not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.content_type}")

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:  # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Cannot decode image")

    # Thumbnail for response preview
    thumb = img.copy()
    thumb.thumbnail((300, 300), Image.NEAREST)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    result = _predict(img)
    return JSONResponse({
        **result,
        "filename"  : file.filename,
        "dimensions": {"width": img.width, "height": img.height},
        "preview"   : f"data:image/jpeg;base64,{b64}",
    })


@app.post("/api/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    if not state.loaded:
        raise HTTPException(status_code=503, detail=state.error or "Model not loaded")
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Max 100 files per batch")

    results = []
    for f in files:
        try:
            raw = await f.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            res = _predict(img)
            results.append({"filename": f.filename, "ok": True, **res})
        except Exception as e:
            results.append({"filename": f.filename, "ok": False, "error": str(e)})

    summary = {
        "pixelart"    : sum(1 for r in results if r.get("label") == "pixelart"),
        "non-pixelart": sum(1 for r in results if r.get("label") == "non-pixelart"),
        "errors"      : sum(1 for r in results if not r.get("ok")),
        "total"       : len(results),
    }
    return JSONResponse({"summary": summary, "results": results})


def main():
    import uvicorn
    uvicorn.run("pixelart_classifier.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)