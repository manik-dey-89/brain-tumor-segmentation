# Brain Tumor Segmentation System

> **Research-grade brain tumour segmentation using Attention U-Net.**  
> End-to-end pipeline: MRI upload → preprocessing → deep learning inference → segmentation overlay → quantitative metrics → export.

---

> ⚠️ **Medical Disclaimer**  
> This system is an **AI-assisted research tool**, not a medical device.  
> Results must **not** replace professional medical diagnosis.  
> Always consult a qualified radiologist or physician.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Inference (CLI)](#inference-cli)
8. [Backend API](#backend-api)
9. [Frontend Dashboard](#frontend-dashboard)
10. [Docker Deployment](#docker-deployment)
11. [Testing](#testing)
12. [Configuration Reference](#configuration-reference)
13. [API Reference](#api-reference)
14. [Model Architecture](#model-architecture)

---

## Features

| Category | Details |
|---|---|
| **Models** | U-Net, Attention U-Net, ResUNet (configurable from `config.yaml`) |
| **Loss** | Dice + BCE + Focal + Tversky (configurable weighted combination) |
| **Training** | AdamW, cosine warm-up LR, mixed precision (AMP), early stopping, TensorBoard |
| **Augmentation** | Albumentations: flips, rotation, elastic, brightness, Gaussian noise, coarse dropout |
| **Preprocessing** | Resize, z-score/min-max/percentile normalisation, Gaussian/median denoising, mask validation |
| **Data split** | Patient-level split to prevent data leakage |
| **Evaluation** | Dice, IoU, Precision, Recall, F1, Sensitivity, Specificity, Hausdorff-95 |
| **Inference** | TTA (flip), MC-Dropout uncertainty estimation, confidence scoring |
| **Backend** | FastAPI: `/predict`, `/segment`, `/health`, `/model-info`, `/metrics`, `/history` |
| **Frontend** | React + Tailwind: drag-drop upload, overlay viewer, metrics panel, history |
| **Docker** | Multi-stage builds for backend (Python) and frontend (Nginx) |

---

## Project Structure

```
brain-tumor-segmentation/
├── configs/
│   └── config.yaml          # Central configuration (model, data, training, inference)
├── models/
│   ├── blocks.py            # Shared building blocks (ConvBnAct, ResidualBlock, AttentionGate …)
│   ├── unet.py              # Classic U-Net
│   ├── attention_unet.py    # Attention U-Net (Oktay et al. 2018)
│   ├── resunet.py           # Residual U-Net + optional pretrained encoder
│   └── factory.py           # build_model(), load_checkpoint(), save_checkpoint()
├── data/
│   ├── preprocessing.py     # Preprocessor: load, resize, normalise, denoise, validate mask
│   ├── augmentation.py      # get_train_transforms / get_val_transforms (albumentations)
│   ├── splitter.py          # Patient-level and image-level split, discover_pairs()
│   └── dataset.py           # BrainTumorDataset, create_dataloaders()
├── training/
│   ├── losses.py            # DiceLoss, BCEDiceLoss, FocalLoss, TverskyLoss, CombinedLoss
│   ├── metrics.py           # SegmentationMetrics (accumulator), MetricTracker
│   ├── trainer.py           # Trainer: full training loop, AMP, scheduler, checkpointing
│   └── visualization.py     # Training curves, segmentation panels, overlay arrays
├── inference/
│   ├── engine.py            # SegmentationEngine: predict_array(), TTA, MC-Dropout
│   ├── postprocessing.py    # Morphological cleanup, encode/decode helpers, contours
│   └── inference.py         # Programmatic API: run_inference(), generate_report()
├── backend/
│   └── app/
│       ├── main.py          # FastAPI app factory + lifespan model loading
│       ├── api/routes.py    # All API endpoints
│       ├── core/            # config.py (pydantic-settings), engine_state.py
│       ├── services/        # prediction_service.py, history_service.py
│       └── utils/           # file_utils.py (upload validation, TempFileContext)
├── frontend/
│   ├── src/
│   │   ├── pages/           # LandingPage, AnalyzePage, HistoryPage, ModelInfoPage
│   │   ├── components/      # Layout, UploadZone, ResultViewer, MetricsPanel, ProgressBar
│   │   ├── hooks/           # usePrediction.js
│   │   └── utils/           # api.js (axios), helpers.js
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   ├── prepare_dataset.py   # Validate + preprocess raw data, generate manifest CSV
│   ├── generate_samples.py  # Synthetic MRI/mask generator for pipeline testing
│   └── export_onnx.py       # Export trained model to ONNX
├── tests/
│   ├── test_preprocessing.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_losses.py
│   └── test_api.py          # FastAPI integration tests (mocked engine)
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_train_and_evaluate.ipynb
│   └── 03_inference_demo.ipynb
├── train.py                 # Training entry point
├── evaluate.py              # Evaluation entry point
├── predict.py               # CLI prediction (single image or batch)
├── requirements.txt
├── .env.example
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd brain-tumor-segmentation

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

For GPU (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Generate synthetic samples (no real data needed to test)

```bash
python scripts/generate_samples.py --n 100 --out data/raw
```

### 3. Train

```bash
python train.py
```

### 4. Predict

```bash
python predict.py --input data/raw/images/SYNTH_0001_slice000.png
```

### 5. Start the API

```bash
python backend/run.py
# Open: http://localhost:8000/docs
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
# Open: http://localhost:3000
```

---

## Dataset Preparation

### Expected layout

```
data/raw/
├── images/
│   ├── BraTS21_001_slice042.png
│   └── ...
└── masks/
    ├── BraTS21_001_slice042.png   # same filename as image
    └── ...
```

Supported formats: **PNG, JPG, TIFF, NIfTI (.nii, .nii.gz)**

### Validate and preprocess

```bash
python scripts/prepare_dataset.py --validate-only     # check only
python scripts/prepare_dataset.py                      # validate + process to data/processed/
```

### Patient-level split

Enable in `configs/config.yaml`:
```yaml
data:
  patient_level_split: true
  patient_id_separator: "_"
  patient_id_index: 1    # "BraTS21_001_slice042" → patient ID = "001"
```

---

## Training

```bash
# Default config
python train.py

# Override key parameters
python train.py --epochs 100 --lr 1e-4 --batch 8 --arch attention_unet

# Resume from checkpoint
python train.py --resume outputs/checkpoints/last_model.pth

# Disable mixed precision (CPU training)
python train.py --no-amp
```

Outputs:
- `outputs/checkpoints/best_model.pth`  – best validation Dice checkpoint
- `outputs/checkpoints/last_model.pth`  – most recent checkpoint
- `outputs/visualizations/training_curves.png`
- `outputs/logs/training_history.json`
- `outputs/logs/tensorboard/`           – TensorBoard logs

Launch TensorBoard:
```bash
tensorboard --logdir outputs/logs/tensorboard
```

---

## Evaluation

```bash
python evaluate.py                        # evaluate best_model on test set
python evaluate.py --split val            # evaluate on validation set
python evaluate.py --model outputs/checkpoints/best_model.pth
```

Outputs:
- `outputs/predictions/evaluation_test.csv`              – per-sample metrics
- `outputs/predictions/evaluation_test_aggregate.json`   – aggregate report
- `outputs/visualizations/<stem>_panel.png`              – visualisation panels

---

## Inference (CLI)

```bash
# Single image
python predict.py --input data/raw/images/mri.png

# With ground-truth mask for metrics
python predict.py --input mri.png --mask mask.png

# Batch (entire folder)
python predict.py --input data/raw/images/ --output outputs/predictions/

# Skip post-processing
python predict.py --input mri.png --no-postprocess
```

---

## Backend API

```bash
python backend/run.py --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: **http://localhost:8000/docs**

### Sample requests

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Model info
curl http://localhost:8000/api/v1/model-info

# Predict (JSON with base64 images)
curl -X POST http://localhost:8000/api/v1/predict \
     -F "file=@data/raw/images/mri.png" \
     -F "gt_mask=@data/raw/masks/mri.png"

# Segment (raw PNG mask bytes)
curl -X POST http://localhost:8000/api/v1/segment \
     -F "file=@mri.png" \
     --output pred_mask.png

# History
curl http://localhost:8000/api/v1/history?limit=10
curl http://localhost:8000/api/v1/metrics
```

---

## Frontend Dashboard

```bash
cd frontend
npm install
npm run dev      # http://localhost:3000

npm run build    # production build → frontend/dist/
```

Pages:
| Route | Description |
|---|---|
| `/` | Landing page — project overview, pipeline diagram |
| `/analyze` | Upload MRI, run segmentation, view results |
| `/history` | Paginated prediction history + statistics |
| `/model` | Architecture details and inference config |

---

## Docker Deployment

```bash
# Build and start both services
docker compose up --build

# Detached
docker compose up --build -d

# GPU support (requires nvidia-container-toolkit)
docker compose --profile gpu up --build

# Stop
docker compose down
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v
pytest tests/test_metrics.py -v
pytest tests/test_preprocessing.py -v
pytest tests/test_losses.py -v
pytest tests/test_api.py -v

# With coverage
pip install pytest-cov
pytest tests/ --cov=. --cov-report=term-missing
```

---

## Configuration Reference

All settings live in `configs/config.yaml`.  
Key sections:

| Section | Key Settings |
|---|---|
| `data` | `image_size`, `batch_size`, `patient_level_split`, `num_classes` |
| `preprocessing` | `normalization_method`, `denoise_method`, `clip_values` |
| `augmentation` | `enabled`, flip probs, rotation, elastic, noise |
| `model` | `architecture` (unet/attention_unet/resunet), `base_filters`, `depth`, `dropout_rate` |
| `loss` | `primary`, `dice_weight`, `bce_weight`, `focal_weight`, `tversky_weight` |
| `training` | `epochs`, `learning_rate`, `scheduler`, `early_stopping`, `mixed_precision` |
| `inference` | `model_path`, `tta`, `mc_dropout`, `mc_samples`, `threshold` |
| `evaluation` | `metrics`, `threshold` |

Override any value via CLI flags or environment variables (see `.env.example`).

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Liveness / readiness probe |
| GET | `/api/v1/model-info` | Architecture, parameters, config |
| POST | `/api/v1/predict` | Upload MRI → JSON (base64 images + metrics) |
| POST | `/api/v1/segment` | Upload MRI → PNG mask bytes |
| GET | `/api/v1/metrics` | Aggregate statistics from history |
| GET | `/api/v1/history` | Paginated prediction history |
| DELETE | `/api/v1/history` | Clear all history |
| GET | `/api/v1/history/{id}` | Single prediction record |

---

## Model Architecture

### Attention U-Net (default)

```
Input MRI  (1ch, 256×256)
    │
    ▼  Stem: Conv→BN→ReLU (×2)  →  64 ch
    ├──────────────────────────────────── skip₀
    ▼  Encoder 1: MaxPool + DoubleConv   128 ch
    ├─────────────────────────────── skip₁
    ▼  Encoder 2: MaxPool + DoubleConv   256 ch
    ├────────────────────────── skip₂
    ▼  Encoder 3: MaxPool + DoubleConv   512 ch
    ├───────────────────── skip₃
    ▼  Bottleneck: MaxPool + DoubleConv  1024 ch
    │
    ▼  Decoder + AttentionGate(g=1024, x=512)  → 512 ch
    ▼  Decoder + AttentionGate(g=512,  x=256)  → 256 ch
    ▼  Decoder + AttentionGate(g=256,  x=128)  → 128 ch
    ▼  Decoder + AttentionGate(g=128,  x=64)   →  64 ch
    │
    ▼  Head: 1×1 Conv → Sigmoid
    │
Segmentation Mask (256×256)
```

Attention gates (Oktay et al. 2018) compute spatial coefficients that
suppress irrelevant background activations and amplify tumour-relevant features,
reducing false positives without extra supervision.

---

*Brain Tumor Segmentation – Built for B.Tech / AI research demonstration.*  
*Model performance depends entirely on the training data used.*  
*No clinical accuracy claims are made.*
