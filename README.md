# PULMONAR AI — Chest X-Ray Classification System

A deep learning system for automated chest X-ray diagnosis, classifying radiographs into four categories: **COVID-19**, **Normal**, **Bacterial Pneumonia**, and **Viral Pneumonia**. The system combines a fine-tuned DenseNet121 classifier with a Siamese neural network cascade for pneumonia sub-type refinement, exposed through a Flask REST API and a React glassmorphism frontend.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [AI Models](#ai-models)
  - [Model 1 — Fine-Tuned DenseNet121](#model-1--fine-tuned-densenet121-4-class-classifier)
  - [Model 2 — Siamese Network + Random Forest](#model-2--siamese-neural-network--random-forest-viral-vs-bacterial)
  - [Cascade Inference Pipeline](#cascade-inference-pipeline)
  - [Explainability — Grad-CAM](#explainability--grad-cam)
- [Frontend](#frontend)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  INFERENCE PIPELINE                      │
│                                                          │
│   Chest X-Ray Input                                      │
│         │                                                │
│         ▼                                                │
│   ┌───────────────────────────────┐                      │
│   │  DenseNet121 (fine-tuned)     │                      │
│   │  4-class softmax classifier   │                      │
│   └───────────┬───────────────────┘                      │
│               │                                          │
│     ┌─────────┴──────────┐                               │
│     │                    │                               │
│  COVID-19 / Normal    Bacterial / Viral                  │
│     │                 Pneumonia                          │
│     │                    │                               │
│     │                    ▼                               │
│     │     ┌──────────────────────────────┐               │
│     │     │  Siamese Network (CheXNet)   │               │
│     │     │  Left lung ↔ Right lung      │               │
│     │     │  Feature similarity metrics  │               │
│     │     │  + Random Forest classifier  │               │
│     │     └──────────────┬───────────────┘               │
│     │                    │                               │
│     └──────────┬─────────┘                               │
│                ▼                                          │
│        Final Diagnosis + Confidence                       │
└─────────────────────────────────────────────────────────┘
```

---

## AI Models

### Model 1 — Fine-Tuned DenseNet121 (4-class Classifier)

**Files**: `dense_net.py` (baseline), `dense_net_fine_tune.py` (production)  
**Weights**: `mon_modele_pulmonar_final.pth`

#### Approach

DenseNet121 was chosen as the backbone due to its dense connectivity pattern, which encourages feature reuse and is particularly well-suited to medical imaging tasks. Weights were initialised from **CheXNet** — a model pre-trained on the NIH ChestX-ray14 dataset covering 14 thoracic diseases — rather than ImageNet, giving the model domain-relevant priors from the start.

The final classifier layer was replaced with a 4-output linear layer and the entire network was fine-tuned end-to-end.

#### Training Details

| Parameter | Value |
|---|---|
| Input size | 224 × 224 |
| Normalisation | ImageNet stats `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |
| Optimiser | Adam, lr = 1e-5 |
| Loss | Cross-entropy with class weights |
| Class weighting | `total / (n_classes × class_count)` to handle imbalance |
| Early stopping | Patience = 5 epochs (monitors validation loss) |
| Max epochs | 100 |

Training plots (loss curves, confusion matrix, per-class ROC curves) are in `ZPlots/baseline_plots/` and `ZPlots/early_stop_plots/`.

---

### Model 2 — Siamese Neural Network + Random Forest (Viral vs Bacterial)

**Files**: `siamese.py`, `siamese_light.py`  
**Weights**: `rf_siamese_model.pkl`, `siamese_scaler.pkl`

#### Motivation

Bacterial and Viral Pneumonia produce visually similar consolidation patterns on X-rays. Rather than treating this as a straightforward classification task, a **Siamese-inspired approach** was used: the model exploits the bilateral symmetry of healthy lungs, measuring how the two lungs *differ* from each other as a discriminative signal.

#### Feature Extraction

The CheXNet DenseNet121 backbone (frozen, classifier replaced with `nn.Identity()`) is used to extract 1024-dimensional feature vectors. Each X-ray is processed in three ways:

1. **Global** — the full 224×224 image → 1024-dim feature vector
2. **Left lung** — left half of the image (`[:, :, :, :112]`)
3. **Right lung** — right half, horizontally flipped for symmetry alignment (`[:, :, :, 112:]`)

Five bilateral similarity metrics are then computed between the left and right lung features:

| Metric | Formula |
|---|---|
| Cosine similarity | `cos(feat_left, feat_right)` |
| L1 distance | `Σ\|feat_left − feat_right\|` |
| L2 distance | `‖feat_left − feat_right‖₂` |
| Chebyshev distance | `max\|feat_left − feat_right\|` |
| Dot product | `feat_left · feat_right` |

These are concatenated with the global features to form a **1029-dimensional feature vector** per X-ray.

#### Random Forest Classifier

Feature vectors are standardised (StandardScaler) then passed to a Random Forest. Hyperparameters were tuned using **Optuna Bayesian optimisation** (30 trials, F1-macro objective) with `class_weight='balanced'`. An optimal decision threshold is computed from the precision-recall curve rather than using the default 0.5 cutoff.

Training plots are in `ZPlots/Random_Forest_plots/`.

---

### Cascade Inference Pipeline

**File**: `server_upgrade.py`

The two models are combined into a two-stage cascade:

1. **Stage 1**: The fine-tuned DenseNet121 classifies the X-ray into one of four classes.
2. **Stage 2 (conditional)**: If Stage 1 predicts **Bacterial** or **Viral Pneumonia**, the Siamese + Random Forest model is invoked to refine the sub-type diagnosis. If Stage 1 returns **COVID-19** or **Normal**, the result is returned directly.

The API response includes a `used_siamese_model` flag indicating which path was taken, and `initial_model_guess` when the Siamese model overrides the first-stage output.

---

### Explainability — Grad-CAM

**File**: `server.py`, `grad_cam_test.py`

Gradient-weighted Class Activation Mapping (Grad-CAM) is generated on request via the `/gradcam` endpoint. The target layer is `model.features.norm5` (the final batch normalisation layer of DenseNet121), which captures high-level spatial features. The resulting heatmap is overlaid on the original X-ray and returned as a base64-encoded PNG, highlighting the lung regions that most influenced the prediction.

---

## Frontend

**Directory**: `venv/Pullmonar/frontend/`  
**Stack**: React 19, Vite 7, Axios, Lucide React

A single-page diagnostic interface with a glassmorphism design and an animated ASCII matrix rain background composed of medical characters (`+`, `╬`, `⊕`, `†`, `░`, …).

### Key Features

- **Drag-and-drop upload** — accepts any image format; previews the X-ray inline before submission
- **Diagnosis panel** — displays the predicted class with a `ShieldCheck` (Normal) or `ShieldAlert` (pathological) icon, a confidence bar, and a glow-pulse animation keyed to the result colour
- **Grad-CAM viewer** — side-by-side comparison of the original X-ray and the attention heatmap, opened on demand
- **Glass button system** — reusable `GlassIcons` component with a shimmer sweep and inner top-edge highlight on hover

### Component Map

```
App.jsx                   ← All application state; upload → predict → Grad-CAM flow
├── AsciiRain.jsx         ← Canvas-based animated background (medical character matrix rain)
├── GlassIcons.jsx        ← Reusable glassmorphism button (used for the main diagnose button)
└── GradCAMViewer.jsx     ← Side-by-side original X-ray + heatmap display
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- (Optional) CUDA-capable GPU — the server runs on CPU by default

### Backend

```bash
# Install Python dependencies
pip install flask flask-cors torch torchvision Pillow pytorch-grad-cam numpy joblib scikit-learn optuna

# Start the basic server (DenseNet only)
cd venv/Pullmonar
python server.py

# Or start the cascade server (DenseNet + Siamese + RF)
python server_upgrade.py
```

The API will be available at `http://127.0.0.1:5000`.

### Frontend

```bash
cd venv/Pullmonar/frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

### API Reference

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/predict` | POST | `multipart/form-data` with `file` field | `{ diagnosis, confidence }` |
| `/gradcam` | POST | *(none — uses last prediction)* | `{ gradcam_image (base64), predicted_class }` |

`server_upgrade.py` also returns `used_siamese_model: bool` and `initial_model_guess: string` in the `/predict` response.

---

## Project Structure

```
Projet PULMONAR/
├── venv/Pullmonar/
│   ├── server.py                  # Flask API — DenseNet only
│   ├── server_upgrade.py          # Flask API — cascade (DenseNet + Siamese + RF)
│   ├── dense_net.py               # Baseline DenseNet121 training
│   ├── dense_net_fine_tune.py     # Fine-tuned DenseNet with early stopping
│   ├── siamese.py                 # Siamese feature extraction + RF training
│   ├── siamese_light.py           # Optimised siamese (Optuna + optimal threshold)
│   ├── viral_bact_test.py         # DenseNet 2-class (Viral vs Bacterial) experiments
│   ├── grad_cam_test.py           # Standalone Grad-CAM visualisation
│   └── frontend/                  # React + Vite application
│
├── ZPlots/
│   ├── baseline_plots/            # Initial training curves, confusion matrix, ROC
│   ├── early_stop_plots/          # Fine-tuned model training plots
│   ├── Random_Forest_plots/       # Siamese RF results (multiple threshold comparisons)
│   ├── vir_bact_plots/            # DenseNet 2-class experiments
│   └── Grad_cam/                  # Grad-CAM overlay examples
│
├── mon_modele_pulmonar_final.pth  # Production DenseNet121 weights
├── chexnet.pth.tar                # CheXNet pre-trained weights (NIH ChestX-ray14)
├── rf_siamese_model.pkl           # Random Forest (Siamese pipeline)
├── siamese_scaler.pkl             # Feature scaler (Siamese pipeline)
│
├── COVID-19/                      # Training data
├── Normal/
├── Pneumonia-Bacterial/
└── Pneumonia-Viral/
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep learning | PyTorch, TorchVision |
| Model backbone | DenseNet121 (CheXNet initialisation) |
| Explainability | pytorch-grad-cam |
| Classical ML | scikit-learn (Random Forest, StandardScaler) |
| Hyperparameter search | Optuna |
| Image processing | Pillow, OpenCV |
| API | Flask, flask-cors |
| Frontend | React 19, Vite 7 |
| HTTP client | Axios |
| Icons | Lucide React |
| Graphics | Canvas API (ASCII rain) |
