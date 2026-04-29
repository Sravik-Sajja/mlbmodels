# Hit Oracle ⚾

A machine learning app that predicts whether a ball in play results in a hit — and if so, what kind — using MLB Statcast data.

**Live demo:** [hitoracle.vercel.app](https://hit-oracle.vercel.app/)

---

## What it does

Given four Statcast inputs — hit coordinates (hc_x, hc_y), exit velocity, and launch angle — Hit Oracle returns:

- **Hit probability** (0–100%)
- **Out probability**
- **Outcome breakdown** — if a hit, the probability split across Single, Double, Triple, and Home Run

The field panel lets you click directly on an SVG baseball diamond to auto-fill coordinates, or enter them manually.

---

## How it works

### Models

Two XGBoost classifiers are trained on 2025 MLB Statcast data and served via a Flask API:

| Model | Task | Output |
|---|---|---|
| `hit_xgb.json` | Binary classification | Hit vs. Out probability |
| `bases_xgb.json` | Multi-class classification | Probabilities across 0–4 bases |

Neural network versions (PyTorch) are also trained for comparison but the XGBoost models are what's deployed — they're faster to load and perform comparably.

Class imbalance (outs >> hits, singles >> triples) is handled with `scale_pos_weight` for the binary model and `compute_class_weight` with cube-root dampening for the multi-class model.

### Features

```
hc_x          — horizontal hit coordinate (Statcast)
hc_y          — vertical hit coordinate (Statcast)
launch_speed  — exit velocity (mph)
launch_angle  — vertical launch angle (degrees)
```

### Data

Fetched via `pybaseball.statcast()` for the 2025 regular season. Only `hit_into_play` events are used. The `num_bases` label is derived from the `events` column (single=1, double=2, triple=3, home_run=4, otherwise 0).

---

## Project structure

```
├── data/
│   └── prepare_data.py       # Statcast fetch, feature engineering, train/test split
├── train/
│   ├── models.py             # HitNN and BasesNN PyTorch model definitions
│   ├── train_models.py       # Training loop for NN and XGBoost models
│   ├── evaluate_models.py    # Evaluation helpers (F1, ROC-AUC, classification report)
│   └── saved_models/         # Trained model files (git-ignored except .json)
├── frontend/
│   ├── index.html            # Main UI
│   ├── field.js              # SVG baseball field rendering and coordinate mapping
│   ├── predict.js            # API call and results rendering
│   └── styles.css            # Styles
├── server.py                 # Flask API server
├── requirements.txt
├── Procfile                  # For Railway deployment
└── vercel.json               # For Vercel static frontend deployment
```

---

## Running locally

**Backend**

```bash
pip install -r requirements.txt
python server.py
```

The API runs on `http://localhost:5000`.

**Frontend**

Update the fetch URL in `frontend/predict.js` to point to `http://localhost:5000/predict`, then open `frontend/index.html` in a browser (or serve it with any static server).

**Retraining**

```bash
cd train && python train_models.py
```

Trained models are saved to `train/saved_models/`.

---

## API

**POST** `/predict`

```json
{
  "hc_x": 120.0,
  "hc_y": 165.0,
  "launch_speed": 95.2,
  "launch_angle": 18.5
}
```

**Response**

```json
{
  "hit_probability": 67.3,
  "out_probability": 32.7,
  "bases_breakdown": {
    "Single": 48.2,
    "Double": 31.5,
    "Triple": 4.1,
    "HR": 16.2
  }
}
```

Rate limited to 30 requests/minute

---

## Deployment

| Layer | Platform |
|---|---|
| Frontend | Vercel (static) |
| API | Railway |

The frontend is purely static and deployed via `vercel.json`. The backend runs via `Procfile` (`python server.py`) on Railway with the `PORT` environment variable set automatically.