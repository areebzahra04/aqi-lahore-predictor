# 🌫️ Pearls AQI Predictor — Lahore

A 100% serverless, end-to-end system that forecasts the Air Quality Index (AQI) for Lahore, Pakistan, up to 3 days in advance. Built as a data science internship project covering the full ML lifecycle: automated data collection, feature engineering, model training, model registry, CI/CD automation, and a live interactive dashboard.

**🔗 Live dashboard:** https://nepgz73usmnjmdfmbmt5wx.streamlit.app/

---

## Overview

Rather than one model predicting 72 hours ahead directly, the forecast is split into **three specialized models** — one each for t+24h, t+48h, and t+72h — since the strength of feature relationships changes with how far ahead you're predicting.

Three model families were trained and compared for every horizon:
- **Ridge Regression** and **RandomForest** (scikit-learn)
- **TensorFlow** — a 5-model neural network ensemble per horizon

The best-performing model per horizon (by R²) is what's deployed live in the dashboard; all models, including the TensorFlow ones, remain registered in the Model Registry for comparison.

## Architecture

```
OpenWeather + Open-Meteo APIs
        │
        ▼
feature_pipeline/fetch_and_store.py  ──(hourly, GitHub Actions)──▶  Hopsworks Feature Store
        │                                                                   │
        │                                                                   ▼
        │                                              training_pipeline/train_model.py
        │                                              (Ridge + RandomForest, daily)
        │                                                                   │
        │                                              training_pipeline/export_training_data.py
        │                                                     │ (CSV handoff)
        │                                                     ▼
        │                                              training_pipeline/train_tf_model.py
        │                                              (TensorFlow, isolated venv)
        │                                                     │
        │                                                     ▼
        │                                              training_pipeline/register_tf_model.py
        │                                                                   │
        │                                                                   ▼
        │                                                     Hopsworks Model Registry
        │                                                                   │
        └───────────────────────────────────────────────────────────────▶  ▼
                                                              dashboard/app.py (Streamlit)
```

## Why the training pipeline is split into 3 CI jobs

TensorFlow and the `hopsworks`/`hsfs` client library pin conflicting versions of `numpy`, `protobuf`, and `grpcio` — they can't reliably coexist in one Python environment. Rather than fight that, the daily training workflow runs as three sequential GitHub Actions jobs, passing data between them as artifacts instead of sharing an environment:

1. **`train-sklearn-and-export`** (hopsworks env) — trains Ridge/RandomForest, registers them, exports a training CSV from the Feature Store
2. **`train-tensorflow`** (isolated TF env, no hopsworks import) — trains a 5-seed ensemble per horizon on the CSV
3. **`register-tensorflow`** (hopsworks env again) — uploads the trained TF ensembles to the Model Registry

## Results

| Horizon | Best model | RMSE | MAE | R² |
|---|---|---|---|---|
| 24h | Ridge Regression | 24.622 | 19.539 | **0.315** |
| 24h | TensorFlow (ensemble) | 26.019 | 21.425 | 0.236 |
| 48h | RandomForest | 26.453 | 21.424 | **0.207** |
| 48h | TensorFlow (ensemble) | 30.414 | 25.427 | -0.048 |
| 72h | RandomForest | 28.228 | 23.063 | **0.093** |
| 72h | TensorFlow (ensemble) | 30.468 | 25.581 | -0.058 |

Classical models outperformed the TensorFlow ensemble at every horizon, with the gap widening as the horizon lengthens — consistent with the limited training data available (~1,300 hourly rows) favoring models with stronger inductive biases on tabular data. See the full project report for details and analysis.

## Repository structure

```
aqi-lahore-predictor/
├── .github/workflows/
│   ├── feature_pipeline.yml      # hourly
│   └── training_pipeline.yml     # daily, 3 jobs (sklearn+export / TF / register)
├── feature_pipeline/
│   └── fetch_and_store.py        # fetches raw data, computes features, writes to Feature Store
├── training_pipeline/
│   ├── train_model.py            # Ridge + RandomForest, registers best per horizon
│   ├── export_training_data.py   # dumps Feature Store data to CSV for the TF step
│   ├── train_tf_model.py         # TensorFlow ensemble training (isolated env)
│   └── register_tf_model.py      # uploads TF ensembles to Model Registry
├── dashboard/
│   └── app.py                    # Streamlit dashboard
├── models/                       # locally cached best model per horizon (.pkl)
├── requirements.txt              # hopsworks / sklearn / dashboard dependencies
├── requirements-tf.txt           # isolated TensorFlow dependencies
├── runtime.txt                   # pinned Python version for Streamlit Cloud
└── .env                          # API keys (not committed — see below)
```

## Tech stack

- **Data sources:** OpenWeather Air Pollution API, Open-Meteo API
- **Feature Store & Model Registry:** Hopsworks (free tier)
- **Modeling:** scikit-learn (Ridge, RandomForest), TensorFlow/Keras
- **Explainability:** SHAP
- **CI/CD:** GitHub Actions
- **Dashboard:** Streamlit
- **Deployment:** Streamlit Community Cloud

## Running locally

This project uses **two separate virtual environments** — one for everything Hopsworks-related, one isolated for TensorFlow — because their dependencies conflict (see above).

### 1. Main environment (feature pipeline, classical models, dashboard)

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```
HOPSWORKS_API_KEY=your_key_here
```

Run the feature pipeline once to backfill data:
```bash
python feature_pipeline/fetch_and_store.py
```

Train the classical models:
```bash
python training_pipeline/train_model.py
```

Export data for TensorFlow:
```bash
python training_pipeline/export_training_data.py
```

Run the dashboard:
```bash
streamlit run dashboard/app.py
```

### 2. TensorFlow environment

```bash
python -m venv venv-tf
venv-tf\Scripts\activate
pip install -r requirements-tf.txt

python training_pipeline/train_tf_model.py
```

Then switch back to the main venv to register the trained TF models:
```bash
deactivate
venv\Scripts\activate
python training_pipeline/register_tf_model.py
```

## Automation

Both GitHub Actions workflows are schedule-driven and require repo secrets:
- `HOPSWORKS_API_KEY`

Set these under **Settings → Secrets and variables → Actions** in your GitHub repo.

- `feature_pipeline.yml` runs hourly
- `training_pipeline.yml` runs daily, across its 3 sequential jobs

Both can also be triggered manually from the **Actions** tab via "Run workflow."

## Limitations & future work

- Training data is limited to a few months of history; a longer collection window would likely improve all models, especially TensorFlow
- A sequence model (LSTM/GRU) using recent AQI history directly, rather than flat rolling-average features, is the natural next step for the TensorFlow approach
- Hopsworks free-tier budget limits constrain how much historical backfill and how frequent retraining is sustainable long-term