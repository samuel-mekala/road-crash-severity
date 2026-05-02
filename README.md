# 🚗 Road Crash Injury Severity Prediction

> **Senior Design Project** · VIT-AP University · May 2025  
> **Team:** Sri Hari Priya Panchumarthi · **Samuel Mekala**  
> **Guide:** Dr. Deepthi Godavarthi · School of Computer Science and Engineering (SCOPE)

-----

## 📌 Overview

Road accidents claim millions of lives annually. Accurate prediction of injury severity immediately after a crash can significantly improve emergency response and resource allocation. This project develops a **hybrid machine learning model** that combines **Spatio-Temporal Graph Neural Networks (ST-GNN)** with an **ExtraTreesClassifier** to predict crash injury severity from real-world accident data.

**Achieved 96.46% accuracy** — one of the highest reported for this class of problem.

-----

## 🏆 Results

|Metric       |Score     |
|-------------|----------|
|**Accuracy** |**96.46%**|
|**Precision**|**97%**   |
|**Recall**   |**96%**   |
|**F1-Score** |**96%**   |
|**ROC-AUC**  |**0.91**  |

-----

## 📊 Dataset

**UK Road Accident Dataset** (Kaggle) — 1.5M+ real accident records with features including:

- Location (latitude/longitude), date/time, road conditions
- Vehicle type, speed limit, junction detail
- Weather conditions, light conditions
- **Target:** Accident Severity (1 = Slight, 2 = Serious, 3 = Fatal)

-----

## 🧠 Model Architecture

### Hybrid Approach: ST-GNN + ExtraTrees

```
Real-World Accident Data (UK, 1.5M+ records)
        ↓
  Preprocessing (cleaning, feature selection,
  label encoding, normalization, SMOTE oversampling)
        ↓
┌──────────────────────────────┐
│  Spatio-Temporal GNN (ST-GNN)│  ← Captures geographic & temporal crash patterns
│  (PyTorch Geometric)         │    kNN graph on lat/lon + time proximity
└──────────────────────────────┘
        +
┌──────────────────────────────┐
│  ExtraTreesClassifier        │  ← Feature-based ensemble learning
│  (Scikit-learn, 200 trees)   │
└──────────────────────────────┘
        ↓
  Logistic Regression Meta-Classifier (stacked ensemble)
        ↓
  Injury Severity Prediction: Slight / Serious / Fatal
```

**Why this hybrid?**

- **ST-GNN** captures *where* and *when* crashes cluster — spatial proximity (dangerous intersections) and temporal patterns (rush hour, weekends)
- **ExtraTrees** captures *feature-level* patterns — vehicle type, speed, road condition, weather
- Together, they cover both **context** and **attributes**

-----

## ⚙️ Preprocessing Pipeline

```
Raw Accident Dataset (1.5M+ records)
    → Drop redundant identifier columns
    → Drop columns with >40% missing values
    → Impute numeric with median
    → Label Encoding (categorical → numeric)
    → StandardScaler normalization
    → SMOTE oversampling (handle class imbalance)
    → Train/Test Split (80/20)
    → kNN Graph Construction (k=5, spatial proximity)
```

-----

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

-----

## 📁 Project Structure

```
road-crash-severity/
├── analysis.py         # EDA — boxplots, correlation heatmap, count plots (Appendix 1)
├── train.py            # Full pipeline: preprocessing → ST-GNN → ExtraTrees → ensemble (Appendix 2)
├── requirements.txt
└── README.md
```

> After running `train.py`, confusion matrix saved to `confusion_matrix.png` and feature importance to `feature_importance.png`.

-----

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/samuel-mekala/road-crash-severity.git
cd road-crash-severity

# Install dependencies
pip install -r requirements.txt

# Download UK Road Accident Dataset from Kaggle → place as data/UK_Accident.csv
# https://www.kaggle.com/datasets/silicon99/dft-accident-data

# Run EDA and visualisations
python analysis.py

# Train the full hybrid model (ST-GNN + ETC + Logistic Regression)
python train.py
```

-----

## 🔮 Future Work

- [ ] Real-time prediction API for emergency services
- [ ] Integration with live traffic data streams
- [ ] Expand to national highway networks
- [ ] Explainability with SHAP for black-box transparency

-----

*VIT-AP University · SCOPE · Senior Design Project · May 2025*