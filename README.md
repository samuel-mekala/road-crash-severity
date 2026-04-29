# 🚗 Road Crash Injury Severity Prediction

> **Academic Project** · VIT-AP University · Jan–Apr 2024  
> **Team Size:** 4 · **Role: Team Lead**

---

## 📌 Overview

Road accidents claim millions of lives annually. Accurate prediction of injury severity immediately after a crash can significantly improve emergency response and resource allocation. This project develops a **hybrid machine learning model** that combines **Spatio-Temporal Graph Neural Networks (ST-GNN)** with an **ExtraTreesClassifier** to predict crash injury severity from real-world accident data.

**Achieved 96.46% accuracy** — one of the highest reported for this class of problem.

---

## 🏆 Results

| Metric | Score |
|---|---|
| **Accuracy** | **96.46%** |
| **Recall** | High |
| **F1-Score** | High |

---

## 🧠 Model Architecture

### Hybrid Approach: ST-GNN + ExtraTrees

```
Real-World Accident Data
        ↓
  Preprocessing (cleaning, feature selection,
  label encoding, normalization, oversampling)
        ↓
┌──────────────────────────────┐
│  Spatio-Temporal GNN (ST-GNN)│  ← Captures geographic & temporal crash patterns
│  (PyTorch Geometric)         │
└──────────────────────────────┘
        +
┌──────────────────────────────┐
│  ExtraTreesClassifier        │  ← Feature-based ensemble learning
│  (Scikit-learn)              │
└──────────────────────────────┘
        ↓
  Hybrid Prediction: Injury Severity Class
```

**Why this hybrid?**
- **ST-GNN** captures *where* and *when* crashes cluster — spatial proximity (dangerous intersections) and temporal patterns (rush hour, weekends)
- **ExtraTrees** captures *feature-level* patterns — vehicle type, speed, road condition, weather
- Together, they cover both **context** and **attributes**

---

## ⚙️ Preprocessing Pipeline

```
Raw Accident Dataset
    → Data Cleaning (remove nulls, fix formats)
    → Feature Selection (drop redundant columns)
    → Label Encoding (categorical → numeric)
    → Normalization (StandardScaler)
    → Class Imbalance Handling (oversampling)
    → Graph Construction (spatial adjacency matrix)
    → Train/Test Split
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

---

## 📁 Project Structure

```
road-crash-severity/
├── data/
│   └── accident_data.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── preprocessing.py       # Cleaning, encoding, oversampling
│   ├── graph_builder.py       # Build spatial graph for ST-GNN
│   ├── stgnn_model.py         # Spatio-Temporal GNN architecture
│   ├── extratrees_model.py    # ExtraTrees classifier
│   └── hybrid_model.py        # Combine ST-GNN + ExtraTrees
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/samuel-mekala/road-crash-severity.git
cd road-crash-severity

# Install dependencies
pip install -r requirements.txt

# Preprocess data
python src/preprocessing.py

# Train the hybrid model
python train.py

# Evaluate
python evaluate.py
```

---

## 🔮 Future Work

- [ ] Real-time prediction API for emergency services
- [ ] Integration with live traffic data streams
- [ ] Expand to national highway networks
- [ ] Explainability with SHAP for black-box transparency

---

*VIT-AP University · Jan–Apr 2024 · Team Lead*
