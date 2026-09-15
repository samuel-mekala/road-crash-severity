# 🚗 Road Crash Injury Severity Prediction System

> **VIT-AP Senior Design Project** · May 2025  
> **Authors:** Sri Hari Priya Panchumarthi, **Samuel Mekala**, Sai Pranav Kothapalli, Meghana Bindem  
> **Guide:** Dr. Deepthi Godavarthi · School of Computer Science and Engineering (SCOPE), VIT-AP University  

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![GitHub](https://img.shields.io/badge/GitHub-samuel--mekala%2Froad--crash--severity-181717?style=flat-square&logo=github)](https://github.com/samuel-mekala/road-crash-severity)

---

## 📌 Project Overview

Traffic accidents claim millions of lives annually and cause substantial economic loss. Immediate, accurate prediction of injury severity immediately following a crash is essential for optimizing emergency medical service (EMS) dispatch, regional hospital preparedness, and traffic safety management.

This project implements a **Hybrid Machine Learning & Spatio-Temporal Deep Learning Architecture** combining:
1. **ExtraTreesClassifier (ETC)**: Captures complex high-dimensional feature interactions across vehicle count, casualties, speed limit, road type, weather, and light conditions.
2. **Spatio-Temporal Graph Neural Network (ST-GNN)**: Captures spatial proximity and temporal crash clustering using k-Nearest Neighbors (kNN) graph representation with 3-layer Graph Convolutional Networks (GCN) + Dropout regularization.
3. **Logistic Regression Meta-Classifier**: Combines predictions from both paradigms to achieve a peak classification accuracy of **96.46%** (Precision: 0.97, Recall: 0.96, F1-Score: 0.96, ROC-AUC: 0.91).

---

## 🏗️ System Architecture & Model Diagrams

### ST-GNN Architecture & Flow
![ST-GNN Sequential Architecture](images/stgnn.png)

![ST-GNN Sequence Flow](images/stgnn_flow.png)

---

### ExtraTrees Classifier Architecture
![ETC Architecture](images/ETC.png)

---

### Hybrid Model Framework (ST-GNN + ETC Meta-Classifier)
![Hybrid Model Framework](images/hybrid_model.png)

---

## 📈 Performance & Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **ST-GNN (Standalone)** | 85.26% | 0.73 | 0.85 | 0.78 | 0.83 |
| **ExtraTrees (Standalone)** | 92.31% | 0.88 | 0.85 | 0.90 | 0.88 |
| **Hybrid Ensemble (Meta-Classifier)** | **96.46%** | **0.97** | **0.96** | **0.96** | **0.91** |

---

## 🚀 Features & Interactive Web App

- **Interactive Crash Assessment Dashboard**: Real-time form for entering crash attributes (vehicles, casualties, speed limit, weather, light, road surface, urban/rural area).
- **Preset Crash Scenarios**: One-click preset loaders for *Highway Pile-up*, *Urban Intersection Incident*, and *Minor Rural Collision*.
- **Severity Classification**: Dynamic severity badges (**Slight**, **Serious**, **Fatal**).
- **Crash Risk Metric Score**: Calculated 0 - 100% crash risk index.
- **Ensemble Model Confidence Breakdown**: Visual progress bars showing probability distributions for Slight, Serious, and Fatal classes.
- **Emergency Response Protocol**: Actionable safety and emergency guidelines (ALS ambulance dispatch, trauma center pre-notification, highway lockdown, air ambulance alerts).

---

## 🛠️ Installation & Local Setup

```bash
# 1. Clone the Repository
git clone https://github.com/samuel-mekala/road-crash-severity.git
cd road-crash-severity

# 2. Create and Activate Virtual Environment (Recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. (Optional) Download UK Road Accident Dataset
# Place UK_Accident.csv in data/ directory to re-train models from scratch
python3 train.py

# 5. Launch the Flask Web Dashboard
python3 app.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 🧪 Running Automated Unit & Model Tests

```bash
python3 test_app.py
```

---

## 🌐 Deploying to Cloud Platforms (Render / Railway / Hugging Face)

### Deploying to Render
1. Push your repository to GitHub:
   ```bash
   git add .
   git commit -m "Restore architecture diagrams in README and deploy"
   git push origin main
   ```
2. Log in to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Web Service** → Connect your GitHub repository `samuel-mekala/road-crash-severity`.
4. Render will automatically detect `render.yaml` and `Procfile` (`web: gunicorn wsgi:app`).
5. Click **Deploy Web Service**!

---

## 📁 Repository Structure

```
road-crash-severity/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions Continuous Integration
├── images/                    # System architecture & flow diagrams
│   ├── ETC.png
│   ├── hybrid_model.png
│   ├── stgnn.png
│   └── stgnn_flow.png
├── models/                    # Serialized ML & PyTorch model artifacts
│   ├── etc_model.pkl
│   ├── meta_classifier.pkl
│   ├── scaler.pkl
│   └── stgnn_model.pt
├── static/
│   └── css/
│       └── style.css          # Custom Dashboard CSS Styling
├── templates/
│   └── index.html             # Interactive Web Dashboard Template
├── analysis.py                # Dataset Analysis & Exploratory Data Analysis (EDA)
├── app.py                     # Flask Web Application Server
├── predict.py                 # Real-time Inference & Emergency Recommendation Engine
├── train.py                   # Model Training Pipeline (ST-GNN + ETC + Meta-Classifier)
├── test_app.py                # Automated Test Suite
├── wsgi.py                    # Gunicorn Production WSGI Entry Point
├── Procfile                   # Process file for Gunicorn execution
├── render.yaml                # Render Cloud Deployment Config
├── requirements.txt           # Python Dependencies
├── .gitignore                 # Git Exclusions
└── README.md                  # System Documentation
```

---

## 📜 Citation & Credits

*Senior Design Project Report — May 2025*  
**School of Computer Science and Engineering (SCOPE), VIT-AP University**  
*Project Team:* Sri Hari Priya Panchumarthi, Samuel Mekala, Sai Pranav Kothapalli, Meghana Bindem  
*Supervisor:* Dr. Deepthi Godavarthi  
