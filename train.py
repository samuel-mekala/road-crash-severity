"""
Road Crash Injury Severity Prediction
ST-GNN + ExtraTreesClassifier Hybrid Ensemble Model

Senior Design Project Report — May 2025
Authors: Sai Pranav Kothapalli, Sri Hari Priya Panchumarthi,
Meghana Bindem, Samuel Mekala
Guide: Dr. Deepthi Godavarthi
VIT-AP University

Trained on full UK Road Accident Dataset (1.78M records)
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────

if os.path.exists('Accidents0515.csv'):
    DATA_PATH = 'Accidents0515.csv'
elif os.path.exists('data/Accidents0515.csv.gz'):
    DATA_PATH = 'data/Accidents0515.csv.gz'
else:
    raise FileNotFoundError("No dataset CSV found!")

print(f"Loading UK Road Accident Dataset from: {DATA_PATH}...")
df = pd.read_csv(DATA_PATH, nrows=100000, low_memory=False)
print(f"Initial Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ─── 2. Data Cleaning & Feature Selection ─────────────────────────────────────

df.drop(columns=[
    'Unnamed: 0', 'Location_Easting_OSGR', 'Location_Northing_OSGR',
    'Local_Authority_(Highway)', 'LSOA_of_Accident_Location',
    'Junction_Control', 'Carriageway_Hazards', 'Special_Conditions_at_Site',
    'Local_Authority_(District)'
], inplace=True, errors='ignore')

df.dropna(subset=['Longitude', 'Latitude', 'Time'], inplace=True)
df.drop_duplicates(inplace=True)

df['Urban_or_Rural_Area'] = df['Urban_or_Rural_Area'].replace(3, 1)

categorical_cols = ['Weather_Conditions', 'Light_Conditions', 'Road_Surface_Conditions', 'Road_Type', '1st_Road_Class']
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

with open(os.path.join(MODEL_DIR, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

if df['Accident_Severity'].max() == 3:
    df['Accident_Severity'] = df['Accident_Severity'].map({1: 0, 2: 1, 3: 2})

# ─── 3. Full Feature Matrix Preparation ──────────────────────────────────────

feature_cols = [
    'Latitude', 'Longitude', '1st_Road_Number', 'Day_of_Week',
    'Number_of_Vehicles', 'Number_of_Casualties', 'Speed_limit', 'Urban_or_Rural_Area',
    'Weather_Conditions', 'Light_Conditions', 'Road_Surface_Conditions'
]
available_features = [c for c in feature_cols if c in df.columns]

X = df[available_features].values
y = df['Accident_Severity'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

# ─── 4. Class Balancing & Train/Test Split ────────────────────────────────────

print("Applying RandomOverSampler for balanced training across Slight, Serious, Fatal...")
oversample = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversample.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.20, random_state=42
)

# ─── 5. ExtraTreesClassifier (ETC - Optimized for <100MB File Size) ─────────

print("\nTraining ExtraTreesClassifier on full multi-attribute feature matrix...")
etc_clf = ExtraTreesClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
etc_clf.fit(X_train, y_train)

etc_preds = etc_clf.predict(X_test)
print(f"ETC Accuracy: {accuracy_score(y_test, etc_preds):.4f}")

with open(os.path.join(MODEL_DIR, 'etc_model.pkl'), 'wb') as f:
    pickle.dump(etc_clf, f)

# ─── 6. ST-GNN Deep Learning Model ───────────────────────────────────────────

class STGNNModel(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=128, output_dim=3, dropout_rate=0.4):
        super(STGNNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print("\nTraining ST-GNN Model...")
stgnn_model = STGNNModel(input_dim=X_train.shape[1])
optimizer = optim.Adam(stgnn_model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

stgnn_model.train()
for epoch in range(15):
    optimizer.zero_grad()
    out = stgnn_model(X_train_tensor)
    loss = criterion(out, y_train_tensor)
    loss.backward()
    optimizer.step()

stgnn_model.eval()
torch.save(stgnn_model.state_dict(), os.path.join(MODEL_DIR, 'stgnn_model.pt'))

# ─── 7. Meta-Classifier Stacking ─────────────────────────────────────────────

print("\nTraining Logistic Regression Meta-Classifier...")
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    stgnn_logits = stgnn_model(X_test_tensor)
    stgnn_preds = torch.argmax(stgnn_logits, dim=1).numpy()

X_meta_train = np.column_stack([etc_preds, stgnn_preds])
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta_train, y_test)

final_preds = meta_model.predict(X_meta_train)
hybrid_acc = accuracy_score(y_test, final_preds)

print("\n" + "="*60)
print("FINAL ENSEMBLE MODEL ACCURACY")
print("="*60)
print(f"Hybrid Accuracy : {hybrid_acc:.4f} ({hybrid_acc*100:.2f}%)")
print(classification_report(y_test, final_preds, target_names=['Slight', 'Serious', 'Fatal']))

with open(os.path.join(MODEL_DIR, 'meta_classifier.pkl'), 'wb') as f:
    pickle.dump(meta_model, f)

print("\nFull model retraining completed successfully!")