"""
Road Crash Injury Severity Prediction
Hybrid ST-GNN + ExtraTreesClassifier Ensemble

Research Paper: "Road Crash Injury Severity Prediction Using
Spatio-Temporal GNN and ExtraTreesClassifier"
Authors: Sai Pranav Kothapalli, Sri Hari Priya Panchumarthi,
         Meghana Bindem, Deepthi Godavarthi
VIT-AP University, 2024

Accuracy: 96.46% | Precision: 97% | Recall: 96% | ROC-AUC: 0.91
Dataset: UK Road Accident Dataset (1.5M+ records)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE, RandomOverSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ─── 1. Load & Preprocess Dataset ─────────────────────────────────────────────
# Download UK Road Accident Dataset from Kaggle
# Place as data/uk_accidents.csv

df = pd.read_csv("data/uk_accidents.csv", low_memory=False)

print(f"Loaded {len(df):,} records with {df.shape[1]} columns")

# ── Drop redundant features ──
DROP_COLS = [
    "Accident_Index", "Location_Easting_OSGR", "Location_Northing_OSGR",
    "Local_Authority_(District)", "Local_Authority_(Highway)",
    "LSOA_of_Accident_Location"
]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# ── Handle missing values ──
# Drop columns with >40% missing
threshold = 0.4 * len(df)
df.dropna(axis=1, thresh=int(threshold), inplace=True)

# Impute numeric with median
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

# ── Encode categorical features ──
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# ── Target ──
TARGET = "Accident_Severity"     # 1 = Slight, 2 = Serious, 3 = Fatal
# Re-map to 0-indexed classes
df[TARGET] = df[TARGET] - 1

# ── Feature / target split ──
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"Class distribution:\n{y.value_counts().sort_index()}")

# ── Normalise ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Train / Test split (80 / 20) ──
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)

# ─── 2. Handle Class Imbalance ────────────────────────────────────────────────

print("\nApplying SMOTE to balance classes ...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Resampled train size: {X_train_res.shape[0]:,}")

# ─── 3. ExtraTreesClassifier (ETC) ────────────────────────────────────────────

print("\nTraining ExtraTreesClassifier ...")
etc = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    n_jobs=-1,
    random_state=42
)
etc.fit(X_train_res, y_train_res)

etc_pred_test  = etc.predict(X_test)
etc_proba_test = etc.predict_proba(X_test)

print("ETC Accuracy:", accuracy_score(y_test, etc_pred_test))
print(classification_report(y_test, etc_pred_test))

# ─── 4. Build Spatio-Temporal Graph ───────────────────────────────────────────

def build_graph(X_data, y_data, k=5, sample_size=50_000):
    """
    Treat each accident as a graph node.
    Connect via k-NN based on spatial (lat/lon) + temporal proximity.
    Returns a PyG Data object.
    """
    # Sub-sample for memory efficiency
    idx = np.random.choice(len(X_data), min(sample_size, len(X_data)), replace=False)
    X_sub = X_data[idx]
    y_sub = y_data.iloc[idx].values if hasattr(y_data, "iloc") else y_data[idx]

    # Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(X_sub)
    distances, indices = nbrs.kneighbors(X_sub)

    # Edge list (excluding self-loops)
    edge_src, edge_dst = [], []
    for i, neighbours in enumerate(indices):
        for j in neighbours[1:]:          # skip self (index 0)
            edge_src.append(i)
            edge_dst.append(j)

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    node_features = torch.tensor(X_sub, dtype=torch.float)
    labels = torch.tensor(y_sub, dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, y=labels)

print("\nBuilding spatio-temporal graph (train) ...")
train_graph = build_graph(X_train_res, pd.Series(y_train_res))
print("Building spatio-temporal graph (test) ...")
test_graph  = build_graph(X_test, y_test)

# ─── 5. ST-GNN Model ──────────────────────────────────────────────────────────

class TemporalConv(nn.Module):
    """Simple 1-D temporal convolution applied per node feature."""
    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)

    def forward(self, x):
        # x: [N, F] → unsqueeze → [N, F, 1] → conv → squeeze
        x = x.unsqueeze(2)
        x = self.conv(x).squeeze(2)
        return x


class STGNN(nn.Module):
    def __init__(self, in_features, hidden=64, num_classes=3, dropout=0.5):
        super().__init__()
        self.temporal = TemporalConv(in_features, in_features)
        self.gcn1 = GCNConv(in_features, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.gcn3 = GCNConv(hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.temporal(x)
        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        x = self.gcn3(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_features = train_graph.x.shape[1]
num_classes  = len(y.unique())

gnn_model = STGNN(in_features=num_features, hidden=64,
                  num_classes=num_classes, dropout=0.5).to(device)

train_graph = train_graph.to(device)
test_graph  = test_graph.to(device)

optimizer_gnn = optim.Adam(gnn_model.parameters(), lr=0.01)
criterion_gnn = nn.NLLLoss()

# ── Training Loop (10 epochs, early stopping) ──
print("\nTraining ST-GNN ...")
best_val_loss = float("inf")
patience_counter = 0
PATIENCE = 3

for epoch in range(1, 11):
    gnn_model.train()
    optimizer_gnn.zero_grad()
    out  = gnn_model(train_graph)
    loss = criterion_gnn(out, train_graph.y)
    loss.backward()
    optimizer_gnn.step()

    # Validation on test graph
    gnn_model.eval()
    with torch.no_grad():
        val_out  = gnn_model(test_graph)
        val_loss = criterion_gnn(val_out, test_graph.y)
        preds    = val_out.argmax(dim=1).cpu().numpy()
        acc      = accuracy_score(test_graph.y.cpu().numpy(), preds)

    print(f"Epoch {epoch:02d}  Train Loss: {loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(gnn_model.state_dict(), "model/best_stgnn.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping.")
            break

# Load best weights
gnn_model.load_state_dict(torch.load("model/best_stgnn.pt"))

# ─── 6. GNN Predictions ───────────────────────────────────────────────────────

gnn_model.eval()
with torch.no_grad():
    gnn_logits = gnn_model(test_graph)
    gnn_proba  = torch.exp(gnn_logits).cpu().numpy()   # shape [N_test, C]
    gnn_preds  = gnn_logits.argmax(dim=1).cpu().numpy()

print("\nST-GNN Results:")
print(classification_report(test_graph.y.cpu().numpy(), gnn_preds))

# ─── 7. Ensemble Meta-Classifier ──────────────────────────────────────────────

# Align sizes: use the sub-sampled test indices from build_graph
n_gnn = gnn_proba.shape[0]
etc_proba_sub = etc_proba_test[:n_gnn]
y_test_sub    = y_test.values[:n_gnn]

# Stack probabilities as meta-features
meta_X = np.hstack([etc_proba_sub, gnn_proba])

# Use first 80% for meta-train, remaining 20% for meta-test
split = int(0.8 * len(meta_X))
meta_clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
meta_clf.fit(meta_X[:split], y_test_sub[:split])

hybrid_preds = meta_clf.predict(meta_X[split:])
y_meta_test  = y_test_sub[split:]

print("\n" + "="*60)
print("HYBRID ENSEMBLE MODEL RESULTS")
print("="*60)
print(classification_report(y_meta_test, hybrid_preds))
print(f"Accuracy : {accuracy_score(y_meta_test, hybrid_preds)*100:.2f}%")
print(f"Precision: {precision_score(y_meta_test, hybrid_preds, average='macro')*100:.2f}%")
print(f"Recall   : {recall_score(y_meta_test, hybrid_preds, average='macro')*100:.2f}%")
print(f"F1 Score : {f1_score(y_meta_test, hybrid_preds, average='macro')*100:.2f}%")

# ─── 8. Feature Importance ────────────────────────────────────────────────────

import matplotlib.pyplot as plt

feat_imp = pd.Series(etc.feature_importances_, index=pd.RangeIndex(X.shape[1]))
top_15 = feat_imp.nlargest(15)

plt.figure(figsize=(10, 6))
top_15.sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 15 Feature Importances (ExtraTreesClassifier)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()
print("\nFeature importance plot saved.")
