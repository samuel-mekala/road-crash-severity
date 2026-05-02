“””
Road Crash Injury Severity Prediction
ST-GNN + ExtraTreesClassifier Hybrid Ensemble Model

Appendix 2 — Senior Design Project Report
Authors: Sai Pranav Kothapalli, Sri Hari Priya Panchumarthi,
Meghana Bindem, Samuel Mekala
Guide: Dr. Deepthi Godavarthi
VIT-AP University, May 2025

Results:
Hybrid Model Accuracy : 96.46%
Precision             : 97%
Recall                : 96%
ROC-AUC               : 0.91

Dataset: UK Road Accident Dataset (1.5M+ records)

- Download from Kaggle and place as data/UK_Accident.csv
  “””

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
confusion_matrix)
from imblearn.over_sampling import RandomOverSampler, SMOTE

# ─── 1. Load Dataset ──────────────────────────────────────────────────────────

df = pd.read_csv(‘data/UK_Accident.csv’, parse_dates=[‘Date’, ‘Time’])

# ─── 2. Drop Unnecessary Columns ─────────────────────────────────────────────

df.drop(columns=[
‘Unnamed: 0’, ‘Location_Easting_OSGR’, ‘Location_Northing_OSGR’,
‘Local_Authority_(Highway)’, ‘LSOA_of_Accident_Location’
], inplace=True, errors=‘ignore’)

df.drop(columns=[
‘Junction_Control’, ‘Carriageway_Hazards’, ‘Special_Conditions_at_Site’
], inplace=True, errors=‘ignore’)

print(“No. of rows: {}”.format(df.shape[0]))
print(“No. of cols: {}”.format(df.shape[1]))

# ─── 3. Handle Missing Values ────────────────────────────────────────────────

df.dropna(subset=[
‘Longitude’, ‘Time’,
‘Pedestrian_Crossing-Human_Control’,
‘Pedestrian_Crossing-Physical_Facilities’
], inplace=True)

# ─── 4. Remove Duplicates ────────────────────────────────────────────────────

dup_rows = df[df.duplicated()]
print(“Duplicate rows:”, dup_rows.shape[0])
df.drop_duplicates(inplace=True)
print(“Rows remaining:”, df.shape[0])

# ─── 5. Feature Engineering ──────────────────────────────────────────────────

# Identify categorical and numerical columns

categorical_data = df.select_dtypes(include=‘object’)
cat_cols = categorical_data.columns
print(“Categorical columns:”, len(cat_cols))

numerical_data = df.select_dtypes(include=‘number’)
num_cols = numerical_data.columns
print(“Numerical columns:”, len(num_cols))

# Drop highly correlated feature (>80% with others — from EDA)

df.drop(columns=[‘Local_Authority_(District)’], axis=1, inplace=True, errors=‘ignore’)

# Fix Urban_or_Rural_Area (replace 3 → 1)

df[‘Urban_or_Rural_Area’].replace(3, 1, inplace=True)

# Label encode all categorical columns

labelencoder = LabelEncoder()
for feature in cat_cols:
if feature in df.columns:
df[feature] = labelencoder.fit_transform(df[feature])

# Drop Accident_Index (identifier — no predictive value)

if ‘Accident_Index’ in df.columns:
df.drop(‘Accident_Index’, axis=1, inplace=True)

# Drop Year (no significance to severity prediction)

if ‘Year’ in df.columns:
df.drop(‘Year’, axis=1, inplace=True)

# Re-map target: 1,2,3 → 0,1,2

df[‘Accident_Severity’] = df[‘Accident_Severity’].map({1: 0, 2: 1, 3: 2})
num_classes = df[‘Accident_Severity’].nunique()
print(“Classes:”, num_classes)

# ─── 6. Prepare Features for ETC ─────────────────────────────────────────────

# Use subset of columns matching the report’s feature selection

dfnew = df[[‘Latitude’, ‘Longitude’, ‘1st_Road_Number’,
‘Day_of_Week’, ‘Accident_Severity’]]

features = [col for col in dfnew.columns if col != ‘Accident_Severity’]
x = dfnew.iloc[0:50000, :-1]
y = dfnew.iloc[0:50000, [-1]]

x = StandardScaler().fit_transform(x)

# ─── 7. Handle Class Imbalance ───────────────────────────────────────────────

oversample = RandomOverSampler()
x, y = oversample.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.25, random_state=0
)
y_train = np.ravel(y_train)
y_test  = np.ravel(y_test)

# ─── 8. ExtraTreesClassifier (ETC) ───────────────────────────────────────────

print(”\nTraining ExtraTreesClassifier…”)
clf = ExtraTreesClassifier()
clf.fit(x_train, y_train)
etc_preds = clf.predict(x_test)
etc_preds = np.array(etc_preds).flatten()

print(f”ETC Accuracy: {accuracy_score(y_test, etc_preds):.4f}”)
print(classification_report(y_test, etc_preds))

# ─── 9. Graph Construction for ST-GNN ────────────────────────────────────────

def preprocess_graph_data(df):
df = df.copy()

```
# Encode any remaining categorical columns
categorical_cols = [
    'Day_of_Week', '1st_Road_Class', 'Road_Type',
    'Weather_Conditions', 'Light_Conditions',
    'Road_Surface_Conditions', 'Urban_or_Rural_Area',
    'Did_Police_Officer_Attend_Scene_of_Accident'
]
for col in categorical_cols:
    if col in df.columns and df[col].dtype == 'object':
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Convert datetime columns to timestamps
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date']).astype(int) // 10**9
if 'Time' in df.columns:
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').astype(int) // 10**9

# Select relevant features for graph
feature_cols = [
    'Longitude', 'Latitude', 'Police_Force', 'Accident_Severity',
    'Number_of_Vehicles', 'Number_of_Casualties', 'Date', 'Day_of_Week',
    'Time', '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',
    '2nd_Road_Class', '2nd_Road_Number',
    'Pedestrian_Crossing-Human_Control',
    'Pedestrian_Crossing-Physical_Facilities', 'Light_Conditions',
    'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area',
    'Did_Police_Officer_Attend_Scene_of_Accident'
]
available = [c for c in feature_cols if c in df.columns]
df = df[available].dropna()
return df
```

def create_graph(df):
num_nodes = len(df)

```
# Sequential edges — connect each accident node to next
edge_index = []
for i in range(num_nodes - 1):
    edge_index.append([i, i + 1])
edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Remove any out-of-bounds edges
valid_mask = edge_index < num_nodes
edge_index = edge_index[:, valid_mask.all(dim=0)]

# Node features
node_feature_cols = [
    'Longitude', 'Latitude', 'Police_Force', 'Accident_Severity',
    'Number_of_Vehicles', 'Number_of_Casualties', 'Day_of_Week',
    '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit',
    'Urban_or_Rural_Area'
]
available = [c for c in node_feature_cols if c in df.columns]
x = torch.tensor(df[available].values, dtype=torch.float)

y = torch.tensor(df['Accident_Severity'].values, dtype=torch.long)
return Data(x=x, edge_index=edge_index, y=y)
```

# Preprocess and create graph dataset

df_graph = preprocess_graph_data(df)
data = create_graph(df_graph)

train_data, test_data = train_test_split(df_graph, test_size=0.2, random_state=42)
train_graph = create_graph(train_data)
test_graph  = create_graph(test_data)

train_loader = DataLoader([train_graph], batch_size=1)
test_loader  = DataLoader([test_graph],  batch_size=1)

# ─── 10. ST-GNN Model ─────────────────────────────────────────────────────────

class STGNN(nn.Module):
def **init**(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
super(STGNN, self).**init**()
self.conv1   = GCNConv(input_dim, hidden_dim)
self.conv2   = GCNConv(hidden_dim, hidden_dim)
self.conv3   = GCNConv(hidden_dim, hidden_dim)
self.dropout = nn.Dropout(dropout_rate)
self.fc      = nn.Linear(hidden_dim, output_dim)

```
def forward(self, data):
    x, edge_index = data.x, data.edge_index
    x = torch.relu(self.conv1(x, edge_index))
    x = self.dropout(x)
    x = torch.relu(self.conv2(x, edge_index))
    x = self.dropout(x)
    x = torch.relu(self.conv3(x, edge_index))
    x = self.fc(x)
    return x
```

model     = STGNN(input_dim=data.x.shape[1], hidden_dim=128,
output_dim=num_classes, dropout_rate=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ─── 11. Train ST-GNN ─────────────────────────────────────────────────────────

def train_stgnn():
model.train()
for batch in train_loader:
optimizer.zero_grad()
out  = model(batch)
loss = criterion(out, batch.y)
loss.backward()
optimizer.step()

def test_stgnn():
model.eval()
all_preds = []
with torch.no_grad():
for batch in test_loader:
out   = model(batch)
preds = out.argmax(dim=1).cpu().numpy()
all_preds.extend(preds)
return np.array(all_preds[:len(y_test)])

print(”\nTraining ST-GNN for 10 epochs…”)
for epoch in range(10):
train_stgnn()
print(f”Epoch {epoch+1}/10 complete”)

stgnn_preds = test_stgnn()

print(f”\nST-GNN Accuracy: {accuracy_score(y_test, stgnn_preds):.4f}”)
print(classification_report(y_test, stgnn_preds))

# ─── 12. Hybrid Ensemble (ST-GNN + ETC → Logistic Regression) ────────────────

print(”\nTraining Hybrid Ensemble (Meta-Classifier)…”)

# Align prediction sizes

min_len = min(len(etc_preds), len(stgnn_preds), len(y_test))
etc_preds_aligned   = etc_preds[:min_len]
stgnn_preds_aligned = stgnn_preds[:min_len]
y_test_aligned      = y_test[:min_len]

# Stack predictions as meta-features

X_meta = np.column_stack((etc_preds_aligned, stgnn_preds_aligned))

meta_model = LogisticRegression()
meta_model.fit(X_meta, y_test_aligned)
final_preds = meta_model.predict(X_meta)

# ─── 13. Final Results ────────────────────────────────────────────────────────

print(”\n” + “=”*60)
print(“HYBRID ENSEMBLE MODEL RESULTS”)
print(”=”*60)
print(f”Accuracy : {accuracy_score(y_test_aligned, final_preds):.4f}”)
print(classification_report(y_test_aligned, final_preds))

# ─── 14. Confusion Matrix ────────────────────────────────────────────────────

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_aligned, final_preds)
sns.heatmap(cm, annot=True, fmt=‘d’, cmap=‘Blues’,
xticklabels=[‘Slight’, ‘Serious’, ‘Fatal’],
yticklabels=[‘Slight’, ‘Serious’, ‘Fatal’])
plt.xlabel(‘Predicted’); plt.ylabel(‘True’)
plt.title(‘Confusion Matrix — Hybrid Model’)
plt.tight_layout()
plt.savefig(‘confusion_matrix.png’, dpi=150)
plt.show()
print(“Confusion matrix saved to confusion_matrix.png”)

# ─── 15. Feature Importance (ETC) ────────────────────────────────────────────

feature_names = [‘Latitude’, ‘Longitude’, ‘1st_Road_Number’, ‘Day_of_Week’]
importances   = pd.Series(clf.feature_importances_, index=feature_names)
importances.sort_values().plot(kind=‘barh’, figsize=(8, 4), color=‘steelblue’)
plt.title(‘Feature Importance — ExtraTreesClassifier’)
plt.tight_layout()
plt.savefig(‘feature_importance.png’, dpi=150)
plt.show()
print(“Feature importance plot saved to feature_importance.png”)