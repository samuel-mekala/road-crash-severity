“””
Road Crash Injury Severity Prediction
Dataset Analysis & Visualisation

Appendix 1 — Senior Design Project Report
Authors: Sai Pranav Kothapalli, Sri Hari Priya Panchumarthi,
Meghana Bindem, Samuel Mekala
Guide: Dr. Deepthi Godavarthi
VIT-AP University, May 2025

Dataset: UK Road Accident Dataset

- Download from Kaggle and place as data/UK_Accident.csv
  “””

import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
classification_report, f1_score,
precision_score, recall_score,
roc_auc_score, roc_curve)

# ─── Load Dataset ─────────────────────────────────────────────────────────────

df = pd.read_csv(‘data/UK_Accident.csv’, parse_dates=[‘Date’, ‘Time’])

print(df.columns)
df.sample(5)
print(“No. of rows: {}”.format(df.shape[0]))
print(“No. of cols: {}”.format(df.shape[1]))
df.info()
df.isna().any()

# ─── Missing Values ───────────────────────────────────────────────────────────

print(df.isnull().sum() / len(df) * 100)

# Null values are very less compared to total rows — safe to drop

# ─── Duplicates ───────────────────────────────────────────────────────────────

dup_rows = df[df.duplicated()]
print(“No. of duplicate rows: “, dup_rows.shape[0])

# Total duplicated rows are 34155 — drop them

df.drop_duplicates(inplace=True)
print(“No. of rows remaining: “, df.shape[0])

# ─── Basic Stats ──────────────────────────────────────────────────────────────

df.describe(include=np.number)
df.describe(include=object)

# ─── Column Type Split ────────────────────────────────────────────────────────

numerical_data = df.select_dtypes(include=‘number’)
num_cols = numerical_data.columns
print(“Numerical columns:”, len(num_cols))

categorical_data = df.select_dtypes(include=‘object’)
cat_cols = categorical_data.columns
print(“Categorical columns:”, len(cat_cols))

# ─── Boxplots for Outliers (Fig-1) ────────────────────────────────────────────

sns.set(style=“whitegrid”)
fig = plt.figure(figsize=(20, 50))
fig.subplots_adjust(right=1.5)
for plot in range(1, len(num_cols) + 1):
plt.subplot(6, 4, plot)
sns.boxplot(y=df[num_cols[plot - 1]])
plt.show()

# ─── Diagnostic Plots: Histogram + Q-Q + Boxplot (Fig-2) ─────────────────────

def diagnostic_plot(data, col):
fig = plt.figure(figsize=(20, 5))
fig.subplots_adjust(right=1.5)

```
plt.subplot(1, 3, 1)
sns.histplot(data[col], kde=True, color='teal')
plt.title('Histogram')

plt.subplot(1, 3, 2)
stats.probplot(data[col], dist='norm', fit=True, plot=plt)
plt.title('Q-Q Plot')

plt.subplot(1, 3, 3)
sns.boxplot(y=data[col], color='teal')
plt.title('Box Plot')

plt.show()
```

dist_lst = [
‘Police_Force’, ‘Accident_Severity’, ‘Number_of_Vehicles’,
‘Number_of_Casualties’, ‘Local_Authority_(District)’,
‘1st_Road_Class’, ‘1st_Road_Number’, ‘Speed_limit’,
‘2nd_Road_Class’, ‘2nd_Road_Number’, ‘Urban_or_Rural_Area’
]
for col in dist_lst:
diagnostic_plot(df, col)

# ─── Spearman Correlation Heatmap (Fig-3) ─────────────────────────────────────

# Spearman preferred for non-parametric relationships

plt.figure(figsize=(15, 10))
corr = df.corr(method=‘spearman’)
mask = np.triu(np.ones_like(corr, dtype=bool))
cormat = sns.heatmap(corr, mask=mask, annot=True, cmap=‘YlGnBu’,
linewidths=1, fmt=”.2f”)
cormat.set_title(‘Correlation Matrix’)
plt.show()

# ─── Drop Highly Correlated Features (>80%) ───────────────────────────────────

def get_corr(data, threshold):
corr_col = set()
cormat = data.corr()
for i in range(len(cormat.columns)):
for j in range(i):
if abs(cormat.iloc[i, j]) > threshold:
col_name = cormat.columns[i]
corr_col.add(col_name)
return corr_col

corr_features = get_corr(df, 0.80)
print(“Highly correlated features to drop:”, corr_features)

# Local_Authority_(District) has >80% correlation — drop it

df.drop(columns=[‘Local_Authority_(District)’], axis=1, inplace=True)

# ─── Vertical Count Plots (Fig-4 to Fig-7) ────────────────────────────────────

def cnt_plot_vertical(data, col):
plt.figure(figsize=(15, 7))
ax = sns.countplot(x=col, data=data, palette=‘rainbow’)
for p in ax.patches:
ax.annotate(’{}’.format(p.get_height()),
(p.get_x() + 0.15, p.get_height() + 1), ha=‘center’)
plt.show()

cnt_lst1 = [
‘Road_Type’, ‘Junction_Control’,
‘Pedestrian_Crossing-Human_Control’, ‘Road_Surface_Conditions’
]
for col in cnt_lst1:
cnt_plot_vertical(df, col)

# ─── Horizontal Count Plots (Fig-8 to Fig-12) ────────────────────────────────

def cnt_plot_horizontal(data, col):
plt.figure(figsize=(10, 7))
sns.countplot(y=col, data=data, palette=‘rainbow’)
plt.show()

cnt_lst2 = [
‘Pedestrian_Crossing-Physical_Facilities’, ‘Light_Conditions’,
‘Weather_Conditions’, ‘Special_Conditions_at_Site’, ‘Carriageway_Hazards’
]
for col in cnt_lst2:
cnt_plot_horizontal(df, col)

# ─── Fix Urban_or_Rural_Area (replace 3 → 1) ─────────────────────────────────

df[‘Urban_or_Rural_Area’].replace(3, 1, inplace=True)

# ─── Drop Accident_Index (no predictive value) ───────────────────────────────

print(“Unique Accident_Index values:”, len(df[‘Accident_Index’].unique()))
df.drop(‘Accident_Index’, axis=1, inplace=True)

# ─── Missing Values Heatmap ───────────────────────────────────────────────────

plt.figure(figsize=(18, 5))
sns.heatmap(df.isna(), yticklabels=False, cbar=False, cmap=‘viridis’)
plt.show()

# ─── Correlation with Accident Severity (Fig-13) ─────────────────────────────

X = df.drop(columns=[‘Accident_Severity’], axis=1)
plt.figure(figsize=(8, 10))
X.corrwith(df[‘Accident_Severity’]).plot(
kind=‘barh’, title=“Correlation with Accident Severity”
)
plt.show()

# Key predictors: Number_of_Vehicles, Number_of_Casualties, Speed_limit