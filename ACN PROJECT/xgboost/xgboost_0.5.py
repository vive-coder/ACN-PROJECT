#!/usr/bin/env python3
"""
XGBoost Classifier (XGBClassifier)
Split: 50% Train, 50% Test (test_size=0.50)
Dataset: CIC-IDS2017 (Tuesday, Wednesday, Thursday)
Outputs terminal metrics and saves visual output to 'xgboost_output_0.5.jpg'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------
# 1. LOAD DATASETS
# ---------------------------------------------------
print("Loading datasets...")
df1 = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=False)
df2 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv", low_memory=False)
df3 = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", low_memory=False)

dataset = pd.concat([df1, df2, df3], ignore_index=True)
print("Combined Dataset Shape:", dataset.shape)

# ---------------------------------------------------
# 2. CLEAN & PREPROCESS FEATURES
# ---------------------------------------------------
dataset.columns = dataset.columns.str.strip()

X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1].copy()
y = y.astype(str).str.strip()

X = X.apply(pd.to_numeric, errors="coerce")
X.replace([np.inf, -np.inf], np.nan, inplace=True)

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
X = imputer.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# ---------------------------------------------------
# 3. TRAIN-TEST SPLIT (TEST SIZE = 0.5)
# ---------------------------------------------------
TEST_SIZE = 0.5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

print(f"\nTraining samples: {X_train.shape[0]:,}")
print(f"Testing samples : {X_test.shape[0]:,}")

# ---------------------------------------------------
# 4. FEATURE SCALING
# ---------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------
# 5. XGBOOST CLASSIFIER
# ---------------------------------------------------
classifier = XGBClassifier(
    random_state=0,
    eval_metric="mlogloss",
    use_label_encoder=False
)
classifier.fit(X_train, y_train)

# ---------------------------------------------------
# 6. EVALUATION & METRICS
# ---------------------------------------------------
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
report_str = classification_report(
    y_test, y_pred, target_names=labelencoder_y.classes_, zero_division=0
)

print("\n" + "=" * 50)
print(f"XGBOOST RESULTS FOR TEST SIZE = {TEST_SIZE} (50:50 SPLIT)")
print("=" * 50)
print("\nCONFUSION MATRIX:\n", cm)
print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nCLASSIFICATION REPORT:\n", report_str)

# ---------------------------------------------------
# 7. EXPORT VISUALIZATION TO JPG
# ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
fig.suptitle(f"XGBoost Classifier Evaluation (Test Size: {TEST_SIZE})", fontsize=16, fontweight="bold")

# Confusion Matrix Heatmap
im = axes[0].imshow(cm, cmap="Purples", interpolation="nearest")
axes[0].set_title(f"Confusion Matrix (Test Size = {TEST_SIZE})")
fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")

# Performance Summary Bar Chart
metrics_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metrics_values = [accuracy, precision, recall, f1]
bars = axes[1].bar(metrics_names, metrics_values, color=["#8b5cf6", "#3b82f6", "#10b981", "#f59e0b"])
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Performance Metrics")
for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
output_jpg = f"xgboost_output_{TEST_SIZE}.jpg"
plt.savefig(output_jpg, format="jpg")
plt.close()
print(f"\nVisual output saved as: {output_jpg}")
