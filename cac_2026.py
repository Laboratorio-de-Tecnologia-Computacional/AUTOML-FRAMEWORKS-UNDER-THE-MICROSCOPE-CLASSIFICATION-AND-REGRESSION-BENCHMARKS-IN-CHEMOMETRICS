# -*- coding: utf-8 -*-
"""CAC_2026.ipynb

AUTOML FRAMEWORKS UNDER THE MICROSCOPE: CLASSIFICATION AND REGRESSION BENCHMARKS IN CHEMOMETRICS

Authors: Aderval S. Luna and Nathália Alvim de Souza

Datasets:

Classification - Wine (https://archive.ics.uci.edu/dataset/109/wine)

Regression - NeurIPS (https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/data)

###INSTALL AND IMPORTS
"""

# =========================
# Core scientific packages
# =========================
!pip install scikit-learn
!pip install seaborn
!pip install missingno
!pip install statsmodels
!pip install joblib
!pip install psutil

# =========================
# AutoML frameworks
# =========================
!pip install tpot
!pip install h2o
!pip install autogluon
!pip install flaml
!pip install lazypredict

# =========================
# Chemistry / SMILES processing
# =========================
!pip install rdkit
!pip install mordred
!pip install deepchem

# =========================
# Dataset handling
# =========================
!pip install ucimlrepo

# =========================
# Core libraries
# =========================
import numpy as np
import pandas as pd
import scipy.stats as stats
import time
from ucimlrepo import fetch_ucirepo
from sklearn.base import clone
import warnings
warnings.filterwarnings("ignore")

# =========================
# Visualization & EDA
# =========================
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# =========================
# Scikit-learn utilities
# =========================
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder,label_binarize, RobustScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# =========================
# AutoML frameworks
# =========================
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import h2o
from h2o.automl import H2OAutoML
from autogluon.tabular import TabularPredictor
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from flaml import AutoML
from flaml import AutoML as FLAML_AutoML

# =========================
# Chemistry / SMILES processing
# =========================
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors
import deepchem as dc

# =========================
# Statistical analysis
# =========================
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf

# =========================
# Performance and resource tracking
# =========================
import psutil
import os

#SETS
warnings.filterwarnings("ignore")
h2o.init(max_mem_size='4G')
SEED = 42
TIME_LIMIT = 300
print("Libraries imported and environment initialized.")

"""##Wine Dataset (Classification)

Objective
The Wine dataset is used to evaluate the performance of AutoML frameworks on a multiclass classification task with a small, well-structured, and noise-free dataset. The goal is to assess how efficiently each framework identifies the correct wine cultivar based on physicochemical properties, serving as a classical benchmark for classification performance and computational overhead.

Description
The dataset contains 178 samples of wines derived from three different cultivars, with 13 continuous physicochemical features describing properties such as alcohol content, acidity, ash, magnesium, phenols, flavonoids, and color intensity. The target variable is a categorical label with three classes. Due to its low dimensionality and clean structure, this dataset is particularly suitable for analyzing training time, model selection behavior, and baseline classification accuracy.
"""

# wine dataset
dfwine = fetch_ucirepo(id=109)

# data (as pandas dataframes)
X = dfwine.data.features
y = dfwine.data.targets

# variable information
print(dfwine.variables)

"""## 1. Basic Dataset Inspection"""

# =========================
# Basic dataset inspection
# =========================
print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

display(X.head())
display(y.head())

"""## 2. Descriptive Statistics"""

# =========================
# Descriptive statistics
# =========================
X.describe()

"""## 3. Missing Values Check"""

# =========================
# Missing values analysis
# =========================
missing_values = X.isnull().sum()
print("Missing values per feature:")
print(missing_values)

# Visual inspection
msno.matrix(X)
plt.show()

"""## 4. Central Tendency Measures"""

# =========================
# Central tendency measures
# =========================
central_tendency = pd.DataFrame({
    "Mean": X.mean(),
    "Median": X.median(),
    "Std": X.std()
})

central_tendency

"""## 5. Correlation Analysis"""

# =========================
# Correlation matrix
# =========================
corr_matrix = X.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Matrix")
plt.show()

"""## 6. Target Encoding (Class Labels)"""

# =========================
# Target encoding
# =========================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.values.ravel())

print("Encoded classes:", np.unique(y_encoded))

"""## 7. Train-Test Split"""

# =========================
# Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

"""## 8. Feature Normalization (Min-Max Scaling)"""

# =========================
# Feature normalization (Min-Max)
# =========================
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

"""## 9. LazyClassifier Benchmark (Baseline)"""

start_time = time.time()

lazy_clf = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    custom_metric=None
)

# 1) Run LazyPredict leaderboard
models_df, predictions = lazy_clf.fit(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

total_time = time.time() - start_time
print(f"LazyClassifier execution time: {total_time:.4f} seconds")

# 2) Pick the best model (you can change criterion if desired)
# For classification, F1-weighted is often safer than Accuracy when class imbalance exists.
sort_metric = "F1 Score" if "F1 Score" in models_df.columns else "Accuracy"
best_model_name = models_df.sort_values(by=sort_metric, ascending=False).index[0]

# 3) Retrieve trained estimators (this is the key to get BestModel)
trained_models = lazy_clf.provide_models(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

best_estimator = trained_models[best_model_name]

# 4) Predict with the best estimator and compute metrics
preds_best = best_estimator.predict(X_test_scaled)

# Ensure consistent dtype for metrics (avoid label-type issues)
y_true = pd.Series(y_test).astype(str).values
preds_best = pd.Series(preds_best.ravel()).astype(str).values

acc = accuracy_score(y_true, preds_best)
f1 = f1_score(y_true, preds_best, average="weighted")
rec = recall_score(y_true, preds_best, average="weighted")

# 5) Time only the best model fit+predict (useful for reporting)
best_start = time.time()
_ = best_estimator.fit(X_train_scaled, y_train)
_ = best_estimator.predict(X_test_scaled)
best_only_time = time.time() - best_start

print(
    f"LazyPredict finished | BestModel={best_model_name} | "
    f"Acc={acc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | "
    f"TotalTime={total_time:.2f}s | BestOnlyTime={best_only_time:.2f}s"
)

# 6) Keep full leaderboard for reporting (optional)
lazy_leaderboard = models_df.copy()
lazy_leaderboard["Framework"] = "LazyClassifier"
display(lazy_leaderboard)

"""## 11. Evaluation Metrics

### Confusion Matrix
"""

# =========================
# Confusion matrix and metrics
# =========================
cm = confusion_matrix(y_test.astype(str), preds_best)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix ({best_model_name})")
plt.show()

print(f"Accuracy: {acc:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Recall: {rec:.4f}")

"""### ROC Curve (Multiclass – One-vs-Rest)"""

# =========================
# ROC Curve (One-vs-Rest)
# =========================
try:
    y_test_bin = label_binarize(y_test, classes=np.unique(y_encoded))
    # Alguns modelos do LazyPredict não suportam predict_proba
    if hasattr(best_estimator, "predict_proba"):
        y_score = best_estimator.predict_proba(X_test_scaled)

        plt.figure(figsize=(8, 6))
        for i in range(len(np.unique(y_encoded))):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            auc_score = roc_auc_score(y_test_bin[:, i], y_score[:, i])
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Multiclass)")
        plt.legend()
        plt.show()
    else:
        print(f"The model {best_model_name} does not support ROC curves (without predict_proba).")
except Exception as e:
    print(f"SKIP ROC Curve: {e}")

"""##12. Sanity Check

### 1 — Label Permutation Test
"""

n_permutations = 30
perm_accuracies = []

for i in range(n_permutations):
    # Shuffle labels
    y_perm = np.random.permutation(y_encoded)

    # Train-test split WITHOUT stratification
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        y_perm,
        test_size=0.2,
        random_state=i
    )

    # Scaling (fit on training data only)
    scaler_p = MinMaxScaler()
    X_train_p_scaled = scaler_p.fit_transform(X_train_p)
    X_test_p_scaled = scaler_p.transform(X_test_p)

    X_train_p_scaled = pd.DataFrame(X_train_p_scaled, columns=X.columns)
    X_test_p_scaled = pd.DataFrame(X_test_p_scaled, columns=X.columns)

    # Clone the best model (preserves hyperparameters)
    perm_model = clone(best_estimator)
    perm_model.fit(X_train_p_scaled, y_train_p)

    # Evaluate
    y_pred_p = perm_model.predict(X_test_p_scaled)
    acc = accuracy_score(y_test_p, y_pred_p)
    perm_accuracies.append(acc)

print(f"Permutation Test Mean Accuracy: {np.mean(perm_accuracies):.4f}")
print(f"Permutation Test Std Accuracy: {np.std(perm_accuracies):.4f}")

"""### 2 - Repeated Stratified Cross-Validation"""

# Repeated stratified k-fold
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=3,
    random_state=42
)

# Scale the entire dataset X for cross-validation
# It's important to use a new scaler or fit on the entire X for proper cross-validation
# that makes its own splits.
scaler_cv = MinMaxScaler()
X_scaled = scaler_cv.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns) # Convert to DataFrame to retain column names

# Cross-validation accuracy
cv_scores = cross_val_score(
    best_estimator,
    X_scaled,
    y_encoded,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)

print("Cross-validation accuracy scores:")
print(cv_scores)
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Std CV accuracy: {cv_scores.std():.4f}")

"""## 13. AutoML Benchmark"""

# =========================
# AutoML benchmark results container
# =========================

benchmark_results_1 = []

"""## 14. H2O AutoML"""

# =========================
# H2O AutoML
# =========================

h2o.init(max_mem_size="2G", nthreads=-1, verbose=False)

# Build pandas frames with consistent schema
train_df = pd.concat(
    [X_train.reset_index(drop=True), pd.Series(y_train, name="target")],
    axis=1
)
test_df = pd.concat(
    [X_test.reset_index(drop=True), pd.Series(y_test, name="target")],
    axis=1
)

# Convert to H2OFrame
train_h2o = h2o.H2OFrame(train_df)
test_h2o = h2o.H2OFrame(test_df)

# Make sure target is categorical in H2O
train_h2o["target"] = train_h2o["target"].asfactor()
test_h2o["target"] = test_h2o["target"].asfactor()

# Feature columns (safer than X_train.columns.tolist() if anything changes)
x_cols = [c for c in train_h2o.columns if c != "target"]

# -------------------------
# Train H2O AutoML (timed)
# -------------------------
start_time = time.time()
h2o_aml = H2OAutoML(
    max_runtime_secs=TIME_LIMIT,
    seed=42,
    verbosity="warn"
)
h2o_aml.train(x=x_cols, y="target", training_frame=train_h2o)
elapsed_time = time.time() - start_time

# -------------------------
# Predict on test
# IMPORTANT: do NOT cast to int.
# 'predict' is the predicted class label.
# -------------------------
pred_frame = h2o_aml.leader.predict(test_h2o).as_data_frame()
preds = pred_frame["predict"].astype(str).values

# Make y_test comparable (also as string)
y_true = pd.Series(y_test).astype(str).values

# -------------------------
# Metrics
# -------------------------
acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average="weighted")
rec = recall_score(y_true, preds, average="weighted")

# Leader model id (this is what you want to show in the final table)
leader_id = h2o_aml.leader.model_id

benchmark_results_1.append({
    "Framework": "H2O AutoML",
    "BestModel": leader_id,                 # NEW: show best model chosen by H2O
    "Accuracy": acc,
    "F1-Score": f1,                         # FIX: use consistent name
    "Recall": rec,
    "Time (s)": elapsed_time
})

print(
    f"H2O finished | BestModel={leader_id} | "
    f"Acc={acc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Time={elapsed_time:.2f}s"
)

"""## 15. AutoGluon"""

# =========================
# AutoGluon
# =========================

# Build train/test dataframes
train_ag = pd.concat(
    [X_train.reset_index(drop=True), pd.Series(y_train, name="target")],
    axis=1
)
test_ag = pd.concat(
    [X_test.reset_index(drop=True), pd.Series(y_test, name="target")],
    axis=1
)

X_test_ag = test_ag.drop(columns="target")
y_true = test_ag["target"].values

# -------------------------
# Train AutoGluon (timed)
# -------------------------
start_time = time.time()

predictor = TabularPredictor(
    label="target",
    problem_type="multiclass",
    eval_metric="accuracy",
    verbosity=0
).fit(
    train_ag,
    presets="medium_quality",
    time_limit=TIME_LIMIT
)

elapsed_time = time.time() - start_time

# -------------------------
# Leader model (best model name)
# IMPORTANT: AutoGluon doesn't have get_model_best() in some versions.
# The most robust method is: predictor.leaderboard(...).iloc[0]
# -------------------------
lb = predictor.leaderboard(data=test_ag, silent=True)
best_model_name = lb.iloc[0]["model"]

# -------------------------
# Predictions
# -------------------------
preds = predictor.predict(X_test_ag).values

# Make sure y_true and preds share the same dtype to avoid metric edge cases
y_true = pd.Series(y_true).astype(str).values
preds = pd.Series(preds).astype(str).values

# Metrics
acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average="weighted")
rec = recall_score(y_true, preds, average="weighted")

benchmark_results_1.append({
    "Framework": "AutoGluon",
    "BestModel": best_model_name,     # NEW: show leader model
    "Accuracy": acc,
    "F1-Score": f1,                   # FIX: consistent key name
    "Recall": rec,
    "Time (s)": elapsed_time
})

print(
    f"AutoGluon finished | BestModel={best_model_name} | "
    f"Acc={acc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Time={elapsed_time:.2f}s"
)

"""##16. TPOT"""

# =========================
# TPOT
# =========================

# Convert time limit to minutes (minimum 1 minute)
tpot_mins = max(1, int(TIME_LIMIT / 60))

start_time = time.time()

# Try to enforce per-pipeline time cap (supported in newer TPOT versions)
# This prevents one slow pipeline from consuming the entire budget.
try:
    per_eval = max(1, int(tpot_mins * 0.6))  # e.g., 60% of total budget per individual
    tpot = TPOTClassifier(
        generations=None,           # run until time budget
        max_time_mins=tpot_mins,
        max_eval_time_mins=per_eval,
        population_size=20,
        random_state=42,
        n_jobs=1,
        verbose=0
    )
except TypeError:
    # Fallback for older TPOT versions (no max_eval_time_mins)
    # Reduce complexity to improve chance of completing within the time budget.
    tpot = TPOTClassifier(
        generations=None,
        max_time_mins=tpot_mins,
        population_size=12,         # reduced to avoid extreme overruns
        random_state=42,
        n_jobs=1,
        verbosity=0
    )

# -------------------------
# Train TPOT (timed)
# -------------------------
tpot.fit(X_train, y_train)
elapsed_time = time.time() - start_time

# -------------------------
# Best pipeline (the "best model" TPOT found)
# -------------------------
best_pipeline = getattr(tpot, "fitted_pipeline_", None)
best_model_str = str(best_pipeline)

# Optional: export pipeline code (useful for your report)
export_path = "tpot_best_pipeline_classification.py"
try:
    tpot.export(export_path)
except Exception:
    export_path = None

# -------------------------
# Predict and compute metrics
# -------------------------
preds = tpot.predict(X_test)

# Ensure consistent dtype for metrics
y_true = pd.Series(y_test).astype(str).values
preds = pd.Series(preds).astype(str).values

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average="weighted")
rec = recall_score(y_true, preds, average="weighted")

# -------------------------
# Log results (with BestModel)
# -------------------------
benchmark_results_1.append({
    "Framework": "TPOT",
    "BestModel": best_model_str,     # NEW: show best pipeline found
    "ExportPath": export_path,       # optional
    "Accuracy": acc,
    "F1-Score": f1,                  # FIX: consistent name
    "Recall": rec,
    "Time (s)": elapsed_time
})

print(
    f"TPOT finished | Acc={acc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | "
    f"Time={elapsed_time:.2f}s | Export={export_path}"
)

"""##17. FLAML"""

# =========================
# FLAML
# =========================

start_time = time.time()

automl_flaml = AutoML()
automl_flaml.fit(
    X_train=X_train,
    y_train=y_train,
    task="classification",
    time_budget=TIME_LIMIT,
    metric="accuracy",   # explicit metric for clearer behavior
    verbose=0
)

elapsed_time = time.time() - start_time

# -------------------------
# Best model info from FLAML
# -------------------------
best_estimator = getattr(automl_flaml, "best_estimator", None)   # e.g., 'lgbm', 'xgboost', 'rf', ...
best_config = getattr(automl_flaml, "best_config", None)         # hyperparameters

# -------------------------
# Predictions + metrics
# -------------------------
preds = automl_flaml.predict(X_test)

# Ensure consistent dtype for metric functions (avoid label type issues)
y_true = pd.Series(y_test).astype(str).values
preds = pd.Series(preds).astype(str).values

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds, average="weighted")
rec = recall_score(y_true, preds, average="weighted")

benchmark_results_1.append({
    "Framework": "FLAML",
    "BestModel": best_estimator,          # NEW: best estimator family/type
    "BestConfig": str(best_config),       # optional, good for reporting
    "Accuracy": acc,
    "F1-Score": f1,                       # FIX: consistent naming
    "Recall": rec,
    "Time (s)": elapsed_time
})

print(
    f"FLAML finished | BestModel={best_estimator} | "
    f"Acc={acc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Time={elapsed_time:.2f}s"
)

"""## 18. Lazy Predict"""

# =========================
# Lazy Predict (baseline)
# =========================

benchmark_results_1.append({
    "Framework": "Lazy Predict (Best)",
    "BestModel": best_model_name,           # NEW: winning model name
    "Accuracy": acc,
    "F1-Score": f1,                   # FIX: consistent naming
    "Recall": rec,
    "Time (s)": total_time,
    "TimeBestOnly(s)": best_only_time
})

"""## 19. Final Comparison"""

# =========================
# Final benchmark table (Classification)
# =========================

final_df = pd.DataFrame(benchmark_results_1).copy()

# -------------------------
# 1) Normalize column names (handle multiple naming conventions)
# -------------------------
col_map = {}

# Framework
if "Framework" in final_df.columns:
    col_map["Framework"] = "Model/Framework"

# Best model chosen by the framework (leader)
if "BestModel" in final_df.columns:
    col_map["BestModel"] = "Best Model"
elif "Best Model" in final_df.columns:
    col_map["Best Model"] = "Best Model"
elif "LeaderModel" in final_df.columns:
    col_map["LeaderModel"] = "Best Model"
elif "model" in final_df.columns:
    col_map["model"] = "Best Model"

# Metrics (classification)
if "Accuracy" in final_df.columns:
    col_map["Accuracy"] = "Accuracy"

if "F1-Score" in final_df.columns:
    col_map["F1-Score"] = "F1-Score"
elif "F1-score" in final_df.columns:
    col_map["F1-score"] = "F1-Score"
elif "F1 Score" in final_df.columns:
    col_map["F1 Score"] = "F1-Score"
elif "F1" in final_df.columns:
    col_map["F1"] = "F1-Score"

if "Recall" in final_df.columns:
    col_map["Recall"] = "Recall"

# Time
if "Time (seconds)" in final_df.columns:
    col_map["Time (seconds)"] = "Time (seconds)"
elif "Time (s)" in final_df.columns:
    col_map["Time (s)"] = "Time (seconds)"
elif "TimeTotal(s)" in final_df.columns:
    col_map["TimeTotal(s)"] = "Time (seconds)"
elif "Time" in final_df.columns:
    col_map["Time"] = "Time (seconds)"

# Status (optional)
if "Status" in final_df.columns:
    col_map["Status"] = "Status"

final_df_display = final_df.rename(columns=col_map)

# -------------------------
# 2) Ensure required columns exist (even if empty)
# -------------------------
required_cols = ["Model/Framework", "Best Model", "Accuracy", "F1-Score", "Recall", "Time (seconds)"]
for c in required_cols:
    if c not in final_df_display.columns:
        final_df_display[c] = np.nan

# -------------------------
# 3) Prefer OK rows over FAILED when duplicates exist
#    (for same framework, keep OK if available)
# -------------------------
if "Status" in final_df_display.columns:
    final_df_display["__is_failed__"] = final_df_display["Status"].astype(str).str.startswith("FAILED")
else:
    final_df_display["__is_failed__"] = False

# Sort so OK rows come first, then keep FIRST occurrence per framework
final_df_display = final_df_display.sort_values(
    by=["Model/Framework", "__is_failed__"],
    ascending=[True, True]   # False (OK) comes before True (FAILED)
)
final_df_display = final_df_display.drop_duplicates(subset=["Model/Framework"], keep="first")
final_df_display = final_df_display.drop(columns="__is_failed__")

# -------------------------
# 4) Ranking (F1 primary, then Accuracy, then Time)
#    Fill NaNs to avoid weird ordering
# -------------------------
metric_primary = "F1-Score" if final_df_display["F1-Score"].notna().any() else "Accuracy"

# Replace NaNs for sorting (worst-case values)
final_df_display["_sort_metric"] = final_df_display[metric_primary].fillna(-1)
final_df_display["_sort_acc"] = final_df_display["Accuracy"].fillna(-1)
final_df_display["_sort_time"] = final_df_display["Time (seconds)"].fillna(np.inf)

final_df_display = final_df_display.sort_values(
    by=["_sort_metric", "_sort_acc", "_sort_time"],
    ascending=[False, False, True]
).drop(columns=["_sort_metric", "_sort_acc", "_sort_time"])

# Mark the winner row
final_df_display["Winner"] = ""
if len(final_df_display) > 0:
    final_df_display.iloc[0, final_df_display.columns.get_loc("Winner")] = "⭐ Best"

# Reorder columns for clean display
cols_out = ["Winner", "Model/Framework", "Best Model", "Accuracy", "F1-Score", "Recall", "Time (seconds)"]
if "Status" in final_df_display.columns:
    cols_out.append("Status")

final_df_display = final_df_display[cols_out]

print("\n=== FINAL RESULTS TABLE (CLASSIFICATION) ===")
display(final_df_display)

# -------------------------
# 5) Comparative Plots: Time vs Performance
# -------------------------
# Build a consistent color mapping for frameworks
frameworks = final_df_display["Model/Framework"].unique().tolist()
palette_map = dict(zip(frameworks, sns.color_palette("tab10", n_colors=len(frameworks))))

plt.figure(figsize=(12, 5))

# Plot 1: Performance (F1 or Accuracy)
plt.subplot(1, 2, 1)
sns.barplot(
    data=final_df_display,
    x="Model/Framework",
    y=metric_primary,
    palette=palette_map
)
plt.title(f"Performance ({metric_primary})")
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Execution Time
plt.subplot(1, 2, 2)
sns.barplot(
    data=final_df_display,
    x="Model/Framework",
    y="Time (seconds)",
    palette=palette_map
)
plt.title("Execution Time (Seconds)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

"""##NeurIPS Open Polymer Prediction Dataset (Regression)

Objective
This dataset is used to evaluate AutoML frameworks on a real-world, high-dimensional regression problem involving the prediction of physicochemical properties of polymers. The primary objective is to assess each framework’s ability to handle complex feature spaces, scalability, and computational efficiency while achieving competitive predictive performance.

Description
The dataset originates from the NeurIPS Open Polymer Prediction Challenge and consists of polymer representations described by hundreds to thousands of numerical features derived from molecular descriptors and embeddings. The task is to predict continuous-valued target properties related to the physical and chemical behavior of polymers. Compared to the Wine dataset, this dataset poses significantly higher challenges in terms of dimensionality, computational cost, and model optimization, making it suitable for benchmarking AutoML frameworks under more demanding conditions.
"""

# Please upload the database

# Load data
df = pd.read_csv("train.csv")
targets = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

print("Calculating RDKit descriptors for the entire dataset... This may take a few minutes.")
desc_names = [desc[0] for desc in Descriptors._descList]

"""## 1. EDA"""

# ==========================================
# TARGET VARIABLES EDA (Central Tendency)
# ==========================================
print("\n--- Central Tendency & Dispersion Analysis ---")

# Calculate Descriptive Statistics
eda_stats = df[targets].describe().T
eda_stats['median'] = df[targets].median()
eda_stats['skewness'] = df[targets].skew()
eda_stats['kurtosis'] = df[targets].kurtosis()
eda_stats['missing_ratio'] = (df[targets].isnull().sum() / len(df)) * 100

# Reorder columns for better scientific reading
eda_stats = eda_stats[['count', 'missing_ratio', 'mean', 'median', 'std', 'min', 'max', 'skewness', 'kurtosis']]
display(eda_stats.round(3))

"""### 1.1 Histograms"""

# Plotting Distributions (Histograms + KDE)
plt.figure(figsize=(15, 10))
for i, col in enumerate(targets, 1):
    plt.subplot(2, 3, i)
    # Drop NaNs just for plotting
    valid_data = df[col].dropna()
    sns.histplot(valid_data, kde=True, bins=30, color='teal')
    plt.title(f"Distribution of {col}\nSkew: {valid_data.skew():.2f}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

"""## 1.2 Matrix of targets"""

# Correlation Matrix of Targets
plt.figure(figsize=(6, 5))
sns.heatmap(df[targets].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Target Correlation Matrix")
plt.show()

# How many rows have both Tg and FFV present?
pair_n = df[["Tg", "FFV"]].dropna().shape[0]
print("Rows with BOTH Tg and FFV:", pair_n)

# Show overlap counts for all pairs (how many rows are usable per pair)
overlap = pd.DataFrame(index=targets, columns=targets, dtype=int)
for a in targets:
    for b in targets:
        overlap.loc[a, b] = df[[a, b]].dropna().shape[0]

display(overlap)

"""## 2. Descriptors"""

# ==========================================
# CHEMOMETRIC FEATURE EXTRACTION (RDKit)
# ==========================================
print("Calculating RDKit molecular descriptors... This will take a moment.")
start = time.time()

desc_names = [desc[0] for desc in Descriptors._descList]

def calculate_descriptors(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [np.nan] * len(desc_names)
        return [desc[1](mol) for desc in Descriptors._descList]
    except:
        return [np.nan] * len(desc_names)

# Extract features
X_raw = pd.DataFrame(df["SMILES"].apply(calculate_descriptors).tolist(), columns=desc_names)

print(f"Feature extraction completed in {time.time() - start:.2f} seconds.")
print(f"Generated {X_raw.shape[1]} descriptors for {X_raw.shape[0]} samples.")

"""## 3. Data Cleaning and Slipt"""

# ==========================================
# DATA CLEANING & TRAIN-TEST SPLIT PER TARGET
# ==========================================
print("--- Preprocessing features for each target ---")

benchmark_results = []
data_splits = {}

for target in targets:
    # 1. Filter rows where the target is not missing
    print(f"\n== Target: {target} ==")
    valid_idx = df[target].dropna().index
    print("valid lines:", len(valid_idx))
    if len(valid_idx) < 50:
        print(f"Skipping {target}: Not enough valid data.")
        continue

    X_target = X_raw.iloc[valid_idx].copy()
    y_target = df.loc[valid_idx, target].values

    print("shape X_target before drop all-NaN:", X_target.shape)
    X_target = X_target.dropna(axis=1, how="all")
    print("shape X_target after drop all-NaN:", X_target.shape)

    # 2. Chemometric Filter: Remove columns that are 100% NaN
    X_target = X_target.dropna(axis=1, how="all")

    # 2.1 Inf -> NaN e clip
    X_target.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_target = X_target.astype(float).clip(lower=-1e15, upper=1e15)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
      X_target, y_target, test_size=0.2, random_state=42
    )

    missing_ratio = X_train.isna().mean()

    # threshold not agressive
    TH_MISS = 0.80

    cols_drop = missing_ratio[missing_ratio > TH_MISS].index
    if len(cols_drop) > 0:
        X_train = X_train.drop(columns=cols_drop)
        X_test  = X_test.drop(columns=cols_drop)

    print(f"Colunms removided for missing > {TH_MISS:.0%}: {len(cols_drop)}")

    # 4. Imputation
    imputer = SimpleImputer(strategy="median", add_indicator=True)

    X_train_arr = imputer.fit_transform(X_train)
    X_test_arr  = imputer.transform(X_test)

    indicator_cols = imputer.indicator_.features_
    indicator_names = [f"{X_train.columns[i]}_missing" for i in indicator_cols]
    new_cols = list(X_train.columns) + indicator_names

    X_train_imputed = pd.DataFrame(X_train_arr, columns=new_cols, index=X_train.index)
    X_test_imputed  = pd.DataFrame(X_test_arr,  columns=new_cols, index=X_test.index)

    print("Before:", X_train.shape[1], "After imput+ind:", X_train_imputed.shape[1])

    # 4.1 Remove strong collinearity
    CORR_TH = 0.98

    corr = X_train_imputed.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop_corr = [col for col in upper.columns if any(upper[col] > CORR_TH)]

    if len(to_drop_corr) > 0:
        X_train_imputed = X_train_imputed.drop(columns=to_drop_corr)
        X_test_imputed  = X_test_imputed.drop(columns=to_drop_corr)

    print(f"Columns removed by corr > {CORR_TH}: {len(to_drop_corr)}")

    # 5. Zero-Variance
    selector = VarianceThreshold(threshold=0.0)
    X_train_filtered = pd.DataFrame(
        selector.fit_transform(X_train_imputed),
        columns=X_train_imputed.columns[selector.get_support()]
    )
    X_test_filtered = pd.DataFrame(
        selector.transform(X_test_imputed),
        columns=X_train_filtered.columns
    )

    # 6. Scaling
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_filtered), columns=X_train_filtered.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test_filtered), columns=X_train_filtered.columns)

    # Salvar no dict
    data_splits[target] = {
        'X_train': X_train_filtered, 'X_test': X_test_filtered,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
        'y_train': y_train, 'y_test': y_test
    }

    print(f"[{target}] Ready: {X_train_filtered.shape[1]} active features | Train: {X_train_filtered.shape[0]} samples | Test: {X_test_filtered.shape[0]} samples")

"""## 4. Lazy Predict"""

# ==========================================
# LAZY PREDICT (BASELINE)
# ==========================================
print("--- Running LazyPredict for all targets ---")

# Store full LazyPredict leaderboards (all tested models) per target
lazy_all_models = {}

# Store best trained estimator per target (in case you want to reuse it later)
lazy_best_estimators = {}

for target, data in data_splits.items():
    print(f"\nEvaluating LazyPredict on target: {target}")

    # Use SCALED data for LazyPredict baseline (as in your setup)
    Xtr = data['X_train_scaled']
    Xte = data['X_test_scaled']
    ytr = data['y_train']
    yte = data['y_test']

    # -----------------------------
    # 1) Run LazyPredict benchmark
    # -----------------------------
    start_time = time.time()

    lazy_reg = LazyRegressor(
        verbose=0,
        ignore_warnings=True

    )
    Xtr2, Xval, ytr2, yval = train_test_split(
        Xtr, ytr, test_size=0.25, random_state=42
    )
    models_df, predictions = lazy_reg.fit(Xtr2, Xval, ytr2, yval)

    total_time = time.time() - start_time

    # Save full leaderboard for reporting (all models tested by LazyPredict)
    # Add metadata columns to make it easier to concatenate later
    leaderboard = models_df.copy()
    leaderboard["Target"] = target
    leaderboard["Framework"] = "LazyPredict"
    lazy_all_models[target] = leaderboard

    # -----------------------------
    # 2) Select best model by RMSE
    # -----------------------------
    best_name = models_df.sort_values(by="RMSE", ascending=True).index[0]
    trained_models = lazy_reg.provide_models(Xtr, Xte, ytr, yte)
    best_estimator = trained_models[best_name]
    lazy_best_estimators[target] = best_estimator

    # -----------------------------
    # 3) Evaluate best model (metrics)
    # -----------------------------
    y_pred = best_estimator.predict(Xte)

    r2 = r2_score(yte, y_pred)
    rmse = np.sqrt(mean_squared_error(yte, y_pred))
    mae = mean_absolute_error(yte, y_pred)

    # Normalized RMSE (scale-free, comparable across targets)
    nrmse = rmse / (np.std(yte) + 1e-12)  # add epsilon to avoid division by zero

    # -----------------------------
    # 4) OPTIONAL: Measure time of best model only
    # (fit + predict). Useful if you want to report
    # "search time" vs "final model time".
    # -----------------------------
    best_start = time.time()
    _ = best_estimator.fit(Xtr, ytr)
    _ = best_estimator.predict(Xte)
    best_only_time = time.time() - best_start

    # -----------------------------
    # 5) Save benchmark row (one row per target)
    # -----------------------------
    benchmark_results.append({
        "Target": target,
        "Framework": "LazyPredict",
        "BestModel": best_name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "TimeTotal(s)": total_time,
        "TimeBestOnly(s)": best_only_time,
        "N_train": Xtr.shape[0],
        "N_test": Xte.shape[0],
        "N_features": Xtr.shape[1],
        "NRMSE_std": nrmse
    })

    print(
        f"Finished {target}. Best model: {best_name} | "
        f"R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | "
        f"TotalTime={total_time:.1f}s | BestOnlyTime={best_only_time:.2f}s |"
        f"NRMSE(std)={nrmse:.4f}"
    )

#Inspect
lazy_leaderboard_full = pd.concat(lazy_all_models.values(), axis=0).reset_index()
lazy_leaderboard_full = lazy_leaderboard_full.rename(columns={"index": "Model"})
display(lazy_leaderboard_full.head())

"""## 5. H2O AutoML"""

# ==========================================
# H2O AUTOML
# ==========================================
print("--- Running H2O AutoML for all targets ---")

# Initialize H2O once
h2o.init(max_mem_size="4G", nthreads=-1, verbose=False)

# Store full leaderboards (optional, useful for reporting)
h2o_leaderboards = {}

# Store leader models (optional, in case you want to inspect later)
h2o_best_models = {}

for target, data in data_splits.items():
    print(f"\nEvaluating H2O AutoML on target: {target}")

    # H2O expects a single training frame with features + target
    # Resetting index avoids issues when converting pandas -> H2OFrame
    train_df = pd.concat(
        [data['X_train'].reset_index(drop=True),
         pd.Series(data['y_train'], name="target_val")],
        axis=1
    )
    test_df = pd.concat(
        [data['X_test'].reset_index(drop=True),
         pd.Series(data['y_test'], name="target_val")],
        axis=1
    )

    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)

    # Define feature columns dynamically (safer than relying on data['X_train'].columns)
    x_cols = [c for c in train_df.columns if c != "target_val"]
    y_col = "target_val"

    # -----------------------------
    # 1) Run H2O AutoML (timed)
    # -----------------------------
    start_time = time.time()

    h2o_aml = H2OAutoML(
       max_runtime_secs=TIME_LIMIT,
        seed=42,
        verbosity="warn",
        sort_metric="RMSE"
    )
    h2o_aml.train(x=x_cols, y=y_col, training_frame=train_h2o)

    total_time = time.time() - start_time

    # -----------------------------
    # 2) Get the best (leader) model info
    # -----------------------------
    leader = h2o_aml.leader
    leader_id = leader.model_id
    h2o_best_models[target] = leader

    # Optional: save leaderboard (top models and metrics computed by H2O)
    lb = h2o_aml.leaderboard.as_data_frame()
    lb["Target"] = target
    lb["Framework"] = "H2O AutoML"
    h2o_leaderboards[target] = lb

    # -----------------------------
    # 3) Predict on test and compute metrics
    # -----------------------------
    preds = leader.predict(test_h2o).as_data_frame()["predict"].values
    y_true = data['y_test']

    r2 = r2_score(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    nrmse = rmse / (np.std(y_true) + 1e-12)

    # -----------------------------
    # 4) Save benchmark row
    # -----------------------------
    benchmark_results.append({
        "Target": target,
        "Framework": "H2O AutoML",
        "BestModel": leader_id,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "TimeTotal(s)": total_time,
        "N_train": data['X_train'].shape[0],
        "N_test": data['X_test'].shape[0],
        "N_features": data['X_train'].shape[1],
        "NRMSE_std": nrmse
    })

    print(
        f"Finished {target}. Leader: {leader_id} | "
        f"R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | "
        f"TotalTime={total_time:.1f}s | NRMSE(std)={nrmse:.4f}"
    )
# You can inspect or export:
h2o_leaderboard_full = pd.concat(h2o_leaderboards.values(), axis=0).reset_index(drop=True)
display(h2o_leaderboard_full.head())

"""## 6. AutoGluon"""

# ==========================================
# AUTOGLUON
# ==========================================
print("--- Running AutoGluon for all targets ---")

# Store full leaderboards per target (useful for reporting)
ag_leaderboards = {}

# Store fitted predictors per target (optional, if you want to reuse)
ag_predictors = {}

for target, data in data_splits.items():
    print(f"\nEvaluating AutoGluon on target: {target}")

    # Build train/test dataframes with the same schema
    # Resetting index avoids any issues with concatenation
    train_ag = pd.concat(
        [data['X_train'].reset_index(drop=True),
         pd.Series(data['y_train'], name="target_val")],
        axis=1
    )
    test_ag = pd.concat(
        [data['X_test'].reset_index(drop=True),
         pd.Series(data['y_test'], name="target_val")],
        axis=1
    )

    X_test_ag = test_ag.drop(columns="target_val")
    y_test_ag = test_ag["target_val"].values

    # -----------------------------
    # 1) Train AutoGluon (timed)
    # -----------------------------
    start_time = time.time()

    predictor = TabularPredictor(
        label="target_val",
        problem_type="regression",
        eval_metric="rmse",
        verbosity=1
    ).fit(
        train_ag,
        presets="medium_quality",
        time_limit=TIME_LIMIT
    )

    total_time = time.time() - start_time

    # Save predictor (optional)
    ag_predictors[target] = predictor

    # -----------------------------
    # 2) Identify the best/leader model
    # -----------------------------
    # AutoGluon exposes the leader model via get_model_best()
    try:
       lb = predictor.leaderboard(silent=True)
    except TypeError:
    # Fallback for older versions that require a dataset
        lb = predictor.leaderboard(data=test_ag, silent=True)
    best_model_name = lb.iloc[0]["model"]

    # Save leaderboard
    lb = lb.copy()
    lb["Target"] = target
    lb["Framework"] = "AutoGluon"
    ag_leaderboards[target] = lb

    # -----------------------------
    # 3) Predict and compute metrics
    # -----------------------------
    # Optional: measure inference time only
    infer_start = time.time()
    preds = predictor.predict(X_test_ag)
    infer_time = time.time() - infer_start

    r2 = r2_score(y_test_ag, preds)
    rmse = np.sqrt(mean_squared_error(y_test_ag, preds))
    mae = mean_absolute_error(y_test_ag, preds)
    nrmse = rmse / (np.std(y_test_ag) + 1e-12)

    # -----------------------------
    # 4) Save benchmark row
    # -----------------------------
    benchmark_results.append({
        "Target": target,
        "Framework": "AutoGluon",
        "BestModel": best_model_name,
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "TimeTotal(s)": total_time,
        "TimeInference(s)": infer_time,
        "N_train": data['X_train'].shape[0],
        "N_test": data['X_test'].shape[0],
        "N_features": data['X_train'].shape[1],
        "NRMSE_std": nrmse
    })

    print(
        f"Finished {target}. Leader: {best_model_name} | "
        f"R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | "
        f"TotalTime={total_time:.1f}s | InferenceTime={infer_time:.3f}s |"
        f"NRMSE(std)={nrmse:.4f}"
    )



# You can inspect or export:
ag_leaderboard_full = pd.concat(ag_leaderboards.values(), axis=0).reset_index(drop=True)
display(ag_leaderboard_full.head())

"""## 7. FLAML"""

# ==========================================
# FLAML
# ==========================================
print("--- Running FLAML for all targets ---")

# Optional: store best configs per target for reporting
flaml_best_configs = {}

for target, data in data_splits.items():
    print(f"\nEvaluating FLAML on target: {target}")

    Xtr = data['X_train']
    ytr = data['y_train']
    Xte = data['X_test']
    yte = data['y_test']

    # -----------------------------
    # 1) Train FLAML (timed)
    # -----------------------------
    start_time = time.time()

    automl = FLAML_AutoML()
    automl.fit(
        X_train=Xtr,
        y_train=ytr,
        task="regression",
        metric="rmse",          # <-- changed from r2 to rmse
        time_budget=TIME_LIMIT,
        verbose=0,
        # eval_method="cv", n_splits=5  # more stable but slower
    )

    total_time = time.time() - start_time

    # -----------------------------
    # 2) Extract best model info
    # -----------------------------
    best_estimator_name = getattr(automl, "best_estimator", None)
    best_config = getattr(automl, "best_config", None)
    flaml_best_configs[target] = best_config

    # -----------------------------
    # 3) Predict and compute metrics (final evaluation on TEST set)
    # -----------------------------
    infer_start = time.time()
    preds = automl.predict(Xte)
    infer_time = time.time() - infer_start

    r2 = r2_score(yte, preds)
    rmse = np.sqrt(mean_squared_error(yte, preds))
    mae = mean_absolute_error(yte, preds)

    # Normalized RMSE (scale-free, comparable across targets)
    nrmse = rmse / (np.std(yte) + 1e-12)

    # -----------------------------
    # 4) Save benchmark row
    # -----------------------------
    benchmark_results.append({
        "Target": target,
        "Framework": "FLAML",
        "BestModel": best_estimator_name,
        "BestConfig": str(best_config),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "NRMSE_std": nrmse,
        "TimeTotal(s)": total_time,
        "TimeInference(s)": infer_time,
        "N_train": Xtr.shape[0],
        "N_test": Xte.shape[0],
        "N_features": Xtr.shape[1]
    })

    print(
        f"Finished {target}. Best estimator: {best_estimator_name} | "
        f"R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | "
        f"NRMSE(std)={nrmse:.4f} | "
        f"TotalTime={total_time:.1f}s | InferenceTime={infer_time:.3f}s"
    )

"""## 8. TPOT"""

# ==========================================
# TPOT (time-capped + RMSE reporting)
# ==========================================
print("--- Running TPOT for all targets ---")

# TPOT uses minutes; convert from seconds
tpot_mins = max(1, int(TIME_LIMIT / 60))

# Optional: store best pipelines per target (string)
tpot_best_pipelines = {}

for target, data in data_splits.items():
    print(f"\nEvaluating TPOT on target: {target}")

    Xtr = data['X_train']
    ytr = data['y_train']
    Xte = data['X_test']
    yte = data['y_test']

    start_time = time.time()

    # -----------------------------
    # 0) Per-evaluation time cap
    # FFV is much larger, so we allow more time per candidate pipeline,
    # otherwise TPOT may fail before evaluating the initial population.
    # -----------------------------
    per_eval = 10 if target == "FFV" else max(2, int(tpot_mins * 0.6))

    # -----------------------------
    # 1) Create TPOT with stricter time control
    # - max_eval_time_mins prevents a single candidate from running forever
    # - verbose=0 reduces console noise
    # -----------------------------
    try:
        tpot = TPOTRegressor(
            generations=None,            # keep searching until time limit
            max_time_mins=tpot_mins,     # total time budget
            max_eval_time_mins=per_eval, # cap time per pipeline evaluation
            population_size=15,
            random_state=42,
            n_jobs=1,
            verbose=0
        )
    except TypeError:
        # Fallback for older TPOT versions that don't support max_eval_time_mins
        # Reduce search complexity to avoid extreme overruns (especially on FFV)
        tpot = TPOTRegressor(
            generations=None,
            max_time_mins=tpot_mins,
            population_size=8,
            random_state=42,
            n_jobs=1,
            verbose=0
        )

    try:
        # -----------------------------
        # 2) Train TPOT (timed)
        # -----------------------------
        tpot.fit(Xtr, ytr)
        total_time = time.time() - start_time

        # -----------------------------
        # 3) Extract best pipeline info
        # -----------------------------
        best_pipeline = getattr(tpot, "fitted_pipeline_", None)
        best_model_str = str(best_pipeline)

        export_path = f"tpot_best_pipeline_{target}.py"
        try:
            tpot.export(export_path)
        except Exception:
            export_path = None

        tpot_best_pipelines[target] = best_model_str

        # -----------------------------
        # 4) Predict and compute metrics (final evaluation on TEST set)
        # -----------------------------
        infer_start = time.time()
        preds = tpot.predict(Xte)
        infer_time = time.time() - infer_start

        r2 = r2_score(yte, preds)
        rmse = np.sqrt(mean_squared_error(yte, preds))
        mae = mean_absolute_error(yte, preds)
        nrmse = rmse / (np.std(yte) + 1e-12)

        # -----------------------------
        # 5) Save benchmark row
        # -----------------------------
        benchmark_results.append({
            "Target": target,
            "Framework": "TPOT",
            "BestModel": best_model_str,
            "ExportPath": export_path,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
            "NRMSE_std": nrmse,
            "TimeTotal(s)": total_time,
            "TimeInference(s)": infer_time,
            "N_train": Xtr.shape[0],
            "N_test": Xte.shape[0],
            "N_features": Xtr.shape[1],
            "Status": "OK"
        })

        print(
            f"Finished {target}. | R2={r2:.4f} | RMSE={rmse:.4f} | MAE={mae:.4f} | "
            f"NRMSE(std)={nrmse:.4f} | "
            f"TotalTime={total_time:.1f}s | InferenceTime={infer_time:.3f}s | "
            f"Export={export_path}"
        )

    except Exception as e:
        elapsed_time = time.time() - start_time
        err_msg = str(e)

        benchmark_results.append({
            "Target": target,
            "Framework": "TPOT",
            "BestModel": None,
            "ExportPath": None,
            "R2": None,
            "RMSE": None,
            "MAE": None,
            "NRMSE_std": None,
            "TimeTotal(s)": elapsed_time,
            "TimeInference(s)": None,
            "N_train": Xtr.shape[0],
            "N_test": Xte.shape[0],
            "N_features": Xtr.shape[1],
            "Status": f"FAILED: {err_msg}"
        })

        print(f"TPOT failed on {target} after {elapsed_time:.1f}s: {err_msg}")

"""## 9. Final Results"""

# ==========================================
# FINAL RESULTS COMPILATION & PLOTTING
# ==========================================
print("\n=== FINAL MULTI-TARGET REGRESSION BENCHMARK ===")

final_df = pd.DataFrame(benchmark_results).copy()

# ------------------------------------------
# 1) Normalize columns (old -> new)
# ------------------------------------------
if "R2-Score" in final_df.columns and "R2" not in final_df.columns:
    final_df["R2"] = final_df["R2-Score"]

if "Time (s)" in final_df.columns and "TimeTotal(s)" not in final_df.columns:
    final_df["TimeTotal(s)"] = final_df["Time (s)"]

# Ensure Status exists
if "Status" not in final_df.columns:
    final_df["Status"] = "OK"

# If NRMSE_std is missing for some rows, leave as NaN (best practice is to compute it during each run)
if "NRMSE_std" not in final_df.columns:
    final_df["NRMSE_std"] = np.nan

# ------------------------------------------
# 2) Prefer OK rows over FAILED when both exist for the same Target+Framework
# ------------------------------------------
final_df["__is_failed__"] = final_df["Status"].astype(str).str.startswith("FAILED")
final_df = final_df.sort_values(by=["Target", "Framework", "__is_failed__"])  # OK first
final_df = final_df.drop(columns="__is_failed__")

# Keep only the LAST occurrence per Target+Framework (latest run)
final_df = final_df.drop_duplicates(subset=["Target", "Framework"], keep="last")

# Optional: sort results inside each target by best (lowest) NRMSE_std
# (Rows with missing NRMSE_std go to the bottom)
final_df = final_df.sort_values(by=["Target", "NRMSE_std"], ascending=[True, True])

display(final_df)

# ------------------------------------------
# 3) Plotting Performance (NRMSE_std or RMSE)
# ------------------------------------------

# ---- Plotting Performance (NRMSE_std) ----
# NRMSE_std is scale-free, so it is comparable across targets.
if final_df["NRMSE_std"].notna().any():
    g_perf = sns.catplot(
        data=final_df, kind="bar",
        x="Framework", y="NRMSE_std", col="Target",
        col_wrap=3, height=4, aspect=1.2, palette="viridis"
    )
    g_perf.set_axis_labels("AutoML Framework", "NRMSE (std-normalized)  ↓ lower is better")
    g_perf.set_titles("Target: {col_name}")

    for ax in g_perf.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.subplots_adjust(top=0.85)
    g_perf.fig.suptitle("Model Performance Comparison (NRMSE_std) per Property", fontsize=16)
    plt.show()
else:
    # Fallback: if NRMSE_std is not available, plot RMSE instead
    g_perf = sns.catplot(
        data=final_df, kind="bar",
        x="Framework", y="RMSE", col="Target",
        col_wrap=3, height=4, aspect=1.2, palette="viridis"
    )
    g_perf.set_axis_labels("AutoML Framework", "RMSE  ↓ lower is better")
    g_perf.set_titles("Target: {col_name}")

    for ax in g_perf.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(45)

    plt.subplots_adjust(top=0.85)
    g_perf.fig.suptitle("Model Performance Comparison (RMSE) per Property", fontsize=16)
    plt.show()

# ------------------------------------------
# 4) Plotting Time Cost (Total Search Time)
# ------------------------------------------
g_time = sns.catplot(
    data=final_df, kind="bar",
    x="Framework", y="TimeTotal(s)", col="Target",
    col_wrap=3, height=4, aspect=1.2, palette="magma"
)
g_time.set_axis_labels("AutoML Framework", "Total search time (s)")
g_time.set_titles("Target: {col_name}")

for ax in g_time.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

plt.subplots_adjust(top=0.85)
g_time.fig.suptitle("Computational Cost Comparison per Property", fontsize=16)
plt.show()
