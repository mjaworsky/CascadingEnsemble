# ============================================================
# SINGLE-STAGE XGBOOST WITH 5-FOLD CV
# No neural network gate
# ============================================================

import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

# -----------------------------
# Repro
# -----------------------------
SEED = 42
np.random.seed(SEED)

# -----------------------------
# Settings
# -----------------------------
TARGET_COL = "CNCRTYP1"
DROP_COLS = ["CNCRTYP1","CNCRAGE"]

CSV_PATH = "/content/LLCP_SMOKE_5YEAR.csv"

TOP_K_MINORITY = 20
N_FOLDS = 5

# XGBoost params
XGB_N_ESTIMATORS = 400
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE_BYTREE = 0.85
XGB_MIN_CHILD_WEIGHT = 4

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)

print("Loaded:",CSV_PATH)
print("Shape:",df.shape)

# -----------------------------
# Clean target
# -----------------------------
df = df.copy()

df = df[~df[TARGET_COL].isin([77,99])]
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL],errors="coerce")

df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

df.replace([np.inf,-np.inf],np.nan,inplace=True)
df.fillna(0,inplace=True)

print("\n========== RAW TARGET DISTRIBUTION ==========")
print(df[TARGET_COL].value_counts())

# -----------------------------
# Select classes
# -----------------------------
class_counts = df[TARGET_COL].value_counts()

majority_class = int(class_counts.idxmax())

minority_candidates = class_counts[class_counts.index != majority_class]

topk_minority = [int(x) for x in minority_candidates.head(TOP_K_MINORITY).index.tolist()]

selected_classes = [majority_class] + topk_minority

df = df[df[TARGET_COL].isin(selected_classes)]

print("\nSelected classes:",selected_classes)

if len(selected_classes) < 2:
    print("\nOnly one class present. No model trained.")
    sys.exit()

# -----------------------------
# Build X / y
# -----------------------------
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns],errors="ignore").values.astype(np.float32)

y = df[TARGET_COL].values.astype(int)

labels = sorted(np.unique(y))

lab_to_idx = {lab:i for i,lab in enumerate(labels)}
idx_to_lab = {i:lab for lab,i in lab_to_idx.items()}

y_enc = np.array([lab_to_idx[v] for v in y])

print("\nX shape:",X.shape)
print("Classes:",labels)

# -----------------------------
# Metric helpers
# -----------------------------
def summarize_multiclass_metrics(y_true,y_pred):

    acc = accuracy_score(y_true,y_pred)
    bacc = balanced_accuracy_score(y_true,y_pred)

    f1_macro = f1_score(y_true,y_pred,average="macro",zero_division=0)
    f1_weighted = f1_score(y_true,y_pred,average="weighted",zero_division=0)

    return {
        "acc":acc,
        "bal_acc":bacc,
        "f1_macro":f1_macro,
        "f1_weighted":f1_weighted
    }

def mean_std(metrics,key):

    vals = np.array([m[key] for m in metrics])
    return vals.mean(),vals.std(ddof=1)

# -----------------------------
# Model builder
# -----------------------------
def make_xgb(num_classes):

    return XGBClassifier(
        objective="multi:softprob" if num_classes>2 else "binary:logistic",
        num_class=num_classes if num_classes>2 else None,
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        random_state=SEED,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
        verbosity=0
    )

# -----------------------------
# CV
# -----------------------------
kf = KFold(n_splits=N_FOLDS,shuffle=True,random_state=SEED)

oof_pred = np.zeros(len(y),dtype=int)

fold_metrics = []

print("\n================================================")
print("XGBoost 5-FOLD CROSS VALIDATION")
print("================================================")

for fold,(tr_idx,va_idx) in enumerate(kf.split(X),1):

    print(f"\nFold {fold}/{N_FOLDS}")

    X_tr,X_va = X[tr_idx],X[va_idx]
    y_tr,y_va = y_enc[tr_idx],y_enc[va_idx]

    # Oversample training fold
    ros = RandomOverSampler(random_state=SEED)
    X_tr_bal,y_tr_bal = ros.fit_resample(X_tr,y_tr)

    # Class weights
    classes = np.unique(y_tr_bal)

    w = compute_class_weight(class_weight="balanced",classes=classes,y=y_tr_bal)

    cw = {c:wi for c,wi in zip(classes,w)}

    sample_weights = np.array([cw[v] for v in y_tr_bal])

    model = make_xgb(len(labels))

    model.fit(X_tr_bal,y_tr_bal,sample_weight=sample_weights)

    proba = model.predict_proba(X_va)

    if len(labels)==2:
        pred_enc = (proba[:,1] >= 0.5).astype(int)
    else:
        pred_enc = np.argmax(proba,axis=1)

    pred = np.array([idx_to_lab[i] for i in pred_enc])

    oof_pred[va_idx] = pred

    y_va_true = np.array([idx_to_lab[i] for i in y_va])

    m = summarize_multiclass_metrics(y_va_true,pred)

    fold_metrics.append(m)

    print(m)

# -----------------------------
# Mean metrics
# -----------------------------
print("\n========== MEAN ± STD ==========")

for key in ["acc","bal_acc","f1_macro","f1_weighted"]:

    mu,sd = mean_std(fold_metrics,key)

    print(f"{key}: {mu:.4f} ± {sd:.4f}")

# -----------------------------
# OOF report
# -----------------------------
print("\n========== OOF CLASSIFICATION REPORT ==========")

print(classification_report(y,oof_pred,labels=labels,zero_division=0))