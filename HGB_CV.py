import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    classification_report,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_sample_weight

# ============================================================
# Repro
# ============================================================
SEED = 42
np.random.seed(SEED)

# ============================================================
# Load data
# ============================================================
CSV_PATH = "/content/LLCP2017_2018_2019_2020_2021XPT_LINEAR_WHOICD_5YEAR.csv"
df = pd.read_csv(CSV_PATH)

TARGET_COL = "CNCRTYP1"
DROP_COLS = ["CNCRTYP1", "CNCRAGE"]

# ------------------------------------------------------------
# Basic cleaning (match your pipeline)
# ------------------------------------------------------------
df = df[~df[TARGET_COL].isin([77, 99])].copy()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ------------------------------------------------------------
# Select majority + TOP-K minority
# ------------------------------------------------------------
TOP_K_MINORITY = 20

class_counts = df[TARGET_COL].value_counts()
majority_class = int(class_counts.idxmax())

minority_labels = (
    class_counts[class_counts.index != majority_class]
    .head(TOP_K_MINORITY)
    .index.astype(int)
    .tolist()
)

selected_classes = [majority_class] + minority_labels
df = df[df[TARGET_COL].isin(selected_classes)].copy()

print("Majority class:", majority_class)
print("Minority labels:", minority_labels)

# ------------------------------------------------------------
# Build X / y
# ------------------------------------------------------------
X = (
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
      .values
      .astype(np.float32)
)
y = df[TARGET_COL].values.astype(int)

# ============================================================
# SAFE minority macro-PRECISION metric
# ============================================================
def safe_precision_macro_minority(y_true, y_pred, minority_labels):
    """
    Macro-averaged PRECISION over selected minority labels only.
    Safe for non-stratified splits where some labels may be missing.
    Missing labels contribute precision = 0.
    """
    return precision_score(
        y_true,
        y_pred,
        labels=minority_labels,
        average="macro",
        zero_division=0
    )

# ============================================================
# 5-FOLD CV (NON-stratified, by design)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Store OOF predictions for pooled report
oof_pred = np.empty_like(y)

# Store per-fold per-class metrics
per_fold_precision = []  # list of arrays (len = n_classes)
per_fold_recall    = []
per_fold_f1        = []
per_fold_support   = []

fold_minor_prec = []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    # -----------------------------
    # Scaling (TRAIN only) - kept to match your pipeline
    # -----------------------------
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # -----------------------------
    # "Balanced" handling for HGB: per-fold sample weights
    # -----------------------------
    sample_w = compute_sample_weight(class_weight="balanced", y=y_train)

    # -----------------------------
    # Vanilla HistGradientBoosting
    # -----------------------------
    hgb = HistGradientBoostingClassifier(
        random_state=SEED
    )

    hgb.fit(X_train_s, y_train, sample_weight=sample_w)
    y_pred = hgb.predict(X_test_s)

    # Save OOF preds
    oof_pred[te_idx] = y_pred

    # --- minority macro precision (per fold)
    minority_macro = safe_precision_macro_minority(y_test, y_pred, minority_labels)
    fold_minor_prec.append(minority_macro)

    # --- per-class metrics (explicit, ALL selected classes)
    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred,
        labels=selected_classes,
        zero_division=0
    )
    per_fold_precision.append(p)
    per_fold_recall.append(r)
    per_fold_f1.append(f)
    per_fold_support.append(s)

    print(f"\n================ Fold {fold} ================")
    print("Minority macro-PRECISION (fold):", minority_macro)
    print("\nClassification report (fold TEST, all selected classes):")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=selected_classes,
            zero_division=0
        )
    )

# ============================================================
# Aggregate per-class metrics across folds (mean ± std)
# ============================================================
per_fold_precision = np.vstack(per_fold_precision)  # shape (5, n_classes)
per_fold_recall    = np.vstack(per_fold_recall)
per_fold_f1        = np.vstack(per_fold_f1)
per_fold_support   = np.vstack(per_fold_support)

mean_p = per_fold_precision.mean(axis=0)
std_p  = per_fold_precision.std(axis=0)

mean_r = per_fold_recall.mean(axis=0)
std_r  = per_fold_recall.std(axis=0)

mean_f = per_fold_f1.mean(axis=0)
std_f  = per_fold_f1.std(axis=0)

# ============================================================
# Summary: class-level mean±std table
# ============================================================
print("\n================ 5-FOLD CV: Per-class mean ± std ================")
print("Label | Precision (mean±std) | Recall (mean±std) | F1 (mean±std)")
for i, lab in enumerate(selected_classes):
    print(
        f"{lab:>5} | "
        f"{mean_p[i]:.4f}±{std_p[i]:.4f} | "
        f"{mean_r[i]:.4f}±{std_r[i]:.4f} | "
        f"{mean_f[i]:.4f}±{std_f[i]:.4f}"
    )

# ============================================================
# Pooled OOF report (ALL classes)
# ============================================================
print("\n================ Pooled OOF classification report (ALL classes) ================")
print(
    classification_report(
        y,
        oof_pred,
        labels=selected_classes,
        zero_division=0
    )
)

# ============================================================
# >>> EXACT LINE YOU REQUESTED (computed on pooled OOF) <<<
# ============================================================
minority_macro = safe_precision_macro_minority(y, oof_pred, minority_labels)
print("\nMinority macro-PRECISION on OOF (over selected minority labels):", minority_macro)
