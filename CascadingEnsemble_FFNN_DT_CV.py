# ============================================================
# Cascade WITH CV (metrics averaged properly)
# 1) Gate NN: 5-FOLD Stratified CV
#    - computes per-fold metrics + mean±std
#    - also prints pooled OOF classification report
#
# 2) Minority DT: RepeatedStratified 3-FOLD CV (minority classes only)
#    - averages METRICS across all folds/repeats
#    - also produces pooled OOF predictions by probability-averaging per sample
#      across all times it appears in validation
#
# 3) (Bonus) Full CASCADE: out-of-fold across the 5 gate folds
#    - computes per-fold cascade metrics + mean±std
#    - prints pooled OOF classification report
#
# Notes:
# - NO SMOTE
# - Fixed thresholds (no tuning)
# - Scaling is fit on TRAIN only inside each fold (no leakage)
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.tree import DecisionTreeClassifier

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Repro
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# User settings
# -----------------------------
TOP_K_MINORITY = 20

# Gate NN hyperparams
GATE_EPOCHS = 50
GATE_BATCH = 256
GATE_LR = 1e-3
GATE_PATIENCE = 6

# Decision Tree hyperparams
DT_MAX_DEPTH = None
DT_MIN_SAMPLES_LEAF = 1
DT_MIN_SAMPLES_SPLIT = 2

# Fixed thresholds (NO tuning)
GATE_THRESHOLD = 0.50          # route to minority DT if P(minority) >= this
MIN_ACCEPT_THRESHOLD = 0.00    # accept DT prediction if DT max-proba >= this (0 = always accept)

# CV settings
GATE_FOLDS = 5

DT_RS_FOLDS = 3
DT_RS_REPEATS = 5  # set higher for stabler averages

# -----------------------------
# Load data (upload in Colab)
# -----------------------------
try:
    from google.colab import files
    uploaded = files.upload()
    csv_path = list(uploaded.keys())[0]
except Exception:
    raise RuntimeError(
        "Upload your CSV in Colab when prompted. If you're not in Colab, set csv_path manually."
    )

df = pd.read_csv(csv_path)
print(f"Loaded: {csv_path}")
print("Shape:", df.shape)

# -----------------------------
# Basic cleaning / target handling
# -----------------------------
TARGET_COL = "CNCRTYP1"
DROP_COLS = ["CNCRTYP1", "CNCRAGE"]  # keep CNCRAGE out of features (as per your original)

if TARGET_COL not in df.columns:
    raise ValueError(f"Expected column '{TARGET_COL}' not found in CSV.")

df = df.copy()

# Filter invalid CNCRTYP1 codes if present
df = df[~df[TARGET_COL].isin([77, 99])].copy()

# Coerce target to int safely
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

# Replace inf, keep NaNs -> fill 0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# -----------------------------
# Select majority + TOP_K minority classes
# -----------------------------
class_counts = df[TARGET_COL].value_counts()
majority_class = int(class_counts.idxmax())

print("\n========== Select Classes ==========")
print("Majority class:", majority_class)

minority_candidates = class_counts[class_counts.index != majority_class]
topk_minority = [int(x) for x in minority_candidates.head(TOP_K_MINORITY).index.tolist()]

selected_classes = [majority_class] + topk_minority
df = df[df[TARGET_COL].isin(selected_classes)].copy()

print("Selected minority labels:", topk_minority)
print("\nClass distribution (selected):")
print(df[TARGET_COL].value_counts())

# -----------------------------
# Build X / y / y_gate
# -----------------------------
print("\n========== Build Features ==========")
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)

# Gate labels: 0 = majority, 1 = minority
y_gate = (y != majority_class).astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Gate minority rate:", y_gate.mean())

# -----------------------------
# Helpers
# -----------------------------
def build_gate_model(input_dim: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        Dropout(0.15),
        Dense(128, activation="relu"),
        Dropout(0.15),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=GATE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def make_dt() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        random_state=SEED,
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        min_samples_split=DT_MIN_SAMPLES_SPLIT,
        class_weight="balanced"
    )

def summarize_binary_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"acc": acc, "bal_acc": bacc, "prec": p, "rec": r, "f1": f1}

def summarize_multiclass_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return {"acc": acc, "bal_acc": bacc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

def mean_std(metrics_list, key):
    vals = np.array([m[key] for m in metrics_list], dtype=float)
    return vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0

def print_mean_std(metrics_list, keys, title):
    print(f"\n{title}")
    for k in keys:
        mu, sd = mean_std(metrics_list, k)
        print(f"  {k:>10s}: {mu:.4f} ± {sd:.4f}")

def cascade_predict_from_probs(
    y_true: np.ndarray,
    gate_probs: np.ndarray,
    dt_classes: np.ndarray,
    dt_proba: np.ndarray,
    majority_class: int,
    gate_t: float,
    accept_t: float
) -> np.ndarray:
    # default majority
    y_pred = np.full(shape=(len(y_true),), fill_value=majority_class, dtype=int)

    route_min = (gate_probs >= gate_t)
    idx = np.where(route_min)[0]
    if len(idx) == 0:
        return y_pred

    probs = dt_proba[idx]
    maxp = probs.max(axis=1)
    pred_class = dt_classes[np.argmax(probs, axis=1)]

    accept = (maxp >= accept_t)
    y_pred[idx[accept]] = pred_class[accept]
    return y_pred

# ============================================================
# (1) Gate NN: 5-fold stratified CV (metrics averaged + pooled report)
# ============================================================
print("\n============================================================")
print("========== (1) Gate NN: 5-FOLD Stratified CV ==========")
print("============================================================")

skf_gate = StratifiedKFold(n_splits=GATE_FOLDS, shuffle=True, random_state=SEED)

oof_gate_probs = np.zeros(len(y), dtype=float)
oof_gate_pred  = np.zeros(len(y), dtype=int)

gate_fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(skf_gate.split(X, y_gate), start=1):
    print(f"\n--- Gate Fold {fold}/{GATE_FOLDS} ---")

    X_tr, X_va = X[tr_idx], X[va_idx]
    yg_tr, yg_va = y_gate[tr_idx], y_gate[va_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    gate_classes = np.unique(yg_tr)
    gate_cw = compute_class_weight(class_weight="balanced", classes=gate_classes, y=yg_tr)
    gate_cw_dict = {int(c): float(w) for c, w in zip(gate_classes, gate_cw)}

    gate = build_gate_model(input_dim=X_tr_s.shape[1])
    early = EarlyStopping(monitor="val_loss", patience=GATE_PATIENCE, restore_best_weights=True)

    gate.fit(
        X_tr_s, yg_tr,
        validation_split=0.2,
        epochs=GATE_EPOCHS,
        batch_size=GATE_BATCH,
        class_weight=gate_cw_dict,
        callbacks=[early],
        verbose=0
    )

    probs = gate.predict(X_va_s, verbose=0).ravel()
    pred  = (probs >= GATE_THRESHOLD).astype(int)

    oof_gate_probs[va_idx] = probs
    oof_gate_pred[va_idx]  = pred

    m = summarize_binary_metrics(yg_va, pred)
    gate_fold_metrics.append(m)
    print("Fold metrics:", {k: round(v, 4) for k, v in m.items()})

print_mean_std(
    gate_fold_metrics,
    keys=["acc", "bal_acc", "prec", "rec", "f1"],
    title="Gate NN: mean ± std across 5 folds (minority=1 positive class)"
)

print("\n========== Gate NN POOLED OOF Report (all folds combined) ==========")
print("Gate threshold:", GATE_THRESHOLD)
print(classification_report(y_gate, oof_gate_pred, zero_division=0))

# ============================================================
# (2) Minority DT: RepeatedStratified 3-fold CV
#     - averages metrics across ALL folds/repeats
#     - pooled OOF predictions via probability averaging per sample
# ============================================================
print("\n============================================================")
print("========== (2) Minority DT: RepeatedStratified 3-FOLD CV (Averaged) ==========")
print("============================================================")

minor_mask = (y != majority_class)
X_min = X[minor_mask]
y_min = y[minor_mask]

minority_labels = sorted(np.unique(y_min).tolist())
lab_to_idx = {lab: i for i, lab in enumerate(minority_labels)}

print("Minority sample count:", len(y_min))
print("Minority label count :", len(minority_labels))

rskf_dt = RepeatedStratifiedKFold(
    n_splits=DT_RS_FOLDS,
    n_repeats=DT_RS_REPEATS,
    random_state=SEED
)

# For pooled OOF: accumulate predicted proba per sample across times it's in validation
proba_sum = np.zeros((len(y_min), len(minority_labels)), dtype=np.float64)
proba_cnt = np.zeros(len(y_min), dtype=np.int32)

dt_fold_metrics = []
total_splits = DT_RS_FOLDS * DT_RS_REPEATS

for split_i, (tr_idx, va_idx) in enumerate(rskf_dt.split(X_min, y_min), start=1):
    X_tr, X_va = X_min[tr_idx], X_min[va_idx]
    y_tr, y_va = y_min[tr_idx], y_min[va_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    dt = make_dt()
    dt.fit(X_tr_s, y_tr)

    # Map DT class probs into fixed minority label order
    dt_classes = dt.classes_
    dt_proba = dt.predict_proba(X_va_s)

    mapped = np.zeros((len(va_idx), len(minority_labels)), dtype=np.float64)
    for j, c in enumerate(dt_classes):
        mapped[:, lab_to_idx[int(c)]] = dt_proba[:, j]

    proba_sum[va_idx] += mapped
    proba_cnt[va_idx] += 1

    pred = dt_classes[np.argmax(dt_proba, axis=1)]
    m = summarize_multiclass_metrics(y_va, pred)
    dt_fold_metrics.append(m)

    if split_i % DT_RS_FOLDS == 0 or split_i == total_splits:
        print(f"Completed {split_i}/{total_splits} splits")

print_mean_std(
    dt_fold_metrics,
    keys=["acc", "bal_acc", "f1_macro", "f1_weighted"],
    title=f"Minority DT: mean ± std across {total_splits} splits (repeated {DT_RS_FOLDS}-fold)"
)

# Pooled OOF predictions from probability averaging
avg_proba = proba_sum / np.maximum(proba_cnt[:, None], 1)
oof_dt_pred_min = np.array(minority_labels, dtype=int)[np.argmax(avg_proba, axis=1)]

print("\n========== Minority DT POOLED OOF Report (probability-averaged) ==========")
print(classification_report(y_min, oof_dt_pred_min, zero_division=0))

# ============================================================
# (3) BONUS: Full cascade OOF using 5 gate folds
#     - per-fold cascade metrics averaged
#     - pooled OOF cascade classification report
# ============================================================
print("\n============================================================")
print("========== (3) BONUS: Full Cascade OOF (5 folds, averaged) ==========")
print("============================================================")

oof_cascade_pred = np.full(len(y), majority_class, dtype=int)
cascade_fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(skf_gate.split(X, y_gate), start=1):
    print(f"\n--- Cascade Fold {fold}/{GATE_FOLDS} ---")

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    yg_tr, yg_va = y_gate[tr_idx], y_gate[va_idx]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)

    # ---- Train Gate ----
    gate_classes = np.unique(yg_tr)
    gate_cw = compute_class_weight(class_weight="balanced", classes=gate_classes, y=yg_tr)
    gate_cw_dict = {int(c): float(w) for c, w in zip(gate_classes, gate_cw)}

    gate = build_gate_model(input_dim=X_tr_s.shape[1])
    early = EarlyStopping(monitor="val_loss", patience=GATE_PATIENCE, restore_best_weights=True)

    gate.fit(
        X_tr_s, yg_tr,
        validation_split=0.2,
        epochs=GATE_EPOCHS,
        batch_size=GATE_BATCH,
        class_weight=gate_cw_dict,
        callbacks=[early],
        verbose=0
    )

    va_gate_probs = gate.predict(X_va_s, verbose=0).ravel()

    # ---- Train DT on TRUE minority TRAIN samples only ----
    min_tr_mask = (y_tr != majority_class)
    X_tr_min_s = X_tr_s[min_tr_mask]
    y_tr_min   = y_tr[min_tr_mask]

    if len(np.unique(y_tr_min)) < 2:
        print("WARNING: Not enough minority variety in this fold. Defaulting to majority.")
        va_pred = np.full(len(y_va), majority_class, dtype=int)
        oof_cascade_pred[va_idx] = va_pred
        # metrics:
        cascade_fold_metrics.append(summarize_multiclass_metrics(y_va, va_pred))
        continue

    dt = make_dt()
    dt.fit(X_tr_min_s, y_tr_min)

    dt_classes = dt.classes_
    dt_proba_va = dt.predict_proba(X_va_s)

    va_pred = cascade_predict_from_probs(
        y_true=y_va,
        gate_probs=va_gate_probs,
        dt_classes=dt_classes,
        dt_proba=dt_proba_va,
        majority_class=majority_class,
        gate_t=GATE_THRESHOLD,
        accept_t=MIN_ACCEPT_THRESHOLD
    )

    oof_cascade_pred[va_idx] = va_pred

    m = summarize_multiclass_metrics(y_va, va_pred)
    cascade_fold_metrics.append(m)
    print("Fold metrics:", {k: round(v, 4) for k, v in m.items()})

print_mean_std(
    cascade_fold_metrics,
    keys=["acc", "bal_acc", "f1_macro", "f1_weighted"],
    title="Cascade: mean ± std across 5 folds"
)

all_labels = [majority_class] + [lab for lab in sorted(np.unique(y).tolist()) if lab != majority_class]

print("\n========== Full Cascade POOLED OOF Report (5 folds) ==========")
print(f"Gate threshold (fixed) = {GATE_THRESHOLD:.3f}")
print(f"Minor accept threshold (fixed) = {MIN_ACCEPT_THRESHOLD:.3f}")
print(classification_report(y, oof_cascade_pred, labels=all_labels, zero_division=0))

# -----------------------------
# Optional: save outputs
# -----------------------------
out_dir = "/content/Cascade_CV_AveragedMetrics"
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "gate_5fold_meanstd_and_oof.txt"), "w") as f:
    f.write(f"Gate threshold (fixed): {GATE_THRESHOLD:.4f}\n")
    f.write("Gate folds: %d\n\n" % GATE_FOLDS)
    for i, m in enumerate(gate_fold_metrics, start=1):
        f.write(f"Fold {i}: {m}\n")
    f.write("\nPOOLED OOF REPORT:\n")
    f.write(classification_report(y_gate, oof_gate_pred, zero_division=0))

with open(os.path.join(out_dir, "minority_dt_repeated3fold_meanstd_and_oof.txt"), "w") as f:
    f.write(f"RepeatedStratifiedKFold: n_splits={DT_RS_FOLDS}, n_repeats={DT_RS_REPEATS}\n\n")
    for i, m in enumerate(dt_fold_metrics, start=1):
        f.write(f"Split {i}: {m}\n")
    f.write("\nMEAN±STD (computed in notebook output)\n")
    f.write("\nPOOLED OOF REPORT (probability-averaged):\n")
    f.write(classification_report(y_min, oof_dt_pred_min, zero_division=0))

with open(os.path.join(out_dir, "cascade_5fold_meanstd_and_oof.txt"), "w") as f:
    f.write(f"Gate threshold (fixed): {GATE_THRESHOLD:.4f}\n")
    f.write(f"Minor accept threshold (fixed): {MIN_ACCEPT_THRESHOLD:.4f}\n")
    f.write("Cascade folds: %d\n\n" % GATE_FOLDS)
    for i, m in enumerate(cascade_fold_metrics, start=1):
        f.write(f"Fold {i}: {m}\n")
    f.write("\nPOOLED OOF REPORT:\n")
    f.write(classification_report(y, oof_cascade_pred, labels=all_labels, zero_division=0))

print(f"\nSaved averaged-metric outputs to: {out_dir}")
print("Done.")
