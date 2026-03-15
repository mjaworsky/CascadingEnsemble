!pip install xgboost

# ============================================================
# CASCADE WITH CV (XGBoost version, multiclass minority)
#
# Use this when:
#   - target has 1 majority class + multiple minority subtypes
#   - e.g. CNCRTYP1 = 0,1,2,...,30
#
# Stages:
#   (1) Gate NN: majority (0) vs minority (!=0)
#   (2) Minority XGBoost: classify minority subtype labels
#   (3) Full cascade OOF using same gate folds
#
# Features:
#   - Gate NN uses scaling
#   - XGBoost uses raw X (no scaling)
#   - RandomOverSampler on training splits only
#   - RepeatedStratifiedKFold for minority XGB stage
#   - Full pooled OOF reports
#   - Saves outputs
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

from imblearn.over_sampling import RandomOverSampler

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from xgboost import XGBClassifier

# -----------------------------
# Repro
# -----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# User settings
# -----------------------------
TARGET_COL = "CNCRTYP1"
DROP_COLS = ["CNCRTYP1", "CNCRAGE"]

CSV_PATH = "/content/LLCP2017_2018_2019_2020_2021XPT_LINEAR_WHOICD_5YEAR.csv"
OUT_DIR = "/content/Cascade_CV_AveragedMetrics_XGB_MULTICLASS"
os.makedirs(OUT_DIR, exist_ok=True)

TOP_K_MINORITY = 20

# Gate NN hyperparams
GATE_EPOCHS = 50
GATE_BATCH = 256
GATE_LR = 1e-3
GATE_PATIENCE = 6

# XGBoost hyperparams
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.85
XGB_COLSAMPLE_BYTREE = 0.85
XGB_MIN_CHILD_WEIGHT = 4
XGB_REG_ALPHA = 0.0
XGB_REG_LAMBDA = 1.0
XGB_GAMMA = 0.0
XGB_N_JOBS = -1

# Thresholds
GATE_THRESHOLD = 0.50
MIN_ACCEPT_THRESHOLD = 0.00

# CV settings
GATE_FOLDS = 5
XGB_RS_FOLDS = 3
XGB_RS_REPEATS = 3   # reduce if runtime is too long; increase later if needed

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH}")
print("Shape:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(f"Expected target column '{TARGET_COL}' not found.")

# -----------------------------
# Clean target
# -----------------------------
df = df.copy()
df = df[~df[TARGET_COL].isin([77, 99])].copy()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

print("\n========== RAW TARGET DISTRIBUTION ==========")
raw_counts = df[TARGET_COL].value_counts().sort_index()
print(raw_counts)
print("Unique classes:", sorted(df[TARGET_COL].unique().tolist()))

# -----------------------------
# Select majority + TOP_K minority classes
# -----------------------------
class_counts = df[TARGET_COL].value_counts()
majority_class = int(class_counts.idxmax())

minority_candidates = class_counts[class_counts.index != majority_class]
topk_minority = [int(x) for x in minority_candidates.head(TOP_K_MINORITY).index.tolist()]

selected_classes = [majority_class] + topk_minority
df = df[df[TARGET_COL].isin(selected_classes)].copy()

print("\n========== SELECTED CLASSES ==========")
print("Majority class:", majority_class)
print("Selected minority labels:", topk_minority)
print("\nSelected class distribution:")
print(df[TARGET_COL].value_counts())

if len(topk_minority) < 2:
    raise ValueError(
        f"Need at least 2 minority subtype labels for the cascade minority stage, "
        f"but found {len(topk_minority)}."
    )

# -----------------------------
# Build X / y / y_gate
# -----------------------------
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)
y_gate = (y != majority_class).astype(int)

print("\n========== BUILD FEATURES ==========")
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


def make_balanced_sample_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_to_weight = {int(c): float(wi) for c, wi in zip(classes, w)}
    return np.array([class_to_weight[int(v)] for v in y_train], dtype=np.float32)


def make_xgb_multiclass(num_classes: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE_BYTREE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        gamma=XGB_GAMMA,
        random_state=SEED,
        n_jobs=XGB_N_JOBS,
        tree_method="hist",
        eval_metric="mlogloss",
        verbosity=0,
    )


def summarize_binary_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
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


def safe_macro_f1_on_labels(y_true, y_pred, labels):
    labels = [int(l) for l in labels]
    present = sorted(set(int(x) for x in np.unique(y_true)).intersection(labels))
    if len(present) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def cascade_predict_from_probs(
    gate_probs: np.ndarray,
    minor_classes: np.ndarray,
    minor_proba: np.ndarray,
    majority_class: int,
    gate_t: float,
    accept_t: float
) -> np.ndarray:
    y_pred = np.full(shape=(len(gate_probs),), fill_value=majority_class, dtype=int)

    route_min = (gate_probs >= gate_t)
    idx = np.where(route_min)[0]
    if len(idx) == 0:
        return y_pred

    probs = minor_proba[idx]
    maxp = probs.max(axis=1)
    pred_class = minor_classes[np.argmax(probs, axis=1)]

    accept = (maxp >= accept_t)
    y_pred[idx[accept]] = pred_class[accept]
    return y_pred


# ============================================================
# (1) Gate NN: 5-fold NON-STRATIFIED CV
# ============================================================
print("\n============================================================")
print("========== (1) Gate NN: 5-FOLD NON-STRATIFIED CV ==========")
print("============================================================")

kf_gate = KFold(n_splits=GATE_FOLDS, shuffle=True, random_state=SEED)

oof_gate_probs = np.zeros(len(y), dtype=float)
oof_gate_pred = np.zeros(len(y), dtype=int)
gate_fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(kf_gate.split(X), start=1):
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
    pred = (probs >= GATE_THRESHOLD).astype(int)

    oof_gate_probs[va_idx] = probs
    oof_gate_pred[va_idx] = pred

    m = summarize_binary_metrics(yg_va, pred)
    gate_fold_metrics.append(m)
    print("Fold metrics:", {k: round(float(v), 4) for k, v in m.items()})

print_mean_std(
    gate_fold_metrics,
    keys=["acc", "bal_acc", "prec", "rec", "f1"],
    title="Gate NN: mean ± std across 5 folds"
)

print("\n========== Gate NN POOLED OOF REPORT ==========")
print(classification_report(y_gate, oof_gate_pred, zero_division=0))

# ============================================================
# (2) Minority XGBoost: RepeatedStratified CV + Oversampling
# ============================================================
print("\n============================================================")
print("========== (2) Minority XGB: RepeatedStratified CV + Oversampling ==========")
print("============================================================")

minor_mask = (y != majority_class)
X_min = X[minor_mask]
y_min = y[minor_mask]

minority_labels = sorted(np.unique(y_min).tolist())
lab_to_idx = {lab: i for i, lab in enumerate(minority_labels)}
idx_to_lab = {i: lab for lab, i in lab_to_idx.items()}

y_min_enc = np.array([lab_to_idx[int(v)] for v in y_min], dtype=int)

print("Minority sample count:", len(y_min))
print("Minority label count :", len(minority_labels))

rskf_xgb = RepeatedStratifiedKFold(
    n_splits=XGB_RS_FOLDS,
    n_repeats=XGB_RS_REPEATS,
    random_state=SEED
)

proba_sum = np.zeros((len(y_min), len(minority_labels)), dtype=np.float64)
proba_cnt = np.zeros(len(y_min), dtype=np.int32)

xgb_split_metrics = []
total_splits = XGB_RS_FOLDS * XGB_RS_REPEATS

for split_i, (tr_idx, va_idx) in enumerate(rskf_xgb.split(X_min, y_min_enc), start=1):
    X_tr, X_va = X_min[tr_idx], X_min[va_idx]
    y_tr_enc, y_va_enc = y_min_enc[tr_idx], y_min_enc[va_idx]

    ros = RandomOverSampler(random_state=SEED)
    X_tr_bal, y_tr_bal_enc = ros.fit_resample(X_tr, y_tr_enc)

    sw = make_balanced_sample_weights(y_tr_bal_enc)

    xgb = make_xgb_multiclass(num_classes=len(minority_labels))
    xgb.fit(X_tr_bal, y_tr_bal_enc, sample_weight=sw)

    xgb_proba = xgb.predict_proba(X_va)

    proba_sum[va_idx] += xgb_proba
    proba_cnt[va_idx] += 1

    pred_enc = np.argmax(xgb_proba, axis=1)
    pred = np.array([idx_to_lab[int(i)] for i in pred_enc], dtype=int)
    y_va = np.array([idx_to_lab[int(i)] for i in y_va_enc], dtype=int)

    m = summarize_multiclass_metrics(y_va, pred)
    xgb_split_metrics.append(m)

    if split_i % XGB_RS_FOLDS == 0 or split_i == total_splits:
        print(f"Completed {split_i}/{total_splits} splits")

print_mean_std(
    xgb_split_metrics,
    keys=["acc", "bal_acc", "f1_macro", "f1_weighted"],
    title=f"Minority XGB: mean ± std across {total_splits} splits"
)

avg_proba = proba_sum / np.maximum(proba_cnt[:, None], 1)
oof_xgb_pred_min_enc = np.argmax(avg_proba, axis=1)
oof_xgb_pred_min = np.array([idx_to_lab[int(i)] for i in oof_xgb_pred_min_enc], dtype=int)

print("\n========== Minority XGB POOLED OOF REPORT ==========")
print(classification_report(y_min, oof_xgb_pred_min, zero_division=0))

minority_macro_f1_minority_stage = safe_macro_f1_on_labels(
    y_true=y_min,
    y_pred=oof_xgb_pred_min,
    labels=topk_minority
)
print("\nMinority macro-F1 on OOF (minority XGB stage, selected labels):",
      minority_macro_f1_minority_stage)

# ============================================================
# (3) Full Cascade OOF using SAME 5 gate folds
# ============================================================
print("\n============================================================")
print("========== (3) Full Cascade OOF (5 folds) + Oversampling ==========")
print("============================================================")

oof_cascade_pred = np.full(len(y), majority_class, dtype=int)
cascade_fold_metrics = []

for fold, (tr_idx, va_idx) in enumerate(kf_gate.split(X), start=1):
    print(f"\n--- Cascade Fold {fold}/{GATE_FOLDS} ---")

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    yg_tr = y_gate[tr_idx]

    # ---- Train gate ----
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

    va_gate_probs = gate.predict(X_va_s, verbose=0).ravel()

    # ---- Train minority XGB on minority train rows only ----
    min_tr_mask = (y_tr != majority_class)
    X_tr_min = X_tr[min_tr_mask]
    y_tr_min = y_tr[min_tr_mask]

    fold_minority_labels = sorted(np.unique(y_tr_min).tolist())
    fold_lab_to_idx = {lab: i for i, lab in enumerate(fold_minority_labels)}
    fold_idx_to_lab = {i: lab for lab, i in fold_lab_to_idx.items()}

    y_tr_min_enc = np.array([fold_lab_to_idx[int(v)] for v in y_tr_min], dtype=int)

    ros = RandomOverSampler(random_state=SEED)
    X_tr_min_bal, y_tr_min_bal_enc = ros.fit_resample(X_tr_min, y_tr_min_enc)

    sw = make_balanced_sample_weights(y_tr_min_bal_enc)

    xgb = make_xgb_multiclass(num_classes=len(fold_minority_labels))
    xgb.fit(X_tr_min_bal, y_tr_min_bal_enc, sample_weight=sw)

    xgb_proba_va = xgb.predict_proba(X_va)
    xgb_pred_labels = np.array(
        [fold_idx_to_lab[int(i)] for i in range(len(fold_minority_labels))],
        dtype=int
    )

    va_pred = cascade_predict_from_probs(
        gate_probs=va_gate_probs,
        minor_classes=xgb_pred_labels,
        minor_proba=xgb_proba_va,
        majority_class=majority_class,
        gate_t=GATE_THRESHOLD,
        accept_t=MIN_ACCEPT_THRESHOLD
    )

    oof_cascade_pred[va_idx] = va_pred

    m = summarize_multiclass_metrics(y_va, va_pred)
    cascade_fold_metrics.append(m)
    print("Fold metrics:", {k: round(float(v), 4) for k, v in m.items()})
    print("Fold minority rate (VAL):", round(float((y_va != majority_class).mean()), 6))

print_mean_std(
    cascade_fold_metrics,
    keys=["acc", "bal_acc", "f1_macro", "f1_weighted"],
    title="Cascade: mean ± std across 5 folds"
)

all_labels = [majority_class] + [lab for lab in sorted(np.unique(y).tolist()) if lab != majority_class]

print("\n========== Full Cascade POOLED OOF REPORT ==========")
print(f"Gate threshold = {GATE_THRESHOLD:.3f}")
print(f"Minor accept threshold = {MIN_ACCEPT_THRESHOLD:.3f}")
print(classification_report(y, oof_cascade_pred, labels=all_labels, zero_division=0))

minority_macro_f1_cascade = safe_macro_f1_on_labels(
    y_true=y,
    y_pred=oof_cascade_pred,
    labels=topk_minority
)
print("\nMinority macro-F1 on OOF (FULL CASCADE, selected labels):",
      minority_macro_f1_cascade)

# -----------------------------
# Save outputs
# -----------------------------
with open(os.path.join(OUT_DIR, "gate_5fold_non_strat_meanstd_and_oof.txt"), "w") as f:
    f.write(f"Gate threshold: {GATE_THRESHOLD:.4f}\n")
    f.write(f"Gate folds: {GATE_FOLDS}\n\n")
    for i, m in enumerate(gate_fold_metrics, start=1):
        f.write(f"Fold {i}: {m}\n")
    f.write("\nPOOLED OOF REPORT:\n")
    f.write(classification_report(y_gate, oof_gate_pred, zero_division=0))

with open(os.path.join(OUT_DIR, "minority_xgb_repeatedcv_meanstd_and_oof.txt"), "w") as f:
    f.write(f"Selected minority labels: {topk_minority}\n")
    f.write(f"RepeatedStratifiedKFold: n_splits={XGB_RS_FOLDS}, n_repeats={XGB_RS_REPEATS}\n")
    f.write("Oversampling: RandomOverSampler on each TRAIN split\n")
    f.write(
        f"XGB params: n_estimators={XGB_N_ESTIMATORS}, max_depth={XGB_MAX_DEPTH}, "
        f"learning_rate={XGB_LEARNING_RATE}, subsample={XGB_SUBSAMPLE}, "
        f"colsample_bytree={XGB_COLSAMPLE_BYTREE}, min_child_weight={XGB_MIN_CHILD_WEIGHT}, "
        f"reg_alpha={XGB_REG_ALPHA}, reg_lambda={XGB_REG_LAMBDA}, gamma={XGB_GAMMA}\n\n"
    )
    for i, m in enumerate(xgb_split_metrics, start=1):
        f.write(f"Split {i}: {m}\n")
    f.write("\nPOOLED OOF REPORT:\n")
    f.write(classification_report(y_min, oof_xgb_pred_min, zero_division=0))
    f.write("\n\nMinority macro-F1 on OOF (minority XGB stage, selected labels): "
            f"{minority_macro_f1_minority_stage:.10f}\n")

with open(os.path.join(OUT_DIR, "cascade_5fold_non_strat_meanstd_and_oof.txt"), "w") as f:
    f.write(f"Gate threshold: {GATE_THRESHOLD:.4f}\n")
    f.write(f"Minor accept threshold: {MIN_ACCEPT_THRESHOLD:.4f}\n")
    f.write(f"Cascade folds: {GATE_FOLDS}\n")
    f.write(f"Selected minority labels: {topk_minority}\n")
    f.write("Oversampling: RandomOverSampler on minority TRAIN per fold\n")
    f.write(
        f"XGB params: n_estimators={XGB_N_ESTIMATORS}, max_depth={XGB_MAX_DEPTH}, "
        f"learning_rate={XGB_LEARNING_RATE}, subsample={XGB_SUBSAMPLE}, "
        f"colsample_bytree={XGB_COLSAMPLE_BYTREE}, min_child_weight={XGB_MIN_CHILD_WEIGHT}, "
        f"reg_alpha={XGB_REG_ALPHA}, reg_lambda={XGB_REG_LAMBDA}, gamma={XGB_GAMMA}\n\n"
    )
    for i, m in enumerate(cascade_fold_metrics, start=1):
        f.write(f"Fold {i}: {m}\n")
    f.write("\nPOOLED OOF REPORT:\n")
    f.write(classification_report(y, oof_cascade_pred, labels=all_labels, zero_division=0))
    f.write("\n\nMinority macro-F1 on OOF (FULL CASCADE, selected labels): "
            f"{minority_macro_f1_cascade:.10f}\n")

with open(os.path.join(OUT_DIR, "selected_class_distribution.txt"), "w") as f:
    f.write(df[TARGET_COL].value_counts().to_string())
    f.write("\n")

print(f"\nSaved outputs to: {OUT_DIR}")
print("Done.")

