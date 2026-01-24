# ============================================================
# FULL CODE: Cascade A ONLY (Gate NN + Minority-only Stage)
# Goal: improve Top-20 minority macro-F1 by restoring performance on big
# minority classes (esp. 21 & 22) while keeping rare classes reasonable.
#
# Key change vs your last run:
# - Prior adjustment is OPTIONAL and defaulted OFF in the sweep (alpha=0)
# - Weighting gamma reduced/disabled in sweep
# - cap_per_class increased in sweep
# - more trees
#
# Tuning:
# - For each outer fold, we tune minority-stage hyperparams using an INNER
#   split from the fold TRAIN data only (no leakage).
# - The chosen config is then evaluated on the fold VAL.
#
# Prints:
# - Per-fold top-20 minority macro-F1, mean±std
# - Pooled OOF per-class precision/recall/F1/support
# - Pooled OOF Top-20 minority macro-F1
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
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
rng = np.random.default_rng(SEED)

# -----------------------------
# Data
# -----------------------------
CSV_PATH   = "/content/LLCP2017_2018_2019_2020_2021XPT_LINEAR_WHOICD_5YEAR.csv"
TARGET_COL = "CNCRTYP1"
DROP_COLS  = ["CNCRTYP1", "CNCRAGE"]

TOP_K_MINORITY = 20

# -----------------------------
# CV
# -----------------------------
FOLDS = 5

# -----------------------------
# Gate NN hyperparams
# -----------------------------
GATE_EPOCHS   = 50
GATE_BATCH    = 256
GATE_LR       = 1e-3
GATE_PATIENCE = 6

# -----------------------------
# Cascade thresholds
# -----------------------------
GATE_THRESHOLD       = 0.50
MIN_ACCEPT_THRESHOLD = 0.00

# ============================================================
# Helpers
# ============================================================
def safe_macro_f1_on_labels(y_true, y_pred, labels):
    labels = [int(l) for l in labels]
    present = sorted(set(np.unique(y_true)).intersection(labels))
    if len(present) == 0:
        return 0.0
    return float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))

def per_class_table(y_true, y_pred):
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(rep).T.reset_index().rename(columns={"index": "class"})
    df = df[df["class"].astype(str).str.fullmatch(r"-?\d+")].copy()
    df["class"] = df["class"].astype(int)
    df = df[["class","precision","recall","f1-score","support"]].sort_values("class")
    return df

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

def compute_sample_weights(y, gamma=0.1):
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    freq = {c: cnt for c, cnt in zip(classes, counts)}
    w = np.array([1.0 / (freq[yi] ** gamma) for yi in y], dtype=np.float64)
    return w / np.maximum(w.mean(), 1e-12)

def class_priors_from_y(y, classes):
    y = np.asarray(y)
    counts = np.array([(y == c).sum() for c in classes], dtype=np.float64)
    priors = counts / np.maximum(counts.sum(), 1.0)
    return np.clip(priors, 1e-12, 1.0)

def adjust_probs_with_priors(probs, priors, alpha=0.0, temp=1.0):
    # alpha=0 means "no prior adjust"
    if alpha == 0 and temp == 1.0:
        return probs
    adj = probs / (priors ** alpha)
    if temp != 1.0:
        adj = np.power(np.clip(adj, 1e-12, 1.0), 1.0 / temp)
    return adj / np.maximum(adj.sum(axis=1, keepdims=True), 1e-12)

def cascade_predict(gate_probs, dt_classes, dt_proba, majority_class, gate_t, accept_t):
    y_pred = np.full(len(gate_probs), majority_class, dtype=int)
    idx = np.where(gate_probs >= gate_t)[0]
    if len(idx) == 0:
        return y_pred
    probs = dt_proba[idx]
    maxp = probs.max(axis=1)
    pred_class = dt_classes[np.argmax(probs, axis=1)].astype(int)
    accept = (maxp >= accept_t)
    y_pred[idx[accept]] = pred_class[accept]
    return y_pred

# ============================================================
# Minority-stage: Downsample-only bagged DT (NO oversampling)
# ============================================================
def downsample_only_indices(y, cap_per_class, rng):
    y = np.asarray(y)
    idx_all = np.arange(len(y))
    picked = []
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        idx_c = idx_all[y == c]
        if n > cap_per_class:
            idx_c = rng.choice(idx_c, size=cap_per_class, replace=False)
        picked.append(idx_c)
    picked = np.concatenate(picked)
    rng.shuffle(picked)
    return picked

def fit_downsample_bagged_dt(
    X_tr_min, y_tr_min, *,
    n_trees=120,
    cap_per_class=6000,
    max_depth=18,
    min_leaf=3,
    max_features="sqrt",
    weight_gamma=0.0,
    rng=None
):
    classes = np.unique(y_tr_min).astype(int)
    trees = []
    for t in range(n_trees):
        idx = downsample_only_indices(y_tr_min, cap_per_class, rng)
        Xb, yb = X_tr_min[idx], y_tr_min[idx]

        dt = DecisionTreeClassifier(
            random_state=SEED + t,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            min_samples_split=2,
            max_features=max_features,
            class_weight=None
        )

        if weight_gamma and weight_gamma > 0:
            sw = compute_sample_weights(yb, gamma=weight_gamma)
            dt.fit(Xb, yb, sample_weight=sw)
        else:
            dt.fit(Xb, yb)

        trees.append(dt)
    return classes, trees

def predict_proba_bagged(X, classes, trees):
    proba_sum = np.zeros((len(X), len(classes)), dtype=np.float64)
    class_to_col = {c: i for i, c in enumerate(classes)}
    for dt in trees:
        dt_classes = dt.classes_.astype(int)
        p = dt.predict_proba(X)
        mapped = np.zeros((len(X), len(classes)), dtype=np.float64)
        for j, c in enumerate(dt_classes):
            mapped[:, class_to_col[int(c)]] = p[:, j]
        proba_sum += mapped
    proba = proba_sum / max(len(trees), 1)
    return proba / np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)

# ============================================================
# Load + clean
# ============================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH}")
print("Shape:", df.shape)

if TARGET_COL not in df.columns:
    raise ValueError(f"Expected column '{TARGET_COL}' not found in CSV.")

df = df.copy()
df = df[~df[TARGET_COL].isin([77, 99])].copy()

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL]).copy()
df[TARGET_COL] = df[TARGET_COL].astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

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

print("\n========== Build Features ==========")
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)
y_gate = (y != majority_class).astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Gate minority rate:", y_gate.mean())

# ============================================================
# Outer CV folds
# ============================================================
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(X))

oof_pred = np.full(len(y), majority_class, dtype=int)
fold_top20_macro = []

# ============================================================
# Minority-stage hyperparam sweep (train-only inner split)
# These are the knobs most likely to recover 21/22 while not destroying rare ones.
# ============================================================
SWEEP = [
    # cap_per_class, n_trees, max_depth, min_leaf, weight_gamma, prior_alpha, temp
    (6000, 120, 18, 3, 0.0, 0.0, 1.0),
    (6000, 120, 18, 3, 0.1, 0.0, 1.0),
    (8000, 120, 18, 3, 0.0, 0.0, 1.0),
    (8000, 120, 18, 3, 0.1, 0.0, 1.0),

    # small, gentle prior adjust variants (optional)
    (8000, 120, 18, 3, 0.0, 0.1, 1.1),
    (8000, 120, 18, 3, 0.1, 0.1, 1.1),
]

INNER_VAL_FRAC = 0.2  # inner validation split from fold TRAIN only

print("\n============================================================")
print("========== Running Cascade A (Gate + Minority-only Stage) ==========")
print("============================================================")
print(f"Gate threshold={GATE_THRESHOLD:.3f}, accept threshold={MIN_ACCEPT_THRESHOLD:.3f}")
print("Minority stage: downsample-only bagged DT with per-fold inner tuning (NO leakage)")

for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
    print(f"\n--- Fold {fold_i}/{FOLDS} ---")

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    yg_tr, yg_va = y_gate[tr_idx], y_gate[va_idx]

    # ---- Gate ----
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

    # ---- Minority-only fold-train ----
    min_mask_tr = (y_tr != majority_class)
    X_tr_min = X_tr[min_mask_tr]
    y_tr_min = y_tr[min_mask_tr]

    if len(np.unique(y_tr_min)) < 2:
        print("WARNING: not enough minority variety in fold-train -> predicting majority.")
        va_pred = np.full(len(y_va), majority_class, dtype=int)
        oof_pred[va_idx] = va_pred
        f1_fold = safe_macro_f1_on_labels(y_va, va_pred, topk_minority)
        fold_top20_macro.append(f1_fold)
        print("Fold top-20 minority macro-F1:", round(float(f1_fold), 4))
        print("VAL minority rate:", round(float((y_va != majority_class).mean()), 6))
        continue

    # ---- Inner split for tuning (TRAIN ONLY) ----
    sss = StratifiedShuffleSplit(n_splits=1, test_size=INNER_VAL_FRAC, random_state=SEED)
    inner_tr_idx, inner_te_idx = next(sss.split(X_tr_min, y_tr_min))
    X_in_tr, y_in_tr = X_tr_min[inner_tr_idx], y_tr_min[inner_tr_idx]
    X_in_te, y_in_te = X_tr_min[inner_te_idx], y_tr_min[inner_te_idx]

    best_cfg = None
    best_score = -1.0

    for (cap, nt, md, ml, wg, pa, tp) in SWEEP:
        dt_classes, trees = fit_downsample_bagged_dt(
            X_in_tr, y_in_tr,
            n_trees=nt,
            cap_per_class=cap,
            max_depth=md,
            min_leaf=ml,
            max_features="sqrt",
            weight_gamma=wg,
            rng=rng
        )

        p_te = predict_proba_bagged(X_in_te, dt_classes, trees)

        if pa != 0.0 or tp != 1.0:
            pri = class_priors_from_y(y_in_tr, dt_classes)
            p_te = adjust_probs_with_priors(p_te, pri, alpha=pa, temp=tp)

        y_te_pred = dt_classes[np.argmax(p_te, axis=1)].astype(int)
        score = safe_macro_f1_on_labels(y_in_te, y_te_pred, topk_minority)

        if score > best_score:
            best_score = score
            best_cfg = (cap, nt, md, ml, wg, pa, tp)

    cap, nt, md, ml, wg, pa, tp = best_cfg
    print(f"Chosen minority-stage cfg (train-only tuning): cap={cap}, trees={nt}, depth={md}, leaf={ml}, "
          f"wg={wg}, prior_alpha={pa}, temp={tp} | inner top-20 macro-F1={best_score:.4f}")

    # ---- Refit minority-stage on FULL fold-train minority with best cfg ----
    dt_classes, trees = fit_downsample_bagged_dt(
        X_tr_min, y_tr_min,
        n_trees=nt,
        cap_per_class=cap,
        max_depth=md,
        min_leaf=ml,
        max_features="sqrt",
        weight_gamma=wg,
        rng=rng
    )

    p_va = predict_proba_bagged(X_va, dt_classes, trees)

    if pa != 0.0 or tp != 1.0:
        pri = class_priors_from_y(y_tr_min, dt_classes)
        p_va = adjust_probs_with_priors(p_va, pri, alpha=pa, temp=tp)

    va_pred = cascade_predict(
        gate_probs=va_gate_probs,
        dt_classes=dt_classes,
        dt_proba=p_va,
        majority_class=majority_class,
        gate_t=GATE_THRESHOLD,
        accept_t=MIN_ACCEPT_THRESHOLD
    )

    oof_pred[va_idx] = va_pred

    f1_fold = safe_macro_f1_on_labels(y_va, va_pred, topk_minority)
    fold_top20_macro.append(f1_fold)

    print("Fold top-20 minority macro-F1:", round(float(f1_fold), 4))
    print("VAL minority rate:", round(float((y_va != majority_class).mean()), 6))

# ============================================================
# Summary
# ============================================================
fold_top20_macro = np.array(fold_top20_macro, dtype=float)
mu = fold_top20_macro.mean()
sd = fold_top20_macro.std(ddof=1) if len(fold_top20_macro) > 1 else 0.0

print("\n============================================================")
print("========== SUMMARY (Top-20 minority macro-F1 across folds) ==========")
print("============================================================")
print("Per-fold:", [round(float(x), 4) for x in fold_top20_macro.tolist()])
print(f"Mean ± std: {mu:.6f} ± {sd:.6f}")

print("\n============================================================")
print("========== POOLED OOF RESULTS ==========")
print("============================================================")

df_f1 = per_class_table(y, oof_pred)
print("\n========== Per-class F1 scores (POOLED OOF) ==========")
print(df_f1.to_string(index=False))

minority_macro = safe_macro_f1_on_labels(y, oof_pred, topk_minority)
print("\n========== Minority-only macro-F1 (top-20) ==========")
print("Minority labels considered:", topk_minority)
print("Minority macro-F1:", minority_macro)

print("\n========== POOLED OOF classification_report (all classes) ==========")
all_labels = [majority_class] + [lab for lab in sorted(np.unique(y).tolist()) if lab != majority_class]
print(classification_report(y, oof_pred, labels=all_labels, zero_division=0))

print("\nDone.")
