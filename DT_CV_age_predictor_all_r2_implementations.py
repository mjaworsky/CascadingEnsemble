# ============================================================
# FULL CODE:
# Cascade A for CNCRTYP1
# +
# CNCRAGE per-class F1 table for positive CNCRTYP1 only
# +
# Added regression-style metrics for CNCRAGE:
#   - Multiple R² implementations / variants
#   - MAE
#   - Residual plots
#
# Final age metrics shown:
#   F1 score of every CNCRAGE
#   evaluated only on rows where CNCRTYP1 != majority_class
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score
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
rng = np.random.default_rng(SEED)

# -----------------------------
# Data
# -----------------------------
CSV_PATH        = "/content/LLCP_SMOKE_5YEAR.csv"
TARGET_COL      = "CNCRTYP1"
AGE_COL         = "CNCRAGE"
DROP_COLS       = ["CNCRTYP1", "CNCRAGE"]

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

# -----------------------------
# Age model hyperparams
# One CNCRAGE model per CNCRTYP1
# -----------------------------
AGE_DT_MAX_DEPTH = 12
AGE_DT_MIN_LEAF  = 5

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
    df = df[["class", "precision", "recall", "f1-score", "support"]].sort_values("class")
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
# Minority-stage: Downsample-only bagged DT
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
# CNCRAGE models: train one age model per CNCRTYP1
# ============================================================
def fit_age_models_by_type(X_train, y_type_train, y_age_train):
    age_models = {}
    fallback_age = {}

    for cancer_type in sorted(np.unique(y_type_train)):
        mask = (y_type_train == cancer_type)
        X_sub = X_train[mask]
        y_sub = y_age_train[mask]

        if len(y_sub) == 0:
            continue

        vals, counts = np.unique(y_sub, return_counts=True)
        fallback_age[int(cancer_type)] = int(vals[np.argmax(counts)])

        # If only one age class exists for this CNCRTYP1 in train,
        # use constant fallback instead of fitting a model.
        if len(vals) < 2:
            age_models[int(cancer_type)] = None
            continue

        dt_age = DecisionTreeClassifier(
            random_state=SEED,
            max_depth=AGE_DT_MAX_DEPTH,
            min_samples_leaf=AGE_DT_MIN_LEAF,
            class_weight="balanced"
        )
        dt_age.fit(X_sub, y_sub)
        age_models[int(cancer_type)] = dt_age

    return age_models, fallback_age

def predict_age_by_true_type(X_val, y_type_val, age_models, fallback_age, global_fallback_age):
    pred_age = np.full(len(X_val), global_fallback_age, dtype=int)

    for cancer_type in np.unique(y_type_val):
        cancer_type = int(cancer_type)
        mask = (y_type_val == cancer_type)

        model = age_models.get(cancer_type, None)
        fallback = fallback_age.get(cancer_type, global_fallback_age)

        if model is None:
            pred_age[mask] = fallback
        else:
            pred_age[mask] = model.predict(X_val[mask]).astype(int)

    return pred_age

def cncrage_f1_positive_only(y_type, y_age_true, y_age_pred, majority_class):
    """
    Compute one-vs-rest F1 for each CNCRAGE class,
    only on rows where CNCRTYP1 != majority_class.
    """
    mask = (y_type != majority_class)

    yt = np.asarray(y_age_true)[mask]
    yp = np.asarray(y_age_pred)[mask]

    rows = []
    for age in sorted(np.unique(yt)):
        yt_bin = (yt == age).astype(int)
        yp_bin = (yp == age).astype(int)

        rows.append({
            "CNCRAGE": int(age),
            "support": int((yt == age).sum()),
            "F1": float(f1_score(yt_bin, yp_bin, zero_division=0))
        })

    return pd.DataFrame(rows).sort_values("CNCRAGE")

def safe_float(x):
    """Convert library metric outputs to plain Python float where possible."""
    try:
        if hasattr(x, "detach"):  # torch tensor
            x = x.detach().cpu().numpy()
        if hasattr(x, "numpy"):   # tensorflow tensor
            x = x.numpy()
        return float(np.asarray(x).reshape(-1)[0])
    except Exception:
        return np.nan


def bounded_predictive_r2(y_true, y_pred):
    """
    Bounded / truncated predictive R²-like score.

    Standard predictive R² is: 1 - SSE/SST.
    It can be negative when predictions are worse than predicting the mean.
    This version clips the standard predictive R² into [0, 1].

    Interpretation:
      1 = perfect prediction
      0 = no improvement over the mean, or worse than the mean

    Important: this is NOT standard OLS in-sample R².
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    if sst <= 0:
        return np.nan

    raw_r2 = 1.0 - (sse / sst)
    return float(np.clip(raw_r2, 0.0, 1.0))


def squared_pearson_r2(y_true, y_pred):
    """
    Squared Pearson correlation between observed and predicted values.

    This is bounded [0, 1], but it measures association/calibration only.
    It is not the same as predictive R² unless the predictions are from an
    OLS model with an intercept and are evaluated in-sample.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r ** 2)


def collect_r2_implementations(y_true, y_pred):
    """
    Compute R² values from available Python implementations.

    Some libraries may not be installed in the runtime environment. Those rows
    are still printed as 'not available' so the output is transparent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rows = []

    def add(name, value, bounded, note):
        rows.append({
            "Metric / library": name,
            "Value": value,
            "Bounded [0,1]": bounded,
            "Notes": note
        })

    # 1) scikit-learn: standard predictive R². Can be negative.
    try:
        add(
            "sklearn.metrics.r2_score",
            float(r2_score(y_true, y_pred)),
            "No",
            "Standard predictive R²; can be negative for poor out-of-sample predictions."
        )
    except Exception as e:
        add("sklearn.metrics.r2_score", np.nan, "No", f"Error: {e}")

    # 2) Manual standard predictive R², equivalent to sklearn for simple 1-output case.
    try:
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        manual = np.nan if sst <= 0 else 1.0 - (sse / sst)
        add(
            "manual predictive R² = 1 - SSE/SST",
            float(manual),
            "No",
            "Formula equivalent to sklearn R² for this single-output case."
        )
    except Exception as e:
        add("manual predictive R² = 1 - SSE/SST", np.nan, "No", f"Error: {e}")

    # 3) Bounded / truncated version requested for reviewer-safe display.
    try:
        add(
            "bounded predictive R² = max(0, 1 - SSE/SST)",
            bounded_predictive_r2(y_true, y_pred),
            "Yes",
            "Truncated score; 0 means no better than mean prediction, or worse. Not standard OLS R²."
        )
    except Exception as e:
        add("bounded predictive R² = max(0, 1 - SSE/SST)", np.nan, "Yes", f"Error: {e}")

    # 4) Squared Pearson correlation: bounded association measure.
    try:
        add(
            "numpy squared Pearson correlation r²",
            squared_pearson_r2(y_true, y_pred),
            "Yes",
            "Bounded association measure; not a predictive error R²."
        )
    except Exception as e:
        add("numpy squared Pearson correlation r²", np.nan, "Yes", f"Error: {e}")

    # 5) scipy squared Pearson correlation, if scipy is installed.
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(y_true, y_pred)
        add(
            "scipy.stats.pearsonr squared r²",
            float(r ** 2),
            "Yes",
            "Same idea as squared Pearson correlation; association, not predictive error."
        )
    except Exception as e:
        add("scipy.stats.pearsonr squared r²", np.nan, "Yes", f"Not available/error: {e}")

    # 6) statsmodels OLS in-sample R²: regress y_true on y_pred with an intercept.
    try:
        import statsmodels.api as sm
        X_sm = sm.add_constant(y_pred)
        ols_model = sm.OLS(y_true, X_sm).fit()
        add(
            "statsmodels OLS rsquared: y_true ~ intercept + y_pred",
            float(ols_model.rsquared),
            "Yes",
            "In-sample OLS association/calibration R² with intercept; bounded [0,1]. Not original model predictive R²."
        )
        add(
            "statsmodels OLS adjusted rsquared",
            float(ols_model.rsquared_adj),
            "No",
            "Adjusted R² can be negative. Included for completeness."
        )
    except Exception as e:
        add("statsmodels OLS rsquared", np.nan, "Yes", f"Not available/error: {e}")
        add("statsmodels OLS adjusted rsquared", np.nan, "No", f"Not available/error: {e}")

    # 7) TensorFlow / Keras R2Score, if available in installed TF/Keras version.
    try:
        metric = tf.keras.metrics.R2Score()
        metric.update_state(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
        add(
            "tf.keras.metrics.R2Score",
            safe_float(metric.result()),
            "No",
            "TensorFlow/Keras standard R²; can be negative."
        )
    except Exception as e:
        add("tf.keras.metrics.R2Score", np.nan, "No", f"Not available/error: {e}")

    # 8) TensorFlow Addons RSquare, if installed.
    try:
        import tensorflow_addons as tfa
        metric = tfa.metrics.RSquare()
        metric.update_state(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))
        add(
            "tensorflow_addons.metrics.RSquare",
            safe_float(metric.result()),
            "No",
            "TensorFlow Addons standard R²; can be negative."
        )
    except Exception as e:
        add("tensorflow_addons.metrics.RSquare", np.nan, "No", f"Not available/error: {e}")

    # 9) TorchMetrics R2Score, if installed.
    try:
        import torch
        from torchmetrics.regression import R2Score
        metric = R2Score()
        value = metric(
            torch.tensor(y_pred, dtype=torch.float32),
            torch.tensor(y_true, dtype=torch.float32)
        )
        add(
            "torchmetrics.regression.R2Score",
            safe_float(value),
            "No",
            "TorchMetrics standard R²; can be negative."
        )
    except Exception as e:
        add("torchmetrics.regression.R2Score", np.nan, "No", f"Not available/error: {e}")

    return pd.DataFrame(rows)


def cncrage_regression_metrics_positive_only(y_type, y_age_true, y_age_pred, majority_class):
    """
    Treat CNCRAGE as numeric ordinal labels and compute regression-style metrics
    only on rows where CNCRTYP1 != majority_class.

    This function now reports multiple R² implementations/variants so that the
    output clearly distinguishes standard predictive R² from bounded association
    or clipped R²-like scores.
    """
    mask = (np.asarray(y_type) != majority_class)

    yt = np.asarray(y_age_true)[mask].astype(float)
    yp = np.asarray(y_age_pred)[mask].astype(float)

    if len(yt) == 0:
        return {
            "R2": np.nan,
            "MAE": np.nan,
            "Residuals": np.array([]),
            "R2_Table": pd.DataFrame()
        }

    residuals = yt - yp
    r2_table = collect_r2_implementations(yt, yp)

    return {
        "R2": float(r2_score(yt, yp)),
        "R2_Bounded": bounded_predictive_r2(yt, yp),
        "R2_Pearson_Squared": squared_pearson_r2(yt, yp),
        "R2_Table": r2_table,
        "MAE": float(mean_absolute_error(yt, yp)),
        "Residuals": residuals,
        "y_true": yt,
        "y_pred": yp
    }

def plot_cncrage_residuals_positive_only(y_type, y_age_true, y_age_pred, majority_class):
    """
    Residual plots for CNCRAGE on positive CNCRTYP1 only.
    Residual = true age code - predicted age code
    """
    mask = (np.asarray(y_type) != majority_class)

    yt = np.asarray(y_age_true)[mask].astype(float)
    yp = np.asarray(y_age_pred)[mask].astype(float)

    if len(yt) == 0:
        print("No positive CNCRTYP1 rows available for residual plots.")
        return

    residuals = yt - yp

    # Plot 1: residuals vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(yp, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted CNCRAGE")
    plt.ylabel("Residual (True - Predicted)")
    plt.title("Residual Plot: CNCRAGE on Positive CNCRTYP1 Only")
    plt.tight_layout()
    plt.show()

    # Plot 2: histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (True - Predicted)")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution: CNCRAGE on Positive CNCRTYP1 Only")
    plt.tight_layout()
    plt.show()

# ============================================================
# Load + clean
# ============================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH}")
print("Shape:", df.shape)

for col in [TARGET_COL, AGE_COL]:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' not found in CSV.")

df = df.copy()

# Remove BRFSS unknown/refused codes
df = df[~df[TARGET_COL].isin([77, 99])].copy()
df = df[~df[AGE_COL].isin([77, 99])].copy()

df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df[AGE_COL]    = pd.to_numeric(df[AGE_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL, AGE_COL]).copy()

df[TARGET_COL] = df[TARGET_COL].astype(int)
df[AGE_COL]    = df[AGE_COL].astype(int)

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
y_age = df[AGE_COL].values.astype(int)
y_gate = (y != majority_class).astype(int)

global_age_mode = int(pd.Series(y_age).mode().iloc[0])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("y_age shape:", y_age.shape)
print("Gate minority rate:", y_gate.mean())

# ============================================================
# Outer CV folds
# ============================================================
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
folds = list(kf.split(X))

oof_pred = np.full(len(y), majority_class, dtype=int)
oof_pred_age = np.full(len(y_age), global_age_mode, dtype=int)

fold_top20_macro = []

# ============================================================
# Minority-stage hyperparam sweep
# ============================================================
SWEEP = [
    # cap_per_class, n_trees, max_depth, min_leaf, weight_gamma, prior_alpha, temp
    (6000, 120, 18, 3, 0.0, 0.0, 1.0),
    (6000, 120, 18, 3, 0.1, 0.0, 1.0),
    (8000, 120, 18, 3, 0.0, 0.0, 1.0),
    (8000, 120, 18, 3, 0.1, 0.0, 1.0),
    (8000, 120, 18, 3, 0.0, 0.1, 1.1),
    (8000, 120, 18, 3, 0.1, 0.1, 1.1),
]

INNER_VAL_FRAC = 0.2

print("\n============================================================")
print("========== Running Cascade A (Gate + Minority-only Stage) ==========")
print("============================================================")
print(f"Gate threshold={GATE_THRESHOLD:.3f}, accept threshold={MIN_ACCEPT_THRESHOLD:.3f}")
print("Minority stage: downsample-only bagged DT with per-fold inner tuning (NO leakage)")
print("Age stage: one DecisionTree CNCRAGE model per CNCRTYP1 inside each outer fold")

for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
    print(f"\n--- Fold {fold_i}/{FOLDS} ---")

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    yage_tr, yage_va = y_age[tr_idx], y_age[va_idx]
    yg_tr, yg_va = y_gate[tr_idx], y_gate[va_idx]

    # ---- Gate model ----
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

    # ---- Minority-only stage ----
    min_mask_tr = (y_tr != majority_class)
    X_tr_min = X_tr[min_mask_tr]
    y_tr_min = y_tr[min_mask_tr]

    if len(np.unique(y_tr_min)) < 2:
        print("WARNING: not enough minority variety in fold-train -> predicting majority.")
        va_pred = np.full(len(y_va), majority_class, dtype=int)
        oof_pred[va_idx] = va_pred
    else:
        # ---- Inner split for tuning (train-only) ----
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
        print(
            f"Chosen minority-stage cfg: cap={cap}, trees={nt}, depth={md}, "
            f"leaf={ml}, wg={wg}, prior_alpha={pa}, temp={tp} | "
            f"inner top-20 macro-F1={best_score:.4f}"
        )

        # ---- Refit on full fold train minority ----
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

    # ---- Age models by TRUE CNCRTYP1 using train fold only ----
    age_models, fallback_age = fit_age_models_by_type(X_tr, y_tr, yage_tr)

    va_pred_age = predict_age_by_true_type(
        X_val=X_va,
        y_type_val=y_va,
        age_models=age_models,
        fallback_age=fallback_age,
        global_fallback_age=global_age_mode
    )
    oof_pred_age[va_idx] = va_pred_age

    # ---- Fold summary ----
    f1_fold = safe_macro_f1_on_labels(y_va, va_pred, topk_minority)
    fold_top20_macro.append(f1_fold)

    print("Fold top-20 minority macro-F1:", round(float(f1_fold), 4))
    print("VAL minority rate:", round(float((y_va != majority_class).mean()), 6))

# ============================================================
# Summary: CNCRTYP1
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
print("========== POOLED OOF RESULTS: CNCRTYP1 ==========")
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

# ============================================================
# Final CNCRAGE output requested:
# F1 of every CNCRAGE for any positive CNCRTYP1
# + R² / MAE / residual plots
# ============================================================
print("\n============================================================")
print("========== CNCRAGE F1 (positive CNCRTYP1 only) ==========")
print("============================================================")

df_age_f1 = cncrage_f1_positive_only(
    y_type=y,
    y_age_true=y_age,
    y_age_pred=oof_pred_age,
    majority_class=majority_class
)

print(df_age_f1.to_string(index=False))

# -----------------------------
# Added regression-style metrics
# -----------------------------
age_metrics = cncrage_regression_metrics_positive_only(
    y_type=y,
    y_age_true=y_age,
    y_age_pred=oof_pred_age,
    majority_class=majority_class
)

print("\n============================================================")
print("========== CNCRAGE Regression-Style Metrics (positive CNCRTYP1 only) ==========")
print("============================================================")
print(f"Standard sklearn predictive R² : {age_metrics['R2']:.6f}")
print(f"Bounded predictive R² [0,1]   : {age_metrics['R2_Bounded']:.6f}")
print(f"Squared Pearson r² [0,1]      : {age_metrics['R2_Pearson_Squared']:.6f}")
print(f"MAE                           : {age_metrics['MAE']:.6f}")

print("\n============================================================")
print("========== All Available R² Implementations / Variants ==========")
print("============================================================")
print(age_metrics["R2_Table"].to_string(index=False))

print("\nMetric interpretation note:")
print("- sklearn / TensorFlow / TorchMetrics R² are standard predictive R² and may be negative.")
print("- The bounded predictive R² clips negative values to 0, so it is reviewer-friendly but not standard OLS R².")
print("- Squared Pearson r² and statsmodels OLS rsquared are bounded association/calibration measures, not direct predictive-error R².")

# -----------------------------
# Residual plots
# -----------------------------
plot_cncrage_residuals_positive_only(
    y_type=y,
    y_age_true=y_age,
    y_age_pred=oof_pred_age,
    majority_class=majority_class
)

print("\nDone.")
