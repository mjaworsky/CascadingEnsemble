# ================================
# Cascade (NO SMOTE, NO K-FOLD, NO VAL/TUNING)
# Gate NN (majority vs minority) + Minority Decision Tree (30 minority classes)
# Prints full TEST classification report for labels: [majority] + minority_labels
# ================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

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
TOP_K_MINORITY = 30
TEST_SIZE = 0.20

GATE_EPOCHS = 50
GATE_BATCH = 256
GATE_LR = 1e-3
GATE_PATIENCE = 6

# Decision Tree hyperparams (start simple; tune later if needed)
DT_MAX_DEPTH = None
DT_MIN_SAMPLES_LEAF = 1
DT_MIN_SAMPLES_SPLIT = 2

# Fixed thresholds (NO VAL tuning)
GATE_THRESHOLD = 0.50          # gate: route to minority DT if P(minority) >= this
MIN_ACCEPT_THRESHOLD = 0.00    # accept DT prediction if DT max-proba >= this (0 = always accept)

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
DROP_COLS = ["CNCRTYP1", "CNCRAGE"]  # keep CNCRAGE out of features (as you’ve been doing)

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
# Select majority and "all 30 minority classes"
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
# Build X/y
# -----------------------------
print("\n========== Build Features ==========")

X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore").values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)

# Gate labels: 0 = majority, 1 = minority
y_gate = (y != majority_class).astype(int)

# -----------------------------
# Split: train/test only (NO CV)
# stratify on gate so minority presence preserved
# -----------------------------
print("\n========== Split (train/test only) ==========")

X_train, X_test, y_train, y_test, yg_train, yg_test = train_test_split(
    X, y, y_gate,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y_gate
)

print("Shapes:", X_train.shape, X_test.shape, yg_train.shape, yg_test.shape)

# -----------------------------
# Scale features (fit TRAIN only!)
# -----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# -----------------------------
# Train Gate NN (binary)
# Uses validation_split on TRAIN for early stopping (not CV)
# -----------------------------
print("\n========== Train Gate NN ==========")

gate_classes = np.unique(yg_train)
gate_class_weights = compute_class_weight(class_weight="balanced", classes=gate_classes, y=yg_train)
gate_class_weight_dict = {int(c): float(w) for c, w in zip(gate_classes, gate_class_weights)}
print("Gate class weights:", gate_class_weight_dict)

gate = Sequential([
    Input(shape=(X_train_s.shape[1],)),
    Dense(256, activation="relu"),
    Dropout(0.15),
    Dense(128, activation="relu"),
    Dropout(0.15),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid"),
])

gate.compile(
    optimizer=Adam(learning_rate=GATE_LR),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early = EarlyStopping(
    monitor="val_loss",
    patience=GATE_PATIENCE,
    restore_best_weights=True
)

gate.fit(
    X_train_s, yg_train,
    validation_split=0.2,
    epochs=GATE_EPOCHS,
    batch_size=GATE_BATCH,
    class_weight=gate_class_weight_dict,
    callbacks=[early],
    verbose=1
)

# -----------------------------
# Gate report on TEST (fixed threshold)
# -----------------------------
test_gate_probs = gate.predict(X_test_s, verbose=0).ravel()
test_gate_pred  = (test_gate_probs >= GATE_THRESHOLD).astype(int)

print("\n========== Gate Report (majority vs minority) [TEST] ==========")
print("Gate threshold:", GATE_THRESHOLD)
print(classification_report(yg_test, test_gate_pred, zero_division=0))

# -----------------------------
# Train Minority Decision Tree on TRUE minority TRAIN samples only
# -----------------------------
print("\n========== Train Minority Model (true minority train only) ==========")

minority_train_mask = (y_train != majority_class)
minority_test_mask  = (y_test  != majority_class)

y_train_min = y_train[minority_train_mask]
X_train_min = X_train_s[minority_train_mask]

y_test_min = y_test[minority_test_mask]
X_test_min = X_test_s[minority_test_mask]

minority_labels_train = sorted(np.unique(y_train_min).tolist())
print("Minority labels in TRAIN:", minority_labels_train)
print("Minority train samples:", len(y_train_min))
print("Minority test samples :", len(y_test_min))

dt = DecisionTreeClassifier(
    random_state=SEED,
    max_depth=DT_MAX_DEPTH,
    min_samples_leaf=DT_MIN_SAMPLES_LEAF,
    min_samples_split=DT_MIN_SAMPLES_SPLIT,
    class_weight="balanced"
)

print("Fitting Decision Tree...")
dt.fit(X_train_min, y_train_min)
print("DT fit done.")

# Minority-only report (on TRUE minority TEST only)
min_pred_test = dt.predict(X_test_min)

print("\n========== Minority DT Report (only true minority samples) ==========")
print(classification_report(y_test_min, min_pred_test, zero_division=0))

# -----------------------------
# Final cascade on TEST (fixed thresholds)
# If gate routes to minority but DT max-proba < MIN_ACCEPT_THRESHOLD, fallback to majority
# -----------------------------
print("\n========== Final Cascade (TEST) ==========")

# DT probabilities for all TEST samples (only used when routed)
dt_classes = dt.classes_
dt_proba_test = dt.predict_proba(X_test_s)

def cascade_predict(y_true, gate_probs, dt_proba, gate_t, accept_t):
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
    # otherwise: remain majority_class
    return y_pred

y_test_pred = cascade_predict(
    y_test,
    test_gate_probs,
    dt_proba_test,
    gate_t=GATE_THRESHOLD,
    accept_t=MIN_ACCEPT_THRESHOLD
)

all_labels_present_test = sorted(np.unique(y_test).tolist())
printed_minority_labels = [lab for lab in all_labels_present_test if lab != majority_class]

print(f"Gate threshold (fixed) = {GATE_THRESHOLD:.3f}")
print(f"Minor accept threshold (fixed) = {MIN_ACCEPT_THRESHOLD:.3f}")

print(classification_report(
    y_test, y_test_pred,
    labels=[majority_class] + printed_minority_labels,
    zero_division=0
))

print("Minority labels present in TEST (printed in report):", printed_minority_labels)

# Optional: save report to file
out_dir = "/content/Cascade_NoSMOTE_NoKFold_NoVal_All30_DT"
os.makedirs(out_dir, exist_ok=True)

report_txt = classification_report(
    y_test, y_test_pred,
    labels=[majority_class] + printed_minority_labels,
    zero_division=0
)

with open(os.path.join(out_dir, "cascade_report_test.txt"), "w") as f:
    f.write("Gate threshold (fixed): %.4f\n" % GATE_THRESHOLD)
    f.write("Minor accept threshold (fixed): %.4f\n\n" % MIN_ACCEPT_THRESHOLD)
    f.write(report_txt)

print(f"\nSaved TEST report to: {os.path.join(out_dir, 'cascade_report_test.txt')}")
print("Done.")
