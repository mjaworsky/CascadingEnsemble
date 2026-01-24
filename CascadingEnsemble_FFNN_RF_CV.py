# ============================================================
# Cascade threshold tuning (NO SMOTE, NO KFold)
# Tunes:
#   1) Gate threshold T_GATE (majority vs minority)
#   2) Minority accept threshold T_MINOR_ACCEPT (reject -> majority)
#
# Objective on VAL:
#   minority-macro-F1 over the selected minority labels
#
# Outputs:
#   - Prints best thresholds + VAL scores
#   - Evaluates on TEST using the chosen thresholds
#   - Saves full reports + thresholds JSON to Drive (optional)
#
# Assumptions:
#   - CSV has column 'CNCRTYP1' as target
#   - Majority class is the most frequent class in CNCRTYP1
#   - You may filter out invalid CNCRTYP1 beforehand if needed
#
# NOTE: This is "full code" runnable in Colab.
# ============================================================

import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# (Optional) Colab Drive + upload
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False


# ----------------------------
# Config
# ----------------------------
RANDOM_STATE = 42

# Threshold grids (adjust if you want finer search)
T_GATE_GRID = np.round(np.linspace(0.01, 0.50, 50), 4)          # 50 points
T_ACCEPT_GRID = np.round(np.linspace(0.05, 0.95, 19), 4)        # 19 points

# Minority model (stage 2) config (RANDOM FOREST)
RF_N_ESTIMATORS = 600
RF_MIN_SAMPLES_LEAF = 2
RF_N_JOBS = -1

# Gate NN config
GATE_EPOCHS = 30
GATE_BATCH = 256
GATE_LR = 1e-3
GATE_PATIENCE = 5

# Output (optional)
SAVE_TO_DRIVE = IN_COLAB
OUTPUT_DIR = "/content/drive/My Drive/ColabOutputs/Cascade_ThresholdTuning"
REPORT_NAME = "cascade_threshold_tuning_report.txt"
THRESH_NAME = "best_thresholds.json"


# ----------------------------
# Utilities
# ----------------------------
def safe_f1_macro_minority(y_true, y_pred, minority_labels):
    """Macro-F1 over minority labels only."""
    return f1_score(
        y_true, y_pred,
        labels=minority_labels,
        average="macro",
        zero_division=0
    )

def train_gate_nn(X_train, y_train_bin, X_val, y_val_bin):
    """Simple MLP gate: outputs P(minority)."""
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(256, activation="relu"),
        Dropout(0.1),
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=Adam(learning_rate=GATE_LR),
        loss="binary_crossentropy",
        metrics=[]
    )

    es = EarlyStopping(
        monitor="val_loss",
        patience=GATE_PATIENCE,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train_bin,
        validation_data=(X_val, y_val_bin),
        epochs=GATE_EPOCHS,
        batch_size=GATE_BATCH,
        callbacks=[es],
        verbose=1
    )
    return model

def train_minority_model_random_forest(X_min_train, y_min_train):
    """
    Train stage-2 minority classifier: RANDOM FOREST.
    """
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE
    )
    clf.fit(X_min_train, y_min_train)
    return clf

def run_cascade_predict(
    y_majority_label,
    gate_probs,
    minority_clf,
    X,
    T_GATE,
    T_ACCEPT,
    minority_labels
):
    """
    Build final predictions:
      - default to majority
      - if gate says minority (>= T_GATE):
          - get minority model proba
          - accept if max proba >= T_ACCEPT else revert to majority
    """
    n = X.shape[0]
    y_pred = np.full(n, y_majority_label, dtype=int)

    gate_is_min = gate_probs >= T_GATE
    idx = np.where(gate_is_min)[0]
    if idx.size == 0:
        return y_pred

    X_sub = X[idx]

    # Some sklearn classifiers may not support predict_proba if configured oddly.
    if not hasattr(minority_clf, "predict_proba"):
        # fallback: no accept threshold; always predict
        y_sub = minority_clf.predict(X_sub)
        y_pred[idx] = y_sub
        return y_pred

    proba = minority_clf.predict_proba(X_sub)
    pmax = proba.max(axis=1)
    class_idx = proba.argmax(axis=1)
    classes = minority_clf.classes_
    y_sub = classes[class_idx]

    accept = pmax >= T_ACCEPT
    # Accept only labels we consider minority (safety)
    accept = accept & np.isin(y_sub, minority_labels)

    y_pred[idx[accept]] = y_sub[accept]
    # else remain majority
    return y_pred

def gate_binary_labels(y, majority_label):
    """Binary labels: 0=majority, 1=minority."""
    return (y != majority_label).astype(int)


# -----------------------------
# Load data (already uploaded / already present)
# -----------------------------
CSV_PATH = "/content/LLCP2017_2018_2019_2020_2021XPT_LINEAR_WHOICD_5YEAR.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV_PATH not found: {CSV_PATH}\n"
        "Fix CSV_PATH to point to your already-uploaded file. "
        "Tip (Colab): run !ls /content to see filenames."
    )

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH}")
print("Shape:", df.shape)

if "CNCRTYP1" not in df.columns:
    raise ValueError("Expected column CNCRTYP1 in the CSV.")

# ----------------------------
# Select classes
# ----------------------------
class_counts = df["CNCRTYP1"].value_counts()
majority_class = int(class_counts.idxmax())

# "All 30 minority classes": take up to 20 most common non-majority labels
minority_candidates = class_counts[class_counts.index != majority_class]
topk_minority = [int(x) for x in minority_candidates.head(20).index.tolist()]

selected_classes = [majority_class] + topk_minority
df = df[df["CNCRTYP1"].isin(selected_classes)].copy()

print("\n========== Select Classes ==========")
print("Majority class:", majority_class)
print("Selected minority labels:", topk_minority)
print("\nClass distribution (selected):")
print(df["CNCRTYP1"].value_counts())

# ----------------------------
# Build features
# ----------------------------
y = df["CNCRTYP1"].astype(int).values

drop_cols = ["CNCRTYP1"]
if "CNCRAGE" in df.columns:
    drop_cols.append("CNCRAGE")

X_df = df.drop(columns=drop_cols)
X_df = X_df.select_dtypes(include=[np.number]).copy()
X = X_df.values.astype(np.float32)

print("\n========== Build Features ==========")
print("X shape:", X.shape, "y shape:", y.shape)

# ----------------------------
# Split into TRAIN / VAL / TEST (stratified)
# ----------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_trainval
)

print("\n========== Split ==========")
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# ----------------------------
# Scale features (train-only)
# ----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ----------------------------
# Train Gate (majority vs minority)
# ----------------------------
y_train_bin = gate_binary_labels(y_train, majority_class)
y_val_bin = gate_binary_labels(y_val, majority_class)

print("\n========== Train Gate NN ==========")
gate_model = train_gate_nn(X_train_s, y_train_bin, X_val_s, y_val_bin)

gate_val_prob = gate_model.predict(X_val_s, verbose=0).reshape(-1)
gate_test_prob = gate_model.predict(X_test_s, verbose=0).reshape(-1)

# ----------------------------
# Train Minority model on TRUE minority TRAIN ONLY (RANDOM FOREST)
# ----------------------------
minority_labels = np.array(topk_minority, dtype=int)

is_min_train = y_train != majority_class
X_min_train = X_train_s[is_min_train]
y_min_train = y_train[is_min_train]

print("\n========== Train Minority Model (Random Forest; true minority train only) ==========")
print("Minority train samples:", X_min_train.shape[0])
print("Minority labels in TRAIN:", sorted(np.unique(y_min_train).tolist()))

minority_model = train_minority_model_random_forest(X_min_train, y_min_train)

# ----------------------------
# Threshold tuning on VAL
# ----------------------------
print("\n========== Threshold Tuning on VAL (objective = minority-macro-F1) ==========")

best = {"score": -1.0, "T_GATE": None, "T_MINOR_ACCEPT": None}

has_proba = hasattr(minority_model, "predict_proba")
if has_proba:
    min_val_proba_full = minority_model.predict_proba(X_val_s)
    min_val_classes = minority_model.classes_
else:
    min_val_proba_full = None
    min_val_classes = None

for T_GATE in T_GATE_GRID:
    gate_idx = np.where(gate_val_prob >= T_GATE)[0]

    if gate_idx.size == 0:
        y_val_pred = np.full_like(y_val, majority_class)
        score = safe_f1_macro_minority(y_val, y_val_pred, minority_labels)
        if score > best["score"]:
            best.update(score=float(score), T_GATE=float(T_GATE), T_MINOR_ACCEPT=float(T_ACCEPT_GRID[0]))
        continue

    if has_proba:
        proba_sub = min_val_proba_full[gate_idx]
        pmax_sub = proba_sub.max(axis=1)
        argmax_sub = proba_sub.argmax(axis=1)
        yhat_sub = min_val_classes[argmax_sub]
    else:
        yhat_sub = minority_model.predict(X_val_s[gate_idx])
        pmax_sub = None

    for T_ACC in T_ACCEPT_GRID:
        y_val_pred = np.full_like(y_val, majority_class)

        if has_proba:
            accept = (pmax_sub >= T_ACC) & np.isin(yhat_sub, minority_labels)
            y_val_pred[gate_idx[accept]] = yhat_sub[accept]
        else:
            y_val_pred[gate_idx] = yhat_sub

        score = safe_f1_macro_minority(y_val, y_val_pred, minority_labels)

        if score > best["score"]:
            best.update(score=float(score), T_GATE=float(T_GATE), T_MINOR_ACCEPT=float(T_ACC))

print("Best VAL minority-macro-F1:", best["score"])
print("Best thresholds: T_GATE =", best["T_GATE"], ", T_MINOR_ACCEPT =", best["T_MINOR_ACCEPT"])

# ----------------------------
# Evaluate on TEST with best thresholds
# ----------------------------
print("\n========== Final Cascade (TEST) ==========")
T_GATE = best["T_GATE"]
T_ACC = best["T_MINOR_ACCEPT"]

y_test_pred = run_cascade_predict(
    y_majority_label=majority_class,
    gate_probs=gate_test_prob,
    minority_clf=minority_model,
    X=X_test_s,
    T_GATE=T_GATE,
    T_ACCEPT=T_ACC,
    minority_labels=minority_labels
)

print("T_GATE =", T_GATE, "T_MINOR_ACCEPT =", T_ACC)

report = classification_report(y_test, y_test_pred, zero_division=0)
print(report)

minority_macro = safe_f1_macro_minority(y_test, y_test_pred, minority_labels)
print("\nMinority macro-F1 on TEST (over selected minority labels):", minority_macro)

# ----------------------------
# Save report (optional)
# ----------------------------
if SAVE_TO_DRIVE:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, REPORT_NAME)
    thresh_path = os.path.join(OUTPUT_DIR, THRESH_NAME)

    with open(report_path, "w") as f:
        f.write("=== Best thresholds (selected on VAL) ===\n")
        f.write(json.dumps(best, indent=2))
        f.write("\n\n=== TEST Classification Report ===\n")
        f.write(report)
        f.write(f"\nMinority macro-F1 on TEST: {minority_macro}\n")

    with open(thresh_path, "w") as f:
        json.dump(best, f, indent=2)

    print("\nSaved report to:", report_path)
    print("Saved thresholds to:", thresh_path)

print("\nDone.")
