# ============================================================
# Cascade threshold tuning (NO SMOTE, NO KFold)
# Gate NN (majority vs minority) + Minority Gradient Boost classifier
# Tunes thresholds on VAL for minority-macro-F1
# ============================================================

import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import HistGradientBoostingClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False


# ----------------------------
# Config
# ----------------------------
RANDOM_STATE = 42

T_GATE_GRID = np.round(np.linspace(0.01, 0.50, 50), 4)
T_ACCEPT_GRID = np.round(np.linspace(0.05, 0.95, 19), 4)

# Minority Gradient Boost config (HistGradientBoostingClassifier)
HGB_MAX_ITER = 400
HGB_LEARNING_RATE = 0.05
HGB_MAX_LEAF_NODES = 31
HGB_MAX_DEPTH = None
HGB_MIN_SAMPLES_LEAF = 20
HGB_L2_REG = 0.0
HGB_EARLY_STOPPING = True
HGB_VALIDATION_FRACTION = 0.1
HGB_N_ITER_NO_CHANGE = 20
HGB_TOL = 1e-7

# Gate NN config
GATE_EPOCHS = 30
GATE_BATCH = 256
GATE_LR = 1e-3
GATE_PATIENCE = 5

SAVE_TO_DRIVE = IN_COLAB
OUTPUT_DIR = "/content/drive/My Drive/ColabOutputs/Cascade_ThresholdTuning"
REPORT_NAME = "cascade_threshold_tuning_report.txt"
THRESH_NAME = "best_thresholds.json"


# ----------------------------
# Utilities
# ----------------------------
def safe_f1_macro_minority(y_true, y_pred, minority_labels):
    return f1_score(
        y_true, y_pred,
        labels=minority_labels,
        average="macro",
        zero_division=0
    )

def gate_binary_labels(y, majority_label):
    return (y != majority_label).astype(int)

def train_gate_nn(X_train, y_train_bin, X_val, y_val_bin):
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

def make_balanced_sample_weights(y):
    """
    Compute inverse-frequency weights per sample:
      w_c = N / (K * n_c)
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    n = y.size
    k = classes.size
    w_by_class = {c: (n / (k * cnt)) for c, cnt in zip(classes, counts)}
    w = np.array([w_by_class[v] for v in y], dtype=np.float32)
    # mild clipping to avoid insane weights if a class is extremely tiny
    return np.clip(w, 0.5, 50.0)

def train_minority_model_hgb(X_min_train, y_min_train):
    """
    Train stage-2 minority classifier: HistGradientBoostingClassifier
    with per-sample balancing weights.
    """
    sample_weight = make_balanced_sample_weights(y_min_train)

    clf = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=HGB_LEARNING_RATE,
        max_iter=HGB_MAX_ITER,
        max_leaf_nodes=HGB_MAX_LEAF_NODES,
        max_depth=HGB_MAX_DEPTH,
        min_samples_leaf=HGB_MIN_SAMPLES_LEAF,
        l2_regularization=HGB_L2_REG,
        early_stopping=HGB_EARLY_STOPPING,
        validation_fraction=HGB_VALIDATION_FRACTION,
        n_iter_no_change=HGB_N_ITER_NO_CHANGE,
        tol=HGB_TOL,
        random_state=RANDOM_STATE
    )
    clf.fit(X_min_train, y_min_train, sample_weight=sample_weight)
    return clf

class AlwaysMajorityFallback:
    """
    Fallback minority model if there are no minority samples/classes to train on.
    """
    def __init__(self, majority_label):
        self.majority_label = int(majority_label)

    def predict(self, X):
        return np.full(X.shape[0], self.majority_label, dtype=int)

def run_cascade_predict(
    y_majority_label,
    gate_probs,
    minority_clf,
    X,
    T_GATE,
    T_ACCEPT,
    minority_labels
):
    n = X.shape[0]
    y_pred = np.full(n, y_majority_label, dtype=int)

    gate_is_min = gate_probs >= T_GATE
    idx = np.where(gate_is_min)[0]
    if idx.size == 0:
        return y_pred

    X_sub = X[idx]

    if not hasattr(minority_clf, "predict_proba"):
        y_sub = minority_clf.predict(X_sub)
        allow = np.isin(y_sub, minority_labels)
        y_pred[idx[allow]] = y_sub[allow]
        return y_pred

    proba = minority_clf.predict_proba(X_sub)
    pmax = proba.max(axis=1)
    class_idx = proba.argmax(axis=1)
    classes = minority_clf.classes_
    y_sub = classes[class_idx]

    accept = (pmax >= T_ACCEPT) & np.isin(y_sub, minority_labels)
    y_pred[idx[accept]] = y_sub[accept]
    return y_pred


# -----------------------------
# Load data
# -----------------------------
CSV_PATH = "/content/LLCP2017_2018_2019_2020_2021XPT_LINEAR_WHOICD_5YEAR.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"CSV_PATH not found: {CSV_PATH}\n"
        "Fix CSV_PATH to point to your already-uploaded file."
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
# Split into TRAIN / VAL / TEST
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
X_trainval_s = scaler.transform(X_trainval)

# ----------------------------
# Train Gate
# ----------------------------
y_train_bin = gate_binary_labels(y_train, majority_class)
y_val_bin = gate_binary_labels(y_val, majority_class)

print("\n========== Train Gate NN ==========")
gate_model = train_gate_nn(X_train_s, y_train_bin, X_val_s, y_val_bin)

gate_val_prob = gate_model.predict(X_val_s, verbose=0).reshape(-1)
gate_test_prob = gate_model.predict(X_test_s, verbose=0).reshape(-1)

# ----------------------------
# Train Minority model on TRUE minority TRAIN+VAL only (Gradient Boost)
# ----------------------------
minority_labels = np.array(topk_minority, dtype=int)

is_min_trainval = y_trainval != majority_class
X_min_train = X_trainval_s[is_min_trainval]
y_min_train = y_trainval[is_min_trainval]

print("\n========== Train Minority Model (HistGradientBoosting; true minority TRAIN+VAL only) ==========")
print("Minority train samples:", X_min_train.shape[0])
print("Minority labels in TRAIN+VAL:", sorted(np.unique(y_min_train).tolist()))

unique_min_classes = np.unique(y_min_train)
if X_min_train.shape[0] == 0 or unique_min_classes.size < 2:
    print("WARNING: Not enough minority data to train stage-2 model. Using fallback model (always majority).")
    minority_model = AlwaysMajorityFallback(majority_class)
else:
    minority_model = train_minority_model_hgb(X_min_train, y_min_train)

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
            allow = np.isin(yhat_sub, minority_labels)
            y_val_pred[gate_idx[allow]] = yhat_sub[allow]

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
