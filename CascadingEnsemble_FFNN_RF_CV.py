# ============================================================
# Cascade threshold tuning (NO SMOTE, NO KFold)
# Gate trained on BALANCED subset (downsample majority in TRAIN)
# Minority stage = RandomForestClassifier
#
# Goal: prevent gate collapsing to "always majority" which causes
#       predicted minority rate = 0% even when T_GATE is tiny.
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

try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    IN_COLAB = False


# ----------------------------
# Config
# ----------------------------
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# Threshold grids
T_GATE_GRID   = np.round(np.linspace(0.001, 0.20, 60), 4)
T_ACCEPT_GRID = np.round(np.linspace(0.00, 0.50, 51), 4)

# Minority RF config
RF_N_ESTIMATORS = 600
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_DEPTH = None
RF_N_JOBS = -1

# Gate NN config
GATE_EPOCHS = 40
GATE_BATCH = 512
GATE_LR = 1e-3
GATE_PATIENCE = 5

# Output (optional)
SAVE_TO_DRIVE = IN_COLAB
OUTPUT_DIR = "/content/drive/My Drive/ColabOutputs/Cascade_ThresholdTuning_RF"
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

def summarize_probs(name, p):
    p = np.asarray(p).reshape(-1)
    qs = np.quantile(p, [0.0, 0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99, 1.0])
    print(f"\n{name} prob summary:")
    print("  min  :", float(qs[0]))
    print("  p01  :", float(qs[1]))
    print("  p05  :", float(qs[2]))
    print("  p10  :", float(qs[3]))
    print("  p50  :", float(qs[4]))
    print("  p90  :", float(qs[5]))
    print("  p95  :", float(qs[6]))
    print("  p99  :", float(qs[7]))
    print("  max  :", float(qs[8]))

def train_gate_nn_balanced(X_train_s, y_train_bin, X_val_s, y_val_bin, majority_downsample_ratio=1.0):
    """
    Train gate on a BALANCED subset:
      - keep ALL minority samples
      - randomly sample majority to match minority count * ratio
    """
    idx_min = np.where(y_train_bin == 1)[0]
    idx_maj = np.where(y_train_bin == 0)[0]

    if idx_min.size == 0:
        raise ValueError("Gate TRAIN has 0 minority samples. Can't train a gate.")
    if idx_maj.size == 0:
        raise ValueError("Gate TRAIN has 0 majority samples. Weird.")

    n_min = idx_min.size
    n_maj_keep = int(n_min * majority_downsample_ratio)
    n_maj_keep = max(1, min(n_maj_keep, idx_maj.size))

    maj_sample = rng.choice(idx_maj, size=n_maj_keep, replace=False)
    idx_bal = np.concatenate([idx_min, maj_sample])
    rng.shuffle(idx_bal)

    Xb = X_train_s[idx_bal]
    yb = y_train_bin[idx_bal]

    print("\n========== Gate balanced training set ==========")
    print("Original gate TRAIN size:", len(y_train_bin),
          "  minority:", int((y_train_bin==1).sum()),
          "  majority:", int((y_train_bin==0).sum()))
    print("Balanced gate TRAIN size:", len(yb),
          "  minority:", int((yb==1).sum()),
          "  majority:", int((yb==0).sum()))

    model = Sequential([
        Input(shape=(X_train_s.shape[1],)),
        Dense(256, activation="relu"),
        Dropout(0.15),
        Dense(128, activation="relu"),
        Dropout(0.15),
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
        Xb, yb,
        validation_data=(X_val_s, y_val_bin),
        epochs=GATE_EPOCHS,
        batch_size=GATE_BATCH,
        callbacks=[es],
        verbose=1
    )
    return model

def train_minority_model_rf(X_min_train, y_min_train):
    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_depth=RF_MAX_DEPTH,
        class_weight="balanced_subsample",
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE
    )
    clf.fit(X_min_train, y_min_train)
    return clf

def run_cascade_predict(y_majority_label, gate_probs, minority_clf, X, T_GATE, T_ACCEPT, minority_labels):
    n = X.shape[0]
    y_pred = np.full(n, y_majority_label, dtype=int)

    idx = np.where(gate_probs >= T_GATE)[0]
    if idx.size == 0:
        return y_pred

    X_sub = X[idx]

    # accept threshold logic needs proba; RF has it.
    proba = minority_clf.predict_proba(X_sub)
    pmax = proba.max(axis=1)
    y_sub = minority_clf.classes_[proba.argmax(axis=1)]

    accept = (pmax >= T_ACCEPT) & np.isin(y_sub, minority_labels)
    y_pred[idx[accept]] = y_sub[accept]
    return y_pred


# -----------------------------
# Load data
# -----------------------------
CSV_PATH = "/content/LLCP_SMOKE_5YEAR.csv"
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV_PATH not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH}")
print("Shape:", df.shape)

if "CNCRTYP1" not in df.columns:
    raise ValueError("Expected column CNCRTYP1 in the CSV.")

# ----------------------------
# Select classes (majority + top-20 minorities)
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

if len(topk_minority) == 0:
    raise ValueError("No minority labels selected. CNCRTYP1 seems to contain only the majority class in this file.")

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
# Split (stratified)
# ----------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=RANDOM_STATE, stratify=y_trainval
)

print("\n========== Split ==========")
print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# ----------------------------
# Scale
# ----------------------------
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

# ----------------------------
# Train Gate on BALANCED TRAIN subset
# ----------------------------
y_train_bin = gate_binary_labels(y_train, majority_class)
y_val_bin   = gate_binary_labels(y_val, majority_class)

print("\n========== Train Gate NN (balanced TRAIN) ==========")
gate_model = train_gate_nn_balanced(X_train_s, y_train_bin, X_val_s, y_val_bin, majority_downsample_ratio=1.0)

gate_val_prob  = gate_model.predict(X_val_s, verbose=0).reshape(-1)
gate_test_prob = gate_model.predict(X_test_s, verbose=0).reshape(-1)

summarize_probs("Gate VAL", gate_val_prob)
summarize_probs("Gate TEST", gate_test_prob)

print("\n========== Gate send-rate diagnostics (VAL) ==========")
for tg in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
    print(f"T_GATE={tg:>6} -> sent: {float((gate_val_prob >= tg).mean())*100:6.2f}%")

# ----------------------------
# Train Minority RF on TRUE minority TRAIN ONLY
# ----------------------------
minority_labels = np.array(topk_minority, dtype=int)

is_min_train = (y_train != majority_class)
X_min_train = X_train_s[is_min_train]
y_min_train = y_train[is_min_train]

print("\n========== Train Minority Model (RF; true minority train only) ==========")
print("Minority train samples:", X_min_train.shape[0])
print("Minority labels in TRAIN:", sorted(np.unique(y_min_train).tolist()))

minority_model = train_minority_model_rf(X_min_train, y_min_train)

# ----------------------------
# Threshold tuning on VAL
# ----------------------------
print("\n========== Threshold Tuning on VAL (objective = minority-macro-F1) ==========")

best = {"score": -1.0, "T_GATE": None, "T_MINOR_ACCEPT": None}

min_val_proba_full = minority_model.predict_proba(X_val_s)
min_val_classes = minority_model.classes_

for T_GATE in T_GATE_GRID:
    gate_idx = np.where(gate_val_prob >= T_GATE)[0]

    if gate_idx.size == 0:
        # everything majority
        y_val_pred = np.full_like(y_val, majority_class)
        score = safe_f1_macro_minority(y_val, y_val_pred, minority_labels)
        if score > best["score"]:
            best.update(score=float(score), T_GATE=float(T_GATE), T_MINOR_ACCEPT=float(T_ACCEPT_GRID[0]))
        continue

    proba_sub = min_val_proba_full[gate_idx]
    pmax_sub = proba_sub.max(axis=1)
    yhat_sub = min_val_classes[proba_sub.argmax(axis=1)]

    for T_ACC in T_ACCEPT_GRID:
        y_val_pred = np.full_like(y_val, majority_class)
        accept = (pmax_sub >= T_ACC) & np.isin(yhat_sub, minority_labels)
        y_val_pred[gate_idx[accept]] = yhat_sub[accept]

        score = safe_f1_macro_minority(y_val, y_val_pred, minority_labels)
        if score > best["score"]:
            best.update(score=float(score), T_GATE=float(T_GATE), T_MINOR_ACCEPT=float(T_ACC))

print("Best VAL minority-macro-F1:", best["score"])
print("Best thresholds: T_GATE =", best["T_GATE"], ", T_MINOR_ACCEPT =", best["T_MINOR_ACCEPT"])

# ----------------------------
# Evaluate on TEST
# ----------------------------
print("\n========== Final Cascade (TEST) ==========")
T_GATE = best["T_GATE"]
T_ACC  = best["T_MINOR_ACCEPT"]

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
print(f"Predicted minority rate on TEST: {float((y_test_pred != majority_class).mean())*100:.2f}%")

report = classification_report(y_test, y_test_pred, zero_division=0)
print(report)

minority_macro = safe_f1_macro_minority(y_test, y_test_pred, minority_labels)
print("\nMinority macro-F1 on TEST (over selected minority labels):", float(minority_macro))

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
