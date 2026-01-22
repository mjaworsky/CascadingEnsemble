# ============================================================
# FAST + LEAKAGE-SAFE VERSION (as you requested)
#   - Train Neural Net ONCE (no CV for NN)
#   - Do 5-fold Stratified CV ONLY for the Decision Tree part
#   - In each fold:
#       * Train DT on fold-train with SMOTE (train fold only)
#       * Evaluate CASCADED predictions on fold-val using fixed NN + fold DT
#   - Also keep an untouched 20% holdout test for final report
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE, RandomOverSampler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from google.colab import files, drive
import joblib

# -----------------------------
# Config
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

N_SPLITS = 5  # your request: 5 folds, no repeats
SMOTE_BASE_K = 2

EPOCHS = 30
BATCH_SIZE = 256
VAL_SPLIT = 0.2
PATIENCE = 3
LEARNING_RATE = 1e-3

# -----------------------------
# Mount Drive + output folder
# -----------------------------
drive.mount('/content/drive')
output_folder = '/content/drive/My Drive/ColabOutputs/'
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Upload + read data
# -----------------------------
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

# -----------------------------
# Filter + select classes
# -----------------------------
df = df[~df['CNCRTYP1'].isin([77, 99])]

class_counts = df['CNCRTYP1'].value_counts()
majority_class = class_counts.idxmax()

top_20_minority_classes = class_counts[class_counts.index != majority_class].head(20).index
selected_classes = list(top_20_minority_classes) + [majority_class]

df_filtered = df[df['CNCRTYP1'].isin(selected_classes)].copy()

print("Class distribution after truncation:")
print(df_filtered['CNCRTYP1'].value_counts())

# Inputs / target
X = df_filtered.drop(columns=['CNCRTYP1', 'CNCRAGE']).values
y = df_filtered['CNCRTYP1'].values

# Encode target ONCE
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

majority_class_idx = label_encoder.transform([majority_class])[0]
num_classes = len(label_encoder.classes_)
input_dim = X.shape[1]

print("\nEncoded classes:", list(label_encoder.classes_))
print("Majority class:", majority_class, "-> idx:", majority_class_idx)
print("Num classes:", num_classes, "Input dim:", input_dim)

# -----------------------------
# Holdout split FIRST (untouched)
# -----------------------------
X_train_full, X_test_holdout, y_train_full, y_test_holdout = train_test_split(
    X, y_enc,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_enc
)

print("\nHoldout split sizes:")
print("Train:", X_train_full.shape, "Test:", X_test_holdout.shape)

# -----------------------------
# Helpers
# -----------------------------
def build_nn(input_dim: int, num_classes: int) -> tf.keras.Model:
    # A smaller NN = MUCH faster. Still does the job.
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def resample_training_fold_for_dt(X_tr, y_tr, random_state=RANDOM_STATE, smote_base_k=SMOTE_BASE_K):
    """
    Apply SMOTE only on DT training fold.
    Auto-adjust k if the fold has small minority counts; otherwise fallback to RandomOverSampler.
    """
    counts = np.bincount(y_tr)
    nonzero = counts[counts > 0]
    min_class_count = int(nonzero.min()) if len(nonzero) else 0

    # SMOTE requires min_class_count >= k_neighbors + 1
    max_k = max(1, min_class_count - 1)

    if min_class_count >= 2:
        k = min(smote_base_k, max_k)
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        X_rs, y_rs = sm.fit_resample(X_tr, y_tr)
        return X_rs, y_rs, f"SMOTE(k_neighbors={k})"
    else:
        ros = RandomOverSampler(random_state=random_state)
        X_rs, y_rs = ros.fit_resample(X_tr, y_tr)
        return X_rs, y_rs, "RandomOverSampler(fallback)"

def cascaded_predictions(nn_probs, dt_preds, majority_class_idx: int):
    """
    Your original cascade logic:
      - If NN predicts majority class -> keep NN prediction
      - Else -> use DT prediction
    """
    nn_preds = np.argmax(nn_probs, axis=1)
    final = np.empty_like(nn_preds)
    for i, nn_pred in enumerate(nn_preds):
        if nn_pred == majority_class_idx:
            final[i] = nn_pred
        else:
            final[i] = dt_preds[i]
    return final

# -----------------------------
# 1) Train Neural Net ONCE (no CV)
# -----------------------------
# Class weights help the NN not just scream "majority class" all day.
classes = np.unique(y_train_full)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_full)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

tf.keras.backend.clear_session()
nn_model = build_nn(input_dim=input_dim, num_classes=num_classes)
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

print("\nTraining Neural Network ONCE (no CV)...")
nn_model.fit(
    X_train_full, y_train_full,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[early_stopping],
    class_weight=class_weight,
    verbose=1
)

# -----------------------------
# 2) 5-Fold CV for DT ONLY (evaluate cascade on each fold)
# -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cv_rows = []
fold_num = 0

print("\nRunning 5-fold CV for Decision Tree ONLY (cascade evaluated per fold)...")
for tr_idx, va_idx in skf.split(X_train_full, y_train_full):
    fold_num += 1
    X_tr, X_va = X_train_full[tr_idx], X_train_full[va_idx]
    y_tr, y_va = y_train_full[tr_idx], y_train_full[va_idx]

    # Resample ONLY the DT training fold
    X_tr_rs, y_tr_rs, resample_desc = resample_training_fold_for_dt(X_tr, y_tr)

    # Train DT on resampled fold-train
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_tr_rs, y_tr_rs)

    # Cascade eval on fold-val (no resampling here)
    nn_probs_va = nn_model.predict(X_va, verbose=0)
    dt_preds_va = dt.predict(X_va)
    final_va = cascaded_predictions(nn_probs_va, dt_preds_va, majority_class_idx)

    f1w = f1_score(y_va, final_va, average='weighted', zero_division=0)
    pw  = precision_score(y_va, final_va, average='weighted', zero_division=0)
    rw  = recall_score(y_va, final_va, average='weighted', zero_division=0)

    cv_rows.append({
        "fold": fold_num,
        "dt_resampling": resample_desc,
        "f1_weighted": f1w,
        "precision_weighted": pw,
        "recall_weighted": rw
    })

    print(f"Fold {fold_num:02d} | {resample_desc:28s} | F1w={f1w:.4f} Pw={pw:.4f} Rw={rw:.4f}")

cv_df = pd.DataFrame(cv_rows)

print("\n================ DT-only CV (Cascade) Summary ================")
print(cv_df[["f1_weighted","precision_weighted","recall_weighted"]].describe().loc[["mean","std","min","max"]])

cv_path = os.path.join(output_folder, "cv_metrics_dt_only_cascade_5fold.csv")
cv_df.to_csv(cv_path, index=False)
print(f"\nSaved CV metrics to: {cv_path}")

# -----------------------------
# 3) Train FINAL DT on full training set (SMOTE on train only), evaluate on holdout test
# -----------------------------
X_train_rs, y_train_rs, resample_desc_final = resample_training_fold_for_dt(X_train_full, y_train_full)
print("\nTraining FINAL DT on full training set with:", resample_desc_final)

dt_final = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_final.fit(X_train_rs, y_train_rs)

# Holdout cascade evaluation
nn_probs_test = nn_model.predict(X_test_holdout, verbose=0)
dt_preds_test = dt_final.predict(X_test_holdout)
final_test = cascaded_predictions(nn_probs_test, dt_preds_test, majority_class_idx)

# Decode for report
y_test_decoded = label_encoder.inverse_transform(y_test_holdout)
final_test_decoded = label_encoder.inverse_transform(final_test)

report = classification_report(y_test_decoded, final_test_decoded, zero_division=0)
print("\n================ Holdout Classification Report (Final Cascade) ================")
print(report)

print("\nAdditional Metrics (Holdout):")
print("F1 (weighted):", f1_score(y_test_holdout, final_test, average='weighted', zero_division=0))
print("Precision (weighted):", precision_score(y_test_holdout, final_test, average='weighted', zero_division=0))
print("Recall (weighted):", recall_score(y_test_holdout, final_test, average='weighted', zero_division=0))

# Save report
report_path = os.path.join(output_folder, "classification_report_holdout_final_cascade.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"\nSaved holdout report to: {report_path}")

# Save models + encoder
dt_path = os.path.join(output_folder, "dt_model_final.pkl")
nn_path = os.path.join(output_folder, "nn_model_trained_once.h5")
le_path = os.path.join(output_folder, "label_encoder.pkl")

joblib.dump(dt_final, dt_path)
nn_model.save(nn_path)
joblib.dump(label_encoder, le_path)

print("\nSaved:")
print(f"Decision Tree (final): {dt_path}")
print(f"Neural Net (trained once): {nn_path}")
print(f"LabelEncoder: {le_path}")
