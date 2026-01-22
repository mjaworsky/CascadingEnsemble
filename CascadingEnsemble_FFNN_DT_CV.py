# ============================================================
# Leakage-safe evaluation:
#   1) Holdout test split (stratified) kept untouched
#   2) Repeated Stratified K-Fold CV on training set only
#   3) SMOTE is applied INSIDE each CV fold (training fold only)
#   4) Train final models on full training set, evaluate on holdout test
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

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

N_SPLITS = 5
N_REPEATS = 3

EPOCHS = 50
BATCH_SIZE = 64
VAL_SPLIT = 0.2
PATIENCE = 5
LEARNING_RATE = 1e-3

SMOTE_BASE_K = 2  # will auto-adjust per fold if needed

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

# Encode target ONCE globally (keeps fold metrics consistent)
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)
majority_class_idx = label_encoder.transform([majority_class])[0]

num_classes = len(label_encoder.classes_)
input_dim = X.shape[1]

print("\nEncoded classes:", list(label_encoder.classes_))
print("Majority class:", majority_class, "-> idx:", majority_class_idx)
print("Num classes:", num_classes, "Input dim:", input_dim)

# -----------------------------
# Holdout test split (untouched)
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
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def resample_training_fold(X_tr, y_tr, random_state=RANDOM_STATE, smote_base_k=SMOTE_BASE_K):
    """
    Apply SMOTE safely INSIDE a training fold.
    If a fold has too few samples in a minority class for SMOTE(k),
    we either reduce k or fall back to RandomOverSampler.
    """
    counts = np.bincount(y_tr)
    # ignore zero-count classes (can happen in a fold)
    nonzero = counts[counts > 0]
    min_class_count = int(nonzero.min()) if len(nonzero) else 0

    # SMOTE requires min_class_count >= k_neighbors + 1
    # Choose k <= (min_class_count - 1), but at least 1.
    max_k = max(1, min_class_count - 1)

    if min_class_count >= 2:
        k = min(smote_base_k, max_k)
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        X_rs, y_rs = sm.fit_resample(X_tr, y_tr)
        return X_rs, y_rs, f"SMOTE(k_neighbors={k})"
    else:
        # If any class has only 1 sample in the fold, SMOTE can't run.
        ros = RandomOverSampler(random_state=random_state)
        X_rs, y_rs = ros.fit_resample(X_tr, y_tr)
        return X_rs, y_rs, "RandomOverSampler(fallback)"

def fallback_predictions(nn_probs, dt_preds, majority_class_idx: int):
    """
    Preserve your original logic:
      - If NN predicts majority class -> keep NN
      - Else -> use DT prediction
    """
    nn_preds = np.argmax(nn_probs, axis=1)
    final = []
    for i, nn_pred in enumerate(nn_preds):
        if nn_pred == majority_class_idx:
            final.append(nn_pred)
        else:
            final.append(dt_preds[i])
    return np.array(final, dtype=int)

# -----------------------------
# Repeated Stratified K-Fold CV
# -----------------------------
rskf = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

cv_rows = []
fold_num = 0

for tr_idx, va_idx in rskf.split(X_train_full, y_train_full):
    fold_num += 1
    X_tr, X_va = X_train_full[tr_idx], X_train_full[va_idx]
    y_tr, y_va = y_train_full[tr_idx], y_train_full[va_idx]

    # Resample ONLY training fold
    X_tr_rs, y_tr_rs, resample_desc = resample_training_fold(X_tr, y_tr)

    # ---- Decision Tree ----
    dt = DecisionTreeClassifier(random_state=RANDOM_STATE)
    dt.fit(X_tr_rs, y_tr_rs)

    # ---- Neural Net ----
    tf.keras.backend.clear_session()
    nn = build_nn(input_dim=input_dim, num_classes=num_classes)
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    nn.fit(
        X_tr_rs, y_tr_rs,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=[es],
        verbose=0
    )

    # Predict on validation fold (no resampling)
    nn_probs = nn.predict(X_va, verbose=0)
    dt_preds = dt.predict(X_va)

    final_pred = fallback_predictions(nn_probs, dt_preds, majority_class_idx)

    # Metrics (weighted)
    f1w = f1_score(y_va, final_pred, average='weighted', zero_division=0)
    pw  = precision_score(y_va, final_pred, average='weighted', zero_division=0)
    rw  = recall_score(y_va, final_pred, average='weighted', zero_division=0)

    cv_rows.append({
        "fold": fold_num,
        "resampling": resample_desc,
        "f1_weighted": f1w,
        "precision_weighted": pw,
        "recall_weighted": rw
    })

    print(f"Fold {fold_num:02d} | {resample_desc:28s} | F1w={f1w:.4f} Pw={pw:.4f} Rw={rw:.4f}")

cv_df = pd.DataFrame(cv_rows)

print("\n================ CV Summary ================")
print(cv_df[["f1_weighted","precision_weighted","recall_weighted"]].describe().loc[
    ["mean","std","min","max"]
])

cv_summary_path = os.path.join(output_folder, "cv_metrics_summary.csv")
cv_df.to_csv(cv_summary_path, index=False)
print(f"\nCV metrics per fold saved to: {cv_summary_path}")

# -----------------------------
# Train FINAL models on full training set, evaluate on holdout test
# -----------------------------
X_train_rs, y_train_rs, resample_desc_final = resample_training_fold(X_train_full, y_train_full)
print("\nFinal training resampling:", resample_desc_final)

# Final DT
dt_final = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_final.fit(X_train_rs, y_train_rs)

# Final NN
tf.keras.backend.clear_session()
nn_final = build_nn(input_dim=input_dim, num_classes=num_classes)
es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

nn_final.fit(
    X_train_rs, y_train_rs,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    callbacks=[es],
    verbose=1
)

# Holdout predictions
nn_probs_test = nn_final.predict(X_test_holdout, verbose=0)
dt_preds_test = dt_final.predict(X_test_holdout)

final_pred_test = fallback_predictions(nn_probs_test, dt_preds_test, majority_class_idx)

# Decode for report
y_test_decoded = label_encoder.inverse_transform(y_test_holdout)
final_pred_decoded = label_encoder.inverse_transform(final_pred_test)

report = classification_report(y_test_decoded, final_pred_decoded, zero_division=0)
print("\n================ Holdout Classification Report ================")
print(report)

# Additional metrics on holdout
print("\nAdditional Metrics (Holdout):")
print("F1 (weighted):", f1_score(y_test_holdout, final_pred_test, average='weighted', zero_division=0))
print("Precision (weighted):", precision_score(y_test_holdout, final_pred_test, average='weighted', zero_division=0))
print("Recall (weighted):", recall_score(y_test_holdout, final_pred_test, average='weighted', zero_division=0))

# Save report
report_path = os.path.join(output_folder, "classification_report_holdout_fallback_model.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"\nHoldout report saved to: {report_path}")

# Save models + encoder
dt_path = os.path.join(output_folder, "dt_model_final.pkl")
nn_path = os.path.join(output_folder, "nn_model_final.h5")
le_path = os.path.join(output_folder, "label_encoder.pkl")

joblib.dump(dt_final, dt_path)
nn_final.save(nn_path)
joblib.dump(label_encoder, le_path)

print(f"\nSaved:")
print(f"Decision Tree: {dt_path}")
print(f"Neural Net:    {nn_path}")
print(f"LabelEncoder:  {le_path}")
