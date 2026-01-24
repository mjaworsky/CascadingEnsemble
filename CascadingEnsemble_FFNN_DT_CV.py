import os
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE, RandomOverSampler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from google.colab import files, drive
import joblib

# -----------------------------
# Mount Google Drive
# -----------------------------
drive.mount('/content/drive')

# Define output folder in Google Drive
output_folder = '/content/drive/My Drive/ColabOutputs/'
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Upload + read file
# -----------------------------
uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])

# -----------------------------
# Filter invalid CNCRTYP1 values
# -----------------------------
df = df[~df['CNCRTYP1'].isin([77, 99])]

# Identify majority + top 20 minority
class_counts = df['CNCRTYP1'].value_counts()
majority_class = class_counts.idxmax()
top_20_minority_classes = class_counts[class_counts.index != majority_class].head(20).index
selected_classes = list(top_20_minority_classes) + [majority_class]

df_filtered = df[df['CNCRTYP1'].isin(selected_classes)].copy()

print("Class distribution after truncation:")
print(df_filtered['CNCRTYP1'].value_counts())

# -----------------------------
# X / y
# -----------------------------
X = df_filtered.drop(columns=['CNCRTYP1', 'CNCRAGE']).values
y = df_filtered['CNCRTYP1'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

majority_class_encoded = label_encoder.transform([majority_class])[0]
print(f"\nMajority class: {majority_class} -> encoded: {majority_class_encoded}")

# -----------------------------
# Helpers
# -----------------------------
def build_ffnn(input_dim: int, num_classes: int) -> tf.keras.Model:
    # RESTORED original architecture/hyperparameters
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.1),
        Dense(128, activation='relu'),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def smote_minority_only(X_min, y_min, random_state=42, base_k=2):
    """
    Apply SMOTE to minority-only training set.
    Auto-adjust k if the smallest class is too small; fallback to RandomOverSampler.
    """
    counts = np.bincount(y_min)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return X_min, y_min, "No minority samples"

    min_count = int(nonzero.min())

    # SMOTE requires min_count >= k_neighbors + 1
    if min_count >= 2:
        k = min(base_k, max(1, min_count - 1))
        sm = SMOTE(random_state=random_state, k_neighbors=k)
        X_rs, y_rs = sm.fit_resample(X_min, y_min)
        return X_rs, y_rs, f"SMOTE(k_neighbors={k})"
    else:
        ros = RandomOverSampler(random_state=random_state)
        X_rs, y_rs = ros.fit_resample(X_min, y_min)
        return X_rs, y_rs, "RandomOverSampler(fallback)"

# -----------------------------
# 2-Fold CV for the cascade
#   - FFNN trained on full training fold
#   - DT trained ONLY on minority (true label != majority), with SMOTE on that subset only
# -----------------------------
kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y_encoded), start=1):
    print(f"\n================ Fold {fold}/2 ================")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

    # ---- Train FFNN on FULL training fold (all classes) ----
    tf.keras.backend.clear_session()
    num_classes = len(np.unique(y_train))
    nn_model = build_ffnn(input_dim=X_train.shape[1], num_classes=num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    nn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    # ---- Build minority-only training set for DT (TRUE minority only) ----
    is_true_minority_train = (y_train != majority_class_encoded)
    X_train_min = X_train[is_true_minority_train]
    y_train_min = y_train[is_true_minority_train]

    print(f"Minority-only DT training samples (before SMOTE): {len(y_train_min)}")

    # ---- SMOTE ONLY on minority training samples ----
    X_train_min_rs, y_train_min_rs, smote_desc = smote_minority_only(
        X_train_min, y_train_min, random_state=42, base_k=2
    )
    print(f"Resampling for DT: {smote_desc}")
    print(f"Minority-only DT training samples (after resampling): {len(y_train_min_rs)}")

    # ---- Train DT on minority-only resampled data ----
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train_min_rs, y_train_min_rs)

    # ---- Predict on test fold with cascade ----
    nn_probs = nn_model.predict(X_test, verbose=0)
    nn_preds = np.argmax(nn_probs, axis=1)

    # For samples predicted as minority by FFNN, use DT; else majority
    final_preds = np.empty_like(nn_preds)

    majority_mask = (nn_preds == majority_class_encoded)
    minority_mask = ~majority_mask

    # If NN says majority -> output majority
    final_preds[majority_mask] = majority_class_encoded

    # If NN says minority -> DT predicts (trained only on minority classes)
    if np.any(minority_mask):
        final_preds[minority_mask] = dt_model.predict(X_test[minority_mask])

    # Pool results
    all_y_true.append(y_test)
    all_y_pred.append(final_preds)

    # Fold weighted metrics (same style as before)
    y_test_dec = label_encoder.inverse_transform(y_test)
    y_pred_dec = label_encoder.inverse_transform(final_preds)

    fold_f1 = f1_score(y_test_dec, y_pred_dec, average='weighted')
    fold_p = precision_score(y_test_dec, y_pred_dec, average='weighted')
    fold_r = recall_score(y_test_dec, y_pred_dec, average='weighted')

    print(f"Fold {fold} weighted metrics: F1={fold_f1:.4f}  P={fold_p:.4f}  R={fold_r:.4f}")

# -----------------------------
# Pooled evaluation (metrics unchanged)
# -----------------------------
all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

y_true_decoded = label_encoder.inverse_transform(all_y_true)
y_pred_decoded = label_encoder.inverse_transform(all_y_pred)

report = classification_report(y_true_decoded, y_pred_decoded)
print("\n================ POOLED 2-FOLD CV: Classification Report ================")
print(report)

report_file_path = os.path.join(output_folder, 'classification_report_cascade_minoritySMOTE_2foldCV.txt')
with open(report_file_path, 'w') as f:
    f.write(report)

print(f"Classification report saved to: {report_file_path}")

print("\nAdditional Metrics (Pooled 2-Fold CV):")
print("F1 Score:", f1_score(y_true_decoded, y_pred_decoded, average='weighted'))
print("Precision:", precision_score(y_true_decoded, y_pred_decoded, average='weighted'))
print("Recall:", recall_score(y_true_decoded, y_pred_decoded, average='weighted'))

# -----------------------------
# (Optional) Train final deployable models on full data
#   - FFNN trained on all data
#   - DT trained on minority-only with SMOTE
# -----------------------------
# Train FFNN final
tf.keras.backend.clear_session()
num_classes_full = len(np.unique(y_encoded))
nn_final = build_ffnn(input_dim=X.shape[1], num_classes=num_classes_full)
early_stopping_final = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

nn_final.fit(
    X, y_encoded,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping_final],
    verbose=1
)

# Train DT final on minority-only + SMOTE
is_true_minority_full = (y_encoded != majority_class_encoded)
X_min_full = X[is_true_minority_full]
y_min_full = y_encoded[is_true_minority_full]

X_min_rs, y_min_rs, smote_desc_final = smote_minority_only(X_min_full, y_min_full, random_state=42, base_k=2)
print(f"\nFinal DT resampling: {smote_desc_final}")

dt_final = DecisionTreeClassifier(random_state=42)
dt_final.fit(X_min_rs, y_min_rs)

# Save final models + encoder
dt_path = os.path.join(output_folder, 'dt_minority_only.pkl')
nn_path = os.path.join(output_folder, 'ffnn_majority_filter.h5')
le_path = os.path.join(output_folder, 'label_encoder.pkl')

joblib.dump(dt_final, dt_path)
nn_final.save(nn_path)
joblib.dump(label_encoder, le_path)

print(f"\nSaved DT (minority-only): {dt_path}")
print(f"Saved FFNN (majority filter): {nn_path}")
print(f"Saved LabelEncoder: {le_path}")
