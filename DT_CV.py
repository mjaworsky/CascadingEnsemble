import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score
)

# ============================================================
# Repro
# ============================================================
SEED = 42
np.random.seed(SEED)

# ============================================================
# Load data
# ============================================================
CSV_PATH = "/content/LLCP_SMOKE_5YEAR.csv"
df = pd.read_csv(CSV_PATH)

TARGET_COL = "CNCRTYP1"
DROP_COLS = ["CNCRTYP1", "CNCRAGE"]

# ------------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------------
df = df[~df[TARGET_COL].isin([77, 99])].copy()
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
df = df.dropna(subset=[TARGET_COL])
df[TARGET_COL] = df[TARGET_COL].astype(int)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# ------------------------------------------------------------
# Select majority + TOP-K minority
# ------------------------------------------------------------
TOP_K_MINORITY = 20

class_counts = df[TARGET_COL].value_counts()
majority_class = int(class_counts.idxmax())

minority_labels = (
    class_counts[class_counts.index != majority_class]
    .head(TOP_K_MINORITY)
    .index.astype(int)
    .tolist()
)

selected_classes = [majority_class] + minority_labels
df = df[df[TARGET_COL].isin(selected_classes)].copy()

print("Majority class:", majority_class)
print("Minority labels:", minority_labels)

# ------------------------------------------------------------
# Build X / y
# ------------------------------------------------------------
X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore") \
      .values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)

# ============================================================
# Train / Test split (NON-stratified)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=True,
    random_state=SEED
)

# ============================================================
# Scaling (TRAIN only)
# ============================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Vanilla Decision Tree (back from RF)
# ============================================================
dt = DecisionTreeClassifier(
    random_state=SEED,
    class_weight="balanced"
)

dt.fit(X_train_s, y_train)
y_test_pred = dt.predict(X_test_s)

# ============================================================
# SAFE minority metrics
# ============================================================
def safe_f1_macro_minority(y_true, y_pred, minority_labels):
    """
    Macro-averaged F1 over selected minority labels only.
    Missing labels contribute F1 = 0.
    """
    return f1_score(
        y_true,
        y_pred,
        labels=minority_labels,
        average="macro",
        zero_division=0
    )

def safe_precision_macro_minority(y_true, y_pred, minority_labels):
    """
    Macro-averaged precision over selected minority labels only.
    Missing labels contribute precision = 0.
    """
    return precision_score(
        y_true,
        y_pred,
        labels=minority_labels,
        average="macro",
        zero_division=0
    )

# ============================================================
# FULL classification report (ALL classes)
# ============================================================
print("\nClassification report (TEST — all selected classes):")
print(
    classification_report(
        y_test,
        y_test_pred,
        labels=selected_classes,
        zero_division=0
    )
)

# ============================================================
# FINAL minority-only summary metrics
# ============================================================
minority_macro_f1 = safe_f1_macro_minority(
    y_test, y_test_pred, minority_labels
)

minority_macro_precision = safe_precision_macro_minority(
    y_test, y_test_pred, minority_labels
)

print("\n================ Minority-only summary ================")
print("Minority macro-F1:", minority_macro_f1)
print("Minority macro-Precision:", minority_macro_precision)
