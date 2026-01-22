README.txt
==========

Project: Cascading Ensemble (Feedforward Neural Network -> Decision Tree) with Leakage‑Safe CV + SMOTE
---------------------------------------------------------------------------------------------------

This repository contains a leakage-safe evaluation pipeline for a *cascading ensemble* classifier:

1) A Feedforward Neural Network (FFNN) first predicts the class.
2) If the FFNN predicts the **majority class**, we keep the FFNN decision.
3) Otherwise (i.e., the sample looks *minority-ish*), we pass it to a **Decision Tree** (DT) and use the DT prediction.

The goal is to reduce majority-class dominance while giving minority classes a second specialist model.

Included Script
---------------
- `CascadingEnsemble_FFNN_DT_CV.py`

Key Features
------------
- **Leakage-safe evaluation**
  - Stratified holdout test split kept untouched
  - Repeated Stratified K‑Fold cross‑validation run on training data only
  - **SMOTE applied only inside each CV fold** on the training fold (never on validation/test)
- **Adaptive resampling**
  - Uses SMOTE with `k_neighbors` adjusted per fold when minority counts are low
  - Falls back to RandomOverSampler if SMOTE cannot run due to too few samples
- **Reproducibility**
  - Uses a fixed `RANDOM_STATE = 42`
- **Artifacts saved (Colab/Drive)**
  - CV fold metrics CSV
  - Holdout classification report TXT
  - Final trained DT model (joblib)
  - Final trained NN model (.h5)
  - Label encoder (joblib)

How the Cascade Works
---------------------
For each sample:
- FFNN outputs class probabilities (softmax), then we take `argmax` as FFNN predicted class.
- If FFNN prediction == majority class index -> **final = FFNN prediction**
- Else -> **final = DT prediction**

This logic is implemented in `fallback_predictions()`.

Cross‑Validation Setup
----------------------
- Holdout split:
  - `train_test_split(..., stratify=y, test_size=0.2)`
- CV on training portion only:
  - `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)`
- Resampling:
  - For each fold: resample **training fold only** via SMOTE / RandomOverSampler
  - Validation fold remains untouched

Data Assumptions / Expected Columns
-----------------------------------
The script expects a CSV where:
- Target label is in column: `CNCRTYP1`
- Another column removed from features: `CNCRAGE`

It filters out invalid values:
- `CNCRTYP1` in {77, 99} are excluded

Then it selects:
- The **majority class** + the **top 20** most frequent remaining classes.

Features `X` are created by dropping `CNCRTYP1` and `CNCRAGE`.

Environment & Dependencies
--------------------------
This script is written to run in **Google Colab** and uses Drive for output.

Python packages used:
- numpy, pandas
- scikit-learn
- imbalanced-learn
- tensorflow (keras)
- joblib
- google.colab (Drive mount + file upload)

Quickstart (Google Colab)
-------------------------
1) Open Colab and upload this repository (or the script).
2) Run the script.
3) When prompted:
   - Authorize Drive mount
   - Upload your CSV dataset
4) Outputs are saved to:
   `/content/drive/My Drive/ColabOutputs/`

Outputs
-------
- `cv_metrics_summary.csv`:
  Per-fold weighted F1/Precision/Recall and resampling method used.
- `classification_report_holdout_fallback_model.txt`:
  Holdout test classification report.
- `dt_model_final.pkl`:
  Final Decision Tree trained on the full resampled training set.
- `nn_model_final.h5`:
  Final NN trained on the full resampled training set.
- `label_encoder.pkl`:
  Encoder mapping between original `CNCRTYP1` labels and integer indices.

Configuration (Edit in Script)
------------------------------
Main settings at the top of the file:
- `TEST_SIZE = 0.2`
- `N_SPLITS = 5`, `N_REPEATS = 3`
- `EPOCHS = 50`, `BATCH_SIZE = 64`, `PATIENCE = 5`
- `SMOTE_BASE_K = 2` (auto-adjusted per fold)


