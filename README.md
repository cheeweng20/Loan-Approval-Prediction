# Loan Approval Prediction

A small educational project demonstrating supervised classification for
predicting loan application outcomes (Approved vs Rejected) using
scikit-learn pipelines and reproducible evaluation.

Important: this repository is for learning and experimentation only — the
models predict historical approval decisions, not creditworthiness, and must
not be used for real lending.

## Features

- Reproducible data preprocessing and train/test split
- Two model pipelines: Logistic Regression and Random Forest
- Training-only feature selection with SelectKBest
- Post-training permutation impact scores for the best model
- Grid search with 5-fold stratified CV, selecting by Approved-class F1-score
- Streamlit demo for interactive predictions

## Quick Start

1. Create a Python environment (recommended):

```bash
python -m venv .venv
# On Windows: .\.venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Analyse features, then prepare data and train models (from project root):

```bash
python src/analyze_features.py
python src/prepare_data.py
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/compare_models.py
python src/analyze_feature_impact.py
```

4. Run the Streamlit demo:

```bash
streamlit run streamlit_app.py
```

## Project Structure

See the top-level `src/` and `data/` folders for code and dataset.

```
data/                 # raw and processed datasets
models/               # trained model artifacts and exports
src/                  # scripts: prepare_data, training, evaluation
streamlit_app.py      # interactive demo
requirements.txt
```

## Dataset

The dataset (`data/loan_approval_dataset.csv`) contains ~4.2k applications and
13 columns; `loan_status` is the target and `loan_id` is kept for traceability
during validation but excluded from features. See `data/README.md` for schema
and provenance notes.

## Preprocessing & Modeling Notes

- `analyze_features.py` ranks all regular input features with
  `SelectKBest(f_classif)` using only the reproducible training split. Its
  report is saved as
  `data/processed/feature_analysis.json`, with a per-feature score table in
  `data/processed/feature_scores.csv`. Features with a SelectKBest score above
  0.50 are marked as selected in this report.
- `prepare_data.py` requires that analysis report, validates ranges, and creates
  a stratified 70:30 train/test split saved to `data/processed/`.
- Both model pipelines apply `SelectKBest` after their preprocessing step inside
  cross-validation, so the final selected features are fitted without test-data
  leakage. Logistic Regression standardizes numeric features; Random Forest
  passes them through unchanged.
- `analyze_feature_impact.py` selects the model with the best saved F1-score
  and calculates permutation importance on the held-out test set. The impact
  score is the F1-score drop after one original feature column is shuffled.
- Grid searches use `StratifiedKFold(n_splits=5)` and record accuracy,
  precision, recall, and F1; selection uses F1 for the Approved class.

## Results & Artifacts

Model artifacts and outputs are saved under `models/` and include:

- `models/logistic_regression_model.joblib`
- `models/random_forest_model.joblib`
- `models/*_grid_search.csv`
- `models/*_training_summary.json`
- `models/confusion_matrix_*.png`
- `models/comparison_table.csv` and `models/comparison_chart.png`
- `models/feature_impact_scores.csv`, `models/feature_impact_analysis.json`,
  and `models/feature_impact_scores.png`

Example final test-set metrics (for reference only):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 98.4% | 98.6% | 98.9% | 98.7% |
| Logistic Regression | 94.0% | 96.8% | 93.5% | 95.1% |

Interpret these results carefully — strong correlation with `cibil_score`
contributes to high performance on this dataset and limits generalizability.

## Running & Development Tips

- Use `python -m pip install -r requirements.txt` to keep dependencies reproducible.
- Run training scripts on a machine with enough memory for scikit-learn jobs.
- If you change feature exclusions or preprocessing, re-run
  `src/analyze_features.py`, `src/prepare_data.py`, both training scripts,
  `src/compare_models.py`, and `src/analyze_feature_impact.py`.

## Contributing & Contact

This repository is maintained as an educational example. For questions or
improvements, open an issue or submit a pull request.
