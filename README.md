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

3. Prepare data and train models (from project root):

```bash
python src/prepare_data.py
python src/train_logistic_regression.py
python src/train_random_forest.py
python src/compare_models.py
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

- `prepare_data.py` normalizes headers, validates ranges, and creates a
  stratified 70:30 train/test split saved to `data/processed/`.
- Logistic Regression pipeline standardizes numeric features and one-hot encodes
  categorical fields such as `education` and `self_employed`.
- Random Forest pipeline passes numeric features through unchanged and
  one-hot encodes categoricals.
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

Example final test-set metrics (for reference only):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 98.4% | 98.3% | 99.3% | 98.8% |
| Logistic Regression | 93.8% | 96.5% | 93.5% | 95.0% |

Interpret these results carefully — strong correlation with `cibil_score`
contributes to high performance on this dataset and limits generalizability.

## Running & Development Tips

- Use `python -m pip install -r requirements.txt` to keep dependencies reproducible.
- Run training scripts on a machine with enough memory for scikit-learn jobs.
- If you change preprocessing, re-run `src/prepare_data.py` before training.

## Contributing & Contact

This repository is maintained as an educational example. For questions or
improvements, open an issue or submit a pull request.