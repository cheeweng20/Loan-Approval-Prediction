"""Shared project paths, schema, and reproducibility settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "loan_approval_dataset.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_ANALYSIS_PATH = PROCESSED_DATA_DIR / "feature_analysis.json"
FEATURE_SCORE_TABLE_PATH = PROCESSED_DATA_DIR / "feature_scores.csv"
FEATURE_IMPACT_ANALYSIS_PATH = MODELS_DIR / "feature_impact_analysis.json"
FEATURE_IMPACT_TABLE_PATH = MODELS_DIR / "feature_impact_scores.csv"
FEATURE_IMPACT_CHART_PATH = MODELS_DIR / "feature_impact_scores.png"

RANDOM_STATE = 42
TEST_SIZE = 0.30
CV_FOLDS = 5

ID_COLUMN = "loan_id"
TARGET_COLUMN = "loan_status"
EXPECTED_LABELS = frozenset({"Approved", "Rejected"})

NUMERIC_FEATURES = (
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
)
CATEGORICAL_FEATURES = (
    "education",
    "self_employed",
)
# These are the columns passed to the model pipeline.  SelectKBest chooses the
# strongest subset during model fitting; do not remove columns here based on a
# result calculated from the held-out test set.
FEATURE_COLUMNS = (
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
)
# A feature is retained when its raw SelectKBest score is above this.
FEATURE_SCORE_THRESHOLD = 0.50

EXCLUDED_SOURCE_FEATURES = ()
