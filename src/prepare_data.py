"""Prepare and validate the loan approval dataset.

The script saves an untouched-feature train/test split. Encoding and scaling are
performed inside each fitted model pipeline so validation and test data cannot
influence preprocessing.

Usage:
    python src/analyze_features.py
    python src/prepare_data.py
"""

import hashlib
import json

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from settings import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATA_PATH,
    EXCLUDED_SOURCE_FEATURES,
    EXPECTED_LABELS,
    FEATURE_ANALYSIS_PATH,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    FEATURE_SCORE_THRESHOLD,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_loan_data(path):
    """Load the CSV and normalize whitespace in headers and text values."""
    try:
        try:
            data = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
        except UnicodeDecodeError:
            data = pd.read_csv(path, encoding="windows-1252", skipinitialspace=True)
    except pd.errors.EmptyDataError as exception:
        raise ValueError(f"The dataset is empty: {path}") from exception
    except pd.errors.ParserError as exception:
        raise ValueError(f"The dataset is not a valid CSV file: {path}") from exception

    data.columns = data.columns.astype(str).str.strip()
    for column in data.select_dtypes(include=["object", "string"]).columns:
        data[column] = data[column].astype("string").str.strip()
    return data


def _normalize_target(data):
    """Normalize the two target labels."""
    status_map = {"approved": "Approved", "rejected": "Rejected"}
    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(status_map)
    )


def remove_excluded_features(data):
    """Remove configured source columns before validation or analysis."""
    excluded = [
        column for column in EXCLUDED_SOURCE_FEATURES if column in data.columns
    ]
    return data.drop(columns=excluded).copy(), excluded


def get_source_sha256(path):
    """Return a fingerprint used to reject stale feature-analysis reports."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_and_clean_data(data):
    """Validate schema and values while preserving documented source anomalies."""
    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"The loan dataset is missing columns: {sorted(missing_columns)}."
        )

    cleaned = data[[ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    original_rows = len(cleaned)

    numeric_columns = [ID_COLUMN, *NUMERIC_FEATURES]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if cleaned[numeric_columns].isna().any().any():
        invalid = cleaned[numeric_columns].isna().sum()
        invalid = invalid[invalid > 0].to_dict()
        raise ValueError(f"Missing or non-numeric values found: {invalid}.")

    for column in CATEGORICAL_FEATURES:
        cleaned[column] = cleaned[column].astype("string").str.strip()
    categorical_missing = cleaned[list(CATEGORICAL_FEATURES)].isna().sum()
    categorical_missing = categorical_missing[categorical_missing > 0].to_dict()
    if categorical_missing:
        raise ValueError(
            f"Missing categorical values found: {categorical_missing}."
        )

    expected_categories = {
        "education": {"Graduate", "Not Graduate"},
        "self_employed": {"Yes", "No"},
    }
    for column, expected_values in expected_categories.items():
        observed_values = set(cleaned[column].dropna())
        if observed_values - expected_values:
            raise ValueError(
                f"{column} contains unexpected values: "
                f"{sorted(observed_values - expected_values)}."
            )

    _normalize_target(cleaned)
    observed_labels = set(cleaned[TARGET_COLUMN].dropna())
    if cleaned[TARGET_COLUMN].isna().any() or observed_labels != EXPECTED_LABELS:
        raise ValueError(
            "loan_status must contain both Approved and Rejected labels only."
        )

    if cleaned[ID_COLUMN].duplicated().any():
        duplicate_ids = cleaned.loc[
            cleaned[ID_COLUMN].duplicated(keep=False), ID_COLUMN
        ].tolist()
        raise ValueError(f"Duplicate loan_id values found: {duplicate_ids[:10]}.")
    if not (cleaned[ID_COLUMN] % 1 == 0).all():
        raise ValueError("loan_id values must be whole numbers.")
    cleaned[ID_COLUMN] = cleaned[ID_COLUMN].astype("int64")

    integer_columns = ("no_of_dependents", "loan_term", "cibil_score")
    for column in integer_columns:
        if not (cleaned[column] % 1 == 0).all():
            raise ValueError(f"{column} values must be whole numbers.")
        cleaned[column] = cleaned[column].astype("int64")

    if (cleaned["no_of_dependents"] < 0).any():
        raise ValueError("no_of_dependents must be zero or greater.")
    if not cleaned["cibil_score"].between(300, 900).all():
        raise ValueError("cibil_score must be between 300 and 900.")
    if (cleaned["income_annum"] <= 0).any():
        raise ValueError("income_annum must be greater than zero.")
    if (cleaned["loan_amount"] <= 0).any():
        raise ValueError("loan_amount must be greater than zero.")
    if (cleaned["loan_term"] <= 0).any():
        raise ValueError("loan_term must be greater than zero.")
    duplicate_subset = [*FEATURE_COLUMNS, TARGET_COLUMN]
    conflicting = cleaned.groupby(list(FEATURE_COLUMNS), dropna=False)[
        TARGET_COLUMN
    ].nunique()
    if (conflicting > 1).any():
        raise ValueError(
            "Identical applications with conflicting loan_status labels were found."
        )
    cleaned = cleaned.drop_duplicates(subset=duplicate_subset).reset_index(drop=True)

    quality_warnings = []
    negative_residential_assets = int(
        (cleaned["residential_assets_value"] < 0).sum()
    )
    if negative_residential_assets:
        quality_warnings.append(
            f"Retained {negative_residential_assets} rows with negative "
            "residential_assets_value values from the source dataset."
        )
    cleaned.attrs["quality_warnings"] = quality_warnings
    cleaned.attrs["removed_duplicate_rows"] = original_rows - len(cleaned)
    return cleaned


def load_feature_analysis(source_sha256):
    """Load the required training-only SelectKBest analysis report."""
    if not FEATURE_ANALYSIS_PATH.is_file():
        raise FileNotFoundError(
            f"Feature analysis not found: {FEATURE_ANALYSIS_PATH}.\n"
            "Run analyze_features.py before prepare_data.py."
        )
    with FEATURE_ANALYSIS_PATH.open("r", encoding="utf-8") as file:
        analysis = json.load(file)

    expected_features = list(FEATURE_COLUMNS)
    if analysis.get("candidate_features") != expected_features:
        raise ValueError(
            "Feature analysis does not match the configured candidate features. "
            "Run analyze_features.py again."
        )
    selected_features = analysis.get("selected_features", [])
    if (
        analysis.get("source_sha256") != source_sha256
        or not selected_features
        or not set(selected_features).issubset(FEATURE_COLUMNS)
    ):
        raise ValueError(
            "Feature analysis is stale or invalid. Run analyze_features.py again."
        )
    if analysis.get("selection_threshold") != FEATURE_SCORE_THRESHOLD:
        raise ValueError("Feature analysis threshold is out of date. Run analyze_features.py again.")
    return analysis


def validate_training_split(X_train, X_test, y_train, y_test):
    """Reject incomplete, unusable, or overlapping stored data splits."""
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label row counts do not match after splitting.")
    if tuple(X_train.columns) != FEATURE_COLUMNS:
        raise ValueError("Training features do not match the expected schema.")
    if tuple(X_test.columns) != FEATURE_COLUMNS:
        raise ValueError("Test features do not match the expected schema.")

    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()
    if set(train_counts.index) != EXPECTED_LABELS:
        raise ValueError("The training split must contain Approved and Rejected.")
    if set(test_counts.index) != EXPECTED_LABELS:
        raise ValueError("The test split must contain Approved and Rejected.")
    if train_counts.min() < CV_FOLDS:
        raise ValueError(
            f"Each training class needs at least {CV_FOLDS} rows for "
            f"{CV_FOLDS}-fold cross-validation. Found: {train_counts.to_dict()}."
        )

    train_hashes = set(
        pd.util.hash_pandas_object(X_train, index=False).astype("uint64")
    )
    test_hashes = set(
        pd.util.hash_pandas_object(X_test, index=False).astype("uint64")
    )
    overlap = train_hashes.intersection(test_hashes)
    if overlap:
        raise ValueError(
            f"Detected {len(overlap)} duplicated applications across train and test."
        )


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training file not found: {DATA_PATH}\n"
            "Put loan_approval_dataset.csv in the data directory."
        )
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading loan data from {DATA_PATH} ...")
    raw_data = load_loan_data(DATA_PATH)
    data, excluded_features = remove_excluded_features(raw_data)
    data = validate_and_clean_data(data)
    analysis = load_feature_analysis(get_source_sha256(DATA_PATH))
    print(f"Loaded {len(raw_data):,} rows; using {len(data):,} validated rows.")
    print(data[TARGET_COLUMN].value_counts())
    print(
        "Candidate model inputs: "
        f"{', '.join(FEATURE_COLUMNS)}"
    )
    if excluded_features:
        print(f"Removed source features: {', '.join(excluded_features)}")
    print(
        "SelectKBest analysis recommends: "
        f"{', '.join(analysis['selected_features'])}"
    )

    X = data.loc[:, FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    validate_training_split(X_train, X_test, y_train, y_test)

    artifacts = {
        "X_train.joblib": X_train,
        "X_test.joblib": X_test,
        "y_train.joblib": y_train,
        "y_test.joblib": y_test,
    }
    for name, value in artifacts.items():
        joblib.dump(value, PROCESSED_DATA_DIR / name)

    summary = {
        "source_rows": len(raw_data),
        "validated_rows": len(data),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "class_counts": {
            str(label): int(count)
            for label, count in data[TARGET_COLUMN].value_counts().items()
        },
        "candidate_features": list(FEATURE_COLUMNS),
        "select_k_best_features": analysis["selected_features"],
        "excluded_source_features": excluded_features,
        "feature_analysis_file": FEATURE_ANALYSIS_PATH.name,
        "quality_warnings": data.attrs.get("quality_warnings", []),
    }
    with (PROCESSED_DATA_DIR / "data_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print(f"Training set: {len(X_train):,} applications")
    print(f"Test set: {len(X_test):,} applications")
    print(f"Processed data saved to {PROCESSED_DATA_DIR}/")


if __name__ == "__main__":
    main()
