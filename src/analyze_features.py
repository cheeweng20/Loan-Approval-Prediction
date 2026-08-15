"""Rank model candidates with SelectKBest before data preparation.

The analysis creates the same reproducible training split used by
``prepare_data.py``. It fits preprocessing and SelectKBest only on that training
split, so the held-out test data cannot influence the feature-selection
recommendation.

Usage:
    python src/analyze_features.py
"""

import json

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from prepare_data import (
    load_loan_data,
    get_source_sha256,
    validate_and_clean_data,
)
from settings import (
    DATA_PATH,
    FEATURE_ANALYSIS_PATH,
    FEATURE_COLUMNS,
    FEATURE_SCORE_TABLE_PATH,
    FEATURE_SCORE_THRESHOLD,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from training_utils import create_preprocessor, get_original_feature_name


def create_scoring_pipeline():
    """Create preprocessing plus SelectKBest for all configured features."""
    return Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric=True)),
        ("selector", SelectKBest(score_func=f_classif, k="all")),
    ])


def summarize_original_feature_scores(pipeline):
    """Aggregate transformed SelectKBest scores into one score per source feature."""
    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps["selector"]
    transformed_scores = pd.DataFrame({
        "transformed_feature": preprocessor.get_feature_names_out(),
        "raw_score": selector.scores_,
        "p_value": selector.pvalues_,
    })
    transformed_scores["feature"] = transformed_scores["transformed_feature"].apply(
        get_original_feature_name
    )
    feature_scores = (
        transformed_scores
        .sort_values("raw_score", ascending=False, kind="stable")
        .groupby("feature", sort=False)
        .agg(
            raw_score=("raw_score", "max"),
            p_value=("p_value", "min"),
            transformed_features=("transformed_feature", lambda values: list(values)),
        )
        .reindex(FEATURE_COLUMNS)
        .reset_index()
    )
    return feature_scores.rename(columns={"index": "feature"})


def analyze_features(data):
    """Return a SelectKBest score report from the training split."""
    X_train, _, y_train, _ = train_test_split(
        data.loc[:, FEATURE_COLUMNS],
        data[TARGET_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[TARGET_COLUMN],
    )
    pipeline = create_scoring_pipeline()
    pipeline.fit(X_train, y_train)
    scores = summarize_original_feature_scores(pipeline)
    scores["select_k_best_score"] = scores["raw_score"]
    scores["selected"] = scores["select_k_best_score"] > FEATURE_SCORE_THRESHOLD
    scores = scores.sort_values(
        "select_k_best_score", ascending=False, kind="stable"
    )
    scores.insert(0, "rank", range(1, len(scores) + 1))
    selected = scores.loc[scores["selected"], "feature"].tolist()
    return {
        "method": "SelectKBest",
        "score_function": "f_classif",
        "k": "all",
        "selection_threshold": FEATURE_SCORE_THRESHOLD,
        "score_normalization": "none",
        "candidate_features": list(FEATURE_COLUMNS),
        "selected_features": selected,
        "training_rows_used": len(X_train),
        "feature_scores": [
            {
                "feature": row.feature,
                "rank": int(row.rank),
                "select_k_best_score": float(row.select_k_best_score),
                "raw_score": float(row.raw_score),
                "transformed_features": list(row.transformed_features),
                "selected": bool(row.selected),
                "p_value": float(row.p_value),
            }
            for row in scores.itertuples(index=False)
        ],
    }


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Training file not found: {DATA_PATH}")

    raw_data = load_loan_data(DATA_PATH)
    data = validate_and_clean_data(raw_data)
    analysis = analyze_features(data)
    analysis["excluded_source_features"] = []
    analysis["source_sha256"] = get_source_sha256(DATA_PATH)

    FEATURE_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_ANALYSIS_PATH.open("w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2)
    score_table = pd.DataFrame(analysis["feature_scores"])[
        [
            "rank",
            "feature",
            "select_k_best_score",
            "selected",
            "p_value",
            "raw_score",
        ]
    ]
    score_table.to_csv(FEATURE_SCORE_TABLE_PATH, index=False)

    print(
        "SelectKBest selected "
        f"{len(analysis['selected_features'])} features: "
        f"{', '.join(analysis['selected_features'])}"
    )
    print(f"Feature analysis saved to {FEATURE_ANALYSIS_PATH}")
    print(f"Feature score table saved to {FEATURE_SCORE_TABLE_PATH}")


if __name__ == "__main__":
    main()
