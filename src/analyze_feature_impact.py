"""Calculate post-training feature impact scores for the best saved model.

This script uses permutation importance on the held-out test split. The fitted
pipeline receives the original feature columns, so each score reflects the F1
drop caused by shuffling one source feature after preprocessing and
SelectKBest have already been applied.

Usage:
    python src/analyze_feature_impact.py
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer

from settings import (
    COMPARISON_TABLE_PATH,
    FEATURE_COLUMNS,
    FEATURE_IMPACT_ANALYSIS_PATH,
    FEATURE_IMPACT_CHART_PATH,
    FEATURE_IMPACT_TABLE_PATH,
    MODEL_PATHS,
    MODEL_SUMMARY_PATHS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)
from training_utils import (
    POSITIVE_LABEL,
    get_selected_model_features,
    load_test_metrics,
    load_test_data,
)


SCORING_METRIC = "approved_f1"
N_REPEATS = 30


def load_best_model_choice():
    """Choose the best trained model from the saved F1 comparison table."""
    if not COMPARISON_TABLE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing model comparison file: {COMPARISON_TABLE_PATH}. "
            "Run compare_models.py after training both models."
        )
    comparison = pd.read_csv(COMPARISON_TABLE_PATH)
    required_columns = {"model", "f1"}
    missing_columns = required_columns - set(comparison.columns)
    if missing_columns:
        raise ValueError(
            "The comparison table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )

    comparison = comparison.sort_values("f1", ascending=False, kind="stable")
    model_name = str(comparison.iloc[0]["model"])
    if model_name not in MODEL_PATHS:
        raise ValueError(f"No model artifact is configured for {model_name}.")
    model_path = MODEL_PATHS[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing trained model file: {model_path}")
    return model_name, model_path


def calculate_permutation_impacts(model, X_test, y_test):
    """Return a sorted table of original-feature permutation impact scores."""
    scorer = make_scorer(
        f1_score,
        pos_label=POSITIVE_LABEL,
        zero_division=0,
    )
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scorer,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    impacts = pd.DataFrame({
        "feature": list(FEATURE_COLUMNS),
        "impact_score_mean": result.importances_mean,
        "impact_score_std": result.importances_std,
    })
    impacts = impacts.sort_values(
        ["impact_score_mean", "impact_score_std", "feature"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    impacts.insert(0, "rank", range(1, len(impacts) + 1))
    return impacts


def load_baseline_metrics(model_name):
    return load_test_metrics(MODEL_SUMMARY_PATHS[model_name])


def save_impact_chart(impacts, output_path):
    """Save a compact horizontal bar chart for the ranked impact scores."""
    chart_data = impacts.sort_values("impact_score_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 6))
    colors = [
        "#2f6f8f" if selected else "#a9b4bd"
        for selected in chart_data["selected_by_model"]
    ]
    axis.barh(
        chart_data["feature"],
        chart_data["impact_score_mean"],
        xerr=chart_data["impact_score_std"],
        color=colors,
        alpha=0.88,
    )
    axis.set_xlabel("F1-score drop after permutation")
    axis.set_ylabel("Feature")
    axis.set_title("Feature Impact Scores - Permutation Importance")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    model_name, model_path = load_best_model_choice()
    X_test, y_test = load_test_data(PROCESSED_DATA_DIR)
    model = joblib.load(model_path)

    baseline_metrics = load_baseline_metrics(model_name)
    selected_features = get_selected_model_features(model)

    impacts = calculate_permutation_impacts(model, X_test, y_test)
    impacts["selected_by_model"] = impacts["feature"].isin(selected_features)
    impacts["scoring_metric"] = SCORING_METRIC
    impacts["n_repeats"] = N_REPEATS
    impacts["baseline_f1"] = baseline_metrics["f1"]

    FEATURE_IMPACT_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    impacts.to_csv(FEATURE_IMPACT_TABLE_PATH, index=False)
    save_impact_chart(impacts, FEATURE_IMPACT_CHART_PATH)

    analysis = {
        "method": "Permutation Feature Importance",
        "rationale": (
            "Permutation importance is model-agnostic, works on the fitted "
            "pipeline with preprocessing and SelectKBest, and measures the "
            "held-out F1 drop caused by corrupting one original feature."
        ),
        "selected_model": model_name,
        "model_path": str(model_path),
        "scoring_metric": SCORING_METRIC,
        "n_repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "baseline_metrics": baseline_metrics,
        "selected_features": selected_features,
        "candidate_features": list(FEATURE_COLUMNS),
        "impact_scores": [
            {
                "rank": int(row.rank),
                "feature": row.feature,
                "impact_score_mean": float(row.impact_score_mean),
                "impact_score_std": float(row.impact_score_std),
                "selected_by_model": bool(row.selected_by_model),
            }
            for row in impacts.itertuples(index=False)
        ],
    }
    with FEATURE_IMPACT_ANALYSIS_PATH.open("w", encoding="utf-8") as file:
        json.dump(analysis, file, indent=2)

    print(
        f"{model_name} selected as the best model for impact analysis "
        f"(baseline F1: {baseline_metrics['f1']:.4f})."
    )
    print(f"Feature impact table saved to {FEATURE_IMPACT_TABLE_PATH}")
    print(f"Feature impact chart saved to {FEATURE_IMPACT_CHART_PATH}")


if __name__ == "__main__":
    main()
