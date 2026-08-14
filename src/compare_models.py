"""Compare the trained Logistic Regression and Random Forest models."""

import json

import matplotlib.pyplot as plt
import pandas as pd

from settings import MODELS_DIR


MODEL_SUMMARIES = {
    "Logistic Regression": MODELS_DIR / "logistic_regression_training_summary.json",
    "Random Forest": MODELS_DIR / "random_forest_training_summary.json",
}
REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1")


def load_metrics_from_summary(model_name, summary_path):
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing training summary for {model_name}: {summary_path}. "
            "Train both models first."
        )
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    test_metrics = summary.get("test_metrics")
    if not isinstance(test_metrics, dict):
        raise ValueError(
            f"{summary_path.name} is missing a valid test_metrics object."
        )
    missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in test_metrics]
    if missing_metrics:
        raise ValueError(
            f"{summary_path.name} is missing metrics: {', '.join(missing_metrics)}."
        )
    return {metric: float(test_metrics[metric]) for metric in REQUIRED_METRICS}


def main():
    results = []
    for model_name, summary_path in MODEL_SUMMARIES.items():
        results.append({
            "model": model_name,
            **load_metrics_from_summary(model_name, summary_path),
        })

    comparison = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("=== Model Comparison ===")
    print(comparison.set_index("model").round(4))
    comparison.to_csv(MODELS_DIR / "comparison_table.csv", index=False)

    chart_columns = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
    }
    chart_data = comparison.set_index("model")[list(chart_columns)].rename(
        columns=chart_columns
    )
    axis = chart_data.plot(kind="bar", figsize=(9, 5.5), rot=0)
    axis.set_xlabel("Model")
    axis.set_ylabel("Score")
    axis.set_title("Logistic Regression vs Random Forest")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Metric", loc="lower right")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    plt.tight_layout()
    figure = axis.get_figure()
    figure.savefig(MODELS_DIR / "comparison_chart.png", dpi=180)
    plt.close(figure)

    winner = comparison.iloc[0]
    print(f"\nBest F1-score: {winner['model']} ({winner['f1']:.4f})")
    print(f"Saved comparison artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
