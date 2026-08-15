"""Compare the trained Logistic Regression and Random Forest models."""

import matplotlib.pyplot as plt
import pandas as pd

from settings import (
    COMPARISON_CHART_PATH,
    COMPARISON_TABLE_PATH,
    MODEL_SUMMARY_PATHS,
    MODELS_DIR,
)
from training_utils import load_test_metrics


def main():
    results = []
    for model_name, summary_path in MODEL_SUMMARY_PATHS.items():
        results.append({
            "model": model_name,
            **load_test_metrics(summary_path),
        })

    comparison = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("=== Model Comparison ===")
    print(comparison.set_index("model").round(4))
    comparison.to_csv(COMPARISON_TABLE_PATH, index=False)

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
    figure.savefig(COMPARISON_CHART_PATH, dpi=180)
    plt.close(figure)

    winner = comparison.iloc[0]
    print(f"\nBest F1-score: {winner['model']} ({winner['f1']:.4f})")
    print(f"Saved comparison artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
