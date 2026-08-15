"""Shared preprocessing, model selection, evaluation, and plotting helpers."""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from settings import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    EXPECTED_LABELS,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    FEATURE_SCORE_THRESHOLD,
)


ARTIFACT_NAMES = (
    "X_train.joblib",
    "X_test.joblib",
    "y_train.joblib",
    "y_test.joblib",
)
POSITIVE_LABEL = "Approved"
NEGATIVE_LABEL = "Rejected"
SELECTION_METRIC = "f1"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


class ThresholdSelectKBest(SelectKBest):
    """Keep features whose raw F-test score exceeds the threshold."""

    def __init__(self, score_func=f_classif, threshold=FEATURE_SCORE_THRESHOLD):
        super().__init__(score_func=score_func, k="all")
        self.threshold = threshold

    def _get_support_mask(self):
        scores = self.scores_
        return pd.notna(scores) & (scores > self.threshold)


def _load_artifacts(processed_dir, artifact_names):
    missing = [
        name for name in artifact_names if not (processed_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing processed files in {processed_dir}: {', '.join(missing)}. "
            "Run prepare_data.py first."
        )
    return tuple(joblib.load(processed_dir / name) for name in artifact_names)


def _validate_features_and_labels(X, y, split_name):
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"{split_name} features must be a pandas DataFrame.")
    if tuple(X.columns) != FEATURE_COLUMNS:
        raise ValueError(f"{split_name} features do not match the expected schema.")
    if len(X) != len(y):
        raise ValueError(f"{split_name} feature and label row counts do not match.")
    if set(pd.Series(y)) != EXPECTED_LABELS:
        raise ValueError(
            f"{split_name} labels must contain both Approved and Rejected."
        )


def load_training_data(processed_dir):
    X_train, X_test, y_train, y_test = _load_artifacts(
        processed_dir, ARTIFACT_NAMES
    )
    _validate_features_and_labels(X_train, y_train, "Training")
    _validate_features_and_labels(X_test, y_test, "Test")
    train_counts = pd.Series(y_train).value_counts()
    if train_counts.min() < CV_FOLDS:
        raise ValueError(
            f"Each training class needs at least {CV_FOLDS} rows for validation."
        )
    train_hashes = set(pd.util.hash_pandas_object(X_train, index=False))
    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False))
    if train_hashes.intersection(test_hashes):
        raise ValueError("Duplicated applications exist across training and test sets.")
    return X_train, X_test, y_train, y_test


def load_test_data(processed_dir):
    X_test, y_test = _load_artifacts(
        processed_dir, ("X_test.joblib", "y_test.joblib")
    )
    _validate_features_and_labels(X_test, y_test, "Test")
    return X_test, y_test


def create_preprocessor(scale_numeric):
    """Create feature-aware preprocessing fitted only inside model training."""
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    transformers = [
        ("numeric", numeric_transformer, list(NUMERIC_FEATURES)),
    ]
    if CATEGORICAL_FEATURES:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            )
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def get_original_feature_name(transformed_feature):
    """Map a preprocessor output name back to its source feature."""
    for feature in FEATURE_COLUMNS:
        if transformed_feature == f"numeric__{feature}":
            return feature
        if transformed_feature.startswith(f"categorical__{feature}_"):
            return feature
    raise ValueError(f"Cannot map transformed feature: {transformed_feature}")


def create_model_pipeline(classifier, scale_numeric):
    """Create a leakage-safe pipeline with fold-specific feature selection."""
    return Pipeline([
        ("preprocessor", create_preprocessor(scale_numeric)),
        (
            "feature_selector",
            ThresholdSelectKBest(threshold=FEATURE_SCORE_THRESHOLD),
        ),
        ("classifier", classifier),
    ])


def get_selected_model_features(model):
    """Return original source features retained by the fitted SelectKBest step."""
    preprocessor = model.named_steps["preprocessor"]
    selector = model.named_steps["feature_selector"]
    feature_names = preprocessor.get_feature_names_out()
    selected_features = []
    for transformed_feature in feature_names[selector.get_support()]:
        original_feature = get_original_feature_name(transformed_feature)
        if original_feature not in selected_features:
            selected_features.append(original_feature)
    return selected_features


def load_test_metrics(summary_path):
    """Load and validate final test metrics from a training summary."""
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing training summary: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    test_metrics = summary.get("test_metrics")
    if not isinstance(test_metrics, dict):
        raise ValueError(f"{summary_path.name} is missing a valid test_metrics object.")
    missing = [name for name in METRIC_NAMES if name not in test_metrics]
    if missing:
        raise ValueError(
            f"{summary_path.name} is missing metrics: {', '.join(missing)}."
        )
    return {name: float(test_metrics[name]) for name in METRIC_NAMES}


def create_grid_search(pipeline, parameter_grid):
    """Create a reproducible five-fold model search using multiple metrics."""
    approved_precision = make_scorer(
        precision_score, pos_label=POSITIVE_LABEL, zero_division=0
    )
    approved_recall = make_scorer(
        recall_score, pos_label=POSITIVE_LABEL, zero_division=0
    )
    approved_f1 = make_scorer(
        f1_score, pos_label=POSITIVE_LABEL, zero_division=0
    )
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    return GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring={
            "accuracy": "accuracy",
            "precision": approved_precision,
            "recall": approved_recall,
            "f1": approved_f1,
        },
        refit=SELECTION_METRIC,
        cv=cross_validation,
        n_jobs=1,
        return_train_score=False,
        verbose=1,
    )


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
    }


def print_results(model_name, y_true, y_pred, metrics):
    print(f"=== {model_name} Results ===")
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title():<18}: {value:.4f}")
    print(
        "\n",
        classification_report(
            y_true,
            y_pred,
            labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
            zero_division=0,
        ),
    )


def save_grid_search_results(
    search, model_name, artifact_stem, output_dir, test_metrics, selected_features
):
    result_columns = [
        "rank_test_f1",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision",
        "std_test_precision",
        "mean_test_recall",
        "std_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "params",
    ]
    results = pd.DataFrame(search.cv_results_)
    results = results[result_columns].sort_values("rank_test_f1")
    results.to_csv(output_dir / f"{artifact_stem}_grid_search.csv", index=False)

    summary = {
        "model": model_name,
        "cv_folds": CV_FOLDS,
        "selection_metric": SELECTION_METRIC,
        "best_parameters": search.best_params_,
        "best_cross_validation_score": search.best_score_,
        "test_metrics": test_metrics,
        "selected_features": selected_features,
        "scikit_learn_version": sklearn.__version__,
    }
    with (output_dir / f"{artifact_stem}_training_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, default=str)


def save_confusion_matrix(y_true, y_pred, title, color_map, output_path):
    matrix = confusion_matrix(
        y_true, y_pred, labels=[NEGATIVE_LABEL, POSITIVE_LABEL]
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=color_map,
        xticklabels=["Predicted Rejected", "Predicted Approved"],
        yticklabels=["Actual Rejected", "Actual Approved"],
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_and_evaluate_model(
    classifier,
    classifier_parameter_grid,
    scale_numeric,
    model_name,
    artifact_stem,
    confusion_matrix_filename,
    color_map,
    processed_dir,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_training_data(processed_dir)
    search = create_grid_search(
        create_model_pipeline(classifier, scale_numeric),
        classifier_parameter_grid,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_

    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    print("Best parameters selected using five-fold cross-validation:")
    for name, value in search.best_params_.items():
        print(f"  {name}: {value}")
    print_results(model_name, y_test, y_pred, metrics)
    selected_features = get_selected_model_features(model)
    print(f"SelectKBest retained: {', '.join(selected_features)}")

    save_grid_search_results(
        search, model_name, artifact_stem, output_dir, metrics, selected_features
    )
    save_confusion_matrix(
        y_test,
        y_pred,
        f"{model_name} - Confusion Matrix",
        color_map,
        output_dir / confusion_matrix_filename,
    )
    joblib.dump(model, output_dir / f"{artifact_stem}_model.joblib")
    print(f"Saved model and evaluation artifacts to {output_dir}/")
