"""Streamlit interface for the trained loan approval classifiers.

Run from the project root with:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import training_utils  # noqa: F401

from src.settings import (
    FEATURE_COLUMNS,
    FEATURE_IMPACT_CHART_PATH,
    FEATURE_IMPACT_TABLE_PATH,
    MODELS_DIR,
)


MODEL_PATHS = {
    "Logistic Regression": MODELS_DIR / "logistic_regression_model.joblib",
    "Random Forest": MODELS_DIR / "random_forest_model.joblib",
}
COMPARISON_TABLE_PATH = MODELS_DIR / "comparison_table.csv"
COMPARISON_CHART_PATH = MODELS_DIR / "comparison_chart.png"
COMPARISON_COLUMNS = (
    "model",
    "accuracy",
    "precision",
    "recall",
    "f1",
)
FEATURE_IMPACT_COLUMNS = (
    "rank",
    "feature",
    "impact_score_mean",
    "impact_score_std",
    "selected_by_model",
)


@st.cache_resource
def load_models():
    """Load and validate the two fitted model pipelines once per app process."""
    models = {
        model_name: joblib.load(model_path)
        for model_name, model_path in MODEL_PATHS.items()
    }
    for model_name, model in models.items():
        if not callable(getattr(model, "predict", None)):
            raise TypeError(f"{model_name} artifact does not provide predict().")
    return models


def load_comparison_table():
    table = pd.read_csv(COMPARISON_TABLE_PATH)
    missing_columns = set(COMPARISON_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(
            "The comparison table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    return table[list(COMPARISON_COLUMNS)].rename(columns={
        "model": "Model",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1-score",
    })


def load_feature_impact_table():
    table = pd.read_csv(FEATURE_IMPACT_TABLE_PATH)
    missing_columns = set(FEATURE_IMPACT_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(
            "The feature impact table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    return table[list(FEATURE_IMPACT_COLUMNS)].rename(columns={
        "rank": "Rank",
        "feature": "Feature",
        "impact_score_mean": "Impact score",
        "impact_score_std": "Score variation",
        "selected_by_model": "Selected",
    })


def build_application_data(values):
    """Build a one-row DataFrame using the exact training feature schema."""
    application = pd.DataFrame([values])
    missing = set(FEATURE_COLUMNS) - set(application.columns)
    if missing:
        raise ValueError(f"Missing application fields: {sorted(missing)}.")
    return application.loc[:, FEATURE_COLUMNS]


def predict_application(application):
    """Return each model's predicted loan status."""
    return {
        model_name: str(model.predict(application)[0])
        for model_name, model in load_models().items()
    }


st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Approval Prediction")
st.write(
    "Enter an applicant's financial and loan information to compare predictions "
    "from Logistic Regression and Random Forest."
)
with st.form("loan_application_form"):
    financial_column, credit_column = st.columns(2)
    with financial_column:
        st.subheader("Financial and loan details")
        income_annum = st.number_input(
            "Annual income",
            min_value=100_000,
            max_value=100_000_000,
            value=5_000_000,
            step=100_000,
            help="Use the same monetary units as the training dataset.",
        )
        loan_amount = st.number_input(
            "Loan amount",
            min_value=100_000,
            max_value=100_000_000,
            value=15_000_000,
            step=100_000,
        )
        loan_term = st.number_input(
            "Loan term", min_value=1, max_value=40, value=10, step=1
        )

    with credit_column:
        st.subheader("Credit details")
        cibil_score = st.number_input(
            "CIBIL score", min_value=300, max_value=900, value=600, step=1
        )

    submitted = st.form_submit_button(
        "Predict loan status", type="primary", width="stretch"
    )

if submitted:
    application = build_application_data({
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
    })
    try:
        predictions = predict_application(application)
    except FileNotFoundError:
        st.error(
            "The trained model files are missing. Run data preparation and both "
            "training scripts first."
        )
    except (OSError, TypeError, ValueError, AttributeError) as exception:
        st.error(f"Could not load compatible model files: {exception}")
    else:
        st.subheader("Prediction")
        result_columns = st.columns(len(predictions))
        for column, (model_name, predicted_label) in zip(
            result_columns, predictions.items()
        ):
            column.metric(model_name, predicted_label.upper())

st.divider()
st.subheader("Model comparison")
if COMPARISON_TABLE_PATH.is_file():
    try:
        comparison = load_comparison_table()
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exception:
        st.warning(f"Could not read the saved model comparison: {exception}")
    else:
        st.dataframe(
            comparison.style.format({
                column: "{:.4f}"
                for column in comparison.columns
                if column != "Model"
            }),
            hide_index=True,
            width="stretch",
        )
else:
    st.info("Run the model-comparison script to generate the results table.")

if COMPARISON_CHART_PATH.is_file():
    st.image(
        str(COMPARISON_CHART_PATH),
        caption="Logistic Regression and Random Forest test-set performance",
        width="stretch",
    )

st.divider()
st.subheader("Feature impact")
if FEATURE_IMPACT_TABLE_PATH.is_file():
    try:
        feature_impact = load_feature_impact_table()
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exception:
        st.warning(f"Could not read the saved feature impact scores: {exception}")
    else:
        st.dataframe(
            feature_impact.style.format({
                "Impact score": "{:.4f}",
                "Score variation": "{:.4f}",
            }),
            hide_index=True,
            width="stretch",
        )
else:
    st.info("Run the feature-impact script to generate the ranked scores.")

if FEATURE_IMPACT_CHART_PATH.is_file():
    st.image(
        str(FEATURE_IMPACT_CHART_PATH),
        caption="Permutation impact scores for the best saved model",
        width="stretch",
    )
