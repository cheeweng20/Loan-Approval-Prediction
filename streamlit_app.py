"""Streamlit interface for the trained loan approval classifiers.

Run from the project root with:
    streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import training_utils

from src.settings import (
    COMPARISON_CHART_PATH,
    COMPARISON_TABLE_PATH,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_IMPACT_CHART_PATH,
    FEATURE_IMPACT_TABLE_PATH,
    MODEL_PATHS,
    NUMERIC_FEATURES,
)


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

# feature -> (min, max, step, caster)
NUMERIC_BOUNDS = {
    "no_of_dependents": (0, 20, 1, int),
    "loan_term": (1, 40, 1, int),
    "cibil_score": (300, 900, 1, int),
}
DEFAULT_NUMERIC_BOUNDS = (0.0, 100_000_000.0, 100_000.0, float)

DEFAULTS = {
    "income_annum": 5_000_000.0,
    "loan_amount": 15_000_000.0,
    "loan_term": 10,
    "no_of_dependents": 0,
    "cibil_score": 600,
    "residential_assets_value": 0.0,
    "commercial_assets_value": 0.0,
    "luxury_assets_value": 0.0,
    "bank_asset_value": 0.0,
    "education": "Graduate",
    "self_employed": "No",
}


def render_numeric_input(container, feature, default):
    """Render a number_input with bounds looked up from NUMERIC_BOUNDS."""
    min_val, max_val, step, caster = NUMERIC_BOUNDS.get(feature, DEFAULT_NUMERIC_BOUNDS)
    label = feature.replace("_", " ").title()
    return container.number_input(
        label, min_value=min_val, max_value=max_val, value=caster(default), step=step
    )


def render_categorical_input(container, feature):
    """Render a selectbox for a known categorical feature."""
    if feature == "education":
        return container.selectbox("Education", options=["Graduate", "Not Graduate"], index=0)
    if feature == "self_employed":
        return container.selectbox("Self-employed", options=["No", "Yes"], index=0)
    raise ValueError(f"Unknown categorical feature: {feature}")


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


@st.cache_data
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


@st.cache_data
def load_feature_impact_table():
    """Load and validate the feature-impact table produced by the workflow."""
    table = pd.read_csv(FEATURE_IMPACT_TABLE_PATH)
    missing_columns = set(FEATURE_IMPACT_COLUMNS) - set(table.columns)
    if missing_columns:
        raise ValueError(
            "The feature-impact table is missing columns: "
            f"{', '.join(sorted(missing_columns))}."
        )
    table = table[list(FEATURE_IMPACT_COLUMNS)].rename(columns={
        "rank": "Rank",
        "feature": "Feature",
        "impact_score_mean": "Impact score",
        "impact_score_std": "Score variation",
        "selected_by_model": "Selected",
    })
    table["Selected"] = table["Selected"].map({True: "Yes", False: "No"})
    return table


def get_selected_features():
    """Return the common feature set retained by both saved pipelines."""
    selections = {
        name: tuple(training_utils.get_selected_model_features(model))
        for name, model in load_models().items()
    }
    unique_selections = set(selections.values())
    if len(unique_selections) != 1:
        raise ValueError(
            "The saved models use different selected features. Retrain both models "
            "with the same project settings."
        )
    return list(next(iter(unique_selections)))


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

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
PAGES = ["Predict", "Model comparison", "Feature impact"]
PAGE_ICONS = ["bank", "bar-chart-line", "bullseye"]  # Bootstrap Icons names

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; padding:4px 4px 16px 4px;">
            <div style="background:#fff; border-radius:10px; width:34px; height:34px;
                        display:flex; align-items:center; justify-content:center;
                        font-weight:700; font-size:18px; color:#000;">🏦</div>
            <span style="font-size:22px; font-weight:600; color:#fff;">Loan Approval</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = option_menu(
        menu_title=None,
        options=PAGES,
        icons=PAGE_ICONS,
        default_index=0,
        styles={
            "container": {"padding": "0", "background-color": "transparent"},
            "icon": {"font-size": "16px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "10px 14px",
                "border-radius": "10px",
                "color": "#e6e6e6",
            },
            "nav-link-selected": {
                "background-color": "#0d6efd",
                "color": "#fff",
                "font-weight": "500",
            },
        },
    )

    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    if "models_ok" not in st.session_state:
        try:
            load_models()
            st.session_state.models_ok = True
        except Exception:
            st.session_state.models_ok = False

    if st.session_state.models_ok:
        st.success("Models loaded", icon="✅")
    else:
        st.error("Models not found", icon="⚠️")


def page_predict():
    st.title("🏦 Loan Approval Prediction")
    st.write(
        "Enter an applicant's information to compare predictions from Logistic Regression and Random Forest."
    )

    try:
        selected_features = get_selected_features()
    except (FileNotFoundError, OSError, TypeError, ValueError, AttributeError) as exception:
        st.error(f"Could not load compatible trained models: {exception}")
        return

    with st.form("loan_application_form"):
        st.markdown("**Inputs used by both trained models**")
        st.caption(
            "Fields removed by feature selection are filled internally and cannot "
            "affect the current predictions."
        )
        cols = st.columns(3)
        inputs = {}

        selected_numeric = [f for f in selected_features if f in NUMERIC_FEATURES]
        for i, feature in enumerate(selected_numeric):
            col = cols[i % len(cols)]
            inputs[feature] = render_numeric_input(col, feature, DEFAULTS.get(feature, 0.0))

        selected_categorical = [f for f in selected_features if f in CATEGORICAL_FEATURES]
        for j, feature in enumerate(selected_categorical):
            col = cols[(len(selected_numeric) + j) % len(cols)]
            inputs[feature] = render_categorical_input(col, feature)

        submitted = st.form_submit_button("Predict loan status")

    if submitted:
        application = {}
        for feature in FEATURE_COLUMNS:
            val = inputs.get(feature, DEFAULTS.get(feature))
            if feature in NUMERIC_FEATURES:
                if feature in ("no_of_dependents", "loan_term", "cibil_score"):
                    application[feature] = int(val)
                else:
                    application[feature] = float(val)
            else:
                application[feature] = str(val)

        try:
            application_df = build_application_data(application)
            predictions = predict_application(application_df)
        except FileNotFoundError:
            st.error(
                "The trained model files are missing. Run data preparation and both "
                "training scripts first."
            )
        except (OSError, TypeError, ValueError, AttributeError) as exception:
            st.error(f"Could not load compatible model files or build application: {exception}")
        else:
            st.subheader("Prediction")
            result_columns = st.columns(len(predictions))
            for column, (model_name, predicted_label) in zip(result_columns, predictions.items()):
                column.metric(model_name, predicted_label.upper())


def page_model_comparison():
    st.header("Model comparison")
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


def page_feature_impact():
    st.header("Feature impact")
    if FEATURE_IMPACT_TABLE_PATH.is_file():
        try:
            table = load_feature_impact_table()
        except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exception:
            st.warning(f"Could not read the saved feature impact scores: {exception}")
        else:
            st.dataframe(
                table.style.format({
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


# Route to selected page
if page == "Predict":
    page_predict()
elif page == "Model comparison":
    page_model_comparison()
else:
    page_feature_impact()
