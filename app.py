"""Streamlit dashboard for the California Housing project."""

import numpy as np
import pandas as pd
import streamlit as st

from src.housing.bounds import outside_mask, training_bounds
from src.housing.data import (
    EXPECTED_FEATURES,
    TARGET_COLUMN,
    TARGET_UNIT_USD,
    load_dataset,
    validate_frame,
)
from src.housing.finalize import MODEL_PATH, load_model
from src.housing.models import build_models
from src.housing.predict import (
    default_features,
    format_usd,
    predict_house_value,
)
from src.housing.reports import AVAILABLE_REPORTS, load_report
from src.housing.train import train_and_evaluate
from src.housing.uncertainty import prediction_interval

CAP_VALUE = 5.0


@st.cache_data(persist="disk", show_spinner="Training models...")
def get_data():
    """Load and validate the dataset once per session."""
    df = load_dataset()
    validate_frame(df)
    return df


@st.cache_data(show_spinner=False)
def get_bounds():
    """Percentile bounds used to flag unusual inputs."""
    return training_bounds(get_data())


@st.cache_resource(show_spinner="Loading the saved model...")
def get_saved_model():
    """Load the saved artifact, or return None when it is missing."""
    if MODEL_PATH.exists():
        return load_model()
    return None


@st.cache_data(persist="disk", show_spinner="Comparing models...")
def get_results():
    """Use saved comparison results, or train the models."""
    saved = get_saved_model()
    if saved is not None and saved.get("results"):
        return saved["results"]
    return train_and_evaluate()


@st.cache_resource(show_spinner="Fitting the best model...")
def get_model(model_name):
    """Use the saved model when it matches, otherwise fit a fresh one."""
    saved = get_saved_model()
    if saved is not None and saved["metadata"]["model"] == model_name:
        return saved["model"]

    df = get_data()
    model = build_models()[model_name]
    model.fit(df[EXPECTED_FEATURES], df[TARGET_COLUMN])
    return model


def clamp(value, low, high):
    """Keep a default value inside a slider's range."""
    return max(low, min(value, high))


def render_predict_tab(df, results):
    """Show the input form and the predicted value."""
    best = results["best_model"]
    best_row = results["test_results"].loc[
        results["test_results"]["model"] == best
    ].iloc[0]
    defaults = default_features(df).iloc[0]

    st.subheader("Predict a census block group")
    st.caption(
        f"Model: **{best}** | holdout MAE: "
        f"**{format_usd(best_row['mae_usd'])}**"
    )

    # Two columns keep the form compact.
    with st.form("prediction_form"):
        left, right = st.columns(2)

        med_inc = left.slider(
            "Median income ($10,000 units)",
            0.5,
            15.0,
            float(clamp(defaults["MedInc"], 0.5, 15.0)),
            0.1,
        )
        house_age = right.slider(
            "House age (years)",
            1,
            52,
            int(clamp(defaults["HouseAge"], 1, 52)),
            1,
        )

        ave_rooms = left.slider(
            "Average rooms per household",
            1.0,
            20.0,
            float(clamp(defaults["AveRooms"], 1.0, 20.0)),
            0.1,
        )
        ave_bedrms = right.slider(
            "Average bedrooms per household",
            0.5,
            10.0,
            float(clamp(defaults["AveBedrms"], 0.5, 10.0)),
            0.1,
        )

        population = left.slider(
            "Block population",
            1,
            10_000,
            int(clamp(defaults["Population"], 1, 10_000)),
            1,
        )
        ave_occup = right.slider(
            "Average occupants per household",
            0.5,
            10.0,
            float(clamp(defaults["AveOccup"], 0.5, 10.0)),
            0.1,
        )

        latitude = left.slider(
            "Latitude",
            32.5,
            42.0,
            float(clamp(defaults["Latitude"], 32.5, 42.0)),
            0.01,
        )
        longitude = right.slider(
            "Longitude",
            -124.5,
            -114.0,
            float(clamp(defaults["Longitude"], -124.5, -114.0)),
            0.01,
        )

        submitted = st.form_submit_button("Predict", type="primary")

    if not submitted:
        st.info("Adjust the inputs, then press **Predict**.")
        return

    features = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }

    model = get_model(best)
    value = predict_house_value(model, features)
    interval = prediction_interval(model, features).iloc[0]

    st.metric("Predicted median house value", format_usd(value))
    st.caption(
        "Likely range: "
        f"**{format_usd(interval['low'] * TARGET_UNIT_USD)} to "
        f"{format_usd(interval['high'] * TARGET_UNIT_USD)}** "
        "(10th to 90th percentile across the forest's trees)."
    )

    flagged = outside_mask(features, get_bounds()).iloc[0]
    unusual = list(flagged.index[flagged])
    if unusual:
        st.warning(
            "Outside the training range: "
            + ", ".join(unusual)
            + ". The estimate may be unreliable."
        )

    if value >= CAP_VALUE * TARGET_UNIT_USD:
        st.info(
            "This estimate is at or above the $500,000 census cap. "
            "High-value districts are harder to separate."
        )

    st.caption(
        "This is a 1990 census block-group median, not a current "
        "individual house price."
    )


def render_models_tab(results):
    """Show cross-validation and holdout scores."""
    best = results["best_model"]
    best_row = results["test_results"].loc[
        results["test_results"]["model"] == best
    ].iloc[0]

    saved = get_saved_model()
    if saved is not None:
        meta = saved["metadata"]
        st.caption(
            f"Saved model: **{meta['model']}** trained on "
            f"{meta['rows']:,} rows at {meta['saved_at']}."
        )
    else:
        st.caption(
            "No saved model artifact found. Run "
            "`python -m src.housing.finalize` to create one."
        )

    st.subheader("Best model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", best)
    col2.metric("Holdout MAE", format_usd(best_row["mae_usd"]))
    col3.metric("Holdout RMSE", format_usd(best_row["rmse_usd"]))
    col4.metric("Holdout R²", f"{best_row['r2']:.3f}")

    st.subheader("Cross-validation on the training set")
    cv = results["cv_results"][
        ["model", "cv_mae_mean", "cv_rmse_mean", "cv_r2_mean"]
    ].copy()
    cv.columns = ["Model", "MAE", "RMSE", "R²"]
    st.dataframe(cv.round(3), hide_index=True)

    st.subheader("Holdout test set")
    test = results["test_results"].copy()
    test["MAE ($)"] = test["mae_usd"].map(format_usd)
    test["RMSE ($)"] = test["rmse_usd"].map(format_usd)
    test = test[["model", "mae", "rmse", "r2", "MAE ($)", "RMSE ($)"]]
    test.columns = ["Model", "MAE", "RMSE", "R²", "MAE ($)", "RMSE ($)"]
    st.dataframe(test.round(3), hide_index=True)

    st.subheader("Holdout MAE by model")
    chart = results["test_results"].set_index("model")["mae"]
    st.bar_chart(chart, y_label="MAE (target units)")


def render_data_tab(df):
    """Show a quick summary and the target distribution."""
    summary = validate_frame(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{summary['rows']:,}")
    col2.metric("Features", summary["features"])
    col3.metric("Rows at $500k cap", f"{summary['cap_count_at_5_0']:,}")

    st.subheader("Target distribution")
    counts, edges = np.histogram(df[TARGET_COLUMN], bins=30)
    hist = pd.DataFrame(
        {
            "House value ($100,000 units)": np.round(edges[:-1], 2),
            "Rows": counts,
        }
    )
    st.bar_chart(
        hist,
        x="House value ($100,000 units)",
        y="Rows",
    )

    st.subheader("Summary statistics")
    st.dataframe(df[EXPECTED_FEATURES + [TARGET_COLUMN]].describe().T.round(3))


def render_reports_tab():
    """Show the committed project reports inside the app."""
    st.subheader("Project reports")
    name = st.selectbox("Report", AVAILABLE_REPORTS, index=0)

    text = load_report(name)
    if text is None:
        st.info("This report is not available in the current checkout.")
        return

    # Escape dollar signs so Streamlit does not read them as LaTeX.
    st.markdown(text.replace("$", "\\$"))
    st.caption("Links inside reports point to files in the repository.")


def main():
    """Build the dashboard."""
    st.set_page_config(
        page_title="California Housing",
        page_icon="🏠",
        layout="wide",
    )

    st.title("California Housing Price Prediction")
    st.caption(
        "Predicted 1990 census block-group median house values. "
        "Target units are $100,000."
    )

    df = get_data()
    results = get_results()

    predict_tab, models_tab, data_tab, reports_tab = st.tabs(
        ["Predict", "Model comparison", "Data overview", "Reports"]
    )

    with predict_tab:
        render_predict_tab(df, results)

    with models_tab:
        render_models_tab(results)

    with data_tab:
        render_data_tab(df)

    with reports_tab:
        render_reports_tab()


if __name__ == "__main__":
    main()
