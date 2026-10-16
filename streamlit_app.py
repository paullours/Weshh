"""Optional local dashboard. Run: python -m streamlit run streamlit_app.py"""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import math

import streamlit as st

from data_utils import display_dataset_summary, load_dataset
from data_preparation import prepare_features, split_and_clean_data
from model_selection import (
    DecisionTreeClassifier,
    KNeighborsClassifier,
    LogisticRegression,
)
from model_training import train_and_evaluate


DATA_FOLDER = Path(__file__).resolve().parent / "data"
DATASET_LABELS = {"Iris": "iris", "Wine": "wine", "Breast Cancer": "breast_cancer"}
MODEL_NAMES = ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors")
DESCRIPTIONS = {
    "Iris": "Explore flower measurements across three iris classes.",
    "Wine": "Explore chemical measurements across three wine classes.",
    "Breast Cancer": "Explore diagnostic measurements across two target classes.",
}


def create_model(name):
    """UI adapter: mirror model_choice settings without its terminal input loop."""
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42)
    if name == "K-Nearest Neighbors":
        return KNeighborsClassifier()
    raise ValueError("Choose one of the three available classifiers.")


def clear_results():
    st.session_state.pop("result", None)
    st.session_state.pop("training_error", None)


def clear_all_results():
    """Dataset changes invalidate both independent experiment types."""
    clear_results()
    st.session_state.pop("comparison", None)


def run_experiment(dataframe, dataset_name, model_name):
    """Delegate all ML operations to the university project's backend."""
    model = create_model(model_name)
    features, target = prepare_features(dataframe)
    output = StringIO()
    with redirect_stdout(output):
        X_train, X_test, y_train, y_test = split_and_clean_data(features, target, model)
        accuracy = train_and_evaluate(
            model, X_train, X_test, y_train, y_test, dataset_name
        )
    if accuracy is None:
        raise ValueError(output.getvalue().strip() or "The model could not be trained.")
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("The backend returned an invalid accuracy. Please try again.")
    return {
        "dataset": dataset_name,
        "model": model_name,
        "accuracy": float(accuracy),
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "output": output.getvalue().strip(),
    }


def run_comparison(dataframe, dataset_name):
    """Run the same backend workflow for each model, retaining partial failures."""
    results = []
    errors = {}
    for model_name in MODEL_NAMES:
        try:
            results.append(run_experiment(dataframe, dataset_name, model_name))
        except Exception as error:
            errors[model_name] = str(error) or type(error).__name__
    return {"dataset": dataset_name, "results": results, "errors": errors}


def comparison_winners(results):
    """Compare actual scores before percentage formatting; preserve every tie."""
    if not results:
        return []
    highest = max(result["accuracy"] for result in results)
    return [result["model"] for result in results if result["accuracy"] == highest]


def apply_style():
    # Scope styles to named containers, semantic roles and component test IDs.
    # Native controls retain their labels, focus states and keyboard behavior.
    st.html("""
    <style>
    :root { color-scheme: light; }
    .stApp { background: #FAFAF8; color: #1C1C1C; font-family: Arial, sans-serif; }
    [data-testid="stHeader"] { background: #FAFAF8; }
    [data-testid="stSidebar"] { background: #F0EFE7; border-right: 1px solid #E6E5E0; }
    [data-testid="stSidebar"] > div { padding-top: 2rem; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px; padding: 2.5rem 3rem 2rem;
    }
    h1, h2, h3, [data-testid="stMetricValue"] { color: #1C1C1C; letter-spacing: -.035em; }
    h1 { font-size: clamp(1.65rem, 2.5vw, 2.1rem) !important; font-weight: 600 !important; }
    h2 { font-size: 1.25rem !important; font-weight: 600 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
    [data-testid="stCaptionContainer"] { color: #73736E; }
    [data-testid="stWidgetLabel"] { color: #555550; }
    [data-testid="stMetricLabel"] { color: #73736E; font-size: .8rem; }
    [data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E6E5E0; border-radius: 8px;
        padding: 1rem 1.2rem; box-shadow: 0 2px 4px #1c1c1c03;
    }
    [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 550; }
    .st-key-feature_card, .st-key-analysis_panel, .st-key-performance_panel,
    .st-key-preview_card, .st-key-details_card, .st-key-comparison_card {
        background: #FFFFFF; border-radius: 10px;
        border-color: #E6E5E0 !important; padding: 1.3rem;
        box-shadow: 0 2px 8px #1c1c1c03;
    }
    [data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: .65rem .8rem; border-radius: 6px; margin: 0;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #FFE4D5; color: #A63200;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #FFFFFF; color: #1C1C1C; border-color: #E6E5E0; border-radius: 6px;
    }
    [data-testid="stButton"] button[kind="primary"] {
        background: #FF5A14; border-color: #FF5A14; color: white;
        border-radius: 6px; min-height: 2.8rem; font-weight: 600;
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        background: #E64908; border-color: #E64908; color: white;
    }
    [data-testid="stTabs"] [aria-selected="true"] { color: #B43800; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #FF5A14; }
    .lab-brand { font-size: 1.9rem; font-weight: 750; color: #1C1C1C;
        letter-spacing: -.07em; margin-bottom: .2rem; }
    .lab-brand span { color: #FF5A14; }
    .lab-note { border: 1px dashed #E6E5E0; background: #FAFAF8;
        color: #64645F; padding: 1.5rem; border-radius: 6px; line-height: 1.8; }
    .lab-kicker { color: #858585; font-size: .7rem; font-weight: 600;
        letter-spacing: .1em; text-transform: uppercase; }
    hr { border-color: #E6E5E0 !important; }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 4.5rem 1rem 1.5rem; }
        .st-key-feature_card, .st-key-analysis_panel, .st-key-performance_panel,
        .st-key-comparison_card {
            padding: .8rem;
        }
        [data-testid="stMetric"] { padding: .8rem; }
    }
    </style>
    """)


def draw_chart(data, encoding, height=210, accuracy_labels=False):
    """Use Streamlit's built-in Vega-Lite renderer; no additional chart library."""
    spec = {
        "mark": {"type": "bar", "color": "#FF5A14", "cornerRadiusEnd": 2},
        "encoding": encoding,
        "height": height,
        "background": "#FFFFFF",
        "config": {
            "view": {"stroke": None},
            "axis": {
                "domain": False, "ticks": False, "labelColor": "#858585",
                "titleColor": "#73736E", "gridColor": "#F0EFEB",
                "labelPadding": 10, "titlePadding": 14,
                "labelFontSize": 11, "titleFontSize": 11, "titleFontWeight": "normal",
            },
        },
    }
    if accuracy_labels:
        # Align labels at the right of the 0–100% scale, including zero scores.
        spec["layer"] = [
            {"mark": spec.pop("mark")},
            {
                "mark": {"type": "text", "align": "right", "dx": -8,
                         "fontSize": 12, "fontWeight": 600},
                "encoding": {
                    "x": {"datum": 1, "type": "quantitative", "scale": {"domain": [0, 1]}},
                    "color": {"value": "#1C1C1C"},
                    "text": {"field": "accuracy", "type": "quantitative", "format": ".2%"},
                },
            },
        ]
    st.vega_lite_chart(data, spec, width="stretch", theme=None)


def show_feature_distribution(dataframe, dataset_name):
    with st.container(border=True, key="feature_card"):
        title, control = st.columns([2, 1], vertical_alignment="center")
        with title:
            st.subheader("Feature distribution")
            st.caption("A closer look at the measurements behind your model.")
        numeric_features = dataframe.drop(columns="target").select_dtypes(include="number").columns.tolist()
        if not numeric_features:
            st.info("No numeric input features are available to visualize.")
            return
        with control:
            # Dataset-specific keys prevent an obsolete feature selection on dataset changes.
            feature = st.selectbox("Numeric feature", numeric_features,
                                   key=f"feature_{dataset_name}")
        values = dataframe[feature].dropna()
        values = values[values.map(math.isfinite)]
        if values.empty:
            st.info("This feature has no finite observations to display. Choose another feature.")
            return
        draw_chart(values.to_frame("value"), {
            "x": {"field": "value", "type": "quantitative", "bin": {"maxbins": 24},
                  "title": feature.replace("_", " "), "axis": {"grid": False}},
            "y": {"aggregate": "count", "type": "quantitative", "title": "Observations",
                  "axis": {"tickMinStep": 1}},
            "tooltip": [{"aggregate": "count", "type": "quantitative", "title": "Observations"}],
        }, height=310)
        st.caption(f"{len(values):,} observed values · {len(dataframe) - len(values):,} missing or "
                   "nonfinite values excluded · original data before preprocessing")


def show_metrics(dataframe):
    metrics = st.columns(4)
    metrics[0].metric("Observations", f"{len(dataframe):,}")
    metrics[1].metric("Input features", len(dataframe.columns) - 1)
    metrics[2].metric("Target classes", dataframe["target"].nunique())
    metrics[3].metric("Missing values", int(dataframe.isna().sum().sum()))


def show_analysis(dataframe):
    st.subheader("Dataset Analysis")
    st.caption("Class balance and data completeness")
    distribution, missing = st.tabs(["Target classes", "Missing values"])
    with distribution:
        counts = dataframe["target"].value_counts().sort_index()
        chart_data = counts.rename_axis("class").reset_index(name="count")
        chart_data["class"] = chart_data["class"].map(lambda value: f"Class {value}")
        draw_chart(chart_data, {
            "x": {"field": "class", "type": "nominal", "title": None,
                  "axis": {"labelAngle": 0, "grid": False}},
            "y": {"field": "count", "type": "quantitative", "title": "Observations",
                  "axis": {"tickMinStep": 1}},
            "tooltip": [{"field": "class", "title": "Target class"},
                        {"field": "count", "type": "quantitative", "title": "Observations"}],
        })
        st.caption("Original numeric target labels. The train/test split preserves class proportions.")
    with missing:
        missing_counts = dataframe.isna().sum()
        if missing_counts.any():
            chart_counts = missing_counts[missing_counts > 0].sort_values(ascending=False).head(8)
            draw_chart(chart_counts.rename_axis("column").reset_index(name="count"), {
                "y": {"field": "column", "type": "nominal", "sort": "-x", "title": None,
                      "axis": {"grid": False, "labelLimit": 130}},
                "x": {"field": "count", "type": "quantitative", "title": "Missing cells",
                      "axis": {"tickMinStep": 1}},
                "tooltip": [{"field": "column", "title": "Column"},
                            {"field": "count", "type": "quantitative", "title": "Missing cells"}],
            })
            st.caption("Up to 8 most affected columns. Complete counts are in Dataset Explorer. "
                       "Missing features are filled using training-set means during preprocessing.")
        else:
            st.success("This dataset has no missing values.")


def show_explorer(dataframe, label):
    show_metrics(dataframe)
    with st.container(border=True, key="preview_card"):
        st.subheader("Explore the data")
        st.caption("Original values · target is the label predicted by the classifier")
        st.dataframe(dataframe, hide_index=True, width="stretch", height=400)
        with st.expander("View exact missing-value counts"):
            st.dataframe(dataframe.isna().sum().rename("Missing cells"), width="stretch")
    with st.expander("View original backend summary"):
        output = StringIO()
        with redirect_stdout(output):
            display_dataset_summary(DATASET_LABELS[label], dataframe)
        st.code(output.getvalue().strip(), language="text")


def show_results(model_name):
    st.subheader("Model Performance")
    st.caption("Selected classifier")
    st.write(f"**{model_name}**")
    error = st.session_state.get("training_error")
    if error:
        st.error("Training could not be completed. Open Technical details below for the error.")
    result = st.session_state.get("result")
    if result:
        st.success(f"Training complete · {result['dataset']} · {result['model']}")
        st.metric("Test accuracy", f"{result['accuracy']:.2%}")
        training, test = st.columns(2)
        training.metric("Training observations", result["training_rows"])
        test.metric("Test observations", result["test_rows"])
        st.caption("Accuracy is measured on the held-out test set. This is one fixed split, "
                   "not a cross-validation estimate.")
    elif not error:
        st.html('<div class="lab-note"><strong>Your next result starts here.</strong><br>'
                'Select Train Model in the sidebar to evaluate this classifier.<br>'
                'Accuracy and split sizes will appear after training.</div>')


def show_technical_details(model_name):
    """Render outside the Overview columns so expanded details use the full width."""
    with st.expander("Technical details · preprocessing and training output"):
        error = st.session_state.get("training_error")
        if error:
            st.error(error)
        result = st.session_state.get("result")
        if result:
            st.write("**Training output**")
            st.code(result["output"], language="text")
        st.write("**How this experiment works**")
        st.markdown(
            "1. Separate the input features from `target`.\n"
            "2. Make a stratified 80/20 train/test split with random seed 42.\n"
            "3. Fill missing feature values with means learned from training data only.\n"
            "4. Standardize features for Logistic Regression and KNN using training data only. "
            "Decision Tree uses unscaled features.\n"
            "5. Train on the training set and evaluate accuracy on the test set."
        )
        st.write("Selected classifier settings")
        st.code(str(create_model(model_name)), language="text")
        st.caption("All classifier settings match the original terminal application.")


def show_comparison(dataframe, label):
    with st.container(border=True, key="comparison_card"):
        st.subheader(f"Model Comparison / {label}")
        st.caption("Three classifiers. One dataset. The same reproducible 80/20 stratified split "
                   "with random seed 42. Training runs only when you select Compare All Models.")
        if st.button("Compare All Models", type="primary", key="compare_models"):
            with st.spinner(f"Evaluating all three classifiers on {label}…"):
                st.session_state["comparison"] = run_comparison(dataframe, label)

        comparison = st.session_state.get("comparison")
        if not comparison:
            st.info("Ready to compare. Each classifier will use its original settings and preprocessing.")
            return

        results = comparison["results"]
        errors = comparison["errors"]
        for model_name, error in errors.items():
            st.error(f"{model_name} failed: {error}")

        winners = comparison_winners(results)
        if results:
            score = max(result["accuracy"] for result in results)
            names = ", ".join(winners)
            if errors:
                st.warning(f"Incomplete comparison · Highest among successful runs: {names} ({score:.2%}). "
                           "A full ranking is unavailable because some models failed.")
            elif len(winners) > 1:
                st.success(f"Tied on this test split: {names} · {score:.2%}")
            else:
                st.success(f"Highest accuracy on this test split: {names} · {score:.2%}")
            st.caption("The highlighted models lead on this fixed test split. This does not establish "
                       "universal model superiority; no cross-validation or tuning is performed.")

            chart_data = [{"model": result["model"], "accuracy": result["accuracy"],
                           "top": result["model"] in winners} for result in results]
            draw_chart(chart_data, {
                "y": {"field": "model", "type": "nominal", "sort": list(MODEL_NAMES),
                      "title": None, "axis": {"grid": False, "labelLimit": 180}},
                "x": {"field": "accuracy", "type": "quantitative", "title": "Test accuracy",
                      "scale": {"domain": [0, 1]},
                      "axis": {"format": ".0%", "values": [0, .25, .5, .75, 1]}},
                "color": {"condition": {"test": "datum.top", "value": "#FF5A14"},
                          "value": "#CDCBC2"},
                "tooltip": [{"field": "model", "title": "Model"},
                            {"field": "accuracy", "type": "quantitative", "format": ".2%",
                             "title": "Test accuracy"}],
            }, height=220, accuracy_labels=True)
        else:
            st.warning("No model completed successfully. Resolve the errors above and try again.")

        by_model = {result["model"]: result for result in results}
        rows = []
        for model_name in MODEL_NAMES:
            result = by_model.get(model_name)
            rows.append({
                "Model": model_name,
                "Training rows": result["training_rows"] if result else None,
                "Test rows": result["test_rows"] if result else None,
                "Test accuracy": f"{result['accuracy']:.2%}" if result else "Unavailable — failed",
            })
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"Training rows": st.column_config.NumberColumn(format="%d"),
                                    "Test rows": st.column_config.NumberColumn(format="%d")})
        with st.expander("Comparison methodology and training output"):
            st.write("Each model uses the existing backend on the same unchanged dataset. "
                     "The stratified split is repeated with seed 42, so the same observations "
                     "are used for training and testing. Missing features are imputed using "
                     "training-set means. Logistic Regression and KNN use training-fitted "
                     "standardization; Decision Tree uses unscaled features. "
                     "Accuracy is evaluated only on held-out test observations.")
            for result in results:
                st.write(f"**{result['model']}**")
                st.code(result["output"], language="text")


def main():
    st.set_page_config(page_title="Weshh — Machine Learning Lab", page_icon="🧪",
                       layout="wide", initial_sidebar_state="expanded")
    apply_style()
    with st.sidebar:
        st.html('<div class="lab-brand">weshh<span>.</span></div>')
        st.caption("Machine Learning Lab")
        st.divider()
        section = st.radio("Workspace", ["Overview", "Dataset Explorer", "Model Results", "Model Comparison"],
                           label_visibility="collapsed")
        st.divider()
        st.html('<div class="lab-kicker">Experiment setup</div>')
        label = st.selectbox("Dataset", list(DATASET_LABELS), key="dataset",
                             on_change=clear_all_results)
        model_name = st.selectbox("Classifier", MODEL_NAMES, key="classifier",
                                  on_change=clear_results)
        st.caption("80/20 stratified split · random seed 42")
        train_clicked = st.button("Train Model", type="primary", width="stretch", key="train_model")
        st.divider()
        st.caption("A project by Paul and Lorenzo")
        st.caption("Local workspace · Reproducible experiments")

    st.title(f"Weshh / {section}")
    st.caption(DESCRIPTIONS[label] + f" Selected dataset: {label}.")

    dataset_name = DATASET_LABELS[label]
    output = StringIO()
    try:
        with redirect_stdout(output):
            dataframe = load_dataset(dataset_name, data_folder=DATA_FOLDER)
        if dataframe is None:
            raise ValueError(output.getvalue().strip() or "Dataset could not be loaded.")
        # Invalidate previous results if CSV contents change while the app is open.
        signature = hashlib.sha256(dataframe.to_csv(index=False).encode()).hexdigest()
        dataset_context = (dataset_name, signature)
        if st.session_state.get("dataset_context") != dataset_context:
            clear_all_results()
            st.session_state["dataset_context"] = dataset_context
        context = (dataset_name, model_name, signature)
        if st.session_state.get("experiment_context") != context:
            clear_results()
            st.session_state["experiment_context"] = context
    except Exception as error:
        clear_all_results()
        st.error("Dataset unavailable. Check that the original CSV is present and readable, "
                 "then reload the page or select another dataset.")
        st.text(str(error))
        st.stop()

    if train_clicked:
        clear_results()
        try:
            with st.spinner(f"Training {model_name} on {label}…"):
                st.session_state["result"] = run_experiment(dataframe, label, model_name)
        except Exception as error:
            # Boundary handler: failures remain visible without breaking the workspace.
            st.session_state["training_error"] = str(error) or type(error).__name__

    if section == "Overview":
        show_feature_distribution(dataframe, dataset_name)
        show_metrics(dataframe)
        analysis, performance = st.columns(2, gap="medium")
        with analysis, st.container(border=True, key="analysis_panel"):
            show_analysis(dataframe)
        with performance, st.container(border=True, key="performance_panel"):
            show_results(model_name)
        show_technical_details(model_name)
    elif section == "Model Results":
        with st.container(border=True, key="details_card"):
            show_results(model_name)
        show_technical_details(model_name)
    elif section == "Model Comparison":
        show_comparison(dataframe, label)
        if train_clicked:
            if st.session_state.get("training_error"):
                st.error("Individual training failed. Open Model Results for details.")
            else:
                st.success("Individual training complete. Open Model Results to inspect the accuracy.")
    else:
        show_explorer(dataframe, label)
    if section == "Dataset Explorer" and train_clicked:
        if st.session_state.get("training_error"):
            st.error("Training failed. Open Model Results for details.")
        else:
            st.success("Training complete. Open Model Results to inspect the accuracy.")
    st.caption("Weshh — Machine Learning Lab · Local data, real results")


if __name__ == "__main__":
    main()
