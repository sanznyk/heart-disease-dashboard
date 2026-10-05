import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

st.set_page_config(
    page_title="Heart Disease Classification",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f4f7fb;
        color: #14243b;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }

    [data-testid="stSidebar"] {
        background: #14243b;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #eef4ff;
    }

    .hero-banner {
        background: linear-gradient(120deg, #14243b, #176b87);
        padding: 32px;
        border-radius: 20px;
        margin-bottom: 24px;
        box-shadow: 0 12px 30px rgba(20, 36, 59, 0.12);
    }

    .hero-banner .eyebrow {
        color: #a9e6e0;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero-banner h1 {
        color: #ffffff;
        font-size: clamp(26px, 4vw, 38px);
        margin: 0;
        padding: 0;
        line-height: 1.25;
    }

    .hero-banner p {
        color: #dce8f4;
        margin: 12px 0 0;
        font-size: 16px;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 20px;
    }

    .hero-tag {
        border: 1px solid rgba(255, 255, 255, 0.25);
        background: rgba(255, 255, 255, 0.08);
        border-radius: 999px;
        padding: 5px 12px;
        color: #ffffff;
        font-size: 12px;
    }

    [data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #e0e7ef;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 5px 20px rgba(20, 36, 59, 0.04);
    }

    [data-testid="stFormSubmitButton"] button {
        background: #176b87;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        padding: 10px 22px;
        font-weight: 600;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background: #12566d;
        color: #ffffff;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e0e7ef;
        border-radius: 14px;
        padding: 18px;
    }

    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #20344e;
        border: 1px solid #334b68;
    }

    [data-testid="stExpander"] {
        background: #ffffff;
        border-radius: 12px;
    }

    h2, h3, h4 {
        color: #14243b;
    }

    .footer-note {
        margin-top: 32px;
        padding-top: 16px;
        border-top: 1px solid #dce4ed;
        color: #52657a;
        font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    metadata = json.loads(
        (MODEL_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    if sklearn.__version__ != metadata["sklearn_version"]:
        raise ValueError(
            f"Model requires scikit-learn "
            f"{metadata['sklearn_version']}; "
            f"installed version is {sklearn.__version__}."
        )

    pipeline = joblib.load(MODEL_DIR / "heart_pipeline.joblib")

    return pipeline, metadata


NUMERIC_LABELS = {
    "age": "Age",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "thalach": "Maximum heart rate",
    "oldpeak": "ST depression",
    "ca": "Major vessel count",
}

CATEGORY_LABELS = {
    "sex": {
        0: "Sex: female",
        1: "Sex: male",
    },
    "cp": {
        1: "Chest pain: typical angina",
        2: "Chest pain: atypical angina",
        3: "Chest pain: non-anginal",
        4: "Chest pain: asymptomatic",
    },
    "fbs": {
        0: "Blood sugar threshold: no",
        1: "Blood sugar threshold: yes",
    },
    "restecg": {
        0: "ECG: normal",
        1: "ECG: ST-T abnormality",
        2: "ECG: hypertrophy category",
    },
    "exang": {
        0: "Exercise angina: no",
        1: "Exercise angina: yes",
    },
    "slope": {
        1: "ST slope: upsloping",
        2: "ST slope: flat",
        3: "ST slope: downsloping",
    },
    "thal": {
        3: "Thal test: normal",
        6: "Thal test: fixed defect",
        7: "Thal test: reversible defect",
    },
}


def readable_feature_name(feature_name):
    if feature_name.startswith("numeric__"):
        column = feature_name.removeprefix("numeric__")
        return NUMERIC_LABELS.get(column, column)

    if feature_name.startswith("categorical__"):
        encoded_name = feature_name.removeprefix("categorical__")
        column, value = encoded_name.rsplit("_", 1)

        try:
            category = float(value)
            return CATEGORY_LABELS.get(column, {}).get(
                category,
                encoded_name,
            )
        except ValueError:
            return encoded_name

    return feature_name


def explain_record(pipeline, record):
    preprocessor = pipeline.named_steps["preprocessing"]
    classifier = pipeline.named_steps["classifier"]

    prepared = preprocessor.transform(record)

    if hasattr(prepared, "toarray"):
        prepared = prepared.toarray()

    values = np.asarray(prepared)[0]
    coefficients = classifier.coef_[0]
    feature_names = preprocessor.get_feature_names_out()

    explanation = pd.DataFrame({
        "feature": feature_names,
        "display_name": [
            readable_feature_name(name)
            for name in feature_names
        ],
        "prepared_value": values,
        "coefficient": coefficients,
        "contribution": values * coefficients,
    })

    explanation["absolute_contribution"] = (
        explanation["contribution"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_contribution",
        ascending=False,
    )

    intercept = float(classifier.intercept_[0])
    log_odds = float(intercept + explanation["contribution"].sum())

    return explanation, intercept, log_odds


try:
    pipeline, metadata = load_artifacts()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


with st.sidebar:
    st.markdown("## 📊 Model overview")
    st.caption("Logistic Regression · UCI Heart Disease")

    st.divider()

    metrics = metadata["test_metrics"]

    st.metric(
        "Held-out test accuracy",
        f"{metrics['accuracy']:.1%}",
    )

    st.metric(
        "Held-out test recall",
        f"{metrics['recall']:.1%}",
    )

    st.metric(
        "Held-out test ROC-AUC",
        f"{metrics['roc_auc']:.3f}",
    )

    st.divider()

    st.caption(
        f"Evaluated on {metadata['test_records']} held-out records. "
        "Results may vary across populations and data splits."
    )

    st.markdown("### How to explore")
    st.markdown(
        "1. Enter a dataset-style record.\n"
        "2. Select **Classify and explain**.\n"
        "3. Inspect the model output and feature contributions."
    )


st.markdown(
    """
    <div class="hero-banner">
        <div class="eyebrow">EXPLAINABLE MACHINE LEARNING</div>
        <h1>Heart Disease Classification</h1>
        <p>Explore a prediction. Understand the features behind it.</p>
        <div class="hero-tags">
            <span class="hero-tag">13 input features</span>
            <span class="hero-tag">Logistic Regression</span>
            <span class="hero-tag">Individual explanations</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Educational demo: this model predicts a historical dataset label. "
    "It is not a diagnosis tool or a validated estimate of future disease risk."
)

st.subheader("Build an example record")
st.caption(
    "Defaults reproduce an example checked during development. "
    "Some fields require clinical test results. "
    "Unknown ca or thal values use saved training-data imputation."
)

with st.form("record_form"):
    column1, column2, column3 = st.columns(3)

    with column1:
        st.markdown("#### Personal & baseline")

        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=59,
        )

        sex = st.selectbox(
            "Sex — dataset coding",
            options=[0, 1],
            index=1,
            format_func=lambda value: {
                0: "Female",
                1: "Male",
            }[value],
        )

        cp = st.selectbox(
            "Chest-pain category",
            options=[1, 2, 3, 4],
            index=3,
            format_func=lambda value: {
                1: "Typical angina",
                2: "Atypical angina",
                3: "Non-anginal pain",
                4: "Asymptomatic",
            }[value],
        )

        trestbps = st.number_input(
            "Resting blood pressure (mmHg)",
            min_value=50,
            max_value=250,
            value=138,
        )

        chol = st.number_input(
            "Serum cholesterol (mg/dL)",
            min_value=50,
            max_value=650,
            value=271,
        )

    with column2:
        st.markdown("#### Exercise & ECG")

        fbs = st.selectbox(
            "Fasting blood sugar above 120 mg/dL",
            options=[0, 1],
            format_func=lambda value: "Yes" if value else "No",
        )

        restecg = st.selectbox(
            "Resting ECG category",
            options=[0, 1, 2],
            index=2,
            format_func=lambda value: {
                0: "Normal",
                1: "ST-T wave abnormality",
                2: "Left ventricular hypertrophy category",
            }[value],
        )

        thalach = st.number_input(
            "Maximum heart rate achieved",
            min_value=50,
            max_value=250,
            value=182,
        )

        exang = st.selectbox(
            "Exercise-induced angina",
            options=[0, 1],
            format_func=lambda value: "Yes" if value else "No",
        )

    with column3:
        st.markdown("#### Test findings")

        oldpeak = st.number_input(
            "Exercise-induced ST depression",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
        )

        slope = st.selectbox(
            "Peak exercise ST-segment slope",
            options=[1, 2, 3],
            format_func=lambda value: {
                1: "Upsloping",
                2: "Flat",
                3: "Downsloping",
            }[value],
        )

        ca = st.selectbox(
            "Major vessels visualized by fluoroscopy",
            options=[0, 1, 2, 3, "Unknown"],
        )

        thal = st.selectbox(
            "Thal test category",
            options=[3, 6, 7, "Unknown"],
            format_func=lambda value: {
                3: "Normal",
                6: "Fixed defect",
                7: "Reversible defect",
                "Unknown": "Unknown",
            }[value],
        )

    st.divider()
    submitted = st.form_submit_button("Classify and explain")


if submitted:
    record = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": np.nan if ca == "Unknown" else float(ca),
        "thal": np.nan if thal == "Unknown" else float(thal),
    }])

    record = record[metadata["feature_columns"]]

    positive_index = list(pipeline.classes_).index(1)

    probability = float(
        pipeline.predict_proba(record)[0, positive_index]
    )

    predicted_label = int(pipeline.predict(record)[0])

    explanation, intercept, log_odds = explain_record(
        pipeline,
        record,
    )

    st.subheader("Prediction overview")

    result_column, score_column, threshold_column = st.columns(3)

    result_column.metric(
        "Predicted dataset label",
        "Present (1)" if predicted_label else "Absent (0)",
    )

    score_column.metric(
        "Model probability for class 1",
        f"{probability:.1%}",
    )

    threshold_column.metric(
        "Classification threshold",
        "0.50",
    )

    st.caption(
        "The probability is a model output, not a calibrated clinical risk."
    )

    if ca == "Unknown" or thal == "Unknown":
        st.warning(
            "This prediction includes imputed values for missing inputs."
        )

    st.subheader("What influenced this prediction?")
    st.write(
        "Positive contributions push the score toward class 1; "
        "negative contributions push it toward class 0. "
        "These describe model associations, not medical causes."
    )

    top = explanation.head(10).sort_values("contribution")

    colors = [
        "#e78768" if value > 0 else "#176b87"
        for value in top["contribution"]
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#f4f7fb")
    ax.set_facecolor("#f4f7fb")

    ax.barh(
        top["display_name"],
        top["contribution"],
        color=colors,
        height=0.65,
    )

    ax.axvline(0, color="#52657a", linewidth=1)
    ax.set_xlabel("Contribution to class-1 log-odds", color="#14243b")
    ax.set_title(
        "Largest feature contributions",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color="#14243b",
        pad=16,
    )

    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)

    ax.spines["bottom"].set_color("#dce4ed")
    ax.grid(axis="x", alpha=0.15, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=10, colors="#14243b")
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Teal: contribution toward class 0 · "
        "Coral: contribution toward class 1"
    )

    with st.expander("Understand the calculation"):
        st.latex(
            r"z = b + \sum_j w_j x_j,\qquad "
            r"p = \frac{1}{1 + e^{-z}}"
        )

        intercept_column, log_odds_column = st.columns(2)

        intercept_column.metric("Intercept", f"{intercept:.3f}")
        log_odds_column.metric("Total log-odds", f"{log_odds:.3f}")

        st.caption(
            "The chart shows the 10 largest contributions. "
            "The total uses all prepared features plus the intercept. "
            "These are coefficient contributions, not SHAP values."
        )

        st.dataframe(
            explanation[
                [
                    "display_name",
                    "feature",
                    "prepared_value",
                    "coefficient",
                    "contribution",
                ]
            ].round(3),
            hide_index=True,
        )

    with st.expander("View the entered record"):
        st.dataframe(record, hide_index=True)

else:
    st.caption(
        "Your prediction and its explanation will appear below "
        "after you submit the form."
    )


st.markdown(
    """
    <div class="footer-note">
        Built with scikit-learn and Streamlit ·
        UCI Heart Disease dataset · Educational portfolio project
    </div>
    """,
    unsafe_allow_html=True,
)