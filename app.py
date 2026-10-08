import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from common import (CAT_FEATURES, CAT_SPEC, DATA_PATH, FEATURE_LABELS,
                    FEATURES, IMPORTANCE_PATH, METRICS_PATH, MODEL_PATH,
                    NUM_FEATURES, NUMERIC_SPEC, TARGET, clean_dataframe)

st.set_page_config(page_title="CKD Prediction System", page_icon="🩺", layout="wide")
sns.set_theme(style="whitegrid")

st.markdown("""
<style>
.main-title {font-size: 2.2rem; font-weight: 700; color: #0b5394; margin-bottom: 0;}
.sub-title {font-size: 1.1rem; color: #555; margin-top: 0; margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)


# ---------- loaders ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)


@st.cache_data
def load_data():
    return clean_dataframe(pd.read_csv(DATA_PATH, skipinitialspace=True))


def artifacts_ready():
    return MODEL_PATH.exists() and METRICS_PATH.exists() and IMPORTANCE_PATH.exists()


# ---------- header ----------
st.markdown('<p class="main-title">🩺 Chronic Kidney Disease Prediction System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Machine Learning Based CKD Risk Classification</p>',
            unsafe_allow_html=True)

# ---------- make sure data + model exist ----------
if not DATA_PATH.exists():
    st.error("`kidney_disease.csv` was not found. Place it in the project's root "
             "folder (next to app.py) and refresh this page.")
    st.stop()

if not artifacts_ready():
    st.warning("The model has not been trained yet.")
    if st.button("Train model now", type="primary"):
        with st.spinner("Training Random Forest..."):
            import train_model
            train_model.main()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.stop()

model = load_model()
metrics = load_metrics()
importance = load_importance()
data = load_data()

page = st.sidebar.radio("Navigation", [
    "Home / Overview", "Patient Prediction", "Model Performance",
    "Dataset Insights", "About the Project"])
st.sidebar.info("Academic demonstration only. Not a medical diagnosis.")


# ---------- pages ----------
def page_home():
    st.header("Overview")
    st.write(
        "Chronic Kidney Disease (CKD) is a gradual loss of kidney function. "
        "This system uses a **Random Forest classifier** trained on the UCI CKD "
        "dataset to classify a patient as **CKD** or **NOT CKD** from 24 clinical "
        "and laboratory features.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(data))
    c2.metric("Features", len(FEATURES))
    c3.metric("Test Accuracy", f"{metrics['accuracy']:.2%}")
    c4.metric("Model", "Random Forest")
    st.subheader("How to use")
    st.markdown("""
1. Open **Patient Prediction** and enter the patient's values.
2. Click **Predict** to see the model's classification.
3. Check **Model Performance** and **Dataset Insights** for details.
""")


def page_prediction():
    st.header("Patient Prediction")
    st.caption("Enter the patient's details and click Predict.")
    with st.form("patient_form"):
        cols = st.columns(3)
        values = {}
        for i, feat in enumerate(FEATURES):
            with cols[i % 3]:
                if feat in NUMERIC_SPEC:
                    label, mn, mx, default, step, fmt = NUMERIC_SPEC[feat]
                    values[feat] = st.number_input(
                        label, min_value=mn, max_value=mx, value=default,
                        step=step, format=fmt)
                else:
                    label, options = CAT_SPEC[feat]
                    values[feat] = st.selectbox(
                        label, options,
                        format_func=lambda x: x.replace("notpresent", "not present").title())
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        input_df = pd.DataFrame([values])[FEATURES]
        pred = int(model.predict(input_df)[0])
        proba = model.predict_proba(input_df)[0]
        ckd_prob = float(proba[list(model.classes_).index(1)])
        if pred == 1:
            st.error("## Prediction: CKD")
        else:
            st.success("## Prediction: NOT CKD")
        st.progress(ckd_prob, text=f"Model's estimated probability of CKD: {ckd_prob:.1%}")
        st.warning(
            "This is a machine-learning model prediction made for academic "
            "demonstration purposes. It is **not a medical diagnosis**. Consult a "
            "qualified healthcare professional for any medical concern.")


def page_performance():
    st.header("Model Performance")
    st.caption(f"Calculated on the held-out test set ({metrics['n_test']} records, "
               f"trained on {metrics['n_train']}; random_state={metrics['random_state']}). "
               "CKD is treated as the positive class.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1-score", f"{metrics['f1']:.4f}")
    st.caption(f"5-fold cross-validation accuracy on training data: "
               f"{metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=["NOT CKD", "CKD"], yticklabels=["NOT CKD", "CKD"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Feature Importance (Top 15)")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.barplot(data=importance.head(15), x="importance", y="label",
                    color="#3b82c4", ax=ax)
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("Classification Report")
    st.code(metrics["classification_report"])


def page_insights():
    st.header("Dataset Insights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Number of records", len(data))
    c2.metric("Number of features", len(FEATURES))
    c3.metric("Missing cells", int(data[FEATURES].isna().sum().sum()))

    plot_df = data.copy()
    plot_df["Diagnosis"] = plot_df[TARGET].map({"ckd": "CKD", "notckd": "NOT CKD"})

    left, right = st.columns(2)
    with left:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=plot_df, x="Diagnosis", order=["CKD", "NOT CKD"],
                      palette=["#d9534f", "#5cb85c"], ax=ax)
        for p in ax.patches:
            ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom")
        st.pyplot(fig)
    with right:
        st.subheader("Missing Values per Feature")
        miss = data[FEATURES].isna().sum().sort_values(ascending=False)
        miss = miss[miss > 0].rename(index=FEATURE_LABELS)
        fig, ax = plt.subplots(figsize=(5, 4))
        if len(miss):
            sns.barplot(x=miss.values, y=miss.index, color="#f0ad4e", ax=ax)
        ax.set_xlabel("Missing count")
        st.pyplot(fig)

    st.subheader("Basic Statistics (numerical features)")
    st.dataframe(data[NUM_FEATURES].describe().T.round(3), use_container_width=True)

    st.subheader("Feature Distribution by Diagnosis")
    feat = st.selectbox("Choose a numerical feature", NUM_FEATURES,
                        format_func=lambda x: FEATURE_LABELS[x])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(data=plot_df, x=feat, hue="Diagnosis", kde=True, bins=25,
                 palette={"CKD": "#d9534f", "NOT CKD": "#5cb85c"}, ax=ax)
    ax.set_xlabel(FEATURE_LABELS[feat])
    st.pyplot(fig)

    st.subheader("Correlation Heatmap (numerical features)")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(data[NUM_FEATURES].corr(), cmap="coolwarm", center=0, ax=ax)
    st.pyplot(fig)

    st.subheader("Cleaned Data Preview")
    st.dataframe(data.head(20), use_container_width=True)


def page_about():
    st.header("About the Project")
    st.markdown("""
**Goal:** classify patients as CKD / NOT CKD from clinical features using machine learning.

**Dataset:** UCI Chronic Kidney Disease dataset (24 features + `class` target).

**Pipeline**
1. Load CSV and clean it (strip whitespace/tabs, lowercase, `?` → missing, safe numeric conversion).
2. Split into 80% train / 20% test (stratified, `random_state=42`).
3. Numerical features: median imputation. Categorical features: most-frequent imputation + ordinal encoding.
4. Random Forest Classifier (300 trees, balanced class weights).
5. Evaluate on the test set: accuracy, precision, recall, F1, confusion matrix.
6. The *same* saved pipeline (preprocessing + model) is used by this web app for predictions.

**Tech stack:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, joblib.

**Disclaimer:** This project is for academic demonstration only and is **not** a medical diagnostic tool.
""")


{"Home / Overview": page_home, "Patient Prediction": page_prediction,
 "Model Performance": page_performance, "Dataset Insights": page_insights,
 "About the Project": page_about}[page]()
