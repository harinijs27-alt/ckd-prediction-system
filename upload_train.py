"""Upload kidney_disease.csv and train the model from the browser."""
import pandas as pd
import streamlit as st

from common import DATA_PATH, MODEL_PATH, clean_dataframe
import train_model

st.set_page_config(page_title="Upload & Train - CKD", page_icon="🩺")
st.title("🩺 Upload Dataset & Train Model")
st.caption("Step 1: upload kidney_disease.csv  |  Step 2: train  |  Step 3: run app.py")

# ---------- Step 1: upload ----------
st.header("Step 1: Upload CSV")
uploaded = st.file_uploader("Choose kidney_disease.csv", type="csv")

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded, skipinitialspace=True)
        cleaned = clean_dataframe(raw)  # raises an error if columns are missing
    except Exception as e:
        st.error(f"Invalid dataset: {e}")
    else:
        DATA_PATH.write_bytes(uploaded.getvalue())
        st.success(f"Saved as {DATA_PATH.name}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw rows", raw.shape[0])
        c2.metric("Rows after cleaning", cleaned.shape[0])
        c3.metric("Columns", raw.shape[1])
        st.write("Class counts:")
        st.write(cleaned["class"].map({"ckd": "CKD", "notckd": "NOT CKD"}).value_counts())
        st.dataframe(cleaned.head(10), use_container_width=True)
elif DATA_PATH.exists():
    st.info(f"`{DATA_PATH.name}` already exists in the project. You can train with it, "
            "or upload a new file to replace it.")
else:
    st.warning("No dataset yet. Upload the CSV above.")

# ---------- Step 2: train ----------
st.header("Step 2: Train Model")
if st.button("Train Random Forest", type="primary", disabled=not DATA_PATH.exists()):
    with st.spinner("Training... please wait"):
        try:
            m = train_model.main()
        except Exception as e:
            st.error(f"Training failed: {e}")
        else:
            st.success("Training complete. Model saved in the artifacts folder.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']:.4f}")
            c2.metric("Precision", f"{m['precision']:.4f}")
            c3.metric("Recall", f"{m['recall']:.4f}")
            c4.metric("F1-score", f"{m['f1']:.4f}")
            st.write("Confusion matrix [rows = actual, cols = predicted] (NOT CKD, CKD):")
            st.write(pd.DataFrame(m["confusion_matrix"],
                                  index=["Actual NOT CKD", "Actual CKD"],
                                  columns=["Pred NOT CKD", "Pred CKD"]))
            st.code(m["classification_report"])

if MODEL_PATH.exists():
    st.header("Step 3: Run the main app")
    st.code("streamlit run app.py --server.port 8501 --server.address 0.0.0.0")
