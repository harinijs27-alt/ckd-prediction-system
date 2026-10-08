"""Shared configuration, data cleaning and pipeline definition."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "kidney_disease.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ckd_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"

RANDOM_STATE = 42
TARGET = "class"

# Feature order (same as the dataset)
FEATURES = [
    "age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu",
    "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc", "htn", "dm", "cad",
    "appet", "pe", "ane",
]

# name: (label, min, max, default, step, display format)
NUMERIC_SPEC = {
    "age": ("Age (years)", 1.0, 100.0, 50.0, 1.0, "%.0f"),
    "bp": ("Blood Pressure (mm/Hg)", 40.0, 200.0, 80.0, 1.0, "%.0f"),
    "sg": ("Specific Gravity", 1.000, 1.030, 1.020, 0.005, "%.3f"),
    "al": ("Albumin (0-5)", 0.0, 5.0, 0.0, 1.0, "%.0f"),
    "su": ("Sugar (0-5)", 0.0, 5.0, 0.0, 1.0, "%.0f"),
    "bgr": ("Blood Glucose Random (mgs/dl)", 20.0, 600.0, 120.0, 1.0, "%.0f"),
    "bu": ("Blood Urea (mgs/dl)", 1.0, 400.0, 40.0, 1.0, "%.1f"),
    "sc": ("Serum Creatinine (mgs/dl)", 0.1, 80.0, 1.2, 0.1, "%.1f"),
    "sod": ("Sodium (mEq/L)", 4.0, 170.0, 138.0, 1.0, "%.0f"),
    "pot": ("Potassium (mEq/L)", 2.0, 50.0, 4.5, 0.1, "%.1f"),
    "hemo": ("Hemoglobin (gms)", 3.0, 20.0, 14.0, 0.1, "%.1f"),
    "pcv": ("Packed Cell Volume", 9.0, 60.0, 42.0, 1.0, "%.0f"),
    "wbcc": ("White Blood Cell Count (cells/cumm)", 2000.0, 30000.0, 8000.0, 100.0, "%.0f"),
    "rbcc": ("Red Blood Cell Count (millions/cmm)", 2.0, 8.0, 4.8, 0.1, "%.1f"),
}

# name: (label, allowed categories)
CAT_SPEC = {
    "rbc": ("Red Blood Cells", ["normal", "abnormal"]),
    "pc": ("Pus Cell", ["normal", "abnormal"]),
    "pcc": ("Pus Cell Clumps", ["notpresent", "present"]),
    "ba": ("Bacteria", ["notpresent", "present"]),
    "htn": ("Hypertension", ["no", "yes"]),
    "dm": ("Diabetes Mellitus", ["no", "yes"]),
    "cad": ("Coronary Artery Disease", ["no", "yes"]),
    "appet": ("Appetite", ["good", "poor"]),
    "pe": ("Pedal Edema", ["no", "yes"]),
    "ane": ("Anemia", ["no", "yes"]),
}

NUM_FEATURES = list(NUMERIC_SPEC.keys())
CAT_FEATURES = list(CAT_SPEC.keys())
FEATURE_LABELS = {**{k: v[0] for k, v in NUMERIC_SPEC.items()},
                  **{k: v[0] for k, v in CAT_SPEC.items()}}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw UCI CKD dataframe (handles '?', tabs, spaces, case, etc.)."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Some Kaggle versions call the target 'classification' and add an 'id' column
    df = df.rename(columns={"classification": TARGET})
    df = df.drop(columns=["id"], errors="ignore")

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")
    df = df[FEATURES + [TARGET]]

    # Strip whitespace/tabs, lowercase, turn placeholders into real NaN
    for col in df.columns:
        s = df[col].astype(str).str.strip().str.lower()
        df[col] = s.replace({"?": np.nan, "": np.nan, "nan": np.nan, "none": np.nan})

    # Numeric columns -> numbers (bad values become NaN)
    for col in NUM_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Categorical columns -> only allowed values, everything else becomes NaN
    for col in CAT_FEATURES:
        allowed = CAT_SPEC[col][1]
        df[col] = df[col].where(df[col].isin(allowed), np.nan)

    # Target: keep only valid labels
    df[TARGET] = df[TARGET].where(df[TARGET].isin(["ckd", "notckd"]), np.nan)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def build_pipeline() -> Pipeline:
    """Preprocessing + Random Forest in ONE pipeline (used for training and prediction)."""
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[CAT_SPEC[c][1] for c in CAT_FEATURES],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUM_FEATURES),
        ("cat", categorical_pipe, CAT_FEATURES),
    ])
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])
