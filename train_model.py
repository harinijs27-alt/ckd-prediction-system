"""Train and evaluate the Random Forest model on kidney_disease.csv."""
import json

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split

from common import (ARTIFACT_DIR, CAT_FEATURES, DATA_PATH, FEATURE_LABELS,
                    FEATURES, IMPORTANCE_PATH, METRICS_PATH, MODEL_PATH,
                    NUM_FEATURES, RANDOM_STATE, TARGET, build_pipeline,
                    clean_dataframe)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"'{DATA_PATH.name}' not found. Place it in the project root folder."
        )

    raw = pd.read_csv(DATA_PATH, skipinitialspace=True)
    print(f"Raw dataset shape: {raw.shape}")
    df = clean_dataframe(raw)
    print(f"Cleaned dataset shape: {df.shape}")
    print("Missing values per column after cleaning:")
    print(df.isna().sum()[df.isna().sum() > 0])

    X = df[FEATURES]
    y = (df[TARGET] == "ckd").astype(int)  # 1 = CKD, 0 = NOT CKD

    # Split BEFORE fitting so imputers never see test data (no leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cv_scores = cross_val_score(clone(build_pipeline()), X_train, y_train,
                                cv=5, scoring="accuracy")

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["NOT CKD", "CKD"], digits=4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "random_state": RANDOM_STATE,
    }

    importances = pipeline.named_steps["model"].feature_importances_
    names = NUM_FEATURES + CAT_FEATURES  # same order as ColumnTransformer output
    imp_df = pd.DataFrame({
        "feature": names,
        "label": [FEATURE_LABELS[n] for n in names],
        "importance": importances,
    }).sort_values("importance", ascending=False)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    imp_df.to_csv(IMPORTANCE_PATH, index=False)

    print("\n=== Test-set results ===")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k:>10}: {metrics[k]:.4f}")
    print(f"5-fold CV accuracy (train): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("\nConfusion matrix [rows=actual, cols=predicted] (NOT CKD, CKD):")
    print(cm)
    print("\n" + metrics["classification_report"])
    print(f"Saved model to {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
