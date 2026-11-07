from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}
TARGET = "target"
st.set_page_config(page_title="AIML Assignment 2 | Classifier Lab",
                   page_icon="🧪", layout="wide")
st.title("🧪 Classifier Lab")
st.caption("BITS Pilani M.Tech (AIML) • Machine Learning Assignment 2")
st.markdown(
    "Compare five trained classifiers on the Wisconsin Diagnostic Breast Cancer "
    "test dataset. The app **loads saved models only**; it does not retrain "
    "models on uploaded test data."
)

with st.sidebar:
    st.header("Experiment controls")
    model_name = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
    uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
    st.caption("CSV must contain all 30 feature columns and a `target` column. "
               "Target: malignant=1, benign=0.")
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button("Download sample test CSV",
                           sample_path.read_bytes(),
                           file_name="test_data.csv",
                           mime="text/csv")

if uploaded is None:
    st.info("Upload `test_data.csv` from the repository (or download the sample in the sidebar) to run evaluation.")
    st.subheader("Included classifiers")
    st.write(", ".join(MODEL_FILES.keys()))
    st.stop()

try:
    data = pd.read_csv(uploaded)
    if TARGET not in data.columns:
        st.error("The uploaded CSV must include the target column named `target`.")
        st.stop()
    y_true = data[TARGET].astype(int)
    X_test = data.drop(columns=[TARGET])
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    if not model_path.exists():
        st.error(f"Saved model not found: {model_path.name}")
        st.stop()
    model = joblib.load(model_path)
    expected = list(getattr(model, "feature_names_in_", []))
    if expected:
        missing = sorted(set(expected) - set(X_test.columns))
        extra = sorted(set(X_test.columns) - set(expected))
        if missing or extra:
            st.error(f"Feature mismatch. Missing: {missing}; unexpected: {extra}")
            st.stop()
        X_test = X_test[expected]

    # Evaluate every saved classifier on the same uploaded test CSV so the
    # comparison across all five models is visible in the app.
    all_results = {}
    comparison_rows = []
    for display_name, filename in MODEL_FILES.items():
        fitted_model = joblib.load(MODEL_DIR / filename)
        y_pred = fitted_model.predict(X_test)
        y_prob = fitted_model.predict_proba(X_test)[:, 1]
        result = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_prob),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred)
        }
        all_results[display_name] = (fitted_model, y_pred, y_prob, result)
        comparison_rows.append({"ML Model Name": display_name, **result})

    st.subheader("Comparison across all five classifiers")
    comparison = pd.DataFrame(comparison_rows).set_index("ML Model Name")
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

    model, predictions, probabilities, metrics = all_results[model_name]
    st.subheader(f"Detailed evaluation: {model_name}")
    cols = st.columns(6)
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, f"{value:.4f}")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        cm = confusion_matrix(y_true, predictions, labels=[0, 1])
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks([0, 1], labels=["Benign (0)", "Malignant (1)"])
        ax.set_yticks([0, 1], labels=["Benign (0)", "Malignant (1)"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for (i, j), value in __import__("numpy").ndenumerate(cm):
            ax.text(j, i, str(value), ha="center", va="center")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("#### Classification report")
        report = classification_report(
            y_true, predictions, labels=[0, 1],
            target_names=["Benign", "Malignant"], zero_division=0,
            output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)
    st.markdown("#### Prediction preview")
    preview = X_test.copy()
    preview["Actual target"] = y_true.values
    preview["Predicted target"] = predictions
    preview["P(malignant)"] = probabilities
    st.dataframe(preview.head(20), use_container_width=True)
    st.download_button("Download predictions CSV",
        preview.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv", mime="text/csv")
except Exception as exc:
    st.exception(exc)
