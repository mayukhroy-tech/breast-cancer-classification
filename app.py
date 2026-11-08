
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Classification")
st.write(
    "An interactive application to compare five machine-learning "
    "classifiers on the Breast Cancer Wisconsin dataset."
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}

# Obtain the expected input feature names
dataset_info = load_breast_cancer()
EXPECTED_FEATURES = list(dataset_info.feature_names)

CLASS_NAMES = ["Malignant", "Benign"]


# --------------------------------------------------
# 2. Load saved models
# --------------------------------------------------

@st.cache_resource
def load_saved_models():
    loaded = {}

    for display_name, filename in MODEL_FILES.items():
        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        loaded[display_name] = joblib.load(model_path)

    return loaded


# --------------------------------------------------
# 3. Evaluation function
# --------------------------------------------------

def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    scores = {
        "Accuracy": accuracy_score(y, predictions),
        "AUC": roc_auc_score(y, probabilities),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
        "MCC": matthews_corrcoef(y, predictions)
    }

    return scores, predictions


# --------------------------------------------------
# 4. CSV upload
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Please upload the test CSV generated during Phase 2. "
        "It must contain all 30 input features and the target column."
    )
    st.stop()

try:
    test_data = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Unable to read the CSV file: {error}")
    st.stop()


# --------------------------------------------------
# 5. Validate uploaded data
# --------------------------------------------------

st.subheader("Dataset Preview")
st.dataframe(test_data.head(), use_container_width=True)

if "target" not in test_data.columns:
    st.error("The uploaded CSV must contain a 'target' column.")
    st.stop()

missing_features = [
    feature for feature in EXPECTED_FEATURES
    if feature not in test_data.columns
]

if missing_features:
    st.error(
        "The following required input features are missing: "
        + ", ".join(missing_features)
    )
    st.stop()

if test_data.empty:
    st.error("The uploaded CSV contains no records.")
    st.stop()

X_test = test_data[EXPECTED_FEATURES].copy()
y_test = test_data["target"].copy()

if X_test.isnull().any().any() or y_test.isnull().any():
    st.error("The uploaded dataset contains missing values.")
    st.stop()

if not set(y_test.unique()).issubset({0, 1}):
    st.error("The target column must contain only 0 and 1.")
    st.stop()

if y_test.nunique() != 2:
    st.error("The test dataset must contain both target classes.")
    st.stop()

try:
    X_test = X_test.astype(float)
    y_test = y_test.astype(int)
except ValueError:
    st.error("Features and target must contain numeric values.")
    st.stop()

st.success(
    f"Dataset validated: {len(test_data)} records and "
    f"{len(EXPECTED_FEATURES)} input features."
)


# --------------------------------------------------
# 6. Load all five saved models
# --------------------------------------------------

st.header("2. Select Classification Model")

try:
    models = load_saved_models()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

selected_model_name = st.selectbox(
    "Choose a classifier",
    list(models.keys())
)

selected_model = models[selected_model_name]


# --------------------------------------------------
# 7. Evaluate all five models
# --------------------------------------------------

st.header("3. Model Performance Comparison")

all_results = {}
all_predictions = {}

try:
    for name, model in models.items():
        scores, predictions = calculate_metrics(
            model, X_test, y_test
        )

        all_results[name] = scores
        all_predictions[name] = predictions

except Exception as error:
    st.error(f"Model evaluation failed: {error}")
    st.stop()

comparison = pd.DataFrame.from_dict(
    all_results,
    orient="index"
)

comparison.index.name = "ML Model Name"

st.dataframe(
    comparison.style.format("{:.4f}"),
    use_container_width=True
)

# Identify the highest MCC score
best_model_name = comparison["MCC"].idxmax()

st.success(
    f"Highest MCC on the uploaded test data: {best_model_name}"
)


# --------------------------------------------------
# 8. Display selected model metrics
# --------------------------------------------------

st.header("4. Selected Model Evaluation")

selected_scores = all_results[selected_model_name]

metric_columns = st.columns(3)

metric_items = list(selected_scores.items())

for index, (metric_name, metric_value) in enumerate(metric_items):
    with metric_columns[index % 3]:
        st.metric(
            label=metric_name,
            value=f"{metric_value:.4f}"
        )


# --------------------------------------------------
# 9. Confusion matrix
# --------------------------------------------------

st.subheader("Confusion Matrix")

selected_predictions = all_predictions[selected_model_name]

fig, ax = plt.subplots(figsize=(6, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    selected_predictions,
    display_labels=CLASS_NAMES,
    cmap="Blues",
    colorbar=False,
    ax=ax
)

ax.set_title(f"Confusion Matrix - {selected_model_name}")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# 10. Classification report
# --------------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    selected_predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# --------------------------------------------------
# 11. Download evaluation results
# --------------------------------------------------

st.header("5. Download Results")

csv_results = comparison.to_csv().encode("utf-8")

st.download_button(
    label="Download Model Comparison CSV",
    data=csv_results,
    file_name="model_comparison.csv",
    mime="text/csv"
)

st.caption(
    "Note: This application evaluates uploaded labelled test data. "
    "It loads previously trained models and does not retrain them."
)
