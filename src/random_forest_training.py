from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

MODEL_DIR = PROJECT_ROOT / "models"

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "random_forest_training"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Output paths
# ---------------------------------------------------------
MODEL_PATH = (
    MODEL_DIR
    / "random_forest.joblib"
)

REPORT_PATH = (
    ARTIFACT_DIR
    / "classification_report.json"
)

CONFUSION_MATRIX_PATH = (
    ARTIFACT_DIR
    / "confusion_matrix.png"
)

FEATURE_IMPORTANCE_PATH = (
    ARTIFACT_DIR
    / "feature_importance.csv"
)


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------
mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

mlflow.set_experiment(
    "customer_churn_pipeline"
)


# ---------------------------------------------------------
# Load processed data
# ---------------------------------------------------------
X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(
    Y_TRAIN_PATH
)["churn"]

y_test = pd.read_csv(
    Y_TEST_PATH
)["churn"]


# ---------------------------------------------------------
# Random Forest hyperparameters
# ---------------------------------------------------------
model_parameters = {
    "n_estimators": 100,
    "max_depth": 5,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
}


# ---------------------------------------------------------
# Start MLflow run
# ---------------------------------------------------------
with mlflow.start_run(
    run_name="random_forest_challenger"
):

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------
    model = RandomForestClassifier(
        **model_parameters
    )

    # -----------------------------------------------------
    # Train model
    # -----------------------------------------------------
    model.fit(
        X_train,
        y_train,
    )

    # -----------------------------------------------------
    # Make predictions
    # -----------------------------------------------------
    y_train_pred = model.predict(
        X_train
    )

    y_test_pred = model.predict(
        X_test
    )

    # -----------------------------------------------------
    # Training metric
    # -----------------------------------------------------
    training_accuracy = accuracy_score(
        y_train,
        y_train_pred,
    )

    # -----------------------------------------------------
    # Testing metrics
    # -----------------------------------------------------
    testing_accuracy = accuracy_score(
        y_test,
        y_test_pred,
    )

    precision = precision_score(
        y_test,
        y_test_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_test_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_test_pred,
        zero_division=0,
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------
    classification_report_data = (
        classification_report(
            y_test,
            y_test_pred,
            output_dict=True,
            zero_division=0,
        )
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            classification_report_data,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------
    cm = confusion_matrix(
        y_test,
        y_test_pred,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "No Churn",
            "Churn",
        ],
    )

    display.plot()

    plt.title(
        "Random Forest Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH
    )

    plt.close()

    # -----------------------------------------------------
    # Feature importance
    # -----------------------------------------------------
    feature_importance_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    )

    feature_importance_df = (
        feature_importance_df.sort_values(
            by="importance",
            ascending=False,
        )
    )

    feature_importance_df.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
    )

    # -----------------------------------------------------
    # Save model using Joblib
    # -----------------------------------------------------
    joblib.dump(
        model,
        MODEL_PATH,
    )

    # -----------------------------------------------------
    # Log parameters
    # -----------------------------------------------------
    mlflow.log_param(
        "model_name",
        "RandomForestClassifier",
    )

    mlflow.log_params(
        model_parameters
    )

    mlflow.log_param(
        "training_data",
        X_TRAIN_PATH.name,
    )

    mlflow.log_param(
        "testing_data",
        X_TEST_PATH.name,
    )

    mlflow.log_param(
        "target_column",
        "churn",
    )

    # -----------------------------------------------------
    # Log metrics
    # -----------------------------------------------------
    mlflow.log_metric(
        "training_accuracy",
        training_accuracy,
    )

    mlflow.log_metric(
        "testing_accuracy",
        testing_accuracy,
    )

    mlflow.log_metric(
        "precision",
        precision,
    )

    mlflow.log_metric(
        "recall",
        recall,
    )

    mlflow.log_metric(
        "f1_score",
        f1,
    )

    mlflow.log_metric(
        "training_rows",
        len(X_train),
    )

    mlflow.log_metric(
        "testing_rows",
        len(X_test),
    )

    # -----------------------------------------------------
    # Set tags
    # -----------------------------------------------------
    mlflow.set_tag(
        "pipeline_stage",
        "model_training",
    )

    mlflow.set_tag(
        "project",
        "customer_churn",
    )

    mlflow.set_tag(
        "model_type",
        "classification",
    )

    mlflow.set_tag(
        "model_role",
        "challenger",
    )

    mlflow.set_tag(
        "status",
        "completed",
    )

    # -----------------------------------------------------
    # Log artifacts
    # -----------------------------------------------------
    mlflow.log_artifact(
        str(REPORT_PATH),
        artifact_path="evaluation",
    )

    mlflow.log_artifact(
        str(CONFUSION_MATRIX_PATH),
        artifact_path="evaluation",
    )

    mlflow.log_artifact(
        str(FEATURE_IMPORTANCE_PATH),
        artifact_path="evaluation",
    )

    mlflow.log_artifact(
        str(MODEL_PATH),
        artifact_path="model_files",
    )

    # -----------------------------------------------------
    # Log model in MLflow model format
    # -----------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=X_train.head(2),
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------
    print(
        "Random Forest training completed successfully."
    )

    print(
        "Training accuracy:",
        round(training_accuracy, 4),
    )

    print(
        "Testing accuracy:",
        round(testing_accuracy, 4),
    )

    print(
        "Precision:",
        round(precision, 4),
    )

    print(
        "Recall:",
        round(recall, 4),
    )

    print(
        "F1 score:",
        round(f1, 4),
    )

    print(
        "Model saved at:",
        MODEL_PATH,
    )