from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DATA_PATH = (
    PROJECT_ROOT / "data" / "customer_churn_cleaned.csv"
)

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "preprocessing"
MODEL_DIR = PROJECT_ROOT / "models"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Output file paths
# ---------------------------------------------------------
X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------
mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

mlflow.set_experiment("customer_churn_pipeline")


# ---------------------------------------------------------
# Start MLflow run
# ---------------------------------------------------------
with mlflow.start_run(run_name="data_preprocessing"):

    # Load cleaned dataset
    df = pd.read_csv(CLEANED_DATA_PATH)

    print("Cleaned dataset loaded.")
    print("Shape:", df.shape)

    # -----------------------------------------------------
    # Separate features and target
    # -----------------------------------------------------
    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    # -----------------------------------------------------
    # Define numerical and categorical columns
    # -----------------------------------------------------
    numerical_columns = [
        "age",
        "monthly_charges",
        "tenure_months",
        "support_calls",
    ]

    categorical_columns = [
        "contract_type",
        "internet_service",
        "payment_method",
    ]

    # -----------------------------------------------------
    # Numerical preprocessing
    # -----------------------------------------------------
    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    # -----------------------------------------------------
    # Categorical preprocessing
    # -----------------------------------------------------
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    # -----------------------------------------------------
    # Combine preprocessing steps
    # -----------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    # -----------------------------------------------------
    # Train-test split
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # -----------------------------------------------------
    # Fit preprocessing only on training data
    # -----------------------------------------------------
    X_train_processed = preprocessor.fit_transform(X_train)

    # Use the fitted preprocessor on test data
    X_test_processed = preprocessor.transform(X_test)

    # -----------------------------------------------------
    # Get transformed feature names
    # -----------------------------------------------------
    feature_names = preprocessor.get_feature_names_out()

    # -----------------------------------------------------
    # Convert processed arrays into DataFrames
    # -----------------------------------------------------
    X_train_processed_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    # Reset indexes
    X_train_processed_df = X_train_processed_df.reset_index(
        drop=True
    )

    X_test_processed_df = X_test_processed_df.reset_index(
        drop=True
    )

    y_train_df = y_train.reset_index(drop=True).to_frame(
        name="churn"
    )

    y_test_df = y_test.reset_index(drop=True).to_frame(
        name="churn"
    )

    # -----------------------------------------------------
    # Save processed datasets
    # -----------------------------------------------------
    X_train_processed_df.to_csv(
        X_TRAIN_PATH,
        index=False,
    )

    X_test_processed_df.to_csv(
        X_TEST_PATH,
        index=False,
    )

    y_train_df.to_csv(
        Y_TRAIN_PATH,
        index=False,
    )

    y_test_df.to_csv(
        Y_TEST_PATH,
        index=False,
    )

    # Save fitted preprocessing object
    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    # -----------------------------------------------------
    # Log parameters
    # -----------------------------------------------------
    mlflow.log_param(
        "input_data",
        CLEANED_DATA_PATH.name,
    )

    mlflow.log_param(
        "target_column",
        "churn",
    )

    mlflow.log_param(
        "dropped_column",
        "customer_id",
    )

    mlflow.log_param(
        "test_size",
        0.20,
    )

    mlflow.log_param(
        "random_state",
        42,
    )

    mlflow.log_param(
        "numeric_scaler",
        "StandardScaler",
    )

    mlflow.log_param(
        "categorical_encoder",
        "OneHotEncoder",
    )

    # -----------------------------------------------------
    # Log metrics
    # -----------------------------------------------------
    mlflow.log_metric(
        "training_rows",
        len(X_train_processed_df),
    )

    mlflow.log_metric(
        "testing_rows",
        len(X_test_processed_df),
    )

    mlflow.log_metric(
        "original_feature_count",
        len(X.columns),
    )

    mlflow.log_metric(
        "processed_feature_count",
        len(feature_names),
    )

    mlflow.log_metric(
        "training_churn_count",
        int(y_train.sum()),
    )

    mlflow.log_metric(
        "testing_churn_count",
        int(y_test.sum()),
    )

    # -----------------------------------------------------
    # Set tags
    # -----------------------------------------------------
    mlflow.set_tag(
        "pipeline_stage",
        "data_preprocessing",
    )

    mlflow.set_tag(
        "project",
        "customer_churn",
    )

    mlflow.set_tag(
        "status",
        "completed",
    )

    # -----------------------------------------------------
    # Create preprocessing report
    # -----------------------------------------------------
    preprocessing_report = {
        "input_file": CLEANED_DATA_PATH.name,
        "target_column": "churn",
        "dropped_columns": ["customer_id"],
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "train_rows": int(len(X_train_processed_df)),
        "test_rows": int(len(X_test_processed_df)),
        "original_feature_count": int(len(X.columns)),
        "processed_feature_count": int(len(feature_names)),
        "processed_feature_names": feature_names.tolist(),
        "test_size": 0.20,
        "random_state": 42,
    }

    report_path = (
        ARTIFACT_DIR / "preprocessing_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            preprocessing_report,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Log artifacts
    # -----------------------------------------------------
    mlflow.log_artifact(
        str(report_path),
        artifact_path="preprocessing",
    )

    mlflow.log_artifact(
        str(PREPROCESSOR_PATH),
        artifact_path="preprocessing",
    )

    mlflow.log_artifact(
        str(X_TRAIN_PATH),
        artifact_path="processed_data",
    )

    mlflow.log_artifact(
        str(X_TEST_PATH),
        artifact_path="processed_data",
    )

    mlflow.log_artifact(
        str(Y_TRAIN_PATH),
        artifact_path="processed_data",
    )

    mlflow.log_artifact(
        str(Y_TEST_PATH),
        artifact_path="processed_data",
    )

    print("\nPreprocessing completed successfully.")
    print("Training shape:", X_train_processed_df.shape)
    print("Testing shape:", X_test_processed_df.shape)
    print("Preprocessor saved at:", PREPROCESSOR_PATH)