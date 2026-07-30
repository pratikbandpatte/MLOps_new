from pathlib import Path
import json

import mlflow
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "data_loading"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")

mlflow.set_experiment("customer_churn_pipeline")


# ---------------------------------------------------------
# Start one MLflow run
# ---------------------------------------------------------
with mlflow.start_run(run_name="data_loading"):

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)
    print("\nFirst five records:")
    print(df.head())

    # -----------------------------------------------------
    # Log parameters
    # -----------------------------------------------------
    mlflow.log_param("data_file", DATA_PATH.name)
    mlflow.log_param("file_type", "CSV")
    mlflow.log_param("target_column", "churn")

    # -----------------------------------------------------
    # Log dataset metrics
    # -----------------------------------------------------
    mlflow.log_metric("number_of_rows", len(df))
    mlflow.log_metric("number_of_columns", len(df.columns))
    mlflow.log_metric("duplicate_rows", df.duplicated().sum())
    mlflow.log_metric("missing_values", df.isnull().sum().sum())

    # -----------------------------------------------------
    # Add descriptive tags
    # -----------------------------------------------------
    mlflow.set_tag("pipeline_stage", "data_loading")
    mlflow.set_tag("project", "customer_churn")
    mlflow.set_tag("data_versioning", "DVC")

    # -----------------------------------------------------
    # Create dataset preview artifact
    # -----------------------------------------------------
    preview_path = ARTIFACT_DIR / "data_preview.csv"

    df.head(10).to_csv(preview_path, index=False)

    # -----------------------------------------------------
    # Create schema artifact
    # -----------------------------------------------------
    schema = {
        column: str(dtype)
        for column, dtype in df.dtypes.items()
    }

    schema_path = ARTIFACT_DIR / "data_schema.json"

    with open(schema_path, "w", encoding="utf-8") as file:
        json.dump(schema, file, indent=4)

    # -----------------------------------------------------
    # Create dataset summary artifact
    # -----------------------------------------------------
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": "churn",
        "target_distribution": {
            str(key): int(value)
            for key, value in df["churn"].value_counts().to_dict().items()
        },
    }

    summary_path = ARTIFACT_DIR / "data_summary.json"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    # -----------------------------------------------------
    # Log files as MLflow artifacts
    # -----------------------------------------------------
    mlflow.log_artifact(str(preview_path), artifact_path="data_loading")
    mlflow.log_artifact(str(schema_path), artifact_path="data_loading")
    mlflow.log_artifact(str(summary_path), artifact_path="data_loading")

    print("\nData-loading information logged successfully in MLflow.")