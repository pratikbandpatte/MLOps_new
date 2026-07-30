from pathlib import Path
import json

import mlflow
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_cleaned.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "data_cleaning"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("customer_churn_pipeline")


# ---------------------------------------------------------
# Start MLflow run
# ---------------------------------------------------------
with mlflow.start_run(run_name="data_cleaning"):

    # Load raw dataset
    df = pd.read_csv(RAW_DATA_PATH)

    rows_before = len(df)
    columns_before = len(df.columns)
    missing_before = int(df.isnull().sum().sum())
    duplicates_before = int(df.duplicated().sum())

    print("Raw dataset loaded.")
    print("Shape before cleaning:", df.shape)

    # -----------------------------------------------------
    # Cleaning operations
    # -----------------------------------------------------

    # Remove extra spaces from column names
    df.columns = df.columns.str.strip().str.lower()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove spaces from text columns
    text_columns = df.select_dtypes(include="object").columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    # Convert numeric columns safely
    numeric_columns = [
        "customer_id",
        "age",
        "monthly_charges",
        "tenure_months",
        "support_calls",
        "churn",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Fill missing numerical values with median
    for column in numeric_columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].median())

    # Fill missing categorical values with mode
    categorical_columns = [
        "contract_type",
        "internet_service",
        "payment_method",
    ]

    for column in categorical_columns:
        if df[column].isnull().any():
            df[column] = df[column].fillna(df[column].mode()[0])

    # Remove invalid records
    df = df[df["age"] > 0]
    df = df[df["monthly_charges"] >= 0]
    df = df[df["tenure_months"] >= 0]
    df = df[df["support_calls"] >= 0]

    # Ensure target contains only 0 and 1
    df = df[df["churn"].isin([0, 1])]

    # Convert integer columns back to integer type
    integer_columns = [
        "customer_id",
        "age",
        "tenure_months",
        "support_calls",
        "churn",
    ]

    for column in integer_columns:
        df[column] = df[column].astype(int)

    # Reset index
    df = df.reset_index(drop=True)

    # Save cleaned dataset
    df.to_csv(CLEANED_DATA_PATH, index=False)

    # -----------------------------------------------------
    # Metrics after cleaning
    # -----------------------------------------------------
    rows_after = len(df)
    columns_after = len(df.columns)
    missing_after = int(df.isnull().sum().sum())
    duplicates_after = int(df.duplicated().sum())
    removed_rows = rows_before - rows_after

    # -----------------------------------------------------
    # Log parameters
    # -----------------------------------------------------
    mlflow.log_param("input_file", RAW_DATA_PATH.name)
    mlflow.log_param("output_file", CLEANED_DATA_PATH.name)
    mlflow.log_param("duplicate_strategy", "drop_duplicates")
    mlflow.log_param("numeric_missing_strategy", "median")
    mlflow.log_param("categorical_missing_strategy", "mode")

    # -----------------------------------------------------
    # Log metrics
    # -----------------------------------------------------
    mlflow.log_metric("rows_before_cleaning", rows_before)
    mlflow.log_metric("rows_after_cleaning", rows_after)
    mlflow.log_metric("columns_before_cleaning", columns_before)
    mlflow.log_metric("columns_after_cleaning", columns_after)
    mlflow.log_metric("missing_before_cleaning", missing_before)
    mlflow.log_metric("missing_after_cleaning", missing_after)
    mlflow.log_metric("duplicates_before_cleaning", duplicates_before)
    mlflow.log_metric("duplicates_after_cleaning", duplicates_after)
    mlflow.log_metric("removed_rows", removed_rows)

    # -----------------------------------------------------
    # Tags
    # -----------------------------------------------------
    mlflow.set_tag("pipeline_stage", "data_cleaning")
    mlflow.set_tag("project", "customer_churn")
    mlflow.set_tag("status", "completed")

    # -----------------------------------------------------
    # Cleaning report artifact
    # -----------------------------------------------------
    cleaning_report = {
        "input_file": RAW_DATA_PATH.name,
        "output_file": CLEANED_DATA_PATH.name,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "removed_rows": removed_rows,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "duplicates_before": duplicates_before,
        "duplicates_after": duplicates_after,
        "cleaning_steps": [
            "Standardized column names",
            "Removed duplicate rows",
            "Trimmed categorical values",
            "Converted numeric columns",
            "Filled missing numeric values with median",
            "Filled missing categorical values with mode",
            "Removed invalid values",
            "Validated churn values",
        ],
    }

    report_path = ARTIFACT_DIR / "cleaning_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(cleaning_report, file, indent=4)

    # Log artifacts
    mlflow.log_artifact(
        str(report_path),
        artifact_path="data_cleaning",
    )

    mlflow.log_artifact(
        str(CLEANED_DATA_PATH),
        artifact_path="cleaned_data",
    )

    print("\nData cleaning completed successfully.")
    print("Shape after cleaning:", df.shape)
    print("Cleaned dataset saved at:", CLEANED_DATA_PATH)