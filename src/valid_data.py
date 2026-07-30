from pathlib import Path
import json

import mlflow
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "customer_churn_cleaned.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "data_validation"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("customer_churn_pipeline")


# ---------------------------------------------------------
# Expected dataset rules
# ---------------------------------------------------------
EXPECTED_COLUMNS = [
    "customer_id",
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls",
    "contract_type",
    "internet_service",
    "payment_method",
    "churn",
]

NUMERIC_COLUMNS = [
    "customer_id",
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls",
    "churn",
]

CATEGORICAL_COLUMNS = [
    "contract_type",
    "internet_service",
    "payment_method",
]


# ---------------------------------------------------------
# Start MLflow run
# ---------------------------------------------------------
with mlflow.start_run(run_name="data_validation"):

    df = pd.read_csv(CLEANED_DATA_PATH)

    validation_checks = {}

    # -----------------------------------------------------
    # Check 1: Required columns
    # -----------------------------------------------------
    validation_checks["required_columns_present"] = (
        list(df.columns) == EXPECTED_COLUMNS
    )

    # -----------------------------------------------------
    # Check 2: Missing values
    # -----------------------------------------------------
    validation_checks["no_missing_values"] = (
        int(df.isnull().sum().sum()) == 0
    )

    # -----------------------------------------------------
    # Check 3: Duplicate rows
    # -----------------------------------------------------
    validation_checks["no_duplicate_rows"] = (
        int(df.duplicated().sum()) == 0
    )

    # -----------------------------------------------------
    # Check 4: Target values
    # -----------------------------------------------------
    validation_checks["valid_target_values"] = (
        set(df["churn"].unique()).issubset({0, 1})
    )

    # -----------------------------------------------------
    # Check 5: Age range
    # -----------------------------------------------------
    validation_checks["valid_age_range"] = (
        df["age"].between(18, 100).all()
    )

    # -----------------------------------------------------
    # Check 6: Monthly charges
    # -----------------------------------------------------
    validation_checks["valid_monthly_charges"] = (
        (df["monthly_charges"] >= 0).all()
    )

    # -----------------------------------------------------
    # Check 7: Tenure
    # -----------------------------------------------------
    validation_checks["valid_tenure"] = (
        (df["tenure_months"] >= 0).all()
    )

    # -----------------------------------------------------
    # Check 8: Support calls
    # -----------------------------------------------------
    validation_checks["valid_support_calls"] = (
        (df["support_calls"] >= 0).all()
    )

    # -----------------------------------------------------
    # Check 9: Numeric column types
    # -----------------------------------------------------
    validation_checks["numeric_columns_valid"] = all(
        pd.api.types.is_numeric_dtype(df[column])
        for column in NUMERIC_COLUMNS
    )

    # -----------------------------------------------------
    # Check 10: Categorical columns contain values
    # -----------------------------------------------------
    validation_checks["categorical_columns_valid"] = all(
        df[column].astype(str).str.strip().ne("").all()
        for column in CATEGORICAL_COLUMNS
    )

    # -----------------------------------------------------
    # Final validation result
    # -----------------------------------------------------
    passed_checks = sum(validation_checks.values())
    total_checks = len(validation_checks)

    validation_passed = all(validation_checks.values())

    validation_status = (
        "PASSED" if validation_passed else "FAILED"
    )

    # -----------------------------------------------------
    # Log parameters
    # -----------------------------------------------------
    mlflow.log_param("input_file", CLEANED_DATA_PATH.name)
    mlflow.log_param("total_validation_checks", total_checks)
    mlflow.log_param("expected_target_values", "0,1")

    # -----------------------------------------------------
    # Log metrics
    # -----------------------------------------------------
    mlflow.log_metric("passed_checks", passed_checks)
    mlflow.log_metric("failed_checks", total_checks - passed_checks)
    mlflow.log_metric("validation_score", passed_checks / total_checks)
    mlflow.log_metric("row_count", len(df))
    mlflow.log_metric("column_count", len(df.columns))

    # Log each check as 1 or 0
    for check_name, result in validation_checks.items():
        mlflow.log_metric(check_name, int(result))

    # -----------------------------------------------------
    # Tags
    # -----------------------------------------------------
    mlflow.set_tag("pipeline_stage", "data_validation")
    mlflow.set_tag("project", "customer_churn")
    mlflow.set_tag("validation_status", validation_status)

    # -----------------------------------------------------
    # Validation report artifact
    # -----------------------------------------------------
    validation_report = {
        "input_file": CLEANED_DATA_PATH.name,
        "validation_status": validation_status,
        "passed_checks": int(passed_checks),
        "failed_checks": int(total_checks - passed_checks),
        "total_checks": int(total_checks),
        "checks": {
            check_name: bool(result)
            for check_name, result in validation_checks.items()
        },
    }
    report_path = ARTIFACT_DIR / "validation_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(validation_report, file, indent=4)

    mlflow.log_artifact(
        str(report_path),
        artifact_path="data_validation",
    )

    print("Data validation completed.")
    print("Validation status:", validation_status)
    print(f"Passed checks: {passed_checks}/{total_checks}")

    if not validation_passed:
        failed_checks = [
            name
            for name, result in validation_checks.items()
            if not result
        ]

        raise ValueError(
            f"Data validation failed: {failed_checks}"
        )
    
