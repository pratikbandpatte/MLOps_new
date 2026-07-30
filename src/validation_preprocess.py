from pathlib import Path
import json

import mlflow
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "preprocessing_validation"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = ARTIFACT_DIR / "preprocessing_validation_report.json"


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
with mlflow.start_run(
    run_name="preprocessing_validation"
):

    # -----------------------------------------------------
    # Check whether all processed files exist
    # -----------------------------------------------------
    file_checks = {
        "x_train_exists": X_TRAIN_PATH.exists(),
        "x_test_exists": X_TEST_PATH.exists(),
        "y_train_exists": Y_TRAIN_PATH.exists(),
        "y_test_exists": Y_TEST_PATH.exists(),
    }

    all_files_exist = all(file_checks.values())

    if not all_files_exist:
        missing_files = [
            check_name
            for check_name, result in file_checks.items()
            if not result
        ]

        raise FileNotFoundError(
            f"Processed data files are missing: {missing_files}"
        )

    # -----------------------------------------------------
    # Load processed datasets
    # -----------------------------------------------------
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH)
    y_test = pd.read_csv(Y_TEST_PATH)

    print("Processed datasets loaded successfully.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    # -----------------------------------------------------
    # Preprocessing validation checks
    # -----------------------------------------------------
    validation_checks = {}

    # Check 1: Train feature-label row count
    validation_checks["train_rows_match"] = (
        len(X_train) == len(y_train)
    )

    # Check 2: Test feature-label row count
    validation_checks["test_rows_match"] = (
        len(X_test) == len(y_test)
    )

    # Check 3: Train and test columns must match
    validation_checks["train_test_columns_match"] = (
        list(X_train.columns) == list(X_test.columns)
    )

    # Check 4: No missing values in training features
    validation_checks["no_missing_x_train"] = (
        int(X_train.isnull().sum().sum()) == 0
    )

    # Check 5: No missing values in testing features
    validation_checks["no_missing_x_test"] = (
        int(X_test.isnull().sum().sum()) == 0
    )

    # Check 6: No missing values in training target
    validation_checks["no_missing_y_train"] = (
        int(y_train.isnull().sum().sum()) == 0
    )

    # Check 7: No missing values in testing target
    validation_checks["no_missing_y_test"] = (
        int(y_test.isnull().sum().sum()) == 0
    )

    # Check 8: Target column must exist
    validation_checks["target_column_exists"] = (
        "churn" in y_train.columns
        and "churn" in y_test.columns
    )

    # Check 9: Training target contains only 0 and 1
    validation_checks["valid_y_train_values"] = (
        set(y_train["churn"].unique()).issubset({0, 1})
    )

    # Check 10: Testing target contains only 0 and 1
    validation_checks["valid_y_test_values"] = (
        set(y_test["churn"].unique()).issubset({0, 1})
    )

    # Check 11: All processed features must be numeric
    validation_checks["all_train_features_numeric"] = all(
        pd.api.types.is_numeric_dtype(X_train[column])
        for column in X_train.columns
    )

    validation_checks["all_test_features_numeric"] = all(
        pd.api.types.is_numeric_dtype(X_test[column])
        for column in X_test.columns
    )

    # Check 12: No duplicate columns
    validation_checks["no_duplicate_train_columns"] = (
        not X_train.columns.duplicated().any()
    )

    validation_checks["no_duplicate_test_columns"] = (
        not X_test.columns.duplicated().any()
    )

    # Check 13: customer_id must not be present
    validation_checks["customer_id_removed"] = (
        "customer_id" not in X_train.columns
        and "customer_id" not in X_test.columns
    )

    # Check 14: Raw categorical columns must not remain
    raw_categorical_columns = [
        "contract_type",
        "internet_service",
        "payment_method",
    ]

    validation_checks["raw_categories_encoded"] = all(
        column not in X_train.columns
        for column in raw_categorical_columns
    )

    # Check 15: Both train and test must contain rows
    validation_checks["train_data_not_empty"] = (
        len(X_train) > 0 and len(y_train) > 0
    )

    validation_checks["test_data_not_empty"] = (
        len(X_test) > 0 and len(y_test) > 0
    )

    # -----------------------------------------------------
    # Final validation result
    # -----------------------------------------------------
    total_checks = len(validation_checks)

    passed_checks = sum(
        bool(result)
        for result in validation_checks.values()
    )

    failed_checks = total_checks - passed_checks

    validation_passed = all(
        bool(result)
        for result in validation_checks.values()
    )

    validation_status = (
        "PASSED"
        if validation_passed
        else "FAILED"
    )

    # -----------------------------------------------------
    # MLflow parameters
    # -----------------------------------------------------
    mlflow.log_param(
        "input_directory",
        PROCESSED_DATA_DIR.name,
    )

    mlflow.log_param(
        "target_column",
        "churn",
    )

    mlflow.log_param(
        "total_validation_checks",
        total_checks,
    )

    mlflow.log_param(
        "expected_target_values",
        "0,1",
    )

    # -----------------------------------------------------
    # MLflow metrics
    # -----------------------------------------------------
    mlflow.log_metric(
        "passed_checks",
        passed_checks,
    )

    mlflow.log_metric(
        "failed_checks",
        failed_checks,
    )

    mlflow.log_metric(
        "validation_score",
        passed_checks / total_checks,
    )

    mlflow.log_metric(
        "training_rows",
        len(X_train),
    )

    mlflow.log_metric(
        "testing_rows",
        len(X_test),
    )

    mlflow.log_metric(
        "processed_feature_count",
        len(X_train.columns),
    )

    # Log every validation check as 1 or 0
    for check_name, result in validation_checks.items():
        mlflow.log_metric(
            check_name,
            int(bool(result)),
        )

    # -----------------------------------------------------
    # MLflow tags
    # -----------------------------------------------------
    mlflow.set_tag(
        "pipeline_stage",
        "preprocessing_validation",
    )

    mlflow.set_tag(
        "project",
        "customer_churn",
    )

    mlflow.set_tag(
        "validation_status",
        validation_status,
    )

    # -----------------------------------------------------
    # Create JSON report
    # -----------------------------------------------------
    validation_report = {
        "validation_status": validation_status,
        "input_directory": str(PROCESSED_DATA_DIR),
        "total_checks": int(total_checks),
        "passed_checks": int(passed_checks),
        "failed_checks": int(failed_checks),
        "training_shape": [
            int(X_train.shape[0]),
            int(X_train.shape[1]),
        ],
        "testing_shape": [
            int(X_test.shape[0]),
            int(X_test.shape[1]),
        ],
        "processed_features": X_train.columns.tolist(),
        "checks": {
            check_name: bool(result)
            for check_name, result in validation_checks.items()
        },
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation_report,
            file,
            indent=4,
        )

    # -----------------------------------------------------
    # Log validation report as artifact
    # -----------------------------------------------------
    mlflow.log_artifact(
        str(REPORT_PATH),
        artifact_path="preprocessing_validation",
    )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------
    print("\nPreprocessing validation completed.")
    print("Validation status:", validation_status)
    print(
        f"Passed checks: {passed_checks}/{total_checks}"
    )

    if not validation_passed:
        failed_check_names = [
            check_name
            for check_name, result in validation_checks.items()
            if not bool(result)
        ]

        print("Failed checks:", failed_check_names)

        raise ValueError(
            f"Preprocessing validation failed: "
            f"{failed_check_names}"
        )