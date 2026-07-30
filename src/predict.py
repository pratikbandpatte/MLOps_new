from pathlib import Path
import json

import joblib
import mlflow
import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "prediction"

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Input model paths
# ---------------------------------------------------------
PREPROCESSOR_PATH = (
    MODEL_DIR / "preprocessor.joblib"
)

CHAMPION_MODEL_PATH = (
    MODEL_DIR / "champion_model.joblib"
)


# ---------------------------------------------------------
# Output paths
# ---------------------------------------------------------
PREDICTION_CSV_PATH = (
    ARTIFACT_DIR / "predictions.csv"
)

PREDICTION_JSON_PATH = (
    ARTIFACT_DIR / "prediction_report.json"
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
# Check model files
# ---------------------------------------------------------
required_files = [
    PREPROCESSOR_PATH,
    CHAMPION_MODEL_PATH,
]

missing_files = [
    str(file_path)
    for file_path in required_files
    if not file_path.exists()
]

if missing_files:
    raise FileNotFoundError(
        f"Required model files are missing: {missing_files}"
    )


# ---------------------------------------------------------
# Load preprocessor and champion model
# ---------------------------------------------------------
preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

champion_model = joblib.load(
    CHAMPION_MODEL_PATH
)


# ---------------------------------------------------------
# New raw customer data
#
# These columns must match the original features used
# during preprocessing.
# ---------------------------------------------------------
new_customers = pd.DataFrame(
    [
        {
            "age": 29,
            "monthly_charges": 85.50,
            "tenure_months": 4,
            "support_calls": 5,
            "contract_type": "Month-to-month",
            "internet_service": "Fiber optic",
            "payment_method": "Electronic check",
        },
        {
            "age": 54,
            "monthly_charges": 42.00,
            "tenure_months": 48,
            "support_calls": 1,
            "contract_type": "Two year",
            "internet_service": "DSL",
            "payment_method": "Credit card",
        },
    ]
)


# ---------------------------------------------------------
# Validate required columns
# ---------------------------------------------------------
required_columns = [
    "age",
    "monthly_charges",
    "tenure_months",
    "support_calls",
    "contract_type",
    "internet_service",
    "payment_method",
]

missing_columns = [
    column
    for column in required_columns
    if column not in new_customers.columns
]

if missing_columns:
    raise ValueError(
        f"Prediction data is missing columns: {missing_columns}"
    )


# ---------------------------------------------------------
# Transform new data
# ---------------------------------------------------------
new_customers_processed = (
    preprocessor.transform(new_customers)
)


# ---------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------
predictions = champion_model.predict(
    new_customers_processed
)


# ---------------------------------------------------------
# Generate prediction probabilities
# ---------------------------------------------------------
if hasattr(
    champion_model,
    "predict_proba",
):
    probabilities = champion_model.predict_proba(
        new_customers_processed
    )[:, 1]

else:
    probabilities = [
        None
        for _ in range(len(predictions))
    ]


# ---------------------------------------------------------
# Create prediction result
# ---------------------------------------------------------
prediction_results = (
    new_customers.copy()
)

prediction_results[
    "churn_prediction"
] = predictions

prediction_results[
    "prediction_label"
] = prediction_results[
    "churn_prediction"
].map(
    {
        0: "No Churn",
        1: "Churn",
    }
)

prediction_results[
    "churn_probability"
] = probabilities


# ---------------------------------------------------------
# Save predictions as CSV
# ---------------------------------------------------------
prediction_results.to_csv(
    PREDICTION_CSV_PATH,
    index=False,
)


# ---------------------------------------------------------
# Create JSON report
# ---------------------------------------------------------
prediction_report = {
    "records_received": int(
        len(new_customers)
    ),
    "records_predicted": int(
        len(prediction_results)
    ),
    "churn_predictions": int(
        prediction_results[
            "churn_prediction"
        ].sum()
    ),
    "no_churn_predictions": int(
        (
            prediction_results[
                "churn_prediction"
            ] == 0
        ).sum()
    ),
    "predictions": (
        prediction_results.to_dict(
            orient="records"
        )
    ),
}


with open(
    PREDICTION_JSON_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        prediction_report,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Track prediction run in MLflow
# ---------------------------------------------------------
with mlflow.start_run(
    run_name="champion_model_prediction"
):

    # Parameters
    mlflow.log_param(
        "preprocessor_file",
        PREPROCESSOR_PATH.name,
    )

    mlflow.log_param(
        "model_file",
        CHAMPION_MODEL_PATH.name,
    )

    mlflow.log_param(
        "prediction_type",
        "batch_prediction",
    )

    # Metrics
    mlflow.log_metric(
        "records_received",
        len(new_customers),
    )

    mlflow.log_metric(
        "records_predicted",
        len(prediction_results),
    )

    mlflow.log_metric(
        "predicted_churn_count",
        int(
            prediction_results[
                "churn_prediction"
            ].sum()
        ),
    )

    mlflow.log_metric(
        "predicted_no_churn_count",
        int(
            (
                prediction_results[
                    "churn_prediction"
                ] == 0
            ).sum()
        ),
    )

    # Tags
    mlflow.set_tag(
        "pipeline_stage",
        "model_inference",
    )

    mlflow.set_tag(
        "project",
        "customer_churn",
    )

    mlflow.set_tag(
        "model_role",
        "champion",
    )

    mlflow.set_tag(
        "status",
        "completed",
    )

    # Artifacts
    mlflow.log_artifact(
        str(PREDICTION_CSV_PATH),
        artifact_path="predictions",
    )

    mlflow.log_artifact(
        str(PREDICTION_JSON_PATH),
        artifact_path="predictions",
    )


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
print(
    "\nPrediction completed successfully."
)

print(
    prediction_results.to_string(
        index=False
    )
)

print(
    "\nPrediction CSV saved at:",
    PREDICTION_CSV_PATH,
)

print(
    "Prediction report saved at:",
    PREDICTION_JSON_PATH,
)