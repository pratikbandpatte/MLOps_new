from pathlib import Path
import json
import shutil

import joblib
import mlflow
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "model_comparison"
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------
# Input data paths
# ---------------------------------------------------------
X_TEST_PATH = (
    PROCESSED_DATA_DIR
    / "X_test.csv"
)

Y_TEST_PATH = (
    PROCESSED_DATA_DIR
    / "y_test.csv"
)


# ---------------------------------------------------------
# Existing model paths
# ---------------------------------------------------------
LOGISTIC_MODEL_PATH = (
    MODEL_DIR
    / "logistic_regression.joblib"
)

RANDOM_FOREST_MODEL_PATH = (
    MODEL_DIR
    / "random_forest.joblib"
)


# ---------------------------------------------------------
# Champion output paths
# ---------------------------------------------------------
CHAMPION_MODEL_PATH = (
    MODEL_DIR
    / "champion_model.joblib"
)

COMPARISON_REPORT_PATH = (
    ARTIFACT_DIR
    / "model_comparison_report.json"
)

COMPARISON_CSV_PATH = (
    ARTIFACT_DIR
    / "model_comparison.csv"
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
# Check required files
# ---------------------------------------------------------
required_files = [
    X_TEST_PATH,
    Y_TEST_PATH,
    LOGISTIC_MODEL_PATH,
    RANDOM_FOREST_MODEL_PATH,
]

missing_files = [
    str(file_path)
    for file_path in required_files
    if not file_path.exists()
]

if missing_files:
    raise FileNotFoundError(
        f"Required files are missing: {missing_files}"
    )


# ---------------------------------------------------------
# Load test data
# ---------------------------------------------------------
X_test = pd.read_csv(
    X_TEST_PATH
)

y_test = pd.read_csv(
    Y_TEST_PATH
)["churn"]


# ---------------------------------------------------------
# Load trained models
# ---------------------------------------------------------
logistic_model = joblib.load(
    LOGISTIC_MODEL_PATH
)

random_forest_model = joblib.load(
    RANDOM_FOREST_MODEL_PATH
)


# ---------------------------------------------------------
# Models to compare
# ---------------------------------------------------------
models = {
    "LogisticRegression": {
        "model": logistic_model,
        "path": LOGISTIC_MODEL_PATH,
        "role": "baseline",
    },
    "RandomForestClassifier": {
        "model": random_forest_model,
        "path": RANDOM_FOREST_MODEL_PATH,
        "role": "challenger",
    },
}


# ---------------------------------------------------------
# Evaluate each model
# ---------------------------------------------------------
comparison_results = []

for model_name, model_details in models.items():

    model = model_details["model"]

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    comparison_results.append(
        {
            "model_name": model_name,
            "model_role": model_details["role"],
            "model_path": str(
                model_details["path"]
            ),
            "testing_accuracy": float(
                accuracy
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1_score": float(
                f1
            ),
        }
    )


# ---------------------------------------------------------
# Convert results into DataFrame
# ---------------------------------------------------------
comparison_df = pd.DataFrame(
    comparison_results
)


# ---------------------------------------------------------
# Rank models
#
# First: highest F1 score
# Second: highest recall
# Third: highest testing accuracy
# ---------------------------------------------------------
comparison_df = comparison_df.sort_values(
    by=[
        "f1_score",
        "recall",
        "testing_accuracy",
    ],
    ascending=[
        False,
        False,
        False,
    ],
).reset_index(drop=True)


# ---------------------------------------------------------
# Select champion
# ---------------------------------------------------------
champion_result = (
    comparison_df.iloc[0]
)

champion_name = (
    champion_result["model_name"]
)

champion_model_path = Path(
    champion_result["model_path"]
)


# ---------------------------------------------------------
# Copy winning model as champion
# ---------------------------------------------------------
shutil.copy2(
    champion_model_path,
    CHAMPION_MODEL_PATH,
)


# ---------------------------------------------------------
# Mark model results
# ---------------------------------------------------------
comparison_df["selection_status"] = (
    comparison_df["model_name"].apply(
        lambda model_name: (
            "champion"
            if model_name == champion_name
            else "not_selected"
        )
    )
)


# ---------------------------------------------------------
# Save comparison CSV
# ---------------------------------------------------------
comparison_df.to_csv(
    COMPARISON_CSV_PATH,
    index=False,
)


# ---------------------------------------------------------
# Create comparison report
# ---------------------------------------------------------
comparison_report = {
    "selection_metric": "f1_score",
    "tie_breaker_1": "recall",
    "tie_breaker_2": "testing_accuracy",
    "test_rows": int(len(X_test)),
    "champion_model": champion_name,
    "champion_model_path": str(
        CHAMPION_MODEL_PATH
    ),
    "champion_metrics": {
        "testing_accuracy": float(
            champion_result[
                "testing_accuracy"
            ]
        ),
        "precision": float(
            champion_result[
                "precision"
            ]
        ),
        "recall": float(
            champion_result[
                "recall"
            ]
        ),
        "f1_score": float(
            champion_result[
                "f1_score"
            ]
        ),
    },
    "all_models": comparison_df.to_dict(
        orient="records"
    ),
}


with open(
    COMPARISON_REPORT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        comparison_report,
        file,
        indent=4,
    )


# ---------------------------------------------------------
# Start MLflow comparison run
# ---------------------------------------------------------
with mlflow.start_run(
    run_name="champion_model_selection"
):

    # Log comparison settings
    mlflow.log_param(
        "models_compared",
        len(comparison_df),
    )

    mlflow.log_param(
        "primary_selection_metric",
        "f1_score",
    )

    mlflow.log_param(
        "first_tie_breaker",
        "recall",
    )

    mlflow.log_param(
        "second_tie_breaker",
        "testing_accuracy",
    )

    mlflow.log_param(
        "champion_model",
        champion_name,
    )

    mlflow.log_param(
        "test_data",
        X_TEST_PATH.name,
    )

    # Log champion metrics
    mlflow.log_metric(
        "champion_testing_accuracy",
        float(
            champion_result[
                "testing_accuracy"
            ]
        ),
    )

    mlflow.log_metric(
        "champion_precision",
        float(
            champion_result[
                "precision"
            ]
        ),
    )

    mlflow.log_metric(
        "champion_recall",
        float(
            champion_result[
                "recall"
            ]
        ),
    )

    mlflow.log_metric(
        "champion_f1_score",
        float(
            champion_result[
                "f1_score"
            ]
        ),
    )

    # Log each model's metrics
    for _, row in comparison_df.iterrows():

        safe_model_name = (
            row["model_name"]
            .lower()
            .replace(
                "classifier",
                ""
            )
        )

        mlflow.log_metric(
            f"{safe_model_name}_accuracy",
            float(
                row["testing_accuracy"]
            ),
        )

        mlflow.log_metric(
            f"{safe_model_name}_precision",
            float(
                row["precision"]
            ),
        )

        mlflow.log_metric(
            f"{safe_model_name}_recall",
            float(
                row["recall"]
            ),
        )

        mlflow.log_metric(
            f"{safe_model_name}_f1",
            float(
                row["f1_score"]
            ),
        )

    # Set tags
    mlflow.set_tag(
        "pipeline_stage",
        "model_selection",
    )

    mlflow.set_tag(
        "project",
        "customer_churn",
    )

    mlflow.set_tag(
        "selected_model",
        champion_name,
    )

    mlflow.set_tag(
        "model_role",
        "champion",
    )

    mlflow.set_tag(
        "status",
        "completed",
    )

    # Log comparison artifacts
    mlflow.log_artifact(
        str(COMPARISON_CSV_PATH),
        artifact_path="model_comparison",
    )

    mlflow.log_artifact(
        str(COMPARISON_REPORT_PATH),
        artifact_path="model_comparison",
    )

    mlflow.log_artifact(
        str(CHAMPION_MODEL_PATH),
        artifact_path="champion_model",
    )


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
print(
    "\nModel comparison completed successfully."
)

print(
    "\nModel comparison:"
)

print(
    comparison_df[
        [
            "model_name",
            "testing_accuracy",
            "precision",
            "recall",
            "f1_score",
            "selection_status",
        ]
    ].to_string(index=False)
)

print(
    "\nChampion model:",
    champion_name,
)

print(
    "Champion model saved at:",
    CHAMPION_MODEL_PATH,
)