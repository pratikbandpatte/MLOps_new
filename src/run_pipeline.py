from pathlib import Path
import subprocess
import sys
import time

import mlflow


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri(
    f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)

mlflow.set_experiment(
    "customer_churn_pipeline"
)


# ---------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------
PIPELINE_STAGES = [
    {
        "name": "Data Creation",
        "module": "src.create_data",
    },
    {
        "name": "Data Loading",
        "module": "src.load_data",
    },
    {
        "name": "Data Cleaning",
        "module": "src.clean_data",
    },
    {
        "name": "Data Validation",
        "module": "src.valid_data",
    },
    {
        "name": "Data Preprocessing",
        "module": "src.data_preprocessing",
    },
    {
        "name": "Preprocessing Validation",
        "module": "src.validation_preprocess",
    },
    {
        "name": "Logistic Regression Training",
        "module": "src.logistic_regression_training",
    },
    {
        "name": "Random Forest Training",
        "module": "src.random_forest_training",
    },
    {
        "name": "Champion Selection",
        "module": "src.select_champion",
    },
    {
        "name": "Prediction",
        "module": "src.predict",
    },
]


# ---------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------
pipeline_start_time = time.time()

completed_stages = []
failed_stage = None


with mlflow.start_run(
    run_name="complete_ml_pipeline"
):

    mlflow.log_param(
        "total_pipeline_stages",
        len(PIPELINE_STAGES),
    )

    mlflow.set_tag(
        "pipeline_type",
        "end_to_end",
    )

    mlflow.set_tag(
        "project",
        "customer_churn",
    )

    mlflow.set_tag(
        "pipeline_status",
        "running",
    )

    print("\nStarting complete MLOps pipeline...\n")

    for stage_number, stage in enumerate(
        PIPELINE_STAGES,
        start=1,
    ):

        stage_name = stage["name"]
        module_name = stage["module"]

        print("=" * 60)
        print(
            f"Stage {stage_number}/{len(PIPELINE_STAGES)}: "
            f"{stage_name}"
        )
        print("=" * 60)

        stage_start_time = time.time()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    module_name,
                ],
                cwd=PROJECT_ROOT,
                check=True,
                text=True,
            )

            stage_duration = (
                time.time() - stage_start_time
            )

            completed_stages.append(
                stage_name
            )

            mlflow.log_metric(
                f"stage_{stage_number}_duration_seconds",
                stage_duration,
            )

            mlflow.log_metric(
                f"stage_{stage_number}_status",
                1,
            )

            print(
                f"\n{stage_name} completed successfully."
            )

        except subprocess.CalledProcessError as error:

            failed_stage = stage_name

            stage_duration = (
                time.time() - stage_start_time
            )

            mlflow.log_metric(
                f"stage_{stage_number}_duration_seconds",
                stage_duration,
            )

            mlflow.log_metric(
                f"stage_{stage_number}_status",
                0,
            )

            mlflow.set_tag(
                "pipeline_status",
                "failed",
            )

            mlflow.set_tag(
                "failed_stage",
                stage_name,
            )

            print(
                f"\nPipeline failed at stage: {stage_name}"
            )

            print(
                f"Error code: {error.returncode}"
            )

            break

    # -----------------------------------------------------
    # Final pipeline metrics
    # -----------------------------------------------------
    pipeline_duration = (
        time.time() - pipeline_start_time
    )

    mlflow.log_metric(
        "pipeline_duration_seconds",
        pipeline_duration,
    )

    mlflow.log_metric(
        "completed_stage_count",
        len(completed_stages),
    )

    mlflow.log_metric(
        "failed_stage_count",
        1 if failed_stage else 0,
    )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------
    if failed_stage is None:

        mlflow.set_tag(
            "pipeline_status",
            "completed",
        )

        print("\n" + "=" * 60)
        print("Complete MLOps pipeline finished successfully.")
        print("=" * 60)

        print(
            "Completed stages:",
            len(completed_stages),
        )

        print(
            "Total duration:",
            round(pipeline_duration, 2),
            "seconds",
        )

    else:

        print("\n" + "=" * 60)
        print("MLOps pipeline did not complete.")
        print("=" * 60)

        print(
            "Completed stages:",
            len(completed_stages),
        )

        print(
            "Failed stage:",
            failed_stage,
        )

        raise RuntimeError(
            f"Pipeline failed during: {failed_stage}"
        )