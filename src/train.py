"""Training script with MLflow tracking and optional hyperparameter search."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import CreditRiskDataPipeline, read_transactions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def customer_train_test_indices(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    customers = df["CustomerId"].unique()
    return train_test_split(customers, test_size=test_size, random_state=random_state)


def xy_from_processed(processed: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = processed["is_high_risk"]
    X = processed.drop(columns=["CustomerId", "is_high_risk", "cluster"])
    return X, y


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    register_model: bool,
    model_registry_name: str,
    random_state: int,
) -> None:
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state, max_depth=8),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced_subsample",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }

    best: Tuple[float, str | None, str | None] = (-1.0, None, None)

    for name, estimator in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            if name == "logistic_regression" and X_train.shape[1] > 1:
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions={
                        "C": np.logspace(-3, 3, 20),
                        "solver": ["lbfgs", "liblinear"],
                    },
                    n_iter=6,
                    scoring="roc_auc",
                    cv=3,
                    random_state=random_state,
                    n_jobs=-1,
                )
                search.fit(X_train, y_train)
                model = search.best_estimator_
                mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
            else:
                model = estimator
                model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            try:
                roc = float(roc_auc_score(y_test, proba))
            except ValueError:
                roc = float("nan")
            metrics = {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": roc,
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="classifier")

            if not np.isnan(roc) and roc > best[0]:
                run_id = mlflow.active_run().info.run_id
                best = (roc, name, run_id)

    best_auc, best_name, best_run_id = best
    logger.info("Best model by ROC-AUC: %s (AUC=%s)", best_name, best_auc)

    if register_model and best_run_id:
        model_uri = f"runs:/{best_run_id}/classifier"
        mv = mlflow.register_model(model_uri=model_uri, name=model_registry_name)
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_registry_name,
            version=mv.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Registered %s version %s as Production", model_registry_name, mv.version)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train credit risk models with MLflow.")
    parser.add_argument(
        "--raw-path",
        default=os.environ.get("CREDIT_RISK_RAW_DATA", "data/raw/Xente_challenge_dataset.csv"),
        help="Path to raw transaction CSV.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--register-model", action="store_true")
    parser.add_argument(
        "--registry-name",
        default=os.environ.get("MLFLOW_MODEL_NAME", "credit_risk_classifier"),
    )
    parser.add_argument(
        "--processed-out",
        default="data/processed/customers_model_ready.parquet",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "Download the Xente dataset and set --raw-path or CREDIT_RISK_RAW_DATA."
        )

    df = read_transactions(str(raw_path))
    train_cust, test_cust = customer_train_test_indices(df, args.test_size, args.random_state)
    train_raw = df[df["CustomerId"].isin(train_cust)]
    test_raw = df[df["CustomerId"].isin(test_cust)]

    pipeline = CreditRiskDataPipeline(random_state=args.random_state)
    pipeline.fit(train_raw)
    train_ready = pipeline.transform(train_raw)
    test_ready = pipeline.transform(test_raw)

    processed_out = Path(args.processed_out)
    processed_out.parent.mkdir(parents=True, exist_ok=True)
    full_ready = pd.concat([train_ready, test_ready], axis=0)
    full_ready.to_parquet(processed_out, index=False)
    logger.info("Wrote processed dataset to %s", processed_out)

    X_train, y_train = xy_from_processed(train_ready)
    X_test, y_test = xy_from_processed(test_ready)

    feature_list_path = processed_out.parent / "feature_columns.json"
    feature_cols: List[str] = list(X_train.columns)
    with feature_list_path.open("w", encoding="utf-8") as f:
        json.dump(feature_cols, f)
    logger.info("Wrote feature column order to %s", feature_list_path)

    artifact_dir = Path(tempfile.mkdtemp())
    pipeline_path = artifact_dir / "data_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    mlflow.set_experiment("credit_risk_experiments")
    with mlflow.start_run(run_name="data_pipeline"):
        mlflow.log_params(
            {
                "random_state": args.random_state,
                "high_risk_cluster": pipeline.high_risk_cluster_,
                "snapshot_date": str(pipeline.snapshot_date_),
            }
        )
        mlflow.log_artifact(str(pipeline_path), artifact_path="pipeline")

    train_models(
        X_train,
        y_train,
        X_test,
        y_test,
        register_model=args.register_model,
        model_registry_name=args.registry_name,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
