"""Training and inference utilities for the customer churn application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).with_name("Telco Customer Churn.csv")
TARGET_COLUMN = "Churn"
IGNORED_COLUMNS = ("customerID",)
NUMERIC_FEATURES = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
CATEGORICAL_FEATURES = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass(frozen=True)
class ModelBundle:
    """A fitted model plus the metadata displayed by the application."""

    pipeline: Pipeline
    training_rows: int


@dataclass(frozen=True)
class Prediction:
    """The model's binary result and estimated probability of churn."""

    will_churn: bool
    churn_probability: float


def load_training_data(data_path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the repository dataset and apply the notebook's data cleaning."""

    data = pd.read_csv(data_path)
    required = set(FEATURES) | {TARGET_COLUMN, *IGNORED_COLUMNS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    features = data.drop(columns=[TARGET_COLUMN, *IGNORED_COLUMNS]).copy()
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")

    target = data[TARGET_COLUMN].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Churn must contain only 'Yes' and 'No' values.")

    return features.loc[:, FEATURES], target.astype(int)


def build_pipeline() -> Pipeline:
    """Build the preprocessing and preferred LightGBM model as one pipeline."""

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ]
    )

    model = LGBMClassifier(
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    return Pipeline(steps=[("preprocessing", preprocessing), ("model", model)])


def train_model(data_path: Path = DATA_PATH) -> ModelBundle:
    """Train the notebook's preferred model on all available labeled records."""

    features, target = load_training_data(data_path)
    pipeline = build_pipeline()
    pipeline.fit(features, target)
    return ModelBundle(pipeline=pipeline, training_rows=len(features))


def predict_customer(pipeline: Pipeline, customer: Mapping[str, Any]) -> Prediction:
    """Predict churn for one customer using the fitted preprocessing pipeline."""

    missing = set(FEATURES).difference(customer)
    if missing:
        raise ValueError(f"Customer is missing required fields: {sorted(missing)}")

    row = pd.DataFrame([{feature: customer[feature] for feature in FEATURES}])
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce")
    probability = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(pipeline.predict(row)[0])
    return Prediction(will_churn=bool(prediction), churn_probability=probability)
