from pathlib import Path

import pandas as pd
import pytest

from churn_predictor import FEATURES, load_training_data, predict_customer, train_model


@pytest.fixture(scope="module")
def model_bundle():
    return train_model()


def test_training_data_matches_repository_dataset():
    features, target = load_training_data()

    assert len(features) == 7043
    assert tuple(features.columns) == FEATURES
    assert set(target.unique()) == {0, 1}
    assert pd.api.types.is_numeric_dtype(features["TotalCharges"])


def test_model_returns_a_valid_prediction(model_bundle):
    customer = {
        "SeniorCitizen": 0,
        "tenure": 1,
        "MonthlyCharges": 89.10,
        "TotalCharges": 89.10,
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    result = predict_customer(model_bundle.pipeline, customer)

    assert isinstance(result.will_churn, bool)
    assert 0.0 <= result.churn_probability <= 1.0


def test_prediction_rejects_missing_fields(model_bundle):
    with pytest.raises(ValueError, match="missing required fields"):
        predict_customer(model_bundle.pipeline, {"tenure": 12})


def test_load_training_data_rejects_invalid_schema(tmp_path: Path):
    invalid_data = tmp_path / "invalid.csv"
    pd.DataFrame({"Churn": ["No"]}).to_csv(invalid_data, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_training_data(invalid_data)
