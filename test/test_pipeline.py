# test/test_pipeline.py
import pytest
import yaml
from src.data_loader import load_and_preprocess_data
from src.trainer_model import train_and_save_model


def load_config():
    with open("config/params.yaml") as f:
        return yaml.safe_load(f)


def test_data_not_empty():
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    assert len(X_train) > 0
    assert len(X_test) > 0


def test_metrics_keys():
    config = load_config()
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)
    assert "accuracy" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics