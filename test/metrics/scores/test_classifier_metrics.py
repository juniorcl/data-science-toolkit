import pytest
import numpy as np
from dstoolkit.metrics.scores.classifier_metrics import get_classifier_metrics


@pytest.fixture
def binary_classification_data():
    """Create sample binary classification data."""
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.15, 0.3, 0.95, 0.1, 0.9])
    return y_true, y_pred, y_score


def test_get_classifier_metrics_returns_dict(binary_classification_data):
    """Test that get_classifier_metrics returns a dictionary."""
    y_true, y_pred, y_score = binary_classification_data
    metrics = get_classifier_metrics(y_true, y_pred, y_score)
    assert isinstance(metrics, dict)


def test_get_classifier_metrics_has_required_keys(binary_classification_data):
    """Test that all required metric keys are present."""
    y_true, y_pred, y_score = binary_classification_data
    metrics = get_classifier_metrics(y_true, y_pred, y_score)
    
    required_keys = [
        'Balanced Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'KS',
        'Brier', 'LogLoss', 'Avg Precision Lift'
    ]
    
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_get_classifier_metrics_values_are_numeric(binary_classification_data):
    """Test that all metric values are numeric."""
    y_true, y_pred, y_score = binary_classification_data
    metrics = get_classifier_metrics(y_true, y_pred, y_score)
    
    for key, value in metrics.items():
        assert isinstance(value, (int, float, np.number)), f"{key} is not numeric"


def test_get_classifier_metrics_ranges(binary_classification_data):
    """Test that metrics are within expected ranges."""
    y_true, y_pred, y_score = binary_classification_data
    metrics = get_classifier_metrics(y_true, y_pred, y_score)
    
    # Most metrics should be between 0 and 1 (except LogLoss which can be > 1)
    for key in ['Balanced Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'KS', 'Brier']:
        assert 0 <= metrics[key] <= 1, f"{key} out of range [0, 1]: {metrics[key]}"
    
    # LogLoss should be >= 0
    assert metrics['LogLoss'] >= 0, f"LogLoss should be >= 0: {metrics['LogLoss']}"
