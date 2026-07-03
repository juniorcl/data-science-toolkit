import pytest
import numpy as np
from dstoolkit.metrics.scores.regressor_metrics import get_regressor_metrics


@pytest.fixture
def regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    y_true = np.array([3, -0.5, 2, 7, 4, 2, 3])
    y_pred = np.array([2.5, 0.0, 2, 8, 4, 2.5, 3])
    return y_true, y_pred


def test_get_regressor_metrics_returns_dict(regression_data):
    """Test that get_regressor_metrics returns a dictionary."""
    y_true, y_pred = regression_data
    metrics = get_regressor_metrics(y_true, y_pred)
    assert isinstance(metrics, dict)


def test_get_regressor_metrics_has_required_keys(regression_data):
    """Test that all required metric keys are present."""
    y_true, y_pred = regression_data
    metrics = get_regressor_metrics(y_true, y_pred)
    
    required_keys = ['R2', 'MAE', 'MadAE', 'MAPE', 'RMSE', 'Explained Variance']
    
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_get_regressor_metrics_values_are_numeric(regression_data):
    """Test that all metric values are numeric."""
    y_true, y_pred = regression_data
    metrics = get_regressor_metrics(y_true, y_pred)
    
    for key, value in metrics.items():
        assert isinstance(value, (int, float, np.number)), f"{key} is not numeric"


def test_get_regressor_metrics_reasonable_values(regression_data):
    """Test that metrics have reasonable values."""
    y_true, y_pred = regression_data
    metrics = get_regressor_metrics(y_true, y_pred)
    
    # MAE, MadAE, RMSE, MAPE should be >= 0
    assert metrics['MAE'] >= 0, "MAE should be >= 0"
    assert metrics['MadAE'] >= 0, "MadAE should be >= 0"
    assert metrics['RMSE'] >= 0, "RMSE should be >= 0"
    assert metrics['MAPE'] >= 0, "MAPE should be >= 0"


def test_get_regressor_metrics_perfect_prediction():
    """Test metrics with perfect predictions."""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    metrics = get_regressor_metrics(y_true, y_pred)
    
    # Perfect prediction should have R2=1, MAE=0
    assert metrics['R2'] == pytest.approx(1.0, abs=0.01)
    assert metrics['MAE'] == pytest.approx(0.0, abs=0.01)
