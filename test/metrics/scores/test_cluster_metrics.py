import pytest
import numpy as np
import pandas as pd
from dstoolkit.metrics.scores.cluster_metrics import get_cluster_metrics


@pytest.fixture
def clustering_data():
    """Create sample clustering data."""
    np.random.seed(42)
    X = np.random.randn(10, 3)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    return X, labels


def test_get_cluster_metrics_returns_dict(clustering_data):
    """Test that get_cluster_metrics returns a dictionary."""
    X, labels = clustering_data
    metrics = get_cluster_metrics(X, labels)
    assert isinstance(metrics, dict)


def test_get_cluster_metrics_has_required_keys(clustering_data):
    """Test that all required metric keys are present."""
    X, labels = clustering_data
    metrics = get_cluster_metrics(X, labels)
    
    required_keys = ['Silhouette', 'Calinski Harabasz', 'Davies-Bouldin']
    
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_get_cluster_metrics_values_are_numeric(clustering_data):
    """Test that all metric values are numeric."""
    X, labels = clustering_data
    metrics = get_cluster_metrics(X, labels)
    
    for key, value in metrics.items():
        assert isinstance(value, (int, float, np.number)), f"{key} is not numeric"


def test_get_cluster_metrics_with_dataframe(clustering_data):
    """Test that get_cluster_metrics works with pandas DataFrame."""
    X, labels = clustering_data
    X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    metrics = get_cluster_metrics(X_df, labels)
    
    assert isinstance(metrics, dict)
    assert len(metrics) == 3
