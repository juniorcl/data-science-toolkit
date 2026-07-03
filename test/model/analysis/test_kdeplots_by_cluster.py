import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.kdeplots_by_cluster import plot_kdeplots_by_cluster


@pytest.fixture
def clustering_data():
    """Create sample clustering data."""
    np.random.seed(42)
    X = np.random.randn(20, 4)
    labels = np.array([0]*7 + [1]*7 + [2]*6)
    return X, labels


def test_plot_kdeplots_by_cluster_returns_fig_and_axes(clustering_data):
    """Test that plot_kdeplots_by_cluster returns figure and axes array."""
    X, labels = clustering_data
    fig, axes = plot_kdeplots_by_cluster(X, labels)
    assert fig is not None
    assert axes is not None
    assert isinstance(axes, np.ndarray)


def test_plot_kdeplots_by_cluster_with_dataframe(clustering_data):
    """Test plot_kdeplots_by_cluster with pandas DataFrame."""
    X, labels = clustering_data
    X_df = pd.DataFrame(X, columns=[f'feature{i}' for i in range(X.shape[1])])
    fig, axes = plot_kdeplots_by_cluster(X_df, labels)
    assert fig is not None
    assert axes is not None


def test_plot_kdeplots_by_cluster_custom_n_cols(clustering_data):
    """Test plot_kdeplots_by_cluster with custom number of columns."""
    X, labels = clustering_data
    fig, axes = plot_kdeplots_by_cluster(X, labels, n_cols=2)
    assert fig is not None
    assert axes is not None


def test_plot_kdeplots_by_cluster_custom_figsize(clustering_data):
    """Test plot_kdeplots_by_cluster with custom figure size."""
    X, labels = clustering_data
    fig, axes = plot_kdeplots_by_cluster(X, labels, figsize=(12, 8))
    assert fig.get_figwidth() == 12
    assert fig.get_figheight() == 8


def test_plot_kdeplots_by_cluster_with_fill(clustering_data):
    """Test plot_kdeplots_by_cluster with fill option."""
    X, labels = clustering_data
    fig, axes = plot_kdeplots_by_cluster(X, labels, fill=True)
    assert fig is not None
    assert axes is not None


def test_plot_kdeplots_by_cluster_raises_on_empty_x():
    """Test that plot_kdeplots_by_cluster raises ValueError on empty X."""
    X = pd.DataFrame()
    labels = np.array([0, 1])
    
    with pytest.raises(ValueError, match="X must not be empty"):
        plot_kdeplots_by_cluster(X, labels)


def test_plot_kdeplots_by_cluster_raises_on_length_mismatch():
    """Test that plot_kdeplots_by_cluster raises ValueError on length mismatch."""
    X = np.random.randn(10, 3)
    labels = np.array([0, 1])
    
    with pytest.raises(ValueError, match="labels must have the same length"):
        plot_kdeplots_by_cluster(X, labels)


def test_plot_kdeplots_by_cluster_raises_on_no_numerical_features():
    """Test that plot_kdeplots_by_cluster raises ValueError with no numerical features."""
    X = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': ['x', 'y', 'z']})
    labels = np.array([0, 1, 0])
    
    with pytest.raises(ValueError, match="X must contain at least one numerical feature"):
        plot_kdeplots_by_cluster(X, labels)
