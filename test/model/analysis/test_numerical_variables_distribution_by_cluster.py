import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from dstoolkit.model.analysis.numerical_variables_distribution_by_cluster import plot_numerical_variables_distribution_by_cluster


@pytest.fixture
def clustering_data():
    """Create sample clustering data."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(20, 4),
        columns=[f'feature{i}' for i in range(4)]
    )
    labels = np.array([0]*7 + [1]*7 + [2]*6)
    return X, labels


@patch('matplotlib.pyplot.show')
def test_plot_numerical_variables_executes(mock_show, clustering_data):
    """Test that plot_numerical_variables_distribution_by_cluster executes without errors."""
    X, labels = clustering_data
    plot_numerical_variables_distribution_by_cluster(X, labels)
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_numerical_variables_custom_n_cols(mock_show, clustering_data):
    """Test with custom number of columns."""
    X, labels = clustering_data
    plot_numerical_variables_distribution_by_cluster(X, labels, n_cols=2)
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_numerical_variables_custom_figsize(mock_show, clustering_data):
    """Test with custom figure size."""
    X, labels = clustering_data
    plot_numerical_variables_distribution_by_cluster(X, labels, figsize=(12, 8))
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_numerical_variables_with_fill(mock_show, clustering_data):
    """Test with fill option."""
    X, labels = clustering_data
    plot_numerical_variables_distribution_by_cluster(X, labels, fill=True)
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_numerical_variables_with_array(mock_show):
    """Test with numpy array input."""
    X = np.random.randn(15, 3)
    labels = np.array([0]*5 + [1]*5 + [2]*5)
    plot_numerical_variables_distribution_by_cluster(X, labels)
    mock_show.assert_called_once()
