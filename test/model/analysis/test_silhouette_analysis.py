import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
from dstoolkit.model.analysis.silhouette_analysis import plot_silhouette_analysis


@pytest.fixture
def clustering_data():
    """Create sample clustering data."""
    np.random.seed(42)
    X = np.random.randn(20, 3)
    labels = np.array([0]*7 + [1]*7 + [2]*6)
    return X, labels


@patch('matplotlib.pyplot.show')
def test_plot_silhouette_analysis_executes(mock_show, clustering_data):
    """Test that plot_silhouette_analysis executes without errors."""
    X, labels = clustering_data
    plot_silhouette_analysis(X, labels)
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_silhouette_analysis_with_custom_metric(mock_show, clustering_data):
    """Test plot_silhouette_analysis with custom distance metric."""
    X, labels = clustering_data
    plot_silhouette_analysis(X, labels, metric='manhattan')
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_silhouette_analysis_with_dataframe(mock_show, clustering_data):
    """Test plot_silhouette_analysis with pandas DataFrame."""
    import pandas as pd
    X, labels = clustering_data
    X_df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    plot_silhouette_analysis(X_df, labels)
    mock_show.assert_called_once()


@patch('matplotlib.pyplot.show')
def test_plot_silhouette_analysis_closes_previous_figures(mock_show, clustering_data):
    """Test that plot_silhouette_analysis creates a new figure."""
    X, labels = clustering_data
    n_figs_before = len(plt.get_fignums())
    plot_silhouette_analysis(X, labels)
    # After execution, should have created a figure
    assert len(plt.get_fignums()) > n_figs_before
