import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.tree_ovr import plot_tree_ovr


@pytest.fixture
def classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(30, 4),
        columns=[f'feature{i}' for i in range(4)]
    )
    labels = np.array([0]*10 + [1]*10 + [2]*10)
    return X, labels


def test_plot_tree_ovr_returns_dict(classification_data):
    """Test that plot_tree_ovr returns a dictionary."""
    X, labels = classification_data
    results = plot_tree_ovr(X, labels)
    assert isinstance(results, dict)
    assert len(results) > 0


def test_plot_tree_ovr_correct_number_of_trees(classification_data):
    """Test that the correct number of one-vs-rest trees are created."""
    X, labels = classification_data
    results = plot_tree_ovr(X, labels)
    # For 3 classes, there should be 3 one-vs-rest trees
    assert len(results) == 3


def test_plot_tree_ovr_result_structure(classification_data):
    """Test that each result entry has correct structure."""
    X, labels = classification_data
    results = plot_tree_ovr(X, labels)
    
    for label, value in results.items():
        assert 'tree' in value
        assert 'fig' in value
        assert 'ax' in value


def test_plot_tree_ovr_with_custom_max_depth(classification_data):
    """Test plot_tree_ovr with custom max_depth."""
    X, labels = classification_data
    results = plot_tree_ovr(X, labels, max_depth=3)
    assert len(results) == 3


def test_plot_tree_ovr_with_class_names(classification_data):
    """Test plot_tree_ovr with custom class names."""
    X, labels = classification_data
    class_names = {0: 'Class A', 1: 'Class B', 2: 'Class C'}
    results = plot_tree_ovr(X, labels, class_names=class_names)
    assert len(results) == 3


def test_plot_tree_ovr_models_are_trained(classification_data):
    """Test that the models are properly trained."""
    X, labels = classification_data
    results = plot_tree_ovr(X, labels)
    
    for result in results.values():
        tree = result['tree']
        # Check that model has been fitted
        assert hasattr(tree, 'tree_')
        assert tree.get_depth() > 0


def test_plot_tree_ovr_creates_figures(classification_data):
    """Test that figures are created for each class."""
    X, labels = classification_data
    results = plot_tree_ovr(X, labels)
    
    for result in results.values():
        fig = result['fig']
        ax = result['ax']
        assert fig is not None
        assert ax is not None
