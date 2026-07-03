import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dstoolkit.model.analysis.tree_ovo import plot_tree_ovo


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


def test_plot_tree_ovo_returns_dict(classification_data):
    """Test that plot_tree_ovo returns a dictionary."""
    X, labels = classification_data
    trees = plot_tree_ovo(X, labels)
    assert isinstance(trees, dict)
    assert len(trees) > 0


def test_plot_tree_ovo_correct_number_of_trees(classification_data):
    """Test that the correct number of one-vs-one tree pairs are created."""
    X, labels = classification_data
    trees = plot_tree_ovo(X, labels)
    # For 3 classes, there should be C(3,2) = 3 pairs
    assert len(trees) == 3


def test_plot_tree_ovo_tree_structure(classification_data):
    """Test that each tree entry has correct structure."""
    X, labels = classification_data
    trees = plot_tree_ovo(X, labels)
    
    for key, value in trees.items():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert 'model' in value
        assert 'fig' in value
        assert 'ax' in value


def test_plot_tree_ovo_with_custom_max_depth(classification_data):
    """Test plot_tree_ovo with custom max_depth."""
    X, labels = classification_data
    trees = plot_tree_ovo(X, labels, max_depth=3)
    assert len(trees) == 3


def test_plot_tree_ovo_with_custom_figsize(classification_data):
    """Test plot_tree_ovo with custom figure size."""
    X, labels = classification_data
    trees = plot_tree_ovo(X, labels, figsize=(12, 10))
    # Check that figures were created
    for tree_data in trees.values():
        assert tree_data['fig'] is not None


def test_plot_tree_ovo_with_class_names(classification_data):
    """Test plot_tree_ovo with custom class names."""
    X, labels = classification_data
    class_names = {0: 'Class A', 1: 'Class B', 2: 'Class C'}
    trees = plot_tree_ovo(X, labels, class_names=class_names)
    assert len(trees) == 3


def test_plot_tree_ovo_models_are_trained(classification_data):
    """Test that the models are properly trained."""
    X, labels = classification_data
    trees = plot_tree_ovo(X, labels)
    
    for tree_data in trees.values():
        model = tree_data['model']
        # Check that model has been fitted
        assert hasattr(model, 'tree_')
        assert model.get_depth() > 0
