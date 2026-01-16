import pytest
import numpy as np
from dstoolkit.metrics.scores import ks_score


def test_ks_score_raises_on_single_class():
    y_true = np.array([1, 1, 1])
    y_scores = np.array([0.2, 0.5, 0.9])

    with pytest.raises(ValueError, match="both positive"):
        ks_score(y_true, y_scores)

def test_ks_score_returns_valid_range():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.6, 0.9])

    ks = ks_score(y_true, y_scores)

    assert 0.0 <= ks <= 1.0