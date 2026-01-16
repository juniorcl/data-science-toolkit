import pytest
import matplotlib.pyplot as plt

@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    yield
    plt.close("all")