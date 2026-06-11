"""Shared fixtures for the eval test suite; sets non-interactive matplotlib backend."""
import matplotlib
matplotlib.use("Agg")  # must come before any pyplot import

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")
