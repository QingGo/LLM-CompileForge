import pytest


def pytest_runtest_setup(item):
    """Enforce 1s timeout on unit-marked tests at setup time."""
    if item.get_closest_marker("unit"):
        item.own_markers = list(item.own_markers)
        item.own_markers.append(pytest.mark.timeout(1))

