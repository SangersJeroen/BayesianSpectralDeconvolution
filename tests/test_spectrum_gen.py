import numpy as np
import pytest

pytest.importorskip("numba")
from bayspecdec.spectrum_gen import step  # noqa: E402


def test_step_is_zero_below_and_height_from_the_edge_on():
    x = np.linspace(0.0, 10.0, 11)
    np.testing.assert_array_equal(step(x, 5.0, 3.0), [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3])


def test_step_edge_is_inclusive_and_handles_off_grid_edges_and_extremes():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(step(x, 1.5, 2.0), [0, 0, 2, 2])
    np.testing.assert_array_equal(
        step(x, -1.0, 2.0), [2, 2, 2, 2]
    )  # edge left of the grid
    np.testing.assert_array_equal(
        step(x, 9.0, 2.0), [0, 0, 0, 0]
    )  # edge right of the grid
    np.testing.assert_array_equal(step(x, 2.0, 0.0), [0, 0, 0, 0])  # zero height
    np.testing.assert_array_equal(
        step(x, 2.0, -1.5), [0, 0, -1.5, -1.5]
    )  # negative height


def test_step_does_not_mutate_input_and_works_for_integer_grids():
    x = np.arange(6)
    before = x.copy()
    np.testing.assert_array_equal(step(x, 3, 1), [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(x, before)
