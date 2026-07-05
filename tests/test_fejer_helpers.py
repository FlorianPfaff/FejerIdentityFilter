import numpy as np
import numpy.testing as npt
import pytest

from fejer_identity_filter.fejer import apply_fejer_weights, centered_coefficients, fejer_weights


def test_fejer_weights_1d():
    npt.assert_allclose(fejer_weights((5,)), np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3]))


def test_fejer_weights_product():
    w = fejer_weights((3, 5))
    expected = np.outer(np.array([0.5, 1.0, 0.5]), np.array([1 / 3, 2 / 3, 1.0, 2 / 3, 1 / 3]))
    npt.assert_allclose(w, expected)


def test_apply_fejer_weights_preserves_center():
    c = np.ones((5,), dtype=complex)
    weighted = apply_fejer_weights(c)
    assert weighted[2] == 1.0
    assert abs(weighted[0]) < abs(weighted[2])


def test_centered_coefficients_crop_and_pad():
    c = np.arange(7)
    npt.assert_array_equal(centered_coefficients(c, (3,)), np.array([2, 3, 4]))
    npt.assert_array_equal(centered_coefficients(c, (9,)), np.array([0, 0, 1, 2, 3, 4, 5, 6, 0]))


def test_even_shape_rejected():
    with pytest.raises(ValueError):
        fejer_weights((4,))
