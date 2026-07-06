"""Utilities for centered Fourier coefficient tensors and Fejer/Cesaro weights."""

from __future__ import annotations

from numbers import Integral
from typing import Iterable

import numpy as np


CoefficientShape = int | Iterable[int]


def normalize_coefficient_shape(shape_like: CoefficientShape, *, dim: int | None = None, name: str = "n_coefficients") -> tuple[int, ...]:
    """Validate and normalize a centered Fourier coefficient shape.

    PyRecEst's hypertoroidal Fourier coefficients use odd side lengths so that
    the central tensor entry is the zero-frequency coefficient.  A scalar shape
    is interpreted as a one-dimensional shape unless ``dim`` is given; with
    ``dim`` it is broadcast to all dimensions.
    """

    if isinstance(shape_like, bool):
        raise ValueError(f"{name} must contain positive odd integers.")

    if isinstance(shape_like, Integral):
        values = (int(shape_like),) if dim is None else (int(shape_like),) * dim
    else:
        try:
            values = tuple(int(value) for value in shape_like)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer or an iterable of integers.") from exc

    if len(values) == 0:
        raise ValueError(f"{name} must contain at least one entry.")
    if dim is not None and len(values) != dim:
        raise ValueError(f"{name} must contain {dim} entries.")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} entries must be positive.")
    if any(value % 2 != 1 for value in values):
        raise ValueError(f"{name} entries must be odd in every dimension.")

    return values


def centered_coefficients(coefficients, target_shape: CoefficientShape):
    """Center-crop or center-pad a Fourier coefficient tensor.

    Parameters
    ----------
    coefficients : array-like
        Centered coefficient tensor. Every axis must have odd length.
    target_shape : int or iterable of int
        Desired centered tensor shape. Every axis must have odd length.
    """

    coeff_arr = np.asarray(coefficients)
    target_shape = normalize_coefficient_shape(target_shape, dim=coeff_arr.ndim, name="target_shape")
    current_shape = normalize_coefficient_shape(coeff_arr.shape, dim=coeff_arr.ndim, name="coefficients.shape")

    if current_shape == target_shape:
        return coeff_arr.copy()

    result = np.zeros(target_shape, dtype=coeff_arr.dtype)
    old_slices = []
    new_slices = []
    for old_len, new_len in zip(current_shape, target_shape):
        overlap = min(old_len, new_len)
        old_start = (old_len - overlap) // 2
        new_start = (new_len - overlap) // 2
        old_slices.append(slice(old_start, old_start + overlap))
        new_slices.append(slice(new_start, new_start + overlap))

    result[tuple(new_slices)] = coeff_arr[tuple(old_slices)]
    return result


def fejer_weights(shape: CoefficientShape, *, dtype=float):
    """Return separable Fejer/Cesaro weights for centered Fourier coefficients.

    For each side length ``n = 2*K + 1`` the one-dimensional weights are
    ``1 - abs(k)/(K + 1)`` for ``k = -K, ..., K``. For multidimensional
    tensors the returned weights are the tensor product of the one-dimensional
    weights.  The central weight is exactly one, so the zero-frequency
    coefficient and therefore the integral are unchanged.
    """

    shape = normalize_coefficient_shape(shape, name="shape")
    weights = np.ones(shape, dtype=dtype)
    for axis, side_length in enumerate(shape):
        order = (side_length - 1) // 2
        if order == 0:
            one_dim = np.ones((1,), dtype=dtype)
        else:
            ks = np.arange(-order, order + 1, dtype=dtype)
            one_dim = 1.0 - np.abs(ks) / (order + 1.0)
        reshape_shape = [1] * len(shape)
        reshape_shape[axis] = side_length
        weights = weights * one_dim.reshape(reshape_shape)
    return weights


def apply_fejer_weights(coefficients):
    """Apply separable Fejer weights to a centered coefficient tensor."""

    coeff_arr = np.asarray(coefficients)
    normalize_coefficient_shape(coeff_arr.shape, dim=coeff_arr.ndim, name="coefficients.shape")
    weights = fejer_weights(coeff_arr.shape, dtype=float)
    return coeff_arr * weights


def fejer_reduce_coefficients(coefficients, target_shape: CoefficientShape | None = None):
    """Reduce centered coefficients by center alignment followed by Fejer weights.

    This is the coefficient-space operation used by the Fejer identity filter in
    place of PyRecEst's sharp truncation.  If ``target_shape`` is smaller than
    the input shape, only the central low-frequency block is retained before the
    Fejer/Cesaro weights are applied.  If it is larger, the coefficient tensor is
    center-padded with zeros before weighting.
    """

    coeff_arr = np.asarray(coefficients)
    if target_shape is None:
        target_shape = coeff_arr.shape
    reduced = centered_coefficients(coeff_arr, target_shape)
    return apply_fejer_weights(reduced)
