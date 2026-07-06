"""Positive-kernel-reduced identity Fourier filtering for PyRecEst."""

from .fejer import (
    adaptive_kernel_reduce_coefficients,
    apply_fejer_weights,
    apply_kernel_weights,
    centered_coefficients,
    coefficient_grid_shape,
    fejer_reduce_coefficients,
    fejer_weights,
    korovkin_weights,
    minimum_on_fft_grid,
    positive_kernel_weights,
    reduce_coefficients,
    values_on_fft_grid,
)

__all__ = [
    "FejerIdentityFilter",
    "FejerHypertoroidalFourierDistribution",
    "adaptive_kernel_reduce_coefficients",
    "apply_fejer_weights",
    "apply_kernel_weights",
    "centered_coefficients",
    "coefficient_grid_shape",
    "fejer_reduce_coefficients",
    "fejer_weights",
    "korovkin_weights",
    "minimum_on_fft_grid",
    "positive_kernel_weights",
    "reduce_coefficients",
    "values_on_fft_grid",
]


def __getattr__(name):
    if name == "FejerIdentityFilter":
        from .filter import FejerIdentityFilter

        return FejerIdentityFilter
    if name == "FejerHypertoroidalFourierDistribution":
        from .distribution import FejerHypertoroidalFourierDistribution

        return FejerHypertoroidalFourierDistribution
    raise AttributeError(name)
