"""Fejer/Cesaro-reduced identity Fourier filtering for PyRecEst."""

from .fejer import apply_fejer_weights, centered_coefficients, fejer_reduce_coefficients, fejer_weights

__all__ = [
    "FejerIdentityFilter",
    "FejerHypertoroidalFourierDistribution",
    "apply_fejer_weights",
    "centered_coefficients",
    "fejer_reduce_coefficients",
    "fejer_weights",
]


def __getattr__(name):
    if name == "FejerIdentityFilter":
        from .filter import FejerIdentityFilter

        return FejerIdentityFilter
    if name == "FejerHypertoroidalFourierDistribution":
        from .distribution import FejerHypertoroidalFourierDistribution

        return FejerHypertoroidalFourierDistribution
    raise AttributeError(name)
