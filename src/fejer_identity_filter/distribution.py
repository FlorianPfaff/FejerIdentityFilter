"""Fejér-reduced identity Fourier distributions on hypertori."""

from __future__ import annotations

import copy
import warnings

import numpy as np
from scipy import signal

from pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution

from .fejer import apply_fejer_weights, centered_coefficients, normalize_coefficient_shape


class FejerHypertoroidalFourierDistribution(HypertoroidalFourierDistribution):
    """Identity Fourier distribution with Fejér/Cesàro coefficient reduction.

    The class intentionally supports only the ``"identity"`` transformation.
    Multiplication of two identity Fourier distributions increases coefficient
    support; this subclass uses a Fejér reduction when reducing that support
    back to the configured coefficient tensor shape.
    """

    def __init__(self, coeff_mat):
        super().__init__(coeff_mat, transformation="identity")

    @classmethod
    def from_fourier_distribution(
        cls,
        distribution: HypertoroidalFourierDistribution,
        n_coefficients: int | tuple[int, ...] | None = None,
        *,
        apply_fejer: bool = True,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Convert an identity HFD to a Fejér identity HFD."""

        if not isinstance(distribution, HypertoroidalFourierDistribution):
            raise TypeError("distribution must be a HypertoroidalFourierDistribution.")
        if distribution.transformation != "identity":
            raise ValueError("FejerHypertoroidalFourierDistribution requires identity coefficients.")

        if n_coefficients is None:
            n_coefficients = distribution.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=distribution.dim)

        result = cls(centered_coefficients(distribution.coeff_mat, n_coefficients))
        if apply_fejer:
            return result.fejer_reduce(n_coefficients)
        return result

    @classmethod
    def from_distribution(
        cls,
        distribution: AbstractHypertoroidalDistribution,
        n_coefficients: int | tuple[int, ...],
        *,
        apply_fejer: bool = True,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Approximate a hypertoroidal distribution in Fejér-reduced identity form."""

        if isinstance(distribution, HypertoroidalFourierDistribution):
            return cls.from_fourier_distribution(distribution, n_coefficients, apply_fejer=apply_fejer)
        if not isinstance(distribution, AbstractHypertoroidalDistribution):
            raise TypeError("distribution must be an AbstractHypertoroidalDistribution.")

        base = HypertoroidalFourierDistribution.from_distribution(distribution, n_coefficients, "identity")
        return cls.from_fourier_distribution(base, n_coefficients, apply_fejer=apply_fejer)

    @classmethod
    def from_function(
        cls,
        fun,
        n_coefficients: int | tuple[int, ...],
        *,
        apply_fejer: bool = True,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Construct a Fejér identity HFD by sampling a vectorized function."""

        base = HypertoroidalFourierDistribution.from_function(fun, n_coefficients, "identity")
        return cls.from_fourier_distribution(base, n_coefficients, apply_fejer=apply_fejer)

    @classmethod
    def from_function_values(
        cls,
        fvals,
        n_coefficients: int | tuple[int, ...] | None = None,
        *,
        already_transformed: bool = False,
        apply_fejer: bool = True,
    ) -> "FejerHypertoroidalFourierDistribution":
        """Construct a Fejér identity HFD from values on a regular grid."""

        base = HypertoroidalFourierDistribution.from_function_values(
            fvals,
            n_coefficients=n_coefficients,
            desired_transformation="identity",
            already_transformed=already_transformed,
        )
        return cls.from_fourier_distribution(base, n_coefficients or base.coeff_mat.shape, apply_fejer=apply_fejer)

    def fejer_reduce(self, n_coefficients: int | tuple[int, ...] | None = None) -> "FejerHypertoroidalFourierDistribution":
        """Center-crop/pad and apply separable Fejér weights.

        This is the Fejér analogue of sharp truncation. The zero-frequency
        coefficient is unchanged by the weights; the constructor normalizes the
        result if the input represented an unnormalized density.
        """

        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)
        coeff = centered_coefficients(self.coeff_mat, n_coefficients)
        coeff = apply_fejer_weights(coeff)
        return type(self)(coeff)

    def truncate(self, n_coefficients: int | tuple[int, ...], force_normalization: bool = False):
        """Return a distribution with the requested centered coefficient shape.

        If the shape changes, reduction is performed with Fejér weights instead
        of sharp truncation. If the shape is unchanged, only optional
        normalization is performed, mirroring PyRecEst's truncate semantics.
        """

        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)
        if tuple(self.coeff_mat.shape) == n_coefficients:
            result = copy.deepcopy(self)
            if force_normalization:
                result.normalize_in_place(warn_unnorm=False)
            return result
        return self.fejer_reduce(n_coefficients)

    def multiply(self, f2: HypertoroidalFourierDistribution, n_coefficients=None):
        """Pointwise multiplication followed by Fejér coefficient reduction."""

        self._validate_compatible_identity(f2, "multiply")
        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)

        conv = signal.fftconvolve(self.coeff_mat, f2.coeff_mat, mode="full")
        unnormalized = type(self)(conv)
        return unnormalized.fejer_reduce(n_coefficients)

    def convolve(self, f2: HypertoroidalFourierDistribution, n_coefficients=None):
        """Topology-aware convolution for additive noise in identity form.

        For equal coefficient shapes this keeps the ordinary IFF Hadamard-product
        prediction. Fejér reduction is only used to align differing coefficient
        shapes.
        """

        self._validate_compatible_identity(f2, "convolve")
        if n_coefficients is None:
            n_coefficients = self.coeff_mat.shape
        n_coefficients = normalize_coefficient_shape(n_coefficients, dim=self.dim)

        f1_aligned = self if tuple(self.coeff_mat.shape) == n_coefficients else self.fejer_reduce(n_coefficients)
        if tuple(f2.coeff_mat.shape) == n_coefficients:
            f2_aligned = f2
        else:
            f2_aligned = type(self).from_fourier_distribution(f2, n_coefficients, apply_fejer=True)

        c_conv = (2.0 * np.pi) ** self.dim * f1_aligned.coeff_mat * f2_aligned.coeff_mat
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Normalization:notNormalized")
            return type(self)(c_conv)

    def shift(self, shift_by):
        """Shift the distribution on the hypertorus and keep the Fejér subclass."""

        shifted = super().shift(shift_by)
        return type(self).from_fourier_distribution(shifted, shifted.coeff_mat.shape, apply_fejer=False)

    def _validate_compatible_identity(self, other, operation: str) -> None:
        if not isinstance(other, HypertoroidalFourierDistribution):
            raise TypeError(f"{operation}: other must be a HypertoroidalFourierDistribution.")
        if self.dim != other.dim:
            raise ValueError(f"{operation}: dimensions must match.")
        if self.transformation != "identity" or other.transformation != "identity":
            raise ValueError(f"{operation}: both distributions must use the identity transformation.")
