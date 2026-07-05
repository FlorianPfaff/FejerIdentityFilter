# Fejér Identity Filter

This repository contains a PyRecEst-style implementation of a Fejér/Cesàro-reduced identity Fourier filter for circular and hypertoroidal recursive Bayesian estimation.

The implementation is intended as a lightweight experimental companion to PyRecEst's `HypertoroidalFourierFilter`. It keeps the computationally attractive identity Fourier representation, but replaces sharp coefficient truncation after multiplicative update steps by a Fejér reduction. For a one-dimensional coefficient vector with indices `k = -K, ..., K`, the retained coefficients are weighted by

```text
lambda_k = 1 - |k| / (K + 1).
```

For `d` hypertoroidal dimensions, separable product weights are used:

```text
lambda_k = prod_j (1 - |k_j| / (K_j + 1)).
```

This corresponds to convolution with a product Fejér kernel. If the untruncated trigonometric polynomial is nonnegative, this positive-kernel reduction preserves nonnegativity while keeping the identity-filter prediction step simple.

## Installation

```bash
pip install -e .
```

The package depends on PyRecEst and follows its filter/distribution conventions.

## Usage

```python
from pyrecest.backend import array
from pyrecest.distributions import WrappedNormalDistribution

from fejer_identity_filter import FejerIdentityFilter

f = FejerIdentityFilter((21,))
f.filter_state = WrappedNormalDistribution(array(1.0), array(0.4))

system_noise = WrappedNormalDistribution(array(0.0), array(0.2))
f.predict_identity(system_noise)

measurement_noise = WrappedNormalDistribution(array(0.0), array(0.6))
f.update_identity(measurement_noise, array([1.2]))

estimate = f.get_point_estimate()
```

## What is implemented

- `fejer_identity_filter.fejer_weights(shape)`: separable Fejér weights for centered Fourier coefficient tensors.
- `FejerHypertoroidalFourierDistribution`: an identity-transformed `HypertoroidalFourierDistribution` with Fejér reduction after coefficient-growth operations.
- `FejerIdentityFilter`: a PyRecEst-style hypertoroidal filter using the Fejér identity distribution.

The current implementation supports the identity transformation only. This is deliberate: the purpose of the filter is to keep the cheap identity-filter prediction step while replacing sharp truncation by Fejér/Cesàro reduction in the steps where coefficient support grows.

## Notes

The Fejér reduction is a positivity-preserving operator for nonnegative trigonometric polynomials. It is not a magic repair step for an already negative identity approximation; in that case it usually damps Gibbs-type oscillations, but a grid or certificate check is still needed if strict nonnegativity must be verified.
