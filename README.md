# Fejer Identity Filter

This repository contains a PyRecEst-style implementation of a Fejer/Cesaro-reduced identity Fourier filter for circular and hypertoroidal Bayesian filtering.

The package is designed as a small extension around PyRecEst's `HypertoroidalFourierDistribution` and `HypertoroidalFourierFilter` classes. It adds:

- `FejerHypertoroidalFourierDistribution`, an identity Fourier distribution whose coefficient reduction uses tensor-product Fejer weights instead of sharp truncation.
- `FejerIdentityFilter`, a hypertoroidal filter with the same identity-model interface as PyRecEst's Fourier filters.
- `examples/compare_with_fourier_filters.py`, a reproducible comparison against PyRecEst's ordinary Fourier identity filter (IFF) and Fourier square-root filter (SqFF).

## Motivation

The ordinary IFF represents the density directly by Fourier coefficients. This makes the additive-noise identity prediction step especially simple: the topology-aware convolution is a Hadamard product of identity coefficients. However, multiplication followed by sharp coefficient truncation can introduce negative density values.

The SqFF represents the square root of the density and therefore keeps the reconstructed density nonnegative. The cost is that the prediction step is more complicated because square roots do not commute with convolution.

The Fejer identity filter keeps the IFF representation and replaces sharp coefficient truncation after pointwise multiplication by a Fejer mean. For a one-dimensional coefficient vector with order `K`, the retained coefficient of index `k` is multiplied by

```text
lambda_k = 1 - |k| / (K + 1),   k = -K, ..., K.
```

For a hypertorus, the implementation uses the tensor-product weights

```text
lambda_k = prod_j (1 - |k_j| / (K_j + 1)).
```

This is equivalent to convolution with a nonnegative Fejer kernel. Hence, when the untruncated multiplication result is a nonnegative trigonometric polynomial, Fejer reduction is positivity-preserving. The trade-off is deliberate smoothing.

## Installation

For editable installation with pip:

```bash
python -m pip install -e .
```

For development with Poetry:

```bash
poetry install --with dev
```

## Basic usage

```python
import pyrecest.backend as backend
from pyrecest.distributions import WrappedNormalDistribution

from fejer_identity_filter import FejerIdentityFilter

n_coefficients = (33,)
filter_ = FejerIdentityFilter(n_coefficients)

process_noise = WrappedNormalDistribution(backend.array(0.0), backend.array(0.25))
measurement_noise = WrappedNormalDistribution(backend.array(0.0), backend.array(0.20))

filter_.predict_identity(process_noise)
filter_.update_identity(measurement_noise, backend.array([1.0]))
estimate = filter_.filter_state.mean_direction()
```

## Comparison with Fourier filters

Run the included comparison script from the repository root:

```bash
python examples/compare_with_fourier_filters.py --coefficients 5 9 15 25 33
```

The script compares:

- PyRecEst IFF: `HypertoroidalFourierFilter(..., transformation="identity")`
- Fejer identity filter: `FejerIdentityFilter`
- PyRecEst SqFF: `HypertoroidalFourierFilter(..., transformation="sqrt")`

It reports runtime per predict/update cycle, circular mean error against a dense-grid reference, integrated absolute density error, minimum PDF value on a diagnostic grid, and negative grid mass.

Qualitatively, the expected trade-off is:

| Filter | Representation | Reduction after update | Additive identity prediction | Nonnegativity behavior | Typical role |
|---|---|---|---|---|---|
| IFF | density coefficients | sharp truncation | cheapest Hadamard product | not guaranteed after truncation | fastest baseline |
| Fejer-IFF | density coefficients | Fejer/Cesaro smoothing | same identity-coefficient structure | positivity-preserving when reducing a nonnegative trigonometric polynomial | positivity-oriented identity variant |
| SqFF | square-root coefficients | square-root coefficient truncation | more expensive conversion through density | nonnegative by construction | strongest positivity baseline |

A representative run shows the intended behavior: the ordinary IFF is usually fastest but may have negative side lobes, SqFF is nonnegative but slower, and Fejer-IFF removes grid-level negativity while introducing smoothing bias.

## Tests

```bash
pytest
```

## Limitations

Fejer reduction is a positive linear operator applied to a nonnegative trigonometric polynomial. It is not, by itself, a certificate that an arbitrary identity coefficient tensor is globally nonnegative. For exact global certificates, one would need an additional nonnegative projection, sum-of-squares certificate, or spectral-factor representation.
