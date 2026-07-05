# Comparison with PyRecEst Fourier filters

The Fejér identity filter is meant to sit between PyRecEst's two Fourier filters for hypertoroidal densities:

- `HypertoroidalFourierFilter(..., transformation="identity")`, i.e. the ordinary Fourier identity filter (IFF), represents the density directly.  Prediction for additive identity dynamics is especially cheap because the topology-aware convolution is a Hadamard product in the identity coefficients.
- `HypertoroidalFourierFilter(..., transformation="sqrt")`, i.e. the Fourier square-root filter (SqFF), represents the square root of the density.  This keeps the reconstructed density nonnegative, but prediction is more expensive because the square root representation is not closed under convolution.
- `FejerIdentityFilter` keeps the identity representation and the cheap additive prediction step, but replaces sharp coefficient truncation after multiplication by separable Fejér/Cesàro coefficient reduction.

## What is being compared

The example script `examples/compare_with_fourier_filters.py` runs a one-dimensional circular identity-model scenario and compares:

1. runtime per predict/update cycle,
2. circular mean error against a high-resolution dense-grid reference,
3. integrated absolute density error against the same reference,
4. minimum reconstructed PDF value on a dense grid,
5. negative probability mass on that grid.

Run it from the repository root:

```bash
python examples/compare_with_fourier_filters.py --coefficients 5 9 15 25 33
```

Machine-readable output is also available:

```bash
python examples/compare_with_fourier_filters.py --format csv
python examples/compare_with_fourier_filters.py --format json
```

## Expected qualitative behavior

The expected trade-off is:

| filter | nonnegativity | additive identity prediction | update/reduction behavior | expected role |
|---|---:|---:|---|---|
| IFF | not guaranteed after truncation | cheapest | sharp truncation | fastest baseline |
| Fejer identity | preserved if reducing a nonnegative untruncated trigonometric polynomial | same cheap identity prediction when shapes match | Fejer/Cesaro smoothing instead of sharp truncation | positivity-oriented identity variant |
| SqFF | guaranteed by construction | more expensive | square-root multiplication and renormalization | strongest positivity baseline |

Fejer reduction deliberately damps high-frequency coefficients.  This usually reduces Gibbs-type oscillation and grid-level negativity compared with sharp truncation, but it also smooths sharp modes.  Consequently, it should not be expected to dominate both IFF and SqFF in every metric.  The intended comparison is whether the reduction provides a useful middle ground: substantially less negativity than IFF while retaining most of the simple identity-filter prediction structure.

## Important limitation

Fejer reduction is positivity-preserving as an operator applied to a nonnegative trigonometric polynomial.  It is not a certificate that an arbitrary already-truncated identity coefficient tensor is globally nonnegative.  The example therefore reports grid-level negativity metrics; stronger guarantees would require analytic certificates or an additional nonnegative projection step.
