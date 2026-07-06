# Comparison with PyRecEst Fourier filters

The positive-kernel identity filter is meant to sit between PyRecEst's two Fourier filters for hypertoroidal densities, but not as a universal accuracy competitor to the square-root filter:

- `HypertoroidalFourierFilter(..., transformation="identity")`, i.e. the ordinary Fourier identity filter (IFF), represents the density directly. Prediction for additive identity dynamics is especially cheap because the topology-aware convolution is a Hadamard product in the identity coefficients.
- `HypertoroidalFourierFilter(..., transformation="sqrt")`, i.e. the Fourier square-root filter (SqFF), represents the square root of the density. This keeps the reconstructed density nonnegative, but prediction is more expensive because the square-root representation is not closed under convolution.
- Plain Fejer identity reduction keeps the identity representation and the cheap additive prediction step, but replaces sharp coefficient truncation after multiplication by unconditional Fejer/Cesaro smoothing.
- Adaptive Fejer-Korovkin identity reduction keeps sharp IFF truncation unless grid negativity is detected. If negativity appears, it activates a positive-kernel safeguard.

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
| Plain Fejer identity | preserved if reducing a nonnegative untruncated trigonometric polynomial | same cheap identity prediction when shapes match | unconditional first-order Fejer/Cesaro smoothing | historical baseline / strawman |
| Adaptive Fejer-Korovkin identity | grid-level safeguard; full FK step is positivity-preserving under the kernel assumptions | same cheap identity prediction when shapes match | sharp unless diagnostic grid goes negative; then FK damping/blending | practical IFF safeguard |
| SqFF | guaranteed by construction | more expensive | square-root multiplication and renormalization | strongest positivity baseline on smooth densities |

Plain Fejer reduction deliberately damps high-frequency coefficients and also damps the first mode at first order in the Fourier order. It usually reduces Gibbs-type oscillation and grid-level negativity compared with sharp truncation, but it pays a permanent smoothing tax. It should therefore not be used as the representative of the whole positive-kernel idea.

Fejer-Korovkin damping is a better positive-kernel baseline. Its first multiplier is `cos(pi/(K+2))`, so the low-frequency bias is second-order in the order `K`. It still should not beat SqFF asymptotically on smooth problems, but it is a more meaningful comparison than plain Fejer.

The adaptive version is the recommended practical variant: if sharp truncation is already nonnegative on the diagnostic grid, it behaves like the IFF. Only when negativity appears does it spend bias to clear the negative values.

## Important limitation

A full positive-kernel reduction is positivity-preserving as an operator applied to a nonnegative trigonometric polynomial. It is not a certificate that an arbitrary already-truncated identity coefficient tensor is globally nonnegative. The adaptive exponent search is even weaker: it is a grid-level safeguard, not an analytic certificate. Stronger guarantees would require analytic certificates or an additional nonnegative projection step, such as a sum-of-squares or Fejer-Riesz/spectral-factor approach.

## Benchmarking guidance

A wrapped-normal identity-model benchmark is SqFF's home turf because the density and its square root are smooth. To test whether identity coefficients plus adaptive positive-kernel safeguards have a practical advantage, include scenarios where the square-root representation is less favorable, such as deeply multimodal posteriors, interval-censoring likelihoods, likelihoods with zeros, or prediction-heavy pipelines with sparse updates.
