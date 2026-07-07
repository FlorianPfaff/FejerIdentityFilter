"""Compare identity and square-root Fourier updates for likelihoods with zeros.

This diagnostic is intentionally one-dimensional and self-contained. It isolates
one update step so that the approximation behavior of the representations is not
hidden by PyRecEst object overhead or by repeated prediction/update cycles.

Two likelihood scenarios are provided:

* ``smooth_zero`` uses ``L(x) = sin((x-z)/2)^2``. The posterior density remains
  smooth, but its square root has a cusp at the likelihood zero. This is a case
  where identity coefficients can be very favorable.
* ``hard_interval`` uses an interval-censoring gate. This produces a discontinuity
  in the posterior and is a stress test for every global Fourier representation.

The output metrics are density-level errors against a dense-grid reference. They
are not intended to replace the PyRecEst-style benchmark in
``compare_with_fourier_filters.py``; they complement it with cases that are less
favorable to square-root spectral convergence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from fejer_identity_filter.fejer import adaptive_kernel_reduce_coefficients, centered_coefficients, reduce_coefficients


@dataclass(frozen=True)
class ZeroLikelihoodResult:
    filter_name: str
    scenario: str
    n_coefficients: int
    runtime_ms_per_update: float
    circular_mean_error_rad: float
    integrated_abs_error: float
    min_pdf_on_grid: float
    negative_mass_on_grid: float
    adaptive_exponent: float | None


def _angle_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _centered_modes(n_coefficients: int) -> np.ndarray:
    order = (n_coefficients - 1) // 2
    return np.arange(-order, order + 1)


def _coefficients_from_values(values: np.ndarray, target_shape: tuple[int, ...] | None = None) -> np.ndarray:
    coeff = np.fft.fftshift(np.fft.fftn(values) / np.prod(values.shape))
    if target_shape is not None:
        coeff = centered_coefficients(coeff, target_shape)
    return coeff


def _evaluate_centered_1d(coefficients: np.ndarray, xs: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients)
    ks = _centered_modes(coefficients.size)
    return np.sum(coefficients[:, None] * np.exp(1j * ks[:, None] * xs[None, :]), axis=0).real


def _normalize_identity(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=complex)
    center = (coefficients.size - 1) // 2
    center_coeff = float(np.real(coefficients[center]))
    if abs(center_coeff) < 1e-300:
        raise ValueError("Cannot normalize identity coefficients with near-zero central coefficient.")
    return coefficients / (2.0 * math.pi * center_coeff)


def _normalize_sqrt(coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=complex)
    norm = np.linalg.norm(coefficients.ravel())
    if norm < 1e-300:
        raise ValueError("Cannot normalize square-root coefficients with near-zero norm.")
    return coefficients / (math.sqrt(2.0 * math.pi) * norm)


def _wrapped_normal_pdf_grid(grid: np.ndarray, mu: float, sigma: float, truncation: int = 8) -> np.ndarray:
    x = ((grid - mu + math.pi) % (2.0 * math.pi)) - math.pi
    normalizer = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    vals = np.zeros_like(grid, dtype=float)
    for k in range(-truncation, truncation + 1):
        vals += np.exp(-0.5 * ((x + 2.0 * math.pi * k) / sigma) ** 2)
    return normalizer * vals


def _wrapped_normal_identity_coefficients(n_coefficients: int, mu: float, sigma: float) -> np.ndarray:
    ks = _centered_modes(n_coefficients)
    return np.exp(-1j * ks * mu - 0.5 * sigma**2 * ks**2) / (2.0 * math.pi)


def _wrapped_normal_sqrt_coefficients(n_coefficients: int, mu: float, sigma: float) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * math.pi, n_coefficients, endpoint=False)
    return _coefficients_from_values(np.sqrt(_wrapped_normal_pdf_grid(grid, mu, sigma)), (n_coefficients,))


def _likelihood_values(grid: np.ndarray, scenario: str, z: float, interval_half_width: float) -> np.ndarray:
    distance = ((grid - z + math.pi) % (2.0 * math.pi)) - math.pi
    if scenario == "smooth_zero":
        return np.sin(0.5 * distance) ** 2
    if scenario == "hard_interval":
        return (np.abs(distance) <= interval_half_width).astype(float)
    raise ValueError(f"Unknown scenario {scenario!r}.")


def _likelihood_coefficients(n_coefficients: int, scenario: str, z: float, interval_half_width: float, *, transformation: str) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * math.pi, n_coefficients, endpoint=False)
    values = _likelihood_values(grid, scenario, z, interval_half_width)
    if transformation == "sqrt":
        values = np.sqrt(values)
    elif transformation != "identity":
        raise ValueError(f"Unknown transformation {transformation!r}.")
    return _coefficients_from_values(values, (n_coefficients,))


def _updated_coefficients(
    filter_name: str,
    n_coefficients: int,
    scenario: str,
    z: float,
    interval_half_width: float,
    prior_mean: float,
    prior_sigma: float,
    oversampling_factor: int,
) -> tuple[np.ndarray, float | None]:
    if filter_name == "IFF":
        prior = _wrapped_normal_identity_coefficients(n_coefficients, prior_mean, prior_sigma)
        likelihood = _likelihood_coefficients(n_coefficients, scenario, z, interval_half_width, transformation="identity")
        posterior = reduce_coefficients(np.convolve(prior, likelihood, mode="full"), (n_coefficients,), kernel="sharp")
        return _normalize_identity(posterior), 0.0

    if filter_name == "PlainFejerIdentityFilter":
        prior = _wrapped_normal_identity_coefficients(n_coefficients, prior_mean, prior_sigma)
        likelihood = _likelihood_coefficients(n_coefficients, scenario, z, interval_half_width, transformation="identity")
        posterior = reduce_coefficients(np.convolve(prior, likelihood, mode="full"), (n_coefficients,), kernel="fejer")
        return _normalize_identity(posterior), 1.0

    if filter_name == "AdaptiveFKIdentityFilter":
        prior = _wrapped_normal_identity_coefficients(n_coefficients, prior_mean, prior_sigma)
        likelihood = _likelihood_coefficients(n_coefficients, scenario, z, interval_half_width, transformation="identity")
        posterior, exponent = adaptive_kernel_reduce_coefficients(
            np.convolve(prior, likelihood, mode="full"),
            (n_coefficients,),
            kernel="korovkin",
            oversampling_factor=oversampling_factor,
            return_exponent=True,
        )
        return _normalize_identity(posterior), exponent

    if filter_name == "SqFF":
        prior = _wrapped_normal_sqrt_coefficients(n_coefficients, prior_mean, prior_sigma)
        likelihood = _likelihood_coefficients(n_coefficients, scenario, z, interval_half_width, transformation="sqrt")
        posterior = centered_coefficients(np.convolve(prior, likelihood, mode="full"), (n_coefficients,))
        return _normalize_sqrt(posterior), None

    raise ValueError(f"Unknown filter_name {filter_name!r}.")


def _pdf_from_coefficients(filter_name: str, coefficients: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = _evaluate_centered_1d(coefficients, grid)
    if filter_name == "SqFF":
        return values**2
    return values


def _circular_mean_from_grid(grid: np.ndarray, pdf_vals: np.ndarray) -> float:
    dx = 2.0 * math.pi / grid.size
    moment = np.sum(np.exp(1j * grid) * pdf_vals) * dx
    return float(np.angle(moment) % (2.0 * math.pi))


def _reference_density(scenario: str, grid: np.ndarray, z: float, interval_half_width: float, prior_mean: float, prior_sigma: float) -> tuple[np.ndarray, float]:
    dx = 2.0 * math.pi / grid.size
    pdf_ref = _wrapped_normal_pdf_grid(grid, prior_mean, prior_sigma) * _likelihood_values(grid, scenario, z, interval_half_width)
    pdf_ref = pdf_ref / (np.sum(pdf_ref) * dx)
    return pdf_ref, _circular_mean_from_grid(grid, pdf_ref)


def compare_zero_likelihood_filters(
    coefficient_counts: Iterable[int] = (9, 17, 33, 65),
    scenarios: Sequence[str] = ("smooth_zero", "hard_interval"),
    repetitions: int = 5,
    grid_size: int = 16384,
    prior_mean: float = 1.0,
    prior_sigma: float = 0.7,
    measurement: float = 1.3,
    interval_half_width: float = 0.8,
    oversampling_factor: int = 4,
) -> list[ZeroLikelihoodResult]:
    """Run the zero-likelihood comparison and return structured results."""

    coefficient_counts = tuple(int(n) for n in coefficient_counts)
    if any(n <= 0 or n % 2 != 1 for n in coefficient_counts):
        raise ValueError("All coefficient counts must be positive odd integers.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    grid = np.linspace(0.0, 2.0 * math.pi, grid_size, endpoint=False)
    dx = 2.0 * math.pi / grid_size
    results: list[ZeroLikelihoodResult] = []
    filter_names = ("IFF", "PlainFejerIdentityFilter", "AdaptiveFKIdentityFilter", "SqFF")

    for scenario in scenarios:
        pdf_ref, mean_ref = _reference_density(scenario, grid, measurement, interval_half_width, prior_mean, prior_sigma)
        for n_coefficients in coefficient_counts:
            for filter_name in filter_names:
                runtimes = []
                coefficients = None
                adaptive_exponent = None
                for _ in range(repetitions):
                    tic = time.perf_counter()
                    coefficients, adaptive_exponent = _updated_coefficients(
                        filter_name,
                        n_coefficients,
                        scenario,
                        measurement,
                        interval_half_width,
                        prior_mean,
                        prior_sigma,
                        oversampling_factor,
                    )
                    runtimes.append(time.perf_counter() - tic)
                assert coefficients is not None
                pdf_vals = _pdf_from_coefficients(filter_name, coefficients, grid)
                negative_vals = np.minimum(pdf_vals, 0.0)
                mean_est = _circular_mean_from_grid(grid, np.maximum(pdf_vals, 0.0))
                results.append(
                    ZeroLikelihoodResult(
                        filter_name=filter_name,
                        scenario=scenario,
                        n_coefficients=n_coefficients,
                        runtime_ms_per_update=statistics.median(runtimes) * 1000.0,
                        circular_mean_error_rad=_angle_distance(mean_est, mean_ref),
                        integrated_abs_error=float(np.sum(np.abs(pdf_vals - pdf_ref)) * dx),
                        min_pdf_on_grid=float(np.min(pdf_vals)),
                        negative_mass_on_grid=float(-np.sum(negative_vals) * dx),
                        adaptive_exponent=adaptive_exponent,
                    )
                )
    return results


def _markdown_table(results: Sequence[ZeroLikelihoodResult]) -> str:
    headers = ["scenario", "filter", "coeffs", "ms/update", "mean err [rad]", "IAE vs ref", "min pdf", "negative mass", "theta"]
    rows = [
        [
            result.scenario,
            result.filter_name,
            str(result.n_coefficients),
            f"{result.runtime_ms_per_update:.4f}",
            f"{result.circular_mean_error_rad:.4e}",
            f"{result.integrated_abs_error:.4e}",
            f"{result.min_pdf_on_grid:.4e}",
            f"{result.negative_mass_on_grid:.4e}",
            "" if result.adaptive_exponent is None else f"{result.adaptive_exponent:.4f}",
        ]
        for result in results
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def _csv_table(results: Sequence[ZeroLikelihoodResult]) -> str:
    headers = list(asdict(results[0]).keys()) if results else list(ZeroLikelihoodResult.__dataclass_fields__.keys())
    lines = [",".join(headers)]
    for result in results:
        row = asdict(result)
        lines.append(",".join(str(row[h]) for h in headers))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", nargs="+", type=int, default=[9, 17, 33, 65], help="Odd coefficient counts to compare.")
    parser.add_argument("--scenarios", nargs="+", choices=("smooth_zero", "hard_interval"), default=["smooth_zero", "hard_interval"], help="Zero-likelihood scenarios to run.")
    parser.add_argument("--repetitions", type=int, default=5, help="Runtime repetitions per filter/coefficient count.")
    parser.add_argument("--grid-size", type=int, default=16384, help="Dense reference grid size.")
    parser.add_argument("--prior-mean", type=float, default=1.0, help="Wrapped-normal prior mean in radians.")
    parser.add_argument("--prior-sigma", type=float, default=0.7, help="Wrapped-normal prior standard deviation.")
    parser.add_argument("--measurement", type=float, default=1.3, help="Center of the zero/gate likelihood in radians.")
    parser.add_argument("--interval-half-width", type=float, default=0.8, help="Half-width of the hard interval likelihood.")
    parser.add_argument("--oversampling-factor", type=int, default=4, help="Diagnostic FFT-grid oversampling for adaptive FK reduction.")
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown", help="Output format.")
    args = parser.parse_args(argv)

    results = compare_zero_likelihood_filters(
        coefficient_counts=args.coefficients,
        scenarios=args.scenarios,
        repetitions=args.repetitions,
        grid_size=args.grid_size,
        prior_mean=args.prior_mean,
        prior_sigma=args.prior_sigma,
        measurement=args.measurement,
        interval_half_width=args.interval_half_width,
        oversampling_factor=args.oversampling_factor,
    )

    if args.format == "json":
        print(json.dumps([asdict(result) for result in results], indent=2))
    elif args.format == "csv":
        print(_csv_table(results))
    else:
        print(_markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
