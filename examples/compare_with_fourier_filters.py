"""Compare positive-kernel identity filters with PyRecEst's Fourier filters.

The scenario is a one-dimensional circular tracking problem with an identity
system model and identity measurement model. A dense-grid Bayesian recursion is
used as the numerical reference for density-level errors.

Run from the repository root with, for example:

    python examples/compare_with_fourier_filters.py --coefficients 5 9 15 25 33

The script prints a Markdown table by default. Use ``--format csv`` or
``--format json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np

from fejer_identity_filter import FejerHypertoroidalFourierDistribution, FejerIdentityFilter
from pyrecest.backend import array
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution import HypertoroidalFourierDistribution
from pyrecest.filters import HypertoroidalFourierFilter


@dataclass(frozen=True)
class ComparisonResult:
    filter_name: str
    n_coefficients: int
    runtime_ms_per_cycle: float
    circular_mean_error_rad: float
    integrated_abs_error: float
    min_pdf_on_grid: float
    negative_mass_on_grid: float


POSITIVE_KERNEL_FILTERS = ("PlainFejerIdentityFilter", "AdaptiveFKIdentityFilter")


def _angle_distance(a: float, b: float) -> float:
    """Return the unsigned circular distance in radians."""

    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def _positive_kernel_options(filter_name: str):
    if filter_name == "PlainFejerIdentityFilter":
        return {"reduction_kernel": "fejer", "adaptive_reduction": False}
    if filter_name == "AdaptiveFKIdentityFilter":
        return {"reduction_kernel": "korovkin", "adaptive_reduction": True}
    raise ValueError(f"Unknown positive-kernel filter {filter_name!r}.")


def _make_filter(filter_name: str, n_coefficients: int):
    if filter_name in POSITIVE_KERNEL_FILTERS:
        return FejerIdentityFilter((n_coefficients,), **_positive_kernel_options(filter_name))
    if filter_name == "IFF":
        return HypertoroidalFourierFilter((n_coefficients,), transformation="identity")
    if filter_name == "SqFF":
        return HypertoroidalFourierFilter((n_coefficients,), transformation="sqrt")
    raise ValueError(f"Unknown filter_name {filter_name!r}.")


def _as_fourier_model_distribution(filter_name: str, distribution, n_coefficients: int):
    """Approximate an i.i.d. model noise once, as in the Fourier filters.

    For positive-kernel identity filters, the initial model-density
    approximation is not damped. This isolates positive-kernel reduction to
    coefficient-growth steps such as posterior multiplication.
    """

    shape = (n_coefficients,)
    if filter_name in POSITIVE_KERNEL_FILTERS:
        return FejerHypertoroidalFourierDistribution.from_distribution(distribution, shape, apply_fejer=False, **_positive_kernel_options(filter_name))
    transformation = "identity" if filter_name == "IFF" else "sqrt"
    return HypertoroidalFourierDistribution.from_distribution(distribution, shape, transformation)


def _as_fourier_prior(filter_name: str, distribution, n_coefficients: int):
    shape = (n_coefficients,)
    if filter_name in POSITIVE_KERNEL_FILTERS:
        return FejerHypertoroidalFourierDistribution.from_distribution(distribution, shape, apply_fejer=False, **_positive_kernel_options(filter_name))
    transformation = "identity" if filter_name == "IFF" else "sqrt"
    return HypertoroidalFourierDistribution.from_distribution(distribution, shape, transformation)


def _run_filter(filter_name: str, n_coefficients: int, measurements: Sequence[float], repetitions: int):
    prior = WrappedNormalDistribution(array(1.0), array(0.8))
    process_noise = WrappedNormalDistribution(array(0.0), array(0.25))
    measurement_noise = WrappedNormalDistribution(array(0.0), array(0.35))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        d_prior = _as_fourier_prior(filter_name, prior, n_coefficients)
        d_sys = _as_fourier_model_distribution(filter_name, process_noise, n_coefficients)
        d_meas = _as_fourier_model_distribution(filter_name, measurement_noise, n_coefficients)

        runtimes = []
        filt = None
        for _ in range(repetitions):
            filt = _make_filter(filter_name, n_coefficients)
            filt.filter_state = d_prior
            tic = time.perf_counter()
            for z in measurements:
                filt.predict_identity(d_sys)
                filt.update_identity(d_meas, array([z]))
            runtimes.append((time.perf_counter() - tic) / len(measurements))
    assert filt is not None
    return filt, statistics.median(runtimes)


def _pdf_on_grid(distribution, grid: np.ndarray) -> np.ndarray:
    vals = np.asarray(distribution.pdf(array(grid)), dtype=float)
    return vals.reshape(grid.shape)


def _circular_mean_from_grid(grid: np.ndarray, pdf_vals: np.ndarray) -> float:
    dx = 2.0 * math.pi / grid.size
    moment = np.sum(np.exp(1j * grid) * pdf_vals) * dx
    return float(np.angle(moment) % (2.0 * math.pi))


def _wrapped_normal_pdf_grid(grid: np.ndarray, mu: float, sigma: float, truncation: int = 7) -> np.ndarray:
    """Evaluate a scalar wrapped normal density on a grid."""

    x = ((grid - mu + math.pi) % (2.0 * math.pi)) - math.pi
    normalizer = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    vals = np.zeros_like(grid, dtype=float)
    for k in range(-truncation, truncation + 1):
        vals += np.exp(-0.5 * ((x + 2.0 * math.pi * k) / sigma) ** 2)
    return normalizer * vals


def _reference_density(measurements: Sequence[float], grid: np.ndarray):
    """Dense-grid reference using periodic convolution and pointwise updates."""

    dx = 2.0 * math.pi / grid.size
    pdf_ref = _wrapped_normal_pdf_grid(grid, mu=1.0, sigma=0.8)
    process_noise = _wrapped_normal_pdf_grid(grid, mu=0.0, sigma=0.25)

    for z in measurements:
        # Circular convolution: pred[k] approximates sum_j prior[j] noise[k-j] dx.
        pdf_ref = np.fft.ifft(np.fft.fft(pdf_ref) * np.fft.fft(process_noise)).real * dx
        likelihood = _wrapped_normal_pdf_grid(z - grid, mu=0.0, sigma=0.35)
        pdf_ref = pdf_ref * likelihood
        pdf_ref = np.maximum(pdf_ref, 0.0)
        pdf_ref = pdf_ref / (np.sum(pdf_ref) * dx)

    mean_ref = _circular_mean_from_grid(grid, pdf_ref)
    return pdf_ref, mean_ref


def compare_filters(
    coefficient_counts: Iterable[int] = (5, 9, 15, 25, 33),
    measurements: Sequence[float] = (1.15, 1.55, 1.9, 2.15),
    repetitions: int = 5,
    grid_size: int = 4096,
) -> list[ComparisonResult]:
    """Run the comparison and return structured results."""

    coefficient_counts = tuple(int(n) for n in coefficient_counts)
    if any(n <= 0 or n % 2 != 1 for n in coefficient_counts):
        raise ValueError("All coefficient counts must be positive odd integers.")

    grid = np.linspace(0.0, 2.0 * math.pi, grid_size, endpoint=False)
    dx = 2.0 * math.pi / grid_size
    pdf_ref, mean_ref = _reference_density(measurements, grid)

    results: list[ComparisonResult] = []
    for n_coefficients in coefficient_counts:
        for filter_name in ("IFF", "PlainFejerIdentityFilter", "AdaptiveFKIdentityFilter", "SqFF"):
            filt, runtime_per_cycle = _run_filter(filter_name, n_coefficients, measurements, repetitions)
            pdf_vals = _pdf_on_grid(filt.filter_state, grid)
            mean_est = _circular_mean_from_grid(grid, np.maximum(pdf_vals, 0.0))
            negative_vals = np.minimum(pdf_vals, 0.0)
            results.append(
                ComparisonResult(
                    filter_name=filter_name,
                    n_coefficients=n_coefficients,
                    runtime_ms_per_cycle=runtime_per_cycle * 1000.0,
                    circular_mean_error_rad=_angle_distance(mean_est, mean_ref),
                    integrated_abs_error=float(np.sum(np.abs(pdf_vals - pdf_ref)) * dx),
                    min_pdf_on_grid=float(np.min(pdf_vals)),
                    negative_mass_on_grid=float(-np.sum(negative_vals) * dx),
                )
            )
    return results


def _markdown_table(results: Sequence[ComparisonResult]) -> str:
    headers = [
        "filter",
        "coeffs",
        "ms/cycle",
        "mean err [rad]",
        "IAE vs ref",
        "min pdf",
        "negative mass",
    ]
    rows = [
        [
            result.filter_name,
            str(result.n_coefficients),
            f"{result.runtime_ms_per_cycle:.4f}",
            f"{result.circular_mean_error_rad:.4e}",
            f"{result.integrated_abs_error:.4e}",
            f"{result.min_pdf_on_grid:.4e}",
            f"{result.negative_mass_on_grid:.4e}",
        ]
        for result in results
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    line = "| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def _csv_table(results: Sequence[ComparisonResult]) -> str:
    headers = list(asdict(results[0]).keys()) if results else list(ComparisonResult.__dataclass_fields__.keys())
    lines = [",".join(headers)]
    for result in results:
        row = asdict(result)
        lines.append(",".join(str(row[h]) for h in headers))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coefficients", nargs="+", type=int, default=[5, 9, 15, 25, 33], help="Odd coefficient counts to compare.")
    parser.add_argument("--measurements", nargs="+", type=float, default=[1.15, 1.55, 1.9, 2.15], help="Synthetic scalar measurements in radians.")
    parser.add_argument("--repetitions", type=int, default=5, help="Runtime repetitions per filter/coefficient count.")
    parser.add_argument("--grid-size", type=int, default=4096, help="Grid size for density metrics.")
    parser.add_argument("--format", choices=("markdown", "csv", "json"), default="markdown", help="Output format.")
    args = parser.parse_args(argv)

    with contextlib.redirect_stderr(None):
        results = compare_filters(
            coefficient_counts=args.coefficients,
            measurements=args.measurements,
            repetitions=args.repetitions,
            grid_size=args.grid_size,
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
