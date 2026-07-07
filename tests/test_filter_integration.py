import importlib

import numpy.testing as npt
import pytest

PACKAGE = "py" + "recest"


def _optional_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.skip("optional recursive-estimation dependency is not installed")


def test_filter_initializes_uniform_identity_state():
    backend = _optional_module(PACKAGE + ".backend")
    if backend.__backend_name__ in ("jax", "pytorch"):
        pytest.skip("not supported on this backend")

    from fejer_identity_filter import FejerHypertoroidalFourierDistribution, FejerIdentityFilter

    filt = FejerIdentityFilter((11,))
    assert isinstance(filt.filter_state, FejerHypertoroidalFourierDistribution)
    assert filt.filter_state.transformation == "identity"
    assert filt.filter_state.reduction_kernel == "korovkin"
    assert filt.filter_state.adaptive_reduction

    xs = backend.linspace(0.0, 2.0 * backend.pi, 256, endpoint=False)
    integral = float(filt.filter_state.pdf(xs).sum()) * float(2.0 * backend.pi / 256)
    npt.assert_allclose(integral, 1.0, atol=1e-4)


def test_predict_and_update_identity_1d_are_normalized():
    backend = _optional_module(PACKAGE + ".backend")
    if backend.__backend_name__ in ("jax", "pytorch"):
        pytest.skip("not supported on this backend")
    distributions = _optional_module(PACKAGE + ".distributions")

    from fejer_identity_filter import FejerIdentityFilter

    wrapped_normal = distributions.WrappedNormalDistribution
    filt = FejerIdentityFilter((15,))
    filt.filter_state = wrapped_normal(backend.array(1.0), backend.array(0.4))
    filt.predict_identity(wrapped_normal(backend.array(0.0), backend.array(0.2)))
    filt.update_identity(wrapped_normal(backend.array(0.0), backend.array(0.7)), backend.array([1.3]))

    xs = backend.linspace(0.0, 2.0 * backend.pi, 256, endpoint=False)
    integral = float(filt.filter_state.pdf(xs).sum()) * float(2.0 * backend.pi / 256)
    npt.assert_allclose(integral, 1.0, atol=1e-4)


def test_model_distribution_reduction_options_are_normalized():
    backend = _optional_module(PACKAGE + ".backend")
    if backend.__backend_name__ in ("jax", "pytorch"):
        pytest.skip("not supported on this backend")

    from fejer_identity_filter import FejerHypertoroidalFourierDistribution, FejerIdentityFilter

    coeff_mat = backend.zeros((7,), dtype=complex)
    coeff_mat[3] = 1.0 / (2.0 * backend.pi)
    model_distribution = FejerHypertoroidalFourierDistribution(coeff_mat, reduction_kernel="fejer", adaptive_reduction=False)
    filt = FejerIdentityFilter((7,), reduction_kernel="korovkin", adaptive_reduction=True)

    with pytest.warns(RuntimeWarning):
        converted = filt._as_fejer_identity_distribution(model_distribution, "conversion expected")

    assert converted.reduction_options == filt.reduction_options
    npt.assert_allclose(converted.coeff_mat, model_distribution.coeff_mat)
