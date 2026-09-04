import numpy as np
import pandas as pd
import pytest

from remflow import available_backends, remify, remstats, remstimate, resolve_backend

JAX_AVAILABLE = bool(available_backends().get("jax", {}).get("available"))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_cpu_exact_fit_matches_numpy_reference():
    history = remify(
        pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0],
                "sender": ["A", "B", "A"],
                "receiver": ["B", "A", "B"],
            }
        ),
        actors=["A", "B"],
    )
    stats = remstats(history, tie_effects="~ 1", first=2)

    numpy_fit = remstimate(history, stats, backend="numpy")
    jax_fit = remstimate(history, stats, backend="jax:cpu")

    np.testing.assert_allclose(jax_fit.coef, numpy_fit.coef, rtol=1e-8, atol=1e-9)
    assert jax_fit.log_likelihood == pytest.approx(numpy_fit.log_likelihood, rel=1e-9)
    assert jax_fit.metadata["device"] == "cpu"
    assert jax_fit.metadata["precision"] == "float64"
    assert jax_fit.metadata["jax_batched"] is True


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_riskset_chunking_matches_batched_and_numpy_objectives():
    history = remify(
        pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0, 4.0],
                "sender": ["A", "B", "C", "A"],
                "receiver": ["B", "A", "A", "C"],
            }
        ),
        actors=["A", "B", "C"],
    )
    stats = remstats(history, tie_effects="~ 1", first=2)

    reference = remstimate(history, stats, backend="numpy")
    batched = remstimate(history, stats, backend="jax:cpu")
    chunked = remstimate(
        history,
        stats,
        backend="jax:cpu",
        riskset_chunk_size=2,
    )

    np.testing.assert_allclose(chunked.coef, reference.coef, rtol=1e-7, atol=1e-9)
    assert chunked.log_likelihood == pytest.approx(reference.log_likelihood, rel=1e-9)
    assert chunked.log_likelihood == pytest.approx(batched.log_likelihood, rel=1e-10)
    assert chunked.metadata["riskset_chunk_size"] == 2
    assert chunked.metadata["jax_batched"] is False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_chunking_matches_exact_tied_ordinal_likelihood():
    history = remify(
        pd.DataFrame(
            {
                "time": [1, 2, 2, 3, 4],
                "sender": ["A", "B", "C", "A", "C"],
                "receiver": ["B", "A", "A", "C", "B"],
            }
        ),
        actors=["A", "B", "C"],
        ordinal=True,
    )
    stats = remstats(history, tie_effects="~ inertia() + reciprocity()", first=1)

    reference = remstimate(history, stats, backend="numpy")
    chunked = remstimate(
        history,
        stats,
        backend="jax:cpu",
        riskset_chunk_size=2,
    )

    np.testing.assert_allclose(chunked.coef, reference.coef, rtol=1e-5, atol=1e-7)
    assert chunked.log_likelihood == pytest.approx(reference.log_likelihood, rel=1e-8)
    assert chunked.metadata["riskset_chunk_size"] == 2


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_sampled_ordinal_fit_matches_numpy_weighted_likelihood():
    history = remify(
        pd.DataFrame(
            {
                "time": range(1, 7),
                "sender": ["A", "B", "C", "A", "C", "B"],
                "receiver": ["B", "A", "A", "C", "B", "C"],
            }
        ),
        actors=["A", "B", "C"],
        ordinal=True,
    )
    stats = remstats(
        history,
        tie_effects="~ inertia()",
        first=2,
        sampling=True,
        samp_num=3,
        seed=4,
    )

    reference = remstimate(history, stats, backend="numpy")
    accelerated = remstimate(history, stats, backend="jax:cpu")

    np.testing.assert_allclose(accelerated.coef, reference.coef, rtol=1e-6, atol=1e-8)
    assert accelerated.log_likelihood == pytest.approx(reference.log_likelihood, rel=1e-8)
    assert accelerated.metadata["case_control_sampling"] is True


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_backend_protocol_hessian_jit_scan_and_vmap():
    backend = resolve_backend("jax:cpu")

    def quadratic(value):
        return (value**2).sum()

    value = backend.asarray([1.0, 2.0])

    np.testing.assert_allclose(backend.to_numpy(backend.hessian(quadratic)(value)), 2 * np.eye(2))
    assert float(backend.jit(quadratic)(value)) == pytest.approx(5.0)
    carry, outputs = backend.scan(
        lambda state, item: (state + item, state + item), backend.asarray(0.0), value
    )
    assert float(carry) == pytest.approx(3.0)
    np.testing.assert_allclose(backend.to_numpy(outputs), [1.0, 3.0])
    np.testing.assert_allclose(backend.to_numpy(backend.vmap(lambda item: item**2)(value)), [1, 4])


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_hmc_uses_accelerated_gradients_and_matches_numpy_draws():
    events = pd.DataFrame(
        {
            "time": range(1, 9),
            "sender": ["A", "B", "C", "A", "C", "B", "A", "C"],
            "receiver": ["B", "A", "A", "C", "B", "C", "B", "A"],
        }
    )
    history = remify(events, actors=["A", "B", "C"], ordinal=True)
    stats = remstats(history, tie_effects="~ inertia()", first=2)
    controls = {"nsim": 8, "burnin": 2, "nchains": 1, "thin": 1, "L": 2}

    reference = remstimate(
        history,
        stats,
        approach="Bayesian",
        backend="numpy",
        bayes=controls,
        seed=17,
    )
    accelerated = remstimate(
        history,
        stats,
        approach="Bayesian",
        backend="jax:cpu",
        bayes=controls,
        seed=17,
    )

    np.testing.assert_allclose(accelerated.draws, reference.draws, rtol=1e-7, atol=1e-9)
    assert accelerated.metadata["hmc_gradient_backend"] == "jax"
    assert accelerated.metadata["device"] == "cpu"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")
def test_jax_actor_hmc_accelerates_both_choice_components():
    events = pd.DataFrame(
        {
            "time": range(1, 9),
            "sender": ["A", "B", "C", "A", "C", "B", "A", "C"],
            "receiver": ["B", "A", "A", "C", "B", "C", "B", "A"],
        }
    )
    history = remify(
        events,
        actors=["A", "B", "C"],
        model="actor",
        ordinal=True,
    )
    stats = remstats(
        history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia()",
        first=2,
    )
    fitted = remstimate(
        history,
        stats,
        approach="Bayesian",
        backend="jax:cpu",
        riskset_chunk_size=2,
        bayes={"nsim": 6, "burnin": 2, "nchains": 1, "thin": 1, "L": 2},
        seed=29,
    )

    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.metadata["riskset_chunk_size"] == 2
    assert fitted.sender_model.metadata["hmc_gradient_backend"] == "jax"
    assert fitted.receiver_model.metadata["hmc_gradient_backend"] == "jax"
    assert fitted.sender_model.draws is not None
    assert fitted.receiver_model.draws is not None
