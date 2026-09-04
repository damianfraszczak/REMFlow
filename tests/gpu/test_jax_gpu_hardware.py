"""Physical-GPU validation; enable with ``REMFLOW_REQUIRE_GPU=1``."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from remflow import available_backends, remify, remstats, remstimate, resolve_backend

GPU_VALIDATION = os.environ.get("REMFLOW_REQUIRE_GPU") == "1"
pytestmark = pytest.mark.skipif(
    not GPU_VALIDATION,
    reason="physical GPU validation requires REMFLOW_REQUIRE_GPU=1",
)


def test_explicit_gpu_and_cpu_devices_are_both_selectable() -> None:
    inventory = available_backends()
    assert inventory["jax"]["available"] is True
    assert "gpu" in inventory["jax"]["devices"]
    assert "cpu" in inventory["jax"]["devices"]

    gpu = resolve_backend("jax:gpu")
    cpu = resolve_backend("jax:cpu")
    values = gpu.asarray([1.0, 2.0])

    assert gpu.device == "gpu"
    assert cpu.device == "cpu"
    assert str(values.dtype) == "float64"
    assert {device.platform for device in values.devices()} == {"gpu"}
    assert gpu.runtime_metadata["jax_enable_x64"] is True
    assert gpu.runtime_metadata["device_kind"]


def test_gpu_tied_mle_and_chunking_match_numpy() -> None:
    events = pd.DataFrame(
        {
            "time": [1, 2, 2, 3, 4, 5],
            "sender": ["A", "B", "C", "A", "C", "B"],
            "receiver": ["B", "A", "A", "C", "B", "C"],
        }
    )
    history = remify(events, actors=["A", "B", "C"], ordinal=True)
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity()",
        first=1,
    )

    reference = remstimate(history, statistics, backend="numpy")
    batched = remstimate(history, statistics, backend="jax:gpu")
    chunked = remstimate(
        history,
        statistics,
        backend="jax:gpu",
        riskset_chunk_size=2,
    )

    np.testing.assert_allclose(batched.coef, reference.coef, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(chunked.coef, reference.coef, rtol=1e-5, atol=1e-7)
    assert batched.log_likelihood == pytest.approx(reference.log_likelihood, rel=1e-8, abs=1e-9)
    assert chunked.log_likelihood == pytest.approx(
        reference.log_likelihood, rel=1e-8, abs=1e-9
    )
    assert batched.metadata["device"] == "gpu"
    assert batched.metadata["device_kind"]
    assert batched.metadata["jax_enable_x64"] is True
    assert chunked.metadata["riskset_chunk_size"] == 2
    assert chunked.metadata["jax_batched"] is False


def test_gpu_actor_hmc_matches_numpy_draws() -> None:
    events = pd.DataFrame(
        {
            "time": range(1, 9),
            "sender": ["A", "B", "C", "A", "C", "B", "A", "C"],
            "receiver": ["B", "A", "A", "C", "B", "C", "B", "A"],
        }
    )
    history = remify(events, actors=["A", "B", "C"], model="actor", ordinal=True)
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia()",
        first=2,
    )
    controls = {"nsim": 6, "burnin": 2, "nchains": 1, "thin": 1, "L": 2}

    reference = remstimate(
        history,
        statistics,
        approach="Bayesian",
        backend="numpy",
        bayes=controls,
        seed=29,
    )
    accelerated = remstimate(
        history,
        statistics,
        approach="Bayesian",
        backend="jax:gpu",
        riskset_chunk_size=2,
        bayes=controls,
        seed=29,
    )

    assert reference.sender_model is not None
    assert reference.receiver_model is not None
    assert accelerated.sender_model is not None
    assert accelerated.receiver_model is not None
    np.testing.assert_allclose(
        accelerated.sender_model.draws,
        reference.sender_model.draws,
        rtol=1e-7,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        accelerated.receiver_model.draws,
        reference.receiver_model.draws,
        rtol=1e-7,
        atol=1e-9,
    )
    assert accelerated.metadata["device"] == "gpu"
    assert accelerated.sender_model.metadata["hmc_gradient_backend"] == "jax"
    assert accelerated.receiver_model.metadata["hmc_gradient_backend"] == "jax"


def test_gpu_ordinal_duration_fit_matches_numpy() -> None:
    events = pd.DataFrame(
        {
            "time": [1, 2, 4, 4, 7, 9],
            "actor1": ["A", "B", "C", "A", "B", "C"],
            "actor2": ["B", "C", "A", "C", "A", "B"],
            "end": [4, 6, 8, 9, 10, 12],
        }
    )
    history = remify(events, duration=True, ordinal=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        first=1,
    )

    reference = remstimate(history, statistics, backend="numpy")
    accelerated = remstimate(
        history,
        statistics,
        backend="jax:gpu",
        riskset_chunk_size=2,
    )

    np.testing.assert_allclose(accelerated.coef, reference.coef, rtol=1e-6, atol=1e-7)
    assert accelerated.log_likelihood == pytest.approx(
        reference.log_likelihood, rel=1e-8, abs=1e-9
    )
    assert accelerated.metadata["device"] == "gpu"
    assert accelerated.metadata["optimizer_device"] == "cpu-reference"
    assert accelerated.metadata["riskset_chunk_size"] == 2
