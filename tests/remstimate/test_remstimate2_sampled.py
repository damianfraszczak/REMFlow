"""Sampled-risk-set estimator regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from remflow import RemEstimate, remify, remstimate, tomstats

EFFECTS = (
    "~ inertia(consider_type=FALSE) + indegreeSender(consider_type=FALSE) + "
    "outdegreeSender(consider_type=FALSE)"
)


def _history():
    rng = np.random.default_rng(1847)
    senders = rng.integers(1, 6, size=60)
    receivers = rng.integers(1, 5, size=60)
    receivers += receivers >= senders
    events = pd.DataFrame(
        {
            "time": np.cumsum(rng.integers(1, 4, size=60)),
            "actor1": senders,
            "actor2": receivers,
            "type": np.where(np.arange(60) % 3 == 0, "work", "social"),
        }
    )
    return remify(events, actors=[1, 2, 3, 4, 5], model="tie", riskset="active")


def _statistics(*, sampling: bool, samp_num: int = 10, seed: int | None = None):
    history = _history()
    statistics = tomstats(
        EFFECTS,
        reh=history,
        memory="decay",
        memory_value=1000,
        first=6,
        last=45,
        sampling=sampling,
        samp_num=samp_num,
        seed=seed,
    )
    return history, statistics


def test_full_sampled_mle_contract_and_full_riskset_sample():
    history, full_stats = _statistics(sampling=False)
    full = remstimate(history, full_stats, method="MLE")

    assert isinstance(full, RemEstimate)
    assert full.approach == "Frequentist"
    assert full.method == "MLE"
    assert full.sampled is False
    assert full.names == [
        "baseline",
        "inertia",
        "indegreeSender",
        "outdegreeSender",
    ]
    assert full.covariance is not None
    assert np.all(np.linalg.eigvalsh(full.covariance) > 0)
    assert np.isfinite(full.log_likelihood) and full.log_likelihood < 0
    assert np.isfinite(full.AIC) and np.isfinite(full.BIC)
    assert full.converged

    sample_size = len(history.risksets[0])
    _, sampled_stats = _statistics(sampling=True, samp_num=sample_size, seed=1)
    sampled = remstimate(sampled_stats.history, sampled_stats, method="MLE")
    assert isinstance(sampled, RemEstimate)
    assert sampled.sampled is True
    np.testing.assert_array_equal(np.sign(sampled.coef), np.sign(full.coef))
    assert np.max(np.abs(sampled.coef - full.coef)) < 5
    assert np.isfinite(sampled.log_likelihood)


def test_partial_sampled_mle_structure_uncertainty_and_seed_semantics():
    history, full_stats = _statistics(sampling=False)
    full = remstimate(history, full_stats, method="MLE")
    _, sampled_stats = _statistics(sampling=True, samp_num=5, seed=42)
    sampled = remstimate(sampled_stats.history, sampled_stats, method="MLE")
    _, alternate_stats = _statistics(sampling=True, samp_num=5, seed=99)
    alternate = remstimate(alternate_stats.history, alternate_stats, method="MLE")

    assert sampled.sampled is True
    assert sampled.names == full.names
    np.testing.assert_array_equal(np.sign(sampled.coef), np.sign(full.coef))
    assert sampled.se is not None and full.se is not None
    assert np.all(sampled.se >= full.se * 0.9)
    assert np.isfinite(sampled.log_likelihood)
    expected_fields = {
        "coefficients",
        "loglik",
        "gradient",
        "hessian",
        "vcov",
        "se",
        "residual.deviance",
        "null.deviance",
        "model.deviance",
        "df.null",
        "df.model",
        "df.residual",
        "AIC",
        "AICC",
        "BIC",
        "converged",
        "iterations",
        "sampled",
        "samp_num",
        "sampling_scheme",
    }
    assert set(sampled.to_dict()) == expected_fields
    assert sampled.df_model == 4
    assert sampled.metadata["samp_num"] == 5
    assert not np.array_equal(sampled.coef, alternate.coef)
    np.testing.assert_array_equal(np.sign(alternate.coef), np.sign(full.coef))


def test_typed_separate_sampled_mle_preserves_coefficient_names():
    history = _history()
    effects = (
        '~ inertia(consider_type="separate") + '
        "outdegreeSender(consider_type=FALSE)"
    )
    full_stats = tomstats(
        effects,
        reh=history,
        memory="decay",
        memory_value=1000,
        first=6,
        last=45,
    )
    sampled_stats = tomstats(
        effects,
        reh=history,
        memory="decay",
        memory_value=1000,
        first=6,
        last=45,
        sampling=True,
        samp_num=5,
        seed=7,
    )
    full = remstimate(history, full_stats, method="MLE")
    sampled = remstimate(history, sampled_stats, method="MLE")

    assert isinstance(full, RemEstimate)
    assert isinstance(sampled, RemEstimate)
    assert "inertia.social" in full.names
    assert "inertia.work" in full.names
    assert sampled.names == full.names


def test_full_statistics_hmc_draws_and_posterior_mean_are_near_mle():
    history, statistics = _statistics(sampling=False)
    mle = remstimate(history, statistics, method="MLE")
    hmc = remstimate(
        history,
        statistics,
        method="HMC",
        nsim=200,
        burnin=100,
        thin=5,
        L=20,
        epsilon=0.002,
        seed=1,
    )

    assert isinstance(hmc, RemEstimate)
    assert hmc.approach == "Bayesian"
    assert hmc.method == "HMC"
    assert hmc.posterior_mean is not None
    assert np.max(np.abs(hmc.posterior_mean - mle.coef)) < 1.0
    assert hmc.draws is not None
    assert hmc.draws.shape[1] == 4


def test_remstimate_rejects_non_history_and_non_statistics_inputs():
    history, statistics = _statistics(sampling=False)
    with pytest.raises(TypeError, match="EventHistory"):
        remstimate([], statistics)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RemStats"):
        remstimate(history, np.ones((3, 3, 3)))  # type: ignore[arg-type]
