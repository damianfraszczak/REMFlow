"""Exact-time actor-oriented estimator regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from remflow import AIC, AICC, BIC, WAIC, diagnostics, remify, remstats, remstimate
from remflow.estimate import ActorDiagnostics, ActorRemEstimate


def _events() -> pd.DataFrame:
    rng = np.random.default_rng(491)
    senders = rng.integers(1, 6, size=48)
    receivers = rng.integers(1, 5, size=48)
    receivers += receivers >= senders
    times = np.cumsum(rng.integers(1, 4, size=48))
    for index in range(5, 48, 5):
        times[index] = times[index - 1]
    return pd.DataFrame({"time": times, "actor1": senders, "actor2": receivers})


def _problem(*, ordinal: bool = False, riskset: str = "full", two_sender: bool = False):
    history = remify(
        _events(),
        actors=[1, 2, 3, 4, 5],
        model="actor",
        ordinal=ordinal,
        riskset=riskset,
    )
    sender = (
        "~ indegreeSender() + outdegreeSender()"
        if two_sender
        else "~ indegreeSender()"
    )
    statistics = remstats(
        history,
        sender_effects=sender,
        receiver_effects="~ inertia() + reciprocity()",
        first=6,
        last=40,
    )
    return history, statistics


def test_exact_actor_mle_component_contract_print_summary_and_information_criteria():
    history, statistics = _problem()
    fitted = remstimate(history, statistics, method="MLE")

    assert isinstance(fitted, ActorRemEstimate)
    assert list(fitted.to_dict()) == ["sender_model", "receiver_model"]
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    expected_fields = [
        "coefficients",
        "loglik",
        "gradient",
        "hessian",
        "vcov",
        "se",
        "residual.deviance",
        "AIC",
        "AICC",
        "BIC",
        "converged",
        "iterations",
        "df.null",
        "df.model",
        "df.residual",
        "null.deviance",
        "model.deviance",
    ]
    assert list(fitted.sender_model.to_dict()) == expected_fields
    assert list(fitted.receiver_model.to_dict()) == expected_fields
    assert fitted.metadata["model"] == "actor"
    assert fitted.metadata["ordinal"] is False
    assert fitted.metadata["method"] == "MLE"
    assert fitted.metadata["approach"] == "Frequentist"
    assert str(fitted)
    assert fitted.summary()
    assert all(np.isfinite(value) for value in (AIC(fitted), AICC(fitted), BIC(fitted)))
    with pytest.raises(ValueError, match="WAIC"):
        WAIC(fitted)


def test_exact_actor_mle_waic_diagnostics_plot_and_ordinal_variant():
    history, statistics = _problem()
    with pytest.warns(UserWarning, match="nsimWAIC"):
        fallback = remstimate(
            history,
            statistics,
            WAIC=True,
            nsimWAIC="text",
            seed=9,
        )
    assert fallback.metadata["nsimWAIC"] == 100
    fitted = remstimate(
        history,
        statistics,
        WAIC=True,
        nsimWAIC=30,
        seed=23929,
    )
    result = diagnostics(fitted, history, statistics)
    assert np.isfinite(WAIC(fitted))
    assert isinstance(result, ActorDiagnostics)
    assert result.plot(
        which=2,
        sender_effects="indegreeSender",
        receiver_effects="inertia",
    ) is result
    assert fitted.plot(reh=history, diagnostics=result, which=1) is result

    ordinal_history, ordinal_statistics = _problem(ordinal=True)
    ordinal = remstimate(ordinal_history, ordinal_statistics, method="MLE")
    assert ordinal.metadata["ordinal"] is True
    assert ordinal.sender_model is not None
    assert ordinal.sender_model.names == ["indegreeSender"]


def test_exact_actor_hmc_contract_diagnostics_chains_and_waic():
    history, statistics = _problem()
    fitted = remstimate(
        history,
        statistics,
        method="HMC",
        nchains=1,
        nsim=10,
        burnin=5,
        L=4,
        epsilon=0.002,
        seed=23929,
    )
    multi = remstimate(
        history,
        statistics,
        method="HMC",
        nchains=2,
        nsim=6,
        burnin=3,
        L=4,
        epsilon=0.002,
        seed=23929,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.metadata["approach"] == "Bayesian"
    expected_fields = [
        "coefficients",
        "post.mean",
        "vcov",
        "sd",
        "loglik",
        "draws",
        "df.null",
        "df.model",
        "df.residual",
    ]
    assert list(fitted.sender_model.to_dict()) == expected_fields
    assert list(fitted.receiver_model.to_dict()) == expected_fields
    result = diagnostics(multi, history, statistics)
    assert isinstance(result, ActorDiagnostics)
    trace = result.plot_data(which=4, object=multi)["sender.panel4"]
    assert trace["chain"].nunique() == 2
    for criterion in (AIC, AICC, BIC):
        with pytest.raises(ValueError, match="Frequentist"):
            criterion(fitted)
    with pytest.raises(ValueError, match="WAIC"):
        WAIC(fitted)

    with_waic = remstimate(
        history,
        statistics,
        method="HMC",
        nchains=1,
        nsim=8,
        burnin=3,
        L=4,
        epsilon=0.002,
        seed=23929,
        WAIC=True,
    )
    assert np.isfinite(WAIC(with_waic))


def test_active_actor_riskset_two_sender_effects_supports_mle_and_hmc():
    history, statistics = _problem(riskset="active", two_sender=True)
    mle = remstimate(history, statistics, method="MLE")
    hmc = remstimate(
        history,
        statistics,
        method="HMC",
        nchains=1,
        nsim=5,
        burnin=2,
        L=3,
        epsilon=0.002,
        seed=23929,
    )

    assert mle.sender_model is not None
    assert mle.sender_model.names == [
        "baseline",
        "indegreeSender",
        "outdegreeSender",
    ]
    assert diagnostics(mle, history, statistics).sender_model is not None
    assert hmc.sender_model is not None and hmc.sender_model.draws is not None
