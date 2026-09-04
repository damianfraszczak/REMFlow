"""Ordinal actor-oriented estimator regression tests."""

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


def _problem(*, riskset: str = "full", two_sender: bool = False):
    history = remify(
        _events(),
        actors=[1, 2, 3, 4, 5],
        model="actor",
        ordinal=True,
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


def test_ordinal_actor_mle_waic_component_contract_and_diagnostics():
    history, statistics = _problem()
    fitted = remstimate(
        history,
        statistics,
        method="MLE",
        WAIC=True,
        nsimWAIC=30,
        seed=23929,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.metadata["ordinal"] is True
    assert fitted.metadata["approach"] == "Frequentist"
    assert fitted.sender_model.names == ["indegreeSender"]
    assert fitted.receiver_model.names == ["inertia", "reciprocity"]
    assert len(fitted.sender_model.to_dict()) == 18
    assert len(fitted.receiver_model.to_dict()) == 18
    assert "WAIC" in fitted.sender_model.to_dict()
    assert "WAIC" in fitted.receiver_model.to_dict()
    assert all(np.isfinite(value) for value in (AIC(fitted), AICC(fitted), BIC(fitted)))
    assert np.isfinite(WAIC(fitted))
    result = diagnostics(fitted, history, statistics)
    assert isinstance(result, ActorDiagnostics)
    assert result.plot(which=(1, 2)) is result


def test_ordinal_actor_hmc_contract_diagnostics_and_information_criterion_errors():
    history, statistics = _problem()
    fitted = remstimate(
        history,
        statistics,
        method="HMC",
        nchains=2,
        nsim=8,
        burnin=4,
        L=4,
        epsilon=0.002,
        seed=23929,
        WAIC=True,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.metadata["approach"] == "Bayesian"
    assert len(fitted.sender_model.to_dict()) == 9
    assert len(fitted.receiver_model.to_dict()) == 9
    assert fitted.sender_model.draws is not None
    assert fitted.sender_model.draws.shape == (16, 1)
    result = diagnostics(fitted, history, statistics)
    assert isinstance(result, ActorDiagnostics)
    trace = result.plot_data(which=(3, 4), object=fitted)["sender.panel4"]
    assert trace["chain"].nunique() == 2
    assert np.isfinite(WAIC(fitted))
    for criterion in (AIC, AICC, BIC):
        with pytest.raises(ValueError, match="Frequentist"):
            criterion(fitted)


def test_ordinal_active_actor_riskset_supports_mle_hmc_and_two_sender_effects():
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
    assert mle.sender_model.names == ["indegreeSender", "outdegreeSender"]
    assert diagnostics(mle, history, statistics).sender_model is not None
    assert hmc.sender_model is not None
    assert hmc.sender_model.draws is not None
    assert hmc.sender_model.draws.shape == (5, 2)
