"""End-to-end workflows through the single public namespace."""

from __future__ import annotations

import numpy as np
import pandas as pd

from remflow import diagnostics, remify, remstats, remstimate
from remflow.estimate import ActorDiagnostics, ActorRemEstimate, Diagnostics, RemEstimate


def _events() -> pd.DataFrame:
    rng = np.random.default_rng(731)
    senders = rng.integers(1, 6, size=36)
    receivers = rng.integers(1, 5, size=36)
    receivers += receivers >= senders
    times = np.cumsum(rng.integers(1, 4, size=36))
    for index in range(5, 36, 5):
        times[index] = times[index - 1]
    return pd.DataFrame({"time": times, "actor1": senders, "actor2": receivers})


def test_tie_directed_interval_mle_pipeline_from_public_namespace():
    history = remify(_events(), actors=[1, 2, 3, 4, 5], model="tie")
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity()",
        first=6,
        last=30,
    )
    fitted = remstimate(history, statistics, method="MLE")
    result = diagnostics(fitted, history, statistics)

    assert isinstance(fitted, RemEstimate)
    assert isinstance(result, Diagnostics)
    assert fitted.model == "tie"
    assert fitted.ordinal is False
    assert fitted.approach == "Frequentist"
    assert result.plot(which=(1, 2)) is result


def test_actor_directed_interval_mle_pipeline_from_public_namespace():
    history = remify(_events(), actors=[1, 2, 3, 4, 5], model="actor")
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
        first=6,
        last=30,
    )
    fitted = remstimate(history, statistics, method="MLE")
    result = diagnostics(fitted, history, statistics)

    assert isinstance(fitted, ActorRemEstimate)
    assert isinstance(result, ActorDiagnostics)
    assert fitted.metadata["model"] == "actor"
    assert fitted.metadata["ordinal"] is False
    assert fitted.metadata["approach"] == "Frequentist"
    assert result.plot(which=(1, 2)) is result


def test_tie_undirected_ordinal_hmc_pipeline_from_public_namespace():
    history = remify(
        _events(),
        actors=[1, 2, 3, 4, 5],
        model="tie",
        directed=False,
        ordinal=True,
    )
    statistics = remstats(
        history,
        tie_effects="~ inertia()",
        first=6,
        last=30,
    )
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
    result = diagnostics(fitted, history, statistics)

    assert isinstance(fitted, RemEstimate)
    assert isinstance(result, Diagnostics)
    assert fitted.model == "tie"
    assert fitted.ordinal is True
    assert fitted.approach == "Bayesian"
    assert result.plot(which=(1, 2, 3, 4), object=fitted) is result


def test_actor_directed_ordinal_hmc_pipeline_from_public_namespace():
    history = remify(
        _events(),
        actors=[1, 2, 3, 4, 5],
        model="actor",
        ordinal=True,
    )
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
        first=6,
        last=30,
    )
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
    result = diagnostics(fitted, history, statistics)

    assert isinstance(fitted, ActorRemEstimate)
    assert isinstance(result, ActorDiagnostics)
    assert fitted.metadata["model"] == "actor"
    assert fitted.metadata["ordinal"] is True
    assert fitted.metadata["approach"] == "Bayesian"
    assert result.plot(which=(1, 2, 3, 4), object=fitted) is result
