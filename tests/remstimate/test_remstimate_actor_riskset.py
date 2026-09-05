"""Actor-oriented risk-set estimator regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

from remflow import AIC, AICC, BIC, WAIC, remify, remstats, remstimate, stack_stats
from remflow.estimate import ActorRemEstimate


def _events() -> pd.DataFrame:
    """Return a deterministic non-separated five-actor history."""

    rng = np.random.default_rng(1847)
    senders = rng.integers(1, 6, size=60)
    receivers = rng.integers(1, 5, size=60)
    receivers += receivers >= senders
    return pd.DataFrame(
        {
            "time": np.cumsum(rng.integers(1, 4, size=60)),
            "actor1": senders,
            "actor2": receivers,
        }
    )


def _manual_riskset(events: pd.DataFrame) -> pd.DataFrame:
    observed = events[["actor1", "actor2"]]
    reversed_pairs = observed.rename(
        columns={"actor1": "actor2", "actor2": "actor1"}
    )[["actor1", "actor2"]]
    return pd.concat([observed, reversed_pairs], ignore_index=True).drop_duplicates()


def _problem(riskset: str):
    events = _events()
    kwargs = {}
    if riskset == "manual":
        kwargs["manual_riskset"] = _manual_riskset(events)
    history = remify(
        events,
        actors=[1, 2, 3, 4, 5],
        model="actor",
        riskset=riskset,
        **kwargs,
    )
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
        first=6,
        last=45,
    )
    return history, statistics


def _poisson_reference(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    design = frame[names].to_numpy(dtype=float)
    response = frame["obs"].to_numpy(dtype=float)
    exposure = np.exp(frame["log_interevent"].to_numpy(dtype=float))

    def objective(beta: np.ndarray) -> float:
        linear = design @ beta
        return float(-(response @ linear - np.sum(exposure * np.exp(linear))))

    result = minimize(objective, np.zeros(len(names)), method="BFGS", options={"gtol": 1e-9})
    assert result.success or np.linalg.norm(result.jac, ord=np.inf) < 2e-5
    return np.asarray(result.x, dtype=float)


def _clogit_reference(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    design = frame[names].to_numpy(dtype=float)
    response = frame["obs"].to_numpy(dtype=float)
    groups = [
        np.asarray(indexes, dtype=int)
        for indexes in frame.groupby("time_index", sort=False).indices.values()
    ]

    def objective(beta: np.ndarray) -> float:
        linear = design @ beta
        value = 0.0
        for indexes in groups:
            group_response = response[indexes]
            assert group_response.sum() == 1.0
            value += float(
                group_response @ linear[indexes] - logsumexp(linear[indexes])
            )
        return -value

    result = minimize(objective, np.zeros(len(names)), method="BFGS", options={"gtol": 1e-9})
    assert result.success or np.linalg.norm(result.jac, ord=np.inf) < 2e-5
    return np.asarray(result.x, dtype=float)


def test_active_actor_riskset_mle_contract_and_information_criteria():
    history, statistics = _problem("active")
    plain = remstimate(history, statistics, method="MLE")
    fitted = remstimate(
        history,
        statistics,
        method="MLE",
        WAIC=True,
        nsimWAIC=30,
        seed=23929,
    )

    assert isinstance(plain, ActorRemEstimate)
    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.metadata["approach"] == "Frequentist"
    assert list(fitted.to_dict()) == ["sender_model", "receiver_model"]
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.sender_model.names == ["baseline", "indegreeSender"]
    assert fitted.receiver_model.names == ["inertia", "reciprocity"]
    assert np.isfinite(fitted.sender_model.coef).all()
    assert np.isfinite(fitted.receiver_model.coef).all()
    assert all(np.isfinite(value) for value in (AIC(fitted), AICC(fitted), BIC(fitted)))
    assert np.isfinite(WAIC(fitted))


def test_active_actor_riskset_hmc_records_expected_metadata():
    history, statistics = _problem("active")
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

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.metadata["approach"] == "Bayesian"
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert fitted.sender_model.draws is not None
    assert fitted.receiver_model.draws is not None
    assert fitted.sender_model.draws.shape == (10, 2)
    assert fitted.receiver_model.draws.shape == (10, 2)


@pytest.mark.parametrize("riskset", ["manual", "active_saturated"])
def test_manual_and_active_saturated_actor_risksets_fit_with_waic(riskset: str):
    history, statistics = _problem(riskset)
    fitted = remstimate(
        history,
        statistics,
        method="MLE",
        WAIC=True,
        nsimWAIC=30,
        seed=23929,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.metadata["approach"] == "Frequentist"
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert np.isfinite(fitted.sender_model.log_likelihood)
    assert np.isfinite(fitted.receiver_model.log_likelihood)
    assert np.isfinite(WAIC(fitted))


def test_active_actor_riskset_matches_independent_poisson_and_clogit_fits():
    history, statistics = _problem("active")
    fitted = remstimate(history, statistics, method="MLE")
    stacked = stack_stats(statistics, history)

    assert isinstance(fitted, ActorRemEstimate)
    assert fitted.sender_model is not None
    assert fitted.receiver_model is not None
    assert stacked.sender_stack is not None
    assert stacked.receiver_stack is not None
    np.testing.assert_allclose(
        fitted.sender_model.coef,
        _poisson_reference(stacked.sender_stack, fitted.sender_model.names),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        fitted.receiver_model.coef,
        _clogit_reference(stacked.receiver_stack, fitted.receiver_model.names),
        rtol=1e-4,
        atol=1e-4,
    )
