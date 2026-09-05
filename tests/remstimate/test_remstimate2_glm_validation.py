"""Independent GLM and conditional-logit checks.

The reference fits below consume the public stacked representation and use a
separate scalar objective.  They therefore validate both ``stack_stats`` and
the estimator without calling REMFlow's internal likelihood helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import logsumexp

from remflow import aomstats, remify, remstats, remstimate, stack_stats
from remflow.estimate import ActorRemEstimate

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "history_glm_validation.json"
)


def _fixture() -> dict[str, Any]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["schema"] == "remflow.test_fixture"
    assert metadata["schema_version"] == 1
    return payload


def _events() -> pd.DataFrame:
    """Return the 40-event fixture used for estimator validation."""

    return pd.DataFrame(_fixture()["history"])


def _actor_attributes() -> pd.DataFrame:
    """Return actor covariates paired with the event fixture."""

    return pd.DataFrame(_fixture()["info"])


def _independent_poisson_fit(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    design = frame[names].to_numpy(dtype=float)
    response = frame["obs"].to_numpy(dtype=float)
    exposure = np.exp(frame["log_interevent"].to_numpy(dtype=float))
    weights = (
        frame["weight"].to_numpy(dtype=float)
        if "weight" in frame
        else np.ones(len(frame), dtype=float)
    )

    def objective(beta: np.ndarray) -> float:
        linear = design @ beta
        return float(-(response @ linear - np.sum(exposure * weights * np.exp(linear))))

    result = minimize(objective, np.zeros(len(names)), method="BFGS", options={"gtol": 1e-9})
    assert result.success or np.linalg.norm(result.jac, ord=np.inf) < 2e-5
    return np.asarray(result.x, dtype=float)


def _independent_clogit_fit(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    design = frame[names].to_numpy(dtype=float)
    response = frame["obs"].to_numpy(dtype=float)
    weights = (
        frame["weight"].to_numpy(dtype=float)
        if "weight" in frame
        else np.ones(len(frame), dtype=float)
    )
    groups = [
        np.asarray(indexes, dtype=int)
        for indexes in frame.groupby("time_index", sort=False).indices.values()
    ]

    def objective(beta: np.ndarray) -> float:
        linear = design @ beta
        log_likelihood = 0.0
        for indexes in groups:
            group_response = response[indexes]
            assert group_response.sum() == 1.0
            group_linear = linear[indexes]
            log_likelihood += float(
                group_response @ group_linear
                - logsumexp(group_linear + np.log(weights[indexes]))
            )
        return -log_likelihood

    result = minimize(objective, np.zeros(len(names)), method="BFGS", options={"gtol": 1e-9})
    assert result.success or np.linalg.norm(result.jac, ord=np.inf) < 2e-5
    return np.asarray(result.x, dtype=float)


def _fit_and_compare(
    *,
    ordinal: bool,
    riskset: str,
    effects: str,
    sampling: bool = False,
) -> tuple[list[str], np.ndarray]:
    history = remify(
        _events(),
        model="tie",
        ordinal=ordinal,
        riskset=riskset,
    )
    with pytest.warns(DeprecationWarning, match="attr_actors"):
        statistics = remstats(
            history,
            tie_effects=effects,
            memory="decay",
            memory_value=1000,
            first=2,
            last=30,
            sampling=sampling,
            samp_num=10,
            seed=1,
            attr_actors=_actor_attributes(),
        )
    fitted = remstimate(history, statistics, method="MLE")
    frame = stack_stats(statistics, history).remstats_stack
    reference = (
        _independent_clogit_fit(frame, fitted.names)
        if ordinal
        else _independent_poisson_fit(frame, fitted.names)
    )
    np.testing.assert_allclose(fitted.coef, reference, rtol=1e-4, atol=1e-4)
    return fitted.names, fitted.coef


@pytest.mark.parametrize("riskset", ["active", "full"])
def test_interval_mle_matches_independent_poisson_glm(riskset: str):
    names, coefficients = _fit_and_compare(
        ordinal=False,
        riskset=riskset,
        effects=(
            "~ inertia(consider_type=False) + "
            "indegreeSender(consider_type=False) + "
            "outdegreeSender(consider_type=False)"
        ),
    )

    assert names == ["baseline", "inertia", "indegreeSender", "outdegreeSender"]
    assert np.isfinite(coefficients).all()


def test_ordinal_mle_matches_independent_conditional_logit():
    names, coefficients = _fit_and_compare(
        ordinal=True,
        riskset="active",
        effects=(
            "~ inertia(consider_type=False) + "
            "indegreeSender(consider_type=False) + "
            "outdegreeSender(consider_type=False)"
        ),
    )

    assert names == ["inertia", "indegreeSender", "outdegreeSender"]
    assert np.isfinite(coefficients).all()


@pytest.mark.parametrize("ordinal", [False, True])
def test_typed_separate_mle_matches_independent_glm(ordinal: bool):
    names, _ = _fit_and_compare(
        ordinal=ordinal,
        riskset="active",
        effects=(
            '~ inertia(consider_type="separate") + '
            "outdegreeSender(consider_type=False)"
        ),
    )

    assert any(name.startswith("inertia.") for name in names)
    assert ("baseline" in names) is (not ordinal)


@pytest.mark.parametrize("ordinal", [False, True])
def test_sampled_mle_matches_independent_weighted_glm(ordinal: bool):
    names, coefficients = _fit_and_compare(
        ordinal=ordinal,
        riskset="full",
        effects=(
            "~ inertia(consider_type=False) + "
            "indegreeSender(consider_type=False) + "
            "outdegreeSender(consider_type=False)"
        ),
        sampling=True,
    )

    assert ("baseline" in names) is (not ordinal)
    assert np.isfinite(coefficients).all()


@pytest.mark.parametrize("ordinal", [False, True])
def test_actor_mle_matches_sender_glm_and_receiver_clogit(ordinal: bool):
    history = remify(
        _events(),
        model="actor",
        ordinal=ordinal,
    )
    statistics = aomstats(
        reh=history,
        sender_effects="~ indegreeSender()",
        receiver_effects=(
            "~ inertia(consider_type=False) + indegreeReceiver(consider_type=False)"
        ),
        memory="decay",
        memory_value=1000,
        first=2,
        last=30,
    )
    fitted = remstimate(history, statistics, method="MLE")
    stacked = stack_stats(statistics, history)

    assert isinstance(fitted, ActorRemEstimate)
    assert stacked.sender_stack is not None
    assert stacked.receiver_stack is not None
    sender_reference = (
        _independent_clogit_fit(stacked.sender_stack, fitted.sender_model.names)
        if ordinal
        else _independent_poisson_fit(stacked.sender_stack, fitted.sender_model.names)
    )
    receiver_reference = _independent_clogit_fit(
        stacked.receiver_stack, fitted.receiver_model.names
    )
    np.testing.assert_allclose(
        fitted.sender_model.coef, sender_reference, rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        fitted.receiver_model.coef, receiver_reference, rtol=1e-4, atol=1e-4
    )
    assert ("baseline" in fitted.sender_model.names) is (not ordinal)
    assert "baseline" not in fitted.receiver_model.names


def test_actor_stack_is_nullable_when_one_formula_is_missing():
    history = remify(
        _events(),
        model="actor",
    )
    receiver_only = aomstats(
        reh=history,
        receiver_effects="~ inertia(consider_type=False)",
        first=2,
        last=30,
    )
    sender_only = aomstats(
        reh=history,
        sender_effects="~ indegreeSender()",
        first=2,
        last=30,
    )

    assert stack_stats(receiver_only, history).sender_stack is None
    assert stack_stats(receiver_only, history).receiver_stack is not None
    assert stack_stats(sender_only, history).sender_stack is not None
    assert stack_stats(sender_only, history).receiver_stack is None

    receiver_fit = remstimate(history, receiver_only)
    sender_fit = remstimate(history, sender_only)
    assert isinstance(receiver_fit, ActorRemEstimate)
    assert receiver_fit.sender_model is None
    assert receiver_fit.receiver_model is not None
    assert isinstance(sender_fit, ActorRemEstimate)
    assert sender_fit.sender_model is not None
    assert sender_fit.receiver_model is None
