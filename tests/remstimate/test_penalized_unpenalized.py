"""Regression tests for penalized and unpenalized estimation."""

import warnings

import numpy as np
import pandas as pd
import pytest

from remflow import diagnostics, remify, rempenalty, remstats, remstimate
from remflow.estimate import (
    ActorRemEstimate,
    RemEstimateDurationGlmnet,
    RemEstimateGlmnet,
    _balanced_cv_folds,
    _check_penalty_names,
    _intercept_like_stats,
    _recall_ranks,
    _resolve_unpenalized,
)


def _design() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "baseline": [1] * 8,
            "baseline.start": [0, 1] * 4,
            "psABAB.end": [0, 0, 1, 0, 1, 0, 0, 1],
            "inertia": [0, 2, 7, 1, 0, 3, 5, 2],
            "reciprocity": np.random.default_rng(1).normal(size=8),
        }
    )


def test_intercept_like_statistics_use_structural_binary_detection():
    design = _design()
    names = design.columns.to_list()
    assert set(_intercept_like_stats(design, names)) == {
        "baseline",
        "baseline.start",
        "psABAB.end",
    }
    assert set(_intercept_like_stats(design, [*names, "does_not_exist"])) == {
        "baseline",
        "baseline.start",
        "psABAB.end",
    }


def test_recall_ranks_average_full_and_partial_probability_ties():
    assert _recall_ranks([0.5, 0.3, 0.2], 1) == {"rank": 1.0, "cum": 0.5}
    assert _recall_ranks([0.5, 0.3, 0.2], 3) == {"rank": 3.0, "cum": 1.0}
    ranks = [_recall_ranks([0.25] * 4, position)["rank"] for position in range(1, 5)]
    np.testing.assert_array_equal(ranks, [2.5] * 4)
    assert 1 - ranks[0] / 4 == 0.375
    assert _recall_ranks([0.25] * 4, 1)["cum"] == 0.25
    partial = _recall_ranks([0.4, 0.2, 0.2, 0.2], 2)
    assert partial == {"rank": 3.0, "cum": pytest.approx(0.6)}


def test_unpenalized_resolver_is_additive_subtractive_and_exact():
    design = _design()
    names = design.columns.to_list()
    defaults = {"baseline", "baseline.start", "psABAB.end"}
    assert set(_resolve_unpenalized(design, names)) == defaults
    assert set(
        _resolve_unpenalized(design, names, penalized="psABAB.end")
    ) == {"baseline", "baseline.start"}
    assert set(
        _resolve_unpenalized(design, names, unpenalized="reciprocity")
    ) == {*defaults, "reciprocity"}
    assert "psABAB.end" not in _resolve_unpenalized(
        design,
        names,
        unpenalized="psABAB.end",
        penalized="psABAB.end",
    )
    with pytest.warns(UserWarning, match="not found"):
        unchanged = _resolve_unpenalized(design, names, penalized="psABAB")
    assert "psABAB.end" in unchanged
    assert "psABAB.end" not in _resolve_unpenalized(
        design, names, penalized="psABAB.end"
    )


def test_penalty_name_validation_warns_only_for_unknown_exact_names():
    names = _design().columns.to_list()
    with pytest.warns(UserWarning, match="not found among the model statistics"):
        _check_penalty_names(penalized="psABAB", valid=names)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _check_penalty_names(
            penalized="psABAB.end", unpenalized="reciprocity", valid=names
        )
    assert not captured


def test_seeded_default_penalty_selection_matches_expected_solution():
    np.testing.assert_array_equal(
        _balanced_cv_folds(300, 10, 1)[:10] + 1,
        [7, 9, 10, 7, 5, 7, 3, 9, 3, 7],
    )
    history, statistics = _penalty_problem()
    fitted = remstimate(history, statistics, penalty={"alpha": 1.0}, seed=1)

    assert isinstance(fitted, RemEstimateGlmnet)
    assert fitted.lambda_select == "1se"
    assert fitted.lambda_min == pytest.approx(0.002803125278516365, rel=1e-12)
    assert fitted.lambda_1se == pytest.approx(0.023819653367016552, rel=1e-12)
    assert fitted.lambda_value == pytest.approx(fitted.lambda_1se, rel=1e-14)
    np.testing.assert_allclose(
        fitted.coef,
        [-2.9957322735539904, 0.0, 0.0, 0.0, 0.0],
        rtol=1e-6,
        atol=1e-7,
    )


def _penalty_problem():
    events = pd.DataFrame(
        {
            "time": range(1, 17),
            "actor1": [1, 2, 1, 3, 2, 1, 3, 2, 4, 1, 2, 3, 4, 1, 2, 5],
            "actor2": [2, 3, 4, 1, 1, 3, 2, 4, 1, 5, 4, 1, 2, 3, 5, 1],
        }
    )
    history = remify(events, actors=[1, 2, 3, 4, 5], model="tie")
    statistics = remstats(
        history,
        tie_effects=(
            "~ inertia() + reciprocity() + indegreeSender() + outdegreeReceiver()"
        ),
        first=2,
    )
    return history, statistics


def test_elastic_net_lasso_ridge_and_override_integration():
    history, statistics = _penalty_problem()
    lasso = remstimate(
        history, statistics, penalty={"alpha": 1.0, "lambda": 0.25}
    )
    ridge = remstimate(
        history, statistics, penalty={"alpha": 0.0, "lambda": 0.25}
    )
    elastic = remstimate(
        history, statistics, penalty={"alpha": 0.5, "lambda": 0.25}
    )

    for fitted in (lasso, ridge, elastic):
        assert isinstance(fitted, RemEstimateGlmnet)
        assert np.isfinite(fitted.coef).all()
        assert fitted.converged
    assert "baseline" in lasso.names
    assert "baseline" in lasso.unpenalized
    assert abs(lasso.coef[lasso.names.index("baseline")]) > 1e-6

    added = remstimate(
        history,
        statistics,
        penalty={
            "alpha": 1.0,
            "lambda": 0.25,
            "unpenalized": "inertia",
        },
    )
    forced = remstimate(
        history,
        statistics,
        penalty={
            "alpha": 1.0,
            "lambda": 0.25,
            "penalized": "inertia",
        },
    )
    assert isinstance(added, RemEstimateGlmnet)
    assert isinstance(forced, RemEstimateGlmnet)
    assert "inertia" in added.unpenalized
    assert "inertia" not in forced.unpenalized


def test_penalty_strength_shrinks_coefficients_and_supports_diagnostics_wrapper():
    history, statistics = _penalty_problem()
    weak = rempenalty(
        history,
        statistics,
        penalty={"alpha": 1.0, "lambda": 0.05},
    )
    strong = rempenalty(
        history,
        statistics,
        penalty={"alpha": 1.0, "lambda": 2.0},
    )
    weak_penalized = [
        index for index, name in enumerate(weak.names) if name not in weak.unpenalized
    ]
    strong_penalized = [
        index for index, name in enumerate(strong.names) if name not in strong.unpenalized
    ]
    assert np.abs(strong.coef[strong_penalized]).sum() <= np.abs(
        weak.coef[weak_penalized]
    ).sum()
    result = diagnostics(strong, history, statistics)
    assert result.recall["per_event"]["event"].notna().all()
    assert strong.plot(reh=history, diagnostics=result, which=(1, 2)) is result


def test_actor_elastic_net_penalizes_sender_and_receiver_components():
    events = pd.concat([_penalty_problem()[0].events] * 3, ignore_index=True)
    events["time"] = range(1, len(events) + 1)
    history = remify(
        events[["time", "sender", "receiver"]],
        actors=[1, 2, 3, 4, 5],
        model="actor",
    )
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender() + outdegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
        first=6,
    )
    fitted = remstimate(
        history,
        statistics,
        penalty={"alpha": 0.5, "lambda": 0.25},
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert isinstance(fitted.sender_model, RemEstimateGlmnet)
    assert isinstance(fitted.receiver_model, RemEstimateGlmnet)
    assert "baseline" in fitted.sender_model.unpenalized
    assert np.isfinite(fitted.sender_model.coef).all()
    assert np.isfinite(fitted.receiver_model.coef).all()
    result = diagnostics(fitted, history, statistics)
    assert result.sender_model is not None
    assert result.receiver_model is not None


def test_duration_elastic_net_requires_two_effects_and_keeps_duration_diagnostics():
    events = pd.DataFrame(
        {
            "time": [1, 2, 5, 8, 11, 14],
            "actor1": ["A", "B", "A", "C", "B", "A"],
            "actor2": ["B", "C", "C", "A", "A", "B"],
            "end": [3, 4, 7, 10, 13, 16],
        }
    )
    history = remify(events, duration=True)
    too_small = remstats(history, start_effects="~ inertia()")
    with pytest.raises(ValueError, match="at least two penalized effects"):
        remstimate(history, too_small, penalty={"alpha": 1.0, "lambda": 0.25})

    statistics = remstats(
        history,
        start_effects='~ inertia() + reciprocity(scaling="std")',
        psi_start=1,
    )
    fitted = remstimate(
        history,
        statistics,
        penalty={"alpha": 1.0, "lambda": 0.25},
    )

    assert isinstance(fitted, RemEstimateDurationGlmnet)
    assert "baseline.start" in fitted.unpenalized
    assert np.isfinite(fitted.coef).all()
    result = diagnostics(fitted, history, statistics)
    assert result.recall_joint
