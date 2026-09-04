"""End-to-end estimator pipeline regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from remflow import (
    bic_table,
    diagnostics,
    dlcrem,
    frailty_rem,
    remfrailty,
    remify,
    remixture,
    rempenalty,
    remstats,
    remstimate,
    remtribute,
    remwindow,
)
from remflow.estimate import (
    ActorRemEstimate,
    MixtureDiagnostics,
    RemEstimateGLMM,
    RemEstimateMixture,
    RemEstimateShrinkage,
    RemEstimateWindow,
    RemTribute,
    WindowDiagnostics,
)


def _events(count: int = 132) -> pd.DataFrame:
    actors = np.asarray(["A", "B", "C", "D"], dtype=object)
    index = np.arange(count)
    sender_id = index % len(actors)
    receiver_id = (sender_id + 1 + (index // len(actors)) % 3) % len(actors)
    return pd.DataFrame(
        {
            "time": np.cumsum(0.5 + (index % 5) / 10.0),
            "actor1": actors[sender_id],
            "actor2": actors[receiver_id],
        }
    )


def _attribute_events(count: int = 132) -> pd.DataFrame:
    frame = _events(count)
    index = np.arange(count)
    frame["type"] = np.where((index // 3) % 2 == 0, "X", "Y")
    frame["score"] = (
        2.0
        + 0.04 * index
        + 0.7 * (frame["actor1"] == "A").to_numpy(dtype=float)
        - 0.5 * (frame["actor2"] == "D").to_numpy(dtype=float)
    )
    frame["severity"] = 1 + ((index // 5) % 3)
    return frame


@pytest.fixture(scope="module")
def tie_inputs():
    history = remify(_events(), actors=["A", "B", "C", "D"])
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    return history, statistics


@pytest.fixture(scope="module")
def actor_inputs():
    history = remify(
        _events(),
        model="actor",
        actors=["A", "B", "C", "D"],
    )
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender() + outdegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
    )
    return history, statistics


def test_remwindow_tie_auto_coefficients_diagnostics_and_plot(tie_inputs):
    history, statistics = tie_inputs
    fitted = remwindow(history, statistics, n_windows=2)

    assert isinstance(fitted, RemEstimateWindow)
    assert fitted.type == "tie"
    assert fitted.mode == "auto"
    assert fitted.n_windows == 2
    assert fitted.windows["n_events"].sum() == len(statistics.event_indices)
    coefficient_block = fitted.coefficients()
    assert coefficient_block["coefficients"].shape == (2, 3)
    assert not fitted.plot_data().empty
    assert fitted.plot() is fitted

    result = diagnostics(fitted, history, statistics)
    assert isinstance(result, WindowDiagnostics)
    assert result.recall is not None
    assert len(result.recall["per_event"]) == len(statistics.event_indices)
    assert np.isfinite(result.recall["summary"]["mean_rel_rank"])
    assert result.plot() is result


def test_remwindow_tie_manual_width_absorbs_remainder(tie_inputs):
    history, statistics = tie_inputs
    fitted = remwindow(history, statistics, window_width=70)

    assert fitted.mode == "manual"
    assert fitted.windows["start_event"].to_list() == [1, 71]
    assert fitted.windows.iloc[-1]["end_event"] == len(statistics.event_indices)
    assert fitted.windows.iloc[-1]["n_events"] == len(statistics.event_indices) - 70


def test_remwindow_actor_auto_and_diagnostics(actor_inputs):
    history, statistics = actor_inputs
    fitted = remwindow(history, statistics, n_windows=2)

    assert isinstance(fitted, RemEstimateWindow)
    assert fitted.type == "actor"
    blocks = fitted.coefficients()
    assert blocks["sender"] is not None
    assert blocks["receiver"] is not None
    assert blocks["sender"]["coefficients"].shape[0] == 2

    result = diagnostics(fitted, history, statistics)
    assert isinstance(result, WindowDiagnostics)
    assert result.sender is not None
    assert result.receiver is not None
    assert not result.plot_data().empty


def test_remwindow_validation_and_auto_window_reduction(tie_inputs):
    history, statistics = tie_inputs
    with pytest.raises(ValueError, match="requires window_width"):
        remwindow(history, statistics, step_size_window=10)
    with pytest.raises(ValueError, match="outside"):
        remwindow(history, statistics, start_point=10_000)
    with pytest.warns(UserWarning, match="reduced n_windows"):
        fitted = remwindow(history, statistics, n_windows=20)
    assert fitted.n_windows == 2


def test_ncores_validation_uses_fallback_and_records_metadata(tie_inputs):
    """Validate the worker-count fallback and metadata."""

    history, statistics = tie_inputs
    with pytest.warns(UserWarning, match="positive integer"):
        fallback = remstimate(history, statistics, ncores=-1)
    assert fallback.metadata["ncores"] == 1

    requested = remstimate(history, statistics, ncores=2)
    assert requested.metadata["ncores"] == 2


def test_dispatch_rejects_incompatible_extended_models(tie_inputs):
    """Reject incompatible extended-model requests."""

    history, statistics = tie_inputs
    with pytest.raises(ValueError, match="cannot be combined"):
        remstimate(
            history,
            statistics,
            random="~ (1 | actor1)",
            mixture={"k": 2, "random": "~ (1 | dyad)"},
        )


def test_deprecated_top_level_controls_are_routed_into_owned_lists():
    """Route deprecated controls without silently discarding them."""

    history = remify(_events(48), actors=["A", "B", "C", "D"])
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity() + outdegreeReceiver()",
    )
    with pytest.warns(DeprecationWarning) as penalty_warnings:
        penalized = remstimate(
            history,
            statistics,
            penalty={"alpha": 1.0},
            nfolds=5,
            lambda_select="min",
            seed=1,
        )
    assert {"nfolds", "lambda_select"}.issubset(
        {str(item.message).split("'")[1] for item in penalty_warnings}
    )
    assert penalized.lambda_select == "min"
    assert penalized.lambda_value == penalized.lambda_min

    with pytest.warns(DeprecationWarning) as mixture_warnings:
        mixed = remstimate(
            history,
            statistics,
            mixture={"k": 2, "random": "~ (1 | dyad)"},
            concomitant="~ 1",
            nrep=1,
            maxiter=8,
            seed=2,
        )
    assert {"concomitant", "nrep", "maxiter"}.issubset(
        {str(item.message).split("'")[1] for item in mixture_warnings}
    )
    assert isinstance(mixed, RemEstimateMixture)


def test_deprecated_frailty_alias_still_dispatches():
    """Keep the documented frailty alias operational."""

    history = remify(_events(36), actors=["A", "B", "C", "D"])
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    with pytest.warns(DeprecationWarning, match="frailty_rem"):
        fitted = frailty_rem(
            history,
            statistics,
            variance_iterations=2,
            maxiter=40,
        )
    assert isinstance(fitted, RemEstimateGLMM)
    assert fitted.backend_fit


def test_remtribute_tie_nominal_precomputed_stats_route():
    events = _attribute_events()
    history = remify(
        events,
        actors=["A", "B", "C", "D"],
        event_type="type",
        event_attributes="type",
    )
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    fitted = remtribute(
        history,
        stats=statistics,
        attribute="type",
        attribute_type="nominal",
    )

    assert isinstance(fitted, RemTribute)
    assert fitted.attribute_type == "nominal"
    assert fitted.levels == ("X", "Y")
    assert fitted.n_events == len(statistics.event_indices)
    assert isinstance(fitted.coef, pd.Series)
    assert np.isfinite(fitted.coef.to_numpy()).all()
    assert fitted.vcov is not None
    assert np.isfinite(fitted.loglik)
    assert fitted.summary()["coefficients"] is not None
    assert set(fitted.to_dict()) >= {
        "coefficients",
        "vcov",
        "loglik",
        "fit",
        "attribute",
        "attribute_type",
        "n_events",
        "stat_names",
        "formula",
        "data",
        "levels",
        "n_levels",
        "AIC",
        "BIC",
    }


def test_remtribute_actor_effects_route_builds_dyad_statistics():
    events = _attribute_events()
    history = remify(
        events,
        model="actor",
        actors=["A", "B", "C", "D"],
        event_attributes="type",
    )
    fitted = remtribute(
        history,
        effects="~ inertia() + reciprocity()",
        attribute="type",
        attribute_type="nominal",
    )

    assert isinstance(fitted, RemTribute)
    assert fitted.n_events == len(events) - 1
    assert fitted.stat_names == ["inertia", "reciprocity"]
    assert fitted.backend_fit["engine"] == "multinomial_logit"


def test_remtribute_numeric_matches_observed_dyad_ols():
    events = _attribute_events()
    history = remify(events, event_attributes="score")
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity() + indegreeSender()",
    )
    fitted = remtribute(
        history,
        stats=statistics,
        attribute="score",
        attribute_type="numeric",
    )

    model = np.column_stack(
        [
            np.ones(fitted.n_events),
            fitted.data[fitted.stat_names].to_numpy(dtype=float),
        ]
    )
    expected = np.linalg.lstsq(
        model, fitted.data[".y"].to_numpy(dtype=float), rcond=None
    )[0]
    np.testing.assert_allclose(fitted.coef.to_numpy(), expected, rtol=1e-12, atol=1e-12)
    assert fitted.backend_fit["engine"] == "gaussian"
    assert fitted.backend_fit["converged"]


def test_remtribute_ordinal_proportional_odds_and_validation():
    events = _attribute_events()
    history = remify(events, event_attributes="severity")
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity() + indegreeSender()",
    )
    fitted = remtribute(
        history,
        stats=statistics,
        attribute="severity",
        attribute_type="ordinal",
        maxiter=400,
    )

    assert fitted.levels == (1, 2, 3)
    assert fitted.backend_fit["engine"] == "proportional_odds"
    assert len(fitted.backend_fit["thresholds"]) == 2
    assert np.all(np.diff(fitted.backend_fit["thresholds"].to_numpy()) > 0)
    assert fitted.backend_fit["fitted_values"].shape == (fitted.n_events, 3)
    np.testing.assert_allclose(
        fitted.backend_fit["fitted_values"].sum(axis=1), 1.0, atol=1e-12
    )

    with pytest.raises(ValueError, match="provide either"):
        remtribute(history, attribute="severity", attribute_type="ordinal")
    with pytest.raises(ValueError, match="was not retained"):
        remtribute(history, statistics, attribute="missing")
    with pytest.raises(TypeError, match="unsupported remtribute"):
        remtribute(
            history,
            statistics,
            attribute="severity",
            attribute_type="ordinal",
            unsupported=True,
        )


def test_glmm_tie_random_intercept_and_slope_retain_blups():
    history = remify(_events(84), actors=["A", "B", "C", "D"])
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    fitted = remstimate(
        history,
        statistics,
        random="~ (1 + inertia | actor1) + (1 | actor2)",
        variance_iterations=8,
        maxiter=200,
    )

    assert isinstance(fitted, RemEstimateGLMM)
    assert fitted.metadata["method"] == "GLMM"
    assert fitted.metadata["engine"] == "scipy-laplace"
    assert fitted.metadata["estimator_engine"] == "scipy"
    assert "baseline" in fitted.names
    assert set(fitted.random_effects) == {
        "actor1::(Intercept)",
        "actor1::inertia",
        "actor2::(Intercept)",
    }
    assert (fitted.variance_components > 0).all()
    assert fitted.backend_fit["joint_parameters"].shape[0] > len(fitted.coef)
    assert np.isfinite(fitted.coef).all()
    fitted_diagnostics = diagnostics(fitted, history, statistics)
    assert fitted_diagnostics.recall
    assert fitted_diagnostics.use_ranef is True
    assert set(fitted_diagnostics.ranef) == set(fitted.random_effects)
    qq = fitted_diagnostics.plot_data(which=6, object=fitted)["panel6"]
    assert set(qq) == {
        "term",
        "level",
        "theoretical_quantile",
        "random_effect",
    }
    assert len(qq) == sum(len(values) for values in fitted.random_effects.values())


def test_glmm_ordinal_uses_conditional_engine_and_no_baseline():
    history = remify(
        _events(84),
        actors=["A", "B", "C", "D"],
        ordinal=True,
    )
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    fitted = remstimate(
        history,
        statistics,
        random="~ (1 | actor1)",
        variance_iterations=6,
        maxiter=150,
    )

    assert isinstance(fitted, RemEstimateGLMM)
    assert fitted.metadata["engine"] == "scipy-conditional-laplace"
    assert "baseline" not in fitted.names
    assert np.isfinite(fitted.log_likelihood)


def test_glmm_actor_sender_and_receiver_use_separate_random_components():
    history = remify(
        _events(84),
        model="actor",
        actors=["A", "B", "C", "D"],
    )
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender() + outdegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
    )
    fitted = remstimate(
        history,
        statistics,
        random={"sender": "~ (1 | actor)", "receiver": "~ (1 | actor)"},
        variance_iterations=6,
        maxiter=150,
    )

    assert isinstance(fitted, ActorRemEstimate)
    assert isinstance(fitted.sender_model, RemEstimateGLMM)
    assert isinstance(fitted.receiver_model, RemEstimateGLMM)
    assert fitted.metadata["engine_receiver"] == "scipy-conditional-laplace"
    assert fitted.sender_model.random_effects
    assert fitted.receiver_model.random_effects
    fitted_diagnostics = diagnostics(fitted, history, statistics)
    assert fitted_diagnostics.sender_model is not None
    assert fitted_diagnostics.receiver_model is not None
    assert fitted_diagnostics.sender_model.use_ranef is True
    assert fitted_diagnostics.receiver_model.use_ranef is True


def test_glmm_undirected_guard_and_remfrailty_dyad_fallback():
    history = remify(_events(84), directed=False, actors=["A", "B", "C", "D"])
    statistics = remstats(history, tie_effects="~ inertia() + totaldegreeDyad()")
    with pytest.raises(ValueError, match="not identified"):
        remstimate(history, statistics, random="~ (1 | actor1)")

    with pytest.warns(UserWarning, match="dyad-level frailty"):
        fitted = remfrailty(
            history,
            statistics,
            variance_iterations=6,
            maxiter=150,
        )
    assert isinstance(fitted, RemEstimateGLMM)
    assert "dyad::(Intercept)" in fitted.random_effects


def test_mixrem_tie_components_memberships_and_diagnostics_are_retained():
    history = remify(_events(72), actors=["A", "B", "C", "D"])
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    fitted = remstimate(
        history,
        statistics,
        mixture={
            "k": 2,
            "random": "~ (1 | dyad)",
            "nrep": 2,
            "maxiter": 40,
        },
        seed=91,
    )

    assert isinstance(fitted, RemEstimateMixture)
    assert fitted.coef.shape == (len(fitted.names), 2)
    assert np.isfinite(fitted.coef).all()
    assert fitted.prior_probs.sum() == pytest.approx(1.0)
    assert np.all(np.diff(fitted.prior_probs) <= 0.0)
    assert set(fitted.assignments.unique()).issubset({1, 2})
    assert fitted.metadata["method"] == "MIXREM"
    fitted_diagnostics = diagnostics(fitted, history, statistics)
    assert isinstance(fitted_diagnostics, MixtureDiagnostics)
    assert fitted_diagnostics.recall
    assert set(fitted_diagnostics.recall_by_component) == {
        "Component.1",
        "Component.2",
    }
    assert not fitted_diagnostics.plot_data(which=3, object=fitted)["panel3"].empty
    assert not fitted_diagnostics.plot_data(which=9, object=fitted)["panel9"].empty


def test_mixrem_ordinal_is_seeded_and_drops_unidentified_baseline():
    history = remify(
        _events(48),
        actors=["A", "B", "C", "D"],
        ordinal=True,
    )
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    controls = {
        "k": 2,
        "random": "~ (1 | dyad)",
        "concomitant": "~ 1",
        "nrep": 1,
        "maxiter": 25,
    }
    first = remstimate(history, statistics, mixture=controls, seed=41)
    second = remstimate(history, statistics, mixture=controls, seed=41)
    assert isinstance(first, RemEstimateMixture)
    assert isinstance(second, RemEstimateMixture)
    assert "baseline" not in first.names
    np.testing.assert_allclose(first.coef, second.coef, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(first.posterior, second.posterior, rtol=0.0, atol=0.0)
    assert all(np.isclose(values.sum(), 1.0) for values in first.event_probabilities)


def test_remixture_dlcrem_and_multiple_k_bic_table():
    history = remify(_events(60), actors=["A", "B", "C", "D"])
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()")
    wrapped = remixture(
        history,
        statistics,
        random="~ (1 | dyad)",
        k=2,
        nrep=1,
        maxiter=30,
        seed=7,
    )
    specialized = dlcrem(
        history,
        statistics,
        k=2,
        nrep=1,
        maxiter=30,
        seed=7,
    )
    assert isinstance(wrapped, RemEstimateMixture)
    assert isinstance(specialized, RemEstimateMixture)

    multiple = remstimate(
        history,
        statistics,
        mixture={
            "k": [1, 2],
            "random": "~ (1 | dyad)",
            "nrep": 1,
            "maxiter": 30,
        },
        seed=8,
    )
    assert isinstance(multiple, dict)
    assert list(multiple) == ["k1", "k2"]
    comparison = bic_table(multiple)
    assert {row["k"] for row in comparison} == {1, 2}
    assert min(row["delta_BIC"] for row in comparison) == pytest.approx(0.0)


def test_mixrem_actor_components_use_actor_grouping():
    history = remify(
        _events(60),
        model="actor",
        actors=["A", "B", "C", "D"],
    )
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
    )
    fitted = remixture(
        history,
        statistics,
        random="~ (1 | actor)",
        k=2,
        nrep=1,
        maxiter=30,
        seed=19,
    )
    assert isinstance(fitted, ActorRemEstimate)
    assert isinstance(fitted.sender_model, RemEstimateMixture)
    assert isinstance(fitted.receiver_model, RemEstimateMixture)
    assert fitted.sender_model.grouping == "actor"
    assert fitted.receiver_model.grouping == "actor"


@pytest.mark.parametrize("prior", ["horseshoe", "lasso", "ridge"])
def test_bayesian_shrinkage_tie_priors_retain_intercept_and_diagnostics(prior):
    history = remify(_events(84), actors=["A", "B", "C", "D"])
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity() + otp()",
    )
    fitted = remstimate(
        history,
        statistics,
        approach="Bayesian",
        penalty={"prior": prior, "lambda": 0.75},
        seed=123,
    )
    assert isinstance(fitted, RemEstimateShrinkage)
    assert fitted.shrinkage_type == prior
    assert "baseline" in fitted.unpenalized
    assert np.isfinite(fitted.coef).all()
    assert list(fitted.estimates.index) == fitted.names
    assert set(fitted.estimates) == {
        "input.est",
        "input.sd",
        "shrunk.mode",
        "posterior.sd",
        "nonzero",
    }
    fitted_diagnostics = diagnostics(fitted, history, statistics)
    assert fitted_diagnostics.recall
    shrinkage_map = fitted_diagnostics.plot_data(which=6, object=fitted)["panel6"]
    assert list(shrinkage_map["effect"]) == fitted.names


def test_bayesian_shrinkage_actor_and_rempenalty_wrapper():
    history = remify(
        _events(84),
        model="actor",
        actors=["A", "B", "C", "D"],
    )
    statistics = remstats(
        history,
        sender_effects="~ indegreeSender() + outdegreeSender()",
        receiver_effects="~ inertia() + reciprocity()",
    )
    fitted = rempenalty(
        history,
        statistics,
        approach="Bayesian",
        penalty={"prior": "horseshoe", "lambda": 0.5},
        seed=17,
    )
    assert isinstance(fitted, ActorRemEstimate)
    assert isinstance(fitted.sender_model, RemEstimateShrinkage)
    assert isinstance(fitted.receiver_model, RemEstimateShrinkage)
    assert diagnostics(fitted, history, statistics).sender_model is not None


def test_bayesian_shrinkage_rejects_collinear_statistics_before_regularizing():
    history = remify(_events(60), actors=["A", "B", "C", "D"])
    statistics = remstats(history, tie_effects="~ inertia() + inertia()")
    with pytest.raises(ValueError, match="duplicated|collinear|rank-deficient"):
        remstimate(
            history,
            statistics,
            approach="Bayesian",
            penalty={"prior": "lasso"},
        )
