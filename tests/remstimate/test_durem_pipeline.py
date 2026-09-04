"""End-to-end duration-model regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from remflow import (
    DurationDiagnostics,
    RemEstimateDuration,
    RemEstimateMixture,
    RemEstimateShrinkage,
    diagnostics,
    remfrailty,
    remify,
    remstats,
    remstimate,
    stack_stats,
)


def _simple() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5, 6],
            "actor1": ["A", "B", "A", "B", "A", "B"],
            "actor2": ["B", "C", "B", "C", "B", "C"],
            "duration": [2, 2, 2, 3, 3, 3],
        }
    )


def _typed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 6, 6],
            "actor1": ["A", "B", "A", "B", "A", "B"],
            "actor2": ["B", "C", "B", "C", "B", "C"],
            "type": ["X", "X", "Y", "Y", "X", "Y"],
            "duration": [2, 2, 2, 3, 3, 3],
        }
    )


def _typed_ordinal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 6, 6, 7, 7, 9, 10, 12, 12],
            "actor1": ["A", "B", "A", "B", "A", "B", "B", "C", "B", "A", "B", "A"],
            "actor2": ["B", "C", "B", "C", "B", "C", "A", "B", "A", "B", "A", "B"],
            "type": ["X", "X", "Y", "Y", "X", "Y", "Y", "Y", "X", "Y", "Y", "Y"],
            "duration": [2, 2, 2, 3, 3, 3] * 2,
        }
    )


def test_basic_interval_and_ordinal_duration_pipelines():
    history = remify(_simple(), duration=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + reciprocity()",
        psi_start=1,
    )
    stacked = stack_stats(statistics, history)
    assert "time_index" in stacked.remstats_stack
    assert "event" not in stacked.remstats_stack
    assert len(stacked.remstats_stack) > 0
    fitted = remstimate(history, statistics, method="MLE")
    assert isinstance(fitted, RemEstimateDuration)
    assert fitted.coef.size > 0

    ordinal_history = remify(_simple(), duration=True, model="tie", ordinal=True)
    ordinal_statistics = remstats(
        ordinal_history,
        start_effects="~ inertia()",
        psi_start=1,
    )
    ordinal_stacked = stack_stats(ordinal_statistics, ordinal_history)
    assert "log_interevent" not in ordinal_stacked.remstats_stack
    ordinal_fit = remstimate(ordinal_history, ordinal_statistics, method="MLE")
    assert isinstance(ordinal_fit, RemEstimateDuration)
    assert ordinal_fit.engine == "clogit"
    assert ordinal_fit.summary()


def test_typed_expanded_interval_and_ordinal_duration_pipelines():
    history = remify(
        _typed(),
        duration=True,
        extend_riskset_by_type=True,
        riskset="active",
        model="tie",
    )
    statistics = remstats(
        history,
        start_effects='~ inertia(consider_type="interact")',
        psi_start=1,
    )
    stacked = stack_stats(statistics, history)
    assert "type" in stacked.remstats_stack
    assert len(stacked.remstats_stack) > 0
    assert set(stacked.remstats_stack["dyad"]) == {1, 2, 3, 4}
    fitted = remstimate(history, statistics, method="MLE")
    assert isinstance(fitted, RemEstimateDuration)
    assert np.isfinite(fitted.coef).all()

    ordinal_history = remify(
        _typed_ordinal(),
        duration=True,
        extend_riskset_by_type=True,
        riskset="active",
        model="tie",
        ordinal=True,
    )
    assert ordinal_history.events["time"].to_list() == [
        1,
        2,
        3,
        4,
        6,
        6,
        7,
        7,
        8,
        9,
        11,
        11,
    ]
    assert ordinal_history.events["end"].to_list() == [
        3,
        4,
        5,
        7,
        8,
        8,
        8,
        8,
        10,
        12,
        13,
        13,
    ]
    ordinal_statistics = remstats(
        ordinal_history,
        start_effects='~ inertia(consider_type="interact")',
        psi_start=1,
    )
    assert len(stack_stats(ordinal_statistics, ordinal_history).remstats_stack) > 0
    ordinal_fit = remstimate(ordinal_history, ordinal_statistics, method="MLE")
    assert isinstance(ordinal_fit, RemEstimateDuration)
    assert np.isfinite(ordinal_fit.coef).all()


def test_type_exclusivity_reduces_duration_riskset_rows():
    common = {
        "duration": True,
        "extend_riskset_by_type": True,
        "riskset": "active",
        "model": "tie",
    }
    exclusive_history = remify(_typed(), dur_type_exclusive=True, **common)
    independent_history = remify(_typed(), dur_type_exclusive=False, **common)
    exclusive = remstats(
        exclusive_history,
        start_effects='~ inertia(consider_type="interact")',
        psi_start=1,
    )
    independent = remstats(
        independent_history,
        start_effects='~ inertia(consider_type="interact")',
        psi_start=1,
    )
    assert len(exclusive.stacked.remstats_stack) < len(independent.stacked.remstats_stack)


def test_duration_diagnostics_joint_and_per_type_recall_and_printing():
    basic_history = remify(_simple(), duration=True, model="tie")
    basic_stats = remstats(
        basic_history,
        start_effects="~ inertia() + reciprocity()",
        psi_start=1,
    )
    basic_fit = remstimate(basic_history, basic_stats)
    basic = diagnostics(basic_fit, basic_history, basic_stats)
    assert isinstance(basic, DurationDiagnostics)
    assert basic.recall_joint
    assert str(basic)

    typed_history = remify(
        _typed(),
        duration=True,
        extend_riskset_by_type=True,
        riskset="active",
        model="tie",
    )
    typed_stats = remstats(
        typed_history,
        start_effects='~ inertia(consider_type="interact")',
        psi_start=1,
    )
    typed_fit = remstimate(typed_history, typed_stats)
    typed = diagnostics(typed_fit, typed_history, typed_stats)
    assert isinstance(typed, DurationDiagnostics)
    assert typed.recall_joint
    assert set(typed.recall_by_type) == {"X", "Y"}
    assert all(len(value["per_event"]) > 0 for value in typed.recall_by_type.values())
    assert set(typed.surprises_by_type) == {"X", "Y"}
    assert str(typed)


def test_duration_specialized_plot_panels_and_coefficient_fallback():
    """Port the duration-specific calls from pipeline section 16.5."""

    history = remify(_simple(), duration=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + reciprocity()",
        end_effects="~ inertia()",
        psi_start=1,
        psi_end=1,
    )
    fitted = remstimate(history, statistics)
    result = diagnostics(fitted, history, statistics)
    panels = result.plot_data(which=(1, 2, 3, 4, 5, 9, 10), object=fitted)
    assert set(panels) == {
        "panel1",
        "panel2",
        "panel3",
        "panel4",
        "panel5",
        "panel9",
        "panel10",
    }
    assert all(isinstance(frame, pd.DataFrame) for frame in panels.values())
    assert fitted.plot(which=0) is fitted
    assert fitted.plot(
        reh=history,
        stats=statistics,
        which=(1, 2, 3, 4, 5, 9, 10),
    ) is not None

    typed_history = remify(
        _typed(),
        duration=True,
        extend_riskset_by_type=True,
        riskset="active",
        model="tie",
    )
    typed_statistics = remstats(
        typed_history,
        start_effects='~ inertia(consider_type="interact")',
        psi_start=1,
    )
    typed_fit = remstimate(typed_history, typed_statistics)
    typed_result = diagnostics(typed_fit, typed_history, typed_statistics)
    typed_panel = typed_result.plot_data(which=6, object=typed_fit)["panel6"]
    assert set(typed_panel["event_type"]) == {"X", "Y"}


def test_duration_stack_has_unique_dyads_per_time_and_subset_time_bounds():
    history = remify(
        _typed(),
        duration=True,
        extend_riskset_by_type=True,
        riskset="active",
        model="tie",
    )
    statistics = remstats(
        history,
        start_effects='~ inertia(consider_type="interact")',
        psi_start=1,
        first=2,
        last=5,
    )
    frame = statistics.stacked.remstats_stack
    for _, group in frame.groupby("time", sort=False):
        assert group["dyad"].is_unique
    assert frame["time"].min() >= statistics.stacked.subset[0]
    assert frame["time"].max() <= statistics.stacked.subset[1]


def test_duration_glmm_retains_random_effects_and_uses_them_in_diagnostics():
    history = remify(_simple(), duration=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + reciprocity()",
        end_effects="~ inertia()",
        psi_start=1,
    )
    fitted = remstimate(
        history,
        statistics,
        random="~ (1 | actor1)",
        variance_iterations=6,
        maxiter=150,
    )

    assert isinstance(fitted, RemEstimateDuration)
    assert fitted.metadata["method"] == "GLMM"
    assert fitted.metadata["engine"] == "scipy-laplace"
    assert fitted.metadata["estimator_engine"] == "scipy"
    assert "actor1::(Intercept)" in fitted.random_effects
    assert (fitted.variance_components > 0).all()
    assert len(fitted.fitted_values) == len(statistics.stacked.remstats_stack)
    assert np.isfinite(fitted.fitted_values).all()
    fitted_diagnostics = diagnostics(fitted, history, statistics)
    assert isinstance(fitted_diagnostics, DurationDiagnostics)
    assert fitted_diagnostics.use_ranef is True
    assert fitted_diagnostics.recall_joint


def test_duration_frailty_wrapper_and_ordinal_conditional_engine():
    exact_history = remify(_simple(), duration=True, model="tie")
    exact_stats = remstats(exact_history, start_effects="~ inertia()")
    exact = remfrailty(
        exact_history,
        exact_stats,
        variance_iterations=4,
        maxiter=100,
    )
    assert isinstance(exact, RemEstimateDuration)
    assert set(exact.random_effects) == {
        "actor1::(Intercept)",
        "actor2::(Intercept)",
    }

    ordinal_history = remify(
        _typed_ordinal(),
        duration=True,
        model="tie",
        ordinal=True,
    )
    ordinal_stats = remstats(ordinal_history, start_effects="~ inertia()")
    ordinal = remstimate(
        ordinal_history,
        ordinal_stats,
        random="~ (1 | actor1)",
        variance_iterations=4,
        maxiter=100,
    )
    assert isinstance(ordinal, RemEstimateDuration)
    assert ordinal.metadata["engine"] == "scipy-conditional-laplace"
    assert np.isfinite(ordinal.log_likelihood)


def test_duration_mixrem_has_component_recall():
    history = remify(_simple(), duration=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + reciprocity()",
        end_effects="~ inertia()",
    )
    fitted = remstimate(
        history,
        statistics,
        mixture={
            "k": 2,
            "random": "~ (1 | dyad)",
            "nrep": 1,
            "maxiter": 30,
        },
        seed=23,
    )
    assert isinstance(fitted, RemEstimateMixture)
    fitted_diagnostics = diagnostics(fitted, history, statistics)
    assert fitted_diagnostics.recall
    assert len(fitted_diagnostics.recall_by_component) == 2


def test_duration_bayesian_shrinkage_preserves_start_end_diagnostics():
    history = remify(_simple(), duration=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + reciprocity()",
        end_effects="~ inertia()",
    )
    fitted = remstimate(
        history,
        statistics,
        approach="Bayesian",
        penalty={"prior": "horseshoe", "lambda": 0.5},
        seed=29,
    )
    assert isinstance(fitted, RemEstimateShrinkage)
    assert {"baseline.start", "baseline.end"}.issubset(fitted.unpenalized)
    fitted_diagnostics = diagnostics(fitted, history, statistics)
    assert isinstance(fitted_diagnostics, DurationDiagnostics)
    assert fitted_diagnostics.recall_joint
    assert fitted_diagnostics.recall_start
    assert fitted_diagnostics.recall_end
