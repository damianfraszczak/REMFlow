from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from remflow import (
    coefficient_table,
    diagnostic_table,
    diagnostics,
    fit_table,
    plot_coefficients,
    plot_diagnostics,
    plot_event_history,
    plot_statistic,
    remify,
    remstats,
    remstimate,
)


def _problem():
    events = pd.DataFrame(
        {
            "time": range(1, 13),
            "sender": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C", "A", "B"],
            "receiver": ["B", "A", "C", "A", "C", "B", "B", "C", "B", "A", "C", "A"],
        }
    )
    history = remify(events, actors=["A", "B", "C"], ordinal=True)
    statistics = remstats(history, tie_effects="~ inertia() + reciprocity()", first=2)
    fit = remstimate(history, statistics)
    return history, statistics, fit, diagnostics(fit, history, statistics)


def test_readable_summary_tables_have_stable_columns():
    _, _, fit, diagnostic = _problem()
    assert list(coefficient_table(fit)) == ["component", "effect", "estimate", "std_error"]
    assert {
        "component",
        "events",
        "parameters",
        "log_likelihood",
        "AIC",
        "BIC",
        "converged",
        "backend",
    } == set(fit_table(fit))
    assert {
        "component",
        "events",
        "mean_observed_probability",
        "mean_relative_rank",
        "median_relative_rank",
        "top_fraction",
        "observed_in_top_fraction",
    } == set(diagnostic_table(diagnostic))


@pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="optional plotting dependency is not installed",
)
def test_plotting_helpers_return_matplotlib_figures():
    import matplotlib

    matplotlib.use("Agg")
    history, statistics, fit, diagnostic = _problem()
    for figure, _ in (
        plot_event_history(history),
        plot_statistic(statistics, "inertia"),
        plot_coefficients(fit),
        plot_diagnostics(diagnostic),
    ):
        assert figure.axes
        figure.canvas.draw()
