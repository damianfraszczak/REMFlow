"""Moving-window estimation for duration relational event models."""

import numpy as np
import pandas as pd
import pytest

from remflow import (
    RemEstimateDuration,
    RemEstimateWindow,
    diagnostics,
    remify,
    remstats,
    remwindow,
)
from remflow.estimate import WindowDiagnostics


def _duration_inputs(*, ordinal: bool):
    event_count = 50
    actors = np.asarray(["A", "B", "C", "D"])
    starts = np.arange(1, event_count + 1, dtype=float)
    events = pd.DataFrame(
        {
            "time": starts,
            "actor1": np.resize(actors, event_count),
            "actor2": np.resize(np.roll(actors, -1), event_count),
            "end": starts + 2.5,
            "type": np.resize(np.asarray(["reply", "mention"]), event_count),
        }
    )
    history = remify(
        events,
        duration=True,
        model="tie",
        ordinal=ordinal,
        extend_riskset_by_type=True,
    )
    effects = "~ 0 + inertia()" if ordinal else "~ 1"
    statistics = remstats(
        history,
        start_effects=effects,
        end_effects=effects,
    )
    return history, statistics


@pytest.mark.parametrize("ordinal", [False, True])
def test_duration_windows_fit_complete_start_end_time_strata(ordinal):
    history, statistics = _duration_inputs(ordinal=ordinal)

    fitted = remwindow(history, statistics, n_windows=2, min_events=1)

    assert isinstance(fitted, RemEstimateWindow)
    assert fitted.type == "duration"
    assert fitted.n_windows == 2
    assert all(isinstance(model, RemEstimateDuration) for model in fitted.fits)
    assert all(model.converged for model in fitted.fits if isinstance(model, RemEstimateDuration))
    assert fitted.windows["n_events"].sum() == statistics.stacked.E
    assert fitted.windows["n_strata"].sum() == statistics.stacked.E
    assert fitted.metadata["window_unit"] == "duration_time_strata"

    source = statistics.stacked.remstats_stack.reset_index(drop=True)
    reconstructed = pd.concat(
        [item.stacked.remstats_stack for item in fitted.window_stats],
        ignore_index=True,
    )
    pd.testing.assert_frame_equal(reconstructed, source)
    first_times = set(fitted.window_stats[0].stacked.remstats_stack["time_index"])
    second_times = set(fitted.window_stats[1].stacked.remstats_stack["time_index"])
    assert first_times.isdisjoint(second_times)
    assert all("type" in item.stacked.remstats_stack for item in fitted.window_stats)

    coefficients = fitted.coefficients()
    assert coefficients["coefficients"].shape == (2, 2)
    assert fitted.summary()["type"] == "duration"
    assert set(fitted.plot_data()["component"]) == {"duration"}


@pytest.mark.parametrize("ordinal", [False, True])
def test_duration_window_diagnostics_report_joint_start_and_end_recall(ordinal):
    history, statistics = _duration_inputs(ordinal=ordinal)
    fitted = remwindow(history, statistics, n_windows=2, min_events=1)

    result = diagnostics(fitted, history, statistics)

    assert isinstance(result, WindowDiagnostics)
    assert result.type == "duration"
    assert result.recall is not None
    assert result.start is not None
    assert result.end is not None
    assert set(result.plot_data()["component"]) == {"duration", "start", "end"}
