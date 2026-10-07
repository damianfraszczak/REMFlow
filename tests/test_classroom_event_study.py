from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.classroom_event_study import (
    PLOTTED_TERMS,
    build_classroom_stats,
    build_history,
    fit_study_model,
    load_classroom_data,
    save_classroom_plot,
    study_model_terms,
)
from remflow import RemEstimate, remstimate


def test_classroom_data_preparation_matches_source_counts():
    date1 = load_classroom_data("date1")
    date2 = load_classroom_data("date2")

    assert len(date1.actors) == 18
    assert len(date1.interactions) == 228
    assert len(date2.interactions) == 284
    assert date1.seating.shape == (18, 18)
    assert date1.friendship.shape == (18, 18)
    assert int(date1.attributes["male"].sum()) == 6
    assert int((date1.attributes["male"] == 0).sum()) == 12


def test_classroom_partial_model_estimates():
    data = load_classroom_data("date1")
    data = replace(data, interactions=data.interactions.iloc[:30].copy())
    history = build_history(data)
    stats = build_classroom_stats(history, data, terms=study_model_terms("mod3b"))

    fit = remstimate(history, stats)

    assert isinstance(fit, RemEstimate)
    assert fit.names == ["Intercept", "Sender_male", "Receiver_male", "Seating", "Friendship"]
    assert fit.converged


def test_classroom_full_intercept_model_estimates():
    _, fit = fit_study_model("date1", study_model_terms("mod1"))

    assert fit.names == ["Intercept"]
    assert fit.converged


def test_classroom_exact_time_partial_model_estimates():
    data = load_classroom_data("date1")
    data = replace(data, interactions=data.interactions.iloc[:30].copy())
    history = build_history(data, ordinal=False)
    stats = build_classroom_stats(history, data, terms=study_model_terms("mod3b"))

    fit = remstimate(history, stats)

    assert fit.converged
    assert fit.ordinal is False
    assert fit.names == ["Intercept", "Sender_male", "Receiver_male", "Seating", "Friendship"]


@pytest.mark.skipif(find_spec("matplotlib") is None, reason="optional plotting dependency missing")
def test_classroom_figure_contains_coefficient_and_rank_panels(tmp_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    frame = pd.DataFrame(
        {
            "effect": PLOTTED_TERMS,
            "estimate": np.arange(len(PLOTTED_TERMS), dtype=float) / 10,
            "std_error": np.full(len(PLOTTED_TERMS), 0.1),
        }
    )
    report = SimpleNamespace(
        ranks=np.array([1, 1, 3, 2]), recall={"summary": {"top_pct_prop": 0.75}}
    )
    output = tmp_path / "figure" / "classroom_diagnostics.svg"

    save_classroom_plot({"date1": (frame, report), "date2": (frame, report)}, output, ordinal=False)

    svg = output.read_text(encoding="utf-8")
    assert "Conditional log-hazard coefficient" in svg
    assert "In-sample event ranking" in svg
    assert "PSAB_BA" in svg
