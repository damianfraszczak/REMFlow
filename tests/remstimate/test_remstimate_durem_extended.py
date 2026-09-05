"""Extended duration-estimator behavior."""

import numpy as np
import pandas as pd
import pytest

from remflow import RemEstimateDuration, remify, remstats, remstimate


def _larger_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": list(range(1, 11)),
            "actor1": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "actor2": ["B", "C", "A", "C", "A", "B", "B", "C", "A", "C"],
            "end": list(range(5, 15)),
        }
    )


def test_duration_psi_zero_mixed_effects_and_full_result_fields():
    history = remify(_larger_events(), duration=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + activeTie()",
        end_effects="~ inertia()",
        psi_start=0,
        psi_end=0,
    )
    fitted = remstimate(history, statistics, method="MLE")

    assert isinstance(fitted, RemEstimateDuration)
    assert "activeTie.start" in fitted.names
    assert np.isfinite(fitted.coef).all()
    assert fitted.vcov is not None
    assert fitted.vcov.shape == (len(fitted.coef), len(fitted.coef))
    assert fitted.se is not None
    assert (fitted.se >= 0).all()
    assert fitted.model_deviance == pytest.approx(fitted.null_deviance - fitted.residual_deviance)
    assert fitted.to_dict()["df.null"] == fitted.stacked_data.E
    assert fitted.summary()["coefsTab"].shape == (len(fitted.coef), 4)


def test_duration_directed_end_larger_multi_effect_model():
    events = _larger_events().assign(who_ended="actor1")
    history = remify(
        events,
        duration=True,
        model="tie",
        dur_directed_end=True,
    )
    statistics = remstats(
        history,
        start_effects="~ inertia() + outdegreeSender()",
        end_effects="~ inertia() + indegreeSender()",
    )
    fitted = remstimate(history, statistics)

    assert isinstance(fitted, RemEstimateDuration)
    assert fitted.names == [
        "baseline.start",
        "inertia.start",
        "outdegreeSender.start",
        "baseline.end",
        "inertia.end",
        "indegreeSender.end",
    ]
    assert fitted.stacked_data is not None
    assert fitted.stacked_data.D_end == fitted.stacked_data.D_start
    assert np.isfinite(fitted.coef).all()


def test_duration_argument_validation_never_silently_ignores_wrong_effect_family():
    events = _larger_events()
    history = remify(events, duration=True, model="tie")
    with pytest.raises(ValueError, match="start_effects.*end_effects"):
        remstats(history, tie_effects="~ inertia()")
    with pytest.raises(NotImplementedError, match="not yet supported"):
        remstats(history, sender_effects="~ inertia()")
    with pytest.raises(ValueError, match="at least one"):
        remstats(history)

    ordinary = remify(events[["time", "actor1", "actor2"]], model="tie", ordinal=True)
    with pytest.raises(ValueError, match="duration history"):
        remstats(ordinary, start_effects="~ inertia()")


def test_duration_interact_without_expanded_riskset_and_overlapping_ties_smoke():
    typed = _larger_events().assign(type=["X", "Y"] * 5)
    typed_history = remify(
        typed,
        duration=True,
        model="tie",
        extend_riskset_by_type=False,
    )
    typed_stats = remstats(
        typed_history,
        start_effects='~ activeTie(consider_type="interact")',
        first=1,
    )
    assert typed_stats.stacked.stat_names == [
        "baseline.start",
        "activeTie.X.start",
        "activeTie.Y.start",
    ]

    overlap = pd.DataFrame(
        {
            "time": [1, 3, 5, 7],
            "actor1": ["A", "A", "B", "A"],
            "actor2": ["B", "B", "C", "B"],
            "duration": [4, 2, 3, 1],
        }
    )
    overlap_history = remify(overlap, duration=True, model="tie")
    overlap_stats = remstats(
        overlap_history,
        start_effects="~ activeTie() + inertia()",
        psi_start=1,
    )
    overlap_fit = remstimate(overlap_history, overlap_stats)
    assert isinstance(overlap_fit, RemEstimateDuration)
    assert np.isfinite(overlap_fit.coef).all()


def test_ordinal_simultaneous_duration_cases_match_jax_exact_likelihood():
    pytest.importorskip("jax")
    events = pd.DataFrame(
        {
            "time": [1, 2, 4, 4, 7, 9],
            "actor1": ["A", "B", "C", "A", "B", "C"],
            "actor2": ["B", "C", "A", "C", "A", "B"],
            "end": [4, 6, 8, 9, 10, 12],
        }
    )
    history = remify(events, duration=True, ordinal=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        first=1,
    )
    assert statistics.stacked.remstats_stack.groupby("time_index")["obs"].sum().max() > 1

    numpy_fit = remstimate(history, statistics, backend="numpy")
    jax_fit = remstimate(history, statistics, backend="jax:cpu")
    chunked_fit = remstimate(
        history,
        statistics,
        backend="jax:cpu",
        riskset_chunk_size=2,
    )
    assert isinstance(numpy_fit, RemEstimateDuration)
    assert isinstance(jax_fit, RemEstimateDuration)
    np.testing.assert_allclose(jax_fit.coef, numpy_fit.coef, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(chunked_fit.coef, numpy_fit.coef, rtol=1e-6, atol=1e-7)
    assert jax_fit.log_likelihood == pytest.approx(numpy_fit.log_likelihood, rel=1e-8, abs=1e-9)
    assert chunked_fit.log_likelihood == pytest.approx(
        numpy_fit.log_likelihood, rel=1e-8, abs=1e-9
    )
    assert chunked_fit.metadata["riskset_chunk_size"] == 2
