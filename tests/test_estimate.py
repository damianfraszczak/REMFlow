import numpy as np
import pandas as pd
import pytest

from remflow import WAIC, RemEstimate, fit_rem, remify, rempenalty, remstats, remstimate


def test_basic_frequentist_estimate_runs_on_numpy_backend():
    history = remify(
        pd.DataFrame(
            {
                "sender": ["A", "B", "A", "A"],
                "receiver": ["B", "A", "B", "B"],
            }
        ),
        actors=["A", "B"],
        riskset="full",
        ordinal=True,
    )
    stats = remstats(history, tie_effects="~ inertia() + reciprocity()", first=2)

    fit = remstimate(history, stats)
    explicit = remstimate(history, stats, engine="scipy")

    assert isinstance(fit, RemEstimate)
    assert fit.names == ["inertia", "reciprocity"]
    assert fit.coef.shape == (2,)
    assert fit.metadata["backend"] == "numpy"
    assert fit.metadata["device"] == "cpu"
    assert fit.metadata["estimator_engine"] == "scipy"
    assert explicit.metadata["estimator_engine"] == "scipy"
    np.testing.assert_allclose(explicit.coef, fit.coef, rtol=0, atol=0)


def test_estimator_engine_registry_rejects_unknown_or_invalid_names():
    history = remify(
        pd.DataFrame(
            {
                "sender": ["A", "B", "A"],
                "receiver": ["B", "A", "B"],
            }
        ),
        actors=["A", "B"],
        ordinal=True,
    )
    stats = remstats(history, tie_effects="~ inertia()", first=2)

    with pytest.raises(ValueError, match="auto, scipy"):
        remstimate(history, stats, engine="unknown")
    with pytest.raises(TypeError, match="engine must be a string"):
        remstimate(history, stats, engine=1)  # type: ignore[arg-type]


def test_pipeline_fit_rem():
    fit = fit_rem(
        pd.DataFrame(
            {
                "sender": ["A", "B", "A", "A"],
                "receiver": ["B", "A", "B", "B"],
            }
        ),
        actors=["A", "B"],
    )

    assert isinstance(fit, RemEstimate)


def test_waic_uses_posterior_event_log_likelihood_draws():
    draws = np.log(np.array([[0.7, 0.4], [0.8, 0.5], [0.6, 0.3]]))
    fit = RemEstimate(
        coef=np.array([0.0]),
        names=["x"],
        log_likelihood=-1.0,
        converged=True,
        covariance=None,
        metadata={"posterior_log_likelihood": draws},
    )
    maxima = draws.max(axis=0)
    expected = -2 * (
        np.sum(maxima + np.log(np.mean(np.exp(draws - maxima), axis=0)))
        - np.sum(np.var(draws, axis=0, ddof=1))
    )

    assert WAIC(fit) == pytest.approx(expected)


def test_penalty_wrapper_validates_history_and_statistics():
    with pytest.raises(TypeError, match="history must be an EventHistory"):
        rempenalty(None, None)  # type: ignore[arg-type]


def test_exact_time_baseline_matches_closed_form_rate():
    history = remify(
        pd.DataFrame(
            {
                "time": [1.0, 2.0, 3.0],
                "sender": ["A", "B", "A"],
                "receiver": ["B", "A", "B"],
            }
        ),
        actors=["A", "B"],
        ordinal=False,
    )
    stats = remstats(history, tie_effects="~ 1", first=2)

    fit = remstimate(history, stats)

    assert fit.converged
    assert fit.names == ["baseline"]
    assert fit.coef[0] == pytest.approx(-np.log(2.0), abs=1e-6)
    assert fit.log_likelihood == pytest.approx(-2.0 - 2.0 * np.log(2.0), abs=1e-6)
    assert fit.metadata["timing"] == "exact"
