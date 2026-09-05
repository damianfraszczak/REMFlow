"""Duration-estimator contract and numerical checks."""

import numpy as np
import pandas as pd
import pytest

from remflow import (
    AIC,
    AICC,
    BIC,
    DurationDiagnostics,
    RemEstimateDuration,
    diagnostics,
    remify,
    remstats,
    remstimate,
)
from remflow.estimate import _duration_loglik_grad_hessian


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 5, 6, 9, 11, 14, 16],
            "actor1": ["A", "B", "A", "C", "B", "D", "C", "A"],
            "actor2": ["B", "C", "C", "A", "D", "A", "D", "D"],
            "end": [4, 7, 8, 10, 13, 15, 18, 20],
        }
    )


def _fit(*, ordinal: bool = False, backend: str = "numpy") -> RemEstimateDuration:
    history = remify(_events(), duration=True, ordinal=ordinal, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia()",
        end_effects="~ inertia()",
        psi_start=1,
        psi_end=1,
    )
    fitted = remstimate(history, statistics, backend=backend)
    assert isinstance(fitted, RemEstimateDuration)
    return fitted


def test_interval_duration_result_contract_information_criteria_and_summary():
    fitted = _fit()

    assert fitted.names == [
        "baseline.start",
        "inertia.start",
        "baseline.end",
        "inertia.end",
    ]
    assert np.isfinite(fitted.coef).all()
    assert np.isfinite(fitted.log_likelihood)
    assert fitted.log_likelihood < 0
    assert fitted.metadata["model"] == "tie"
    assert fitted.metadata["method"] == "MLE"
    assert fitted.metadata["engine"] == "glm"
    assert fitted.metadata["ordinal"] is False
    assert fitted.stacked_data is not None
    assert fitted.backend_fit["engine"] == "glm"
    assert fitted.gradient is None
    assert fitted.hessian is None
    assert fitted.vcov is not None
    assert fitted.vcov.shape == (len(fitted.coef), len(fitted.coef))
    assert fitted.se is not None
    assert fitted.summary()["coefficients"].shape == (4, 4)
    assert fitted.summary()["coefsTab"].shape == (4, 4)
    assert "Duration relational-event model" in str(fitted)

    assert np.isfinite(AIC(fitted))
    assert np.isfinite(AICC(fitted))
    assert np.isfinite(BIC(fitted))
    assert AICC(fitted) >= AIC(fitted)
    assert fitted.AIC == AIC(fitted)
    assert fitted.AICC == AICC(fitted)
    assert fitted.BIC == BIC(fitted)
    assert fitted.df_null == fitted.stacked_data.E
    assert fitted.df_model == len(fitted.coef)
    assert fitted.df_residual == fitted.df_null - fitted.df_model
    assert fitted.residual_deviance == pytest.approx(-2 * fitted.log_likelihood)
    assert fitted.model_deviance == pytest.approx(fitted.null_deviance - fitted.residual_deviance)
    as_dict = fitted.to_dict()
    assert as_dict["loglik"] == fitted.log_likelihood
    assert as_dict["stacked_data"] is fitted.stacked_data


def test_duration_diagnostics_expose_joint_start_end_recall_and_residuals():
    fitted = _fit()
    result = diagnostics(fitted)

    assert isinstance(result, DurationDiagnostics)
    assert result.deviance_residuals.shape == fitted.fitted_values.shape
    assert result.pearson_residuals.shape == fitted.fitted_values.shape
    for recall in (result.recall_joint, result.recall_start, result.recall_end):
        assert recall is not None
        assert list(recall["summary"]) == [
            "mean_rel_rank",
            "median_rel_rank",
            "mean_cum_prob",
            "mean_prob_ratio",
            "mean_log_loss",
            "top_pct",
            "top_pct_prop",
        ]
        mean_rank = recall["summary"].iloc[0]["mean_rel_rank"]
        assert 0 <= mean_rank <= 1
        assert len(recall["per_event"]) > 0
    assert result.residual_summary is not None
    assert list(result.residual_summary) == ["min", "q1", "median", "q3", "max"]
    assert result.surprise_threshold == 0.2
    assert "Joint" in str(result)
    assert "Start" in str(result)
    assert "End" in str(result)


@pytest.mark.parametrize("ordinal", [False, True])
def test_duration_score_and_hessian_match_central_finite_differences(ordinal):
    design = np.asarray(
        [
            [1.0, 0.0, -0.5],
            [1.0, 1.0, 0.2],
            [1.0, -0.5, 0.7],
            [0.0, 1.0, 0.1],
            [0.0, 1.0, -0.3],
            [0.0, 1.0, 0.8],
        ]
    )
    response = (
        np.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        if ordinal
        else np.asarray([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    )
    groups = [np.arange(3), np.arange(3, 6)]
    offset = None if ordinal else np.log(np.asarray([0.5, 0.5, 0.5, 1.2, 1.2, 1.2]))
    beta = np.asarray([0.2, -0.35, 0.1])
    loglik, gradient, hessian = _duration_loglik_grad_hessian(
        beta, design, response, offset=offset, groups=groups
    )
    epsilon = 1e-6
    numerical_gradient = np.empty_like(beta)
    numerical_hessian = np.empty_like(hessian)
    for column in range(len(beta)):
        step = np.zeros_like(beta)
        step[column] = epsilon
        plus = _duration_loglik_grad_hessian(
            beta + step, design, response, offset=offset, groups=groups
        )
        minus = _duration_loglik_grad_hessian(
            beta - step, design, response, offset=offset, groups=groups
        )
        numerical_gradient[column] = (plus[0] - minus[0]) / (2 * epsilon)
        numerical_hessian[:, column] = (plus[1] - minus[1]) / (2 * epsilon)

    assert np.isfinite(loglik)
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12)


def test_interval_duration_loglik_removes_observed_offset_constant():
    design = np.ones((2, 1), dtype=float)
    response = np.asarray([1.0, 0.0])
    offset = np.log(np.asarray([2.0, 2.0]))
    loglik, gradient, hessian = _duration_loglik_grad_hessian(
        np.zeros(1), design, response, offset=offset, groups=[np.arange(2)]
    )

    # Poisson means are two, while the documented likelihood subtracts
    # sum(log(dt)) over observed rows from the backend GLM likelihood.
    assert loglik == pytest.approx(-4.0)
    np.testing.assert_allclose(gradient, [-3.0])
    np.testing.assert_allclose(hessian, [[-4.0]])


def test_ordinal_duration_uses_conditional_likelihood_without_time_offset():
    fitted = _fit(ordinal=True)

    assert fitted.stacked_data is not None
    assert fitted.stacked_data.ordinal
    assert "log_interevent" not in fitted.stacked_data.remstats_stack
    assert fitted.metadata["engine"] == "clogit"
    assert np.isfinite(fitted.coef).all()
    assert np.isfinite(fitted.log_likelihood)


def test_duration_directed_end_censoring_start_only_and_multiple_effects():
    events = _events()
    events.loc[2, "end"] = np.nan
    history = remify(events, duration=True, dur_directed_end=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + outdegreeSender()",
        end_effects="~ inertia()",
    )
    fitted = remstimate(history, statistics)

    assert isinstance(fitted, RemEstimateDuration)
    assert fitted.stacked_data is not None
    assert fitted.stacked_data.D_end == fitted.stacked_data.D_start
    assert "outdegreeSender.start" in fitted.names
    assert np.isfinite(fitted.coef).all()

    start_statistics = remstats(history, start_effects="~ inertia()")
    start_fit = remstimate(history, start_statistics)
    assert isinstance(start_fit, RemEstimateDuration)
    assert not any(name.endswith(".end") for name in start_fit.names)


def test_typed_duration_separate_and_interaction_models_have_expected_columns():
    events = _events().assign(type=["X", "X", "Y", "Y", "X", "Y", "X", "Y"])
    separate_history = remify(events, duration=True, model="tie")
    separate_stats = remstats(
        separate_history,
        start_effects='~ inertia(consider_type="separate")',
    )
    separate_fit = remstimate(separate_history, separate_stats)
    assert isinstance(separate_fit, RemEstimateDuration)
    assert "inertia.X.start" in separate_fit.names
    assert "inertia.Y.start" in separate_fit.names

    interaction_history = remify(events, duration=True, model="tie", extend_riskset_by_type=True)
    interaction_stats = remstats(
        interaction_history,
        start_effects='~ inertia(consider_type="interact")',
    )
    interaction_fit = remstimate(interaction_history, interaction_stats)
    assert isinstance(interaction_fit, RemEstimateDuration)
    interaction_names = [name for name in interaction_fit.names if name.startswith("inertia.")]
    assert len(interaction_names) == 4
    assert "type" in interaction_stats.stacked.remstats_stack
    assert np.isfinite(interaction_fit.coef).all()


def test_duration_numpy_and_jax_cpu_fits_agree():
    pytest.importorskip("jax")
    numpy_fit = _fit(backend="numpy")
    jax_fit = _fit(backend="jax:cpu")

    np.testing.assert_allclose(jax_fit.coef, numpy_fit.coef, rtol=1e-7, atol=1e-8)
    assert jax_fit.log_likelihood == pytest.approx(numpy_fit.log_likelihood, rel=1e-9, abs=1e-9)
    assert jax_fit.metadata["backend"] == "jax"
    assert jax_fit.metadata["device"] == "cpu"
