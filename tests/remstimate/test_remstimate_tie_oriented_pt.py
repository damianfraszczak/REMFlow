"""Exact-time tie-oriented estimator regression tests."""

import numpy as np
import pandas as pd
import pytest

from remflow import AIC, AICC, BIC, WAIC, RemEstimate, diagnostics, remify, remstats, remstimate
from remflow.estimate import _tie_loglik_grad_hessian


def _events(*, simultaneous: bool = True) -> pd.DataFrame:
    times = [1, 2, 3, 3, 4, 5, 6, 7, 8, 9] if simultaneous else list(range(1, 11))
    return pd.DataFrame(
        {
            "time": times,
            "actor1": [1, 2, 1, 3, 2, 1, 3, 2, 4, 1],
            "actor2": [2, 3, 4, 5, 1, 3, 2, 4, 1, 5],
        }
    )


def _stable_events() -> pd.DataFrame:
    rng = np.random.default_rng(1847)
    senders = rng.integers(1, 6, size=60)
    receivers = rng.integers(1, 5, size=60)
    receivers += receivers >= senders
    return pd.DataFrame(
        {
            "time": np.cumsum(rng.integers(1, 4, size=60)),
            "actor1": senders,
            "actor2": receivers,
        }
    )


def test_tie_mle_result_contract_default_method_and_simultaneous_events():
    history = remify(_events(), model="tie", ordinal=False)
    statistics = remstats(
        history,
        tie_effects="~ indegreeSender() + inertia() + reciprocity()",
        first=1,
    )
    default = remstimate(history, statistics)
    fitted = remstimate(history, statistics, method="MLE")

    assert isinstance(fitted, RemEstimate)
    np.testing.assert_allclose(default.coef, fitted.coef, rtol=0, atol=0)
    assert fitted.names == [
        "baseline",
        "indegreeSender",
        "inertia",
        "reciprocity",
    ]
    assert fitted.approach == "Frequentist"
    assert fitted.method == "MLE"
    assert fitted.model == "tie"
    assert fitted.ordinal is False
    assert fitted.sampled is False
    assert np.isfinite(fitted.coef).all()
    assert np.isfinite(fitted.loglik)
    assert fitted.gradient is not None
    assert fitted.hessian is not None
    assert fitted.gradient.shape == fitted.coef.shape
    assert fitted.hessian.shape == (len(fitted.coef), len(fitted.coef))
    np.testing.assert_allclose(fitted.hessian, fitted.hessian.T, rtol=0, atol=1e-10)
    assert fitted.vcov is not None
    assert fitted.se is not None
    assert fitted.vcov.shape == fitted.hessian.shape
    assert fitted.se.shape == fitted.coef.shape
    assert fitted.residual_deviance == pytest.approx(-2 * fitted.loglik)
    assert fitted.model_deviance == pytest.approx(
        fitted.null_deviance - fitted.residual_deviance
    )
    assert fitted.df_null == history.M
    assert fitted.df_model == len(fitted.coef)
    assert fitted.df_residual == fitted.df_null - fitted.df_model
    assert fitted.AIC == AIC(fitted)
    assert fitted.AICC == AICC(fitted)
    assert fitted.BIC == BIC(fitted)
    assert len(fitted.event_probabilities) == history.E

    expected_fields = {
        "coefficients",
        "loglik",
        "gradient",
        "hessian",
        "vcov",
        "se",
        "residual.deviance",
        "null.deviance",
        "model.deviance",
        "df.null",
        "df.model",
        "df.residual",
        "AIC",
        "AICC",
        "BIC",
        "converged",
        "iterations",
        "sampled",
    }
    assert expected_fields.issubset(fitted.to_dict())


def test_ordinal_tie_mle_drops_baseline_and_handles_exact_tied_cases_on_jax():
    pytest.importorskip("jax")
    history = remify(_events(), model="tie", ordinal=True)
    statistics = remstats(
        history, tie_effects="~ inertia() + reciprocity()", first=1
    )
    numpy_fit = remstimate(history, statistics, backend="numpy")
    jax_fit = remstimate(history, statistics, backend="jax:cpu")

    assert numpy_fit.names == ["inertia", "reciprocity"]
    assert numpy_fit.metadata["where_is_baseline"] is None
    assert any(len(group) > 1 for group in statistics.observed_index_groups)
    np.testing.assert_allclose(jax_fit.coef, numpy_fit.coef, rtol=1e-6, atol=1e-7)
    assert jax_fit.loglik == pytest.approx(numpy_fit.loglik, rel=1e-8, abs=1e-9)
    assert len(numpy_fit.event_probabilities) == history.E


@pytest.mark.parametrize("ordinal", [False, True])
def test_tie_score_and_hessian_match_finite_differences_with_simultaneous_cases(
    ordinal,
):
    designs = [
        np.asarray(
            [[1.0, 0.0, -0.2], [1.0, 0.5, 0.4], [1.0, 1.0, -0.1], [1.0, -0.3, 0.7]]
        ),
        np.asarray([[1.0, 0.2, 0.1], [1.0, -0.4, 0.8], [1.0, 0.7, -0.5]]),
    ]
    observed = [[0, 2], [1]]
    exposures = None if ordinal else np.asarray([0.4, 1.2])
    beta = np.asarray([-0.25, 0.3, -0.15])
    loglik, gradient, hessian = _tie_loglik_grad_hessian(
        beta,
        designs,
        observed,
        exposures=exposures,
        sampling_weights=None,
    )
    epsilon = 1e-6
    numerical_gradient = np.empty_like(beta)
    numerical_hessian = np.empty_like(hessian)
    for column in range(len(beta)):
        step = np.zeros_like(beta)
        step[column] = epsilon
        plus = _tie_loglik_grad_hessian(
            beta + step,
            designs,
            observed,
            exposures=exposures,
            sampling_weights=None,
        )
        minus = _tie_loglik_grad_hessian(
            beta - step,
            designs,
            observed,
            exposures=exposures,
            sampling_weights=None,
        )
        numerical_gradient[column] = (plus[0] - minus[0]) / (2 * epsilon)
        numerical_hessian[:, column] = (plus[1] - minus[1]) / (2 * epsilon)

    assert np.isfinite(loglik)
    np.testing.assert_allclose(gradient, numerical_gradient, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(hessian, numerical_hessian, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0, atol=1e-12)


def test_baseline_only_tie_diagnostics_are_nonempty():
    history = remify(_events(simultaneous=False), model="tie", ordinal=False)
    statistics = remstats(history, tie_effects="~ 1", first=1)
    fitted = remstimate(history, statistics)
    result = diagnostics(fitted, history, statistics)

    assert result.residuals.shape == (history.E,)
    assert result.observed_probabilities.shape == (history.E,)
    assert np.isfinite(result.observed_probabilities).all()


def test_hmc_result_contract_is_reproducible_and_supports_waic():
    history = remify(
        _stable_events(),
        actors=[1, 2, 3, 4, 5],
        model="tie",
        ordinal=False,
    )
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity()",
        first=6,
    )
    controls = {
        "nsim": 8,
        "nchains": 2,
        "burnin": 4,
        "thin": 1,
        "L": 4,
        "epsilon": 0.002,
    }
    fitted = remstimate(
        history,
        statistics,
        method="HMC",
        bayes=controls,
        WAIC=True,
        seed=827,
    )
    repeated = remstimate(
        history,
        statistics,
        approach="Bayesian",
        bayes=controls,
        WAIC=True,
        seed=827,
    )

    assert fitted.method == "HMC"
    assert fitted.approach == "Bayesian"
    assert fitted.draws is not None
    assert fitted.log_posterior is not None
    assert fitted.posterior_mean is not None
    assert fitted.posterior_sd is not None
    assert fitted.draws.shape == (16, len(fitted.names))
    assert fitted.log_posterior.shape == (16,)
    assert np.isfinite(fitted.draws).all()
    assert np.isfinite(fitted.log_posterior).all()
    assert 0.0 <= fitted.metadata["acceptance_rate"] <= 1.0
    assert fitted.metadata["rng"] == "numpy.random.Generator"
    assert fitted.metadata["seed"] == 827
    assert list(fitted.to_dict()) == [
        "draws",
        "log_posterior",
        "coefficients",
        "post.mean",
        "vcov",
        "sd",
        "loglik",
        "sampled",
        "df.null",
    ]
    assert np.isfinite(WAIC(fitted))
    np.testing.assert_array_equal(fitted.draws, repeated.draws)
    np.testing.assert_array_equal(fitted.log_posterior, repeated.log_posterior)

    for information_criterion in (AIC, AICC, BIC):
        with pytest.raises(ValueError, match="Frequentist"):
            information_criterion(fitted)


def test_hmc_supports_case_control_statistics_and_stays_near_the_mle():
    history = remify(
        _stable_events(),
        actors=[1, 2, 3, 4, 5],
        model="tie",
        ordinal=True,
    )
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity()",
        first=6,
        sampling=True,
        samp_num=6,
        seed=91,
    )
    mle = remstimate(history, statistics)
    fitted = remstimate(
        history,
        statistics,
        method="HMC",
        nsim=12,
        nchains=1,
        burnin=4,
        thin=1,
        L=4,
        epsilon=0.002,
        seed=304,
    )

    assert fitted.draws is not None
    assert fitted.posterior_mean is not None
    assert fitted.sampled is True
    assert fitted.draws.shape == (12, len(mle.coef))
    assert np.max(np.abs(fitted.posterior_mean - mle.coef)) < 1.0
    assert mle.to_dict()["samp_num"] == 6
    assert mle.to_dict()["sampling_scheme"] == "case-control"
    assert len(mle.to_dict()) == 20


def test_hmc_supports_an_ordinal_baseline_only_zero_parameter_model():
    history = remify(_events(simultaneous=False), model="tie", ordinal=True)
    statistics = remstats(history, tie_effects="~ 1", first=1)
    fitted = remstimate(
        history,
        statistics,
        method="HMC",
        nsim=4,
        nchains=1,
        burnin=2,
        L=2,
        seed=19,
    )

    assert fitted.draws is not None
    assert fitted.draws.shape == (4, 0)
    assert fitted.coef.shape == (0,)
    assert fitted.vcov is not None
    assert fitted.vcov.shape == (0, 0)
    with pytest.raises(ValueError, match="posterior_log_likelihood"):
        WAIC(fitted)


@pytest.mark.parametrize("ordinal", [False, True])
def test_frequentist_waic_is_reproducible_and_exposed_on_the_result(ordinal):
    history = remify(
        _stable_events(),
        actors=[1, 2, 3, 4, 5],
        model="tie",
        ordinal=ordinal,
    )
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity()",
        first=6,
    )
    first = remstimate(
        history,
        statistics,
        WAIC=True,
        nsimWAIC=40,
        seed=912,
    )
    repeated = remstimate(
        history,
        statistics,
        WAIC=True,
        nsimWAIC=40,
        seed=912,
    )

    assert np.isfinite(WAIC(first))
    assert WAIC(first) == pytest.approx(WAIC(repeated), rel=0, abs=0)
    assert first.metadata["WAIC"] == WAIC(first)
    assert first.metadata["nsimWAIC"] == 40


def test_waic_and_nsim_waic_validation_matches_documented_meaning():
    history = remify(_stable_events(), actors=[1, 2, 3, 4, 5], model="tie")
    statistics = remstats(history, tie_effects="~ inertia()")

    with pytest.raises(TypeError, match="WAIC"):
        remstimate(history, statistics, WAIC=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="WAIC"):
        remstimate(history, statistics, WAIC=20)  # type: ignore[arg-type]
    with pytest.warns(UserWarning, match="nsimWAIC"):
        fitted = remstimate(
            history,
            statistics,
            WAIC=True,
            nsimWAIC="text",
            seed=8,
        )
    assert fitted.metadata["nsimWAIC"] == 100


def test_tie_plot_dispatch_effect_validation_and_hmc_panels():
    history = remify(
        _stable_events(), actors=[1, 2, 3, 4, 5], model="tie", ordinal=False
    )
    statistics = remstats(
        history, tie_effects="~ inertia() + reciprocity()", first=6, last=35
    )
    mle = remstimate(history, statistics)
    result = diagnostics(mle, history, statistics)

    assert result.plot(which=2, effects="inertia") is result
    assert mle.plot(reh=history, diagnostics=result, which=(1, 2)) is result
    assert isinstance(mle.plot(reh=history, stats=statistics, which=1), type(result))
    with pytest.raises(ValueError, match="not found"):
        result.plot(which=2, effects="INERTIA")

    hmc = remstimate(
        history,
        statistics,
        method="HMC",
        nchains=2,
        nsim=6,
        burnin=3,
        L=4,
        epsilon=0.002,
        seed=23929,
    )
    hmc_result = diagnostics(hmc, history, statistics)
    panels = hmc_result.plot_data(which=(3, 4), object=hmc)
    assert set(panels) == {"panel3", "panel4"}
    assert panels["panel4"]["chain"].nunique() == 2


def test_ordinal_statistics_can_drive_equivalent_exact_time_history():
    events = _stable_events()
    ordinal_history = remify(
        events, actors=[1, 2, 3, 4, 5], model="tie", ordinal=True
    )
    exact_history = remify(
        events, actors=[1, 2, 3, 4, 5], model="tie", ordinal=False
    )
    ordinal_statistics = remstats(
        ordinal_history,
        tie_effects="~ inertia() + reciprocity()",
        first=6,
        last=35,
    )
    fitted = remstimate(
        exact_history,
        ordinal_statistics,
        WAIC=True,
        nsimWAIC=20,
        seed=44,
    )

    assert fitted.ordinal is False
    assert fitted.names == ["baseline", "inertia", "reciprocity"]
    assert np.isfinite(WAIC(fitted))
    result = diagnostics(fitted, exact_history, ordinal_statistics)
    assert result.residuals.size == len(ordinal_statistics.observed_indices)


def test_active_riskset_supports_tie_mle_diagnostics_and_hmc():
    history = remify(
        _stable_events(),
        actors=[1, 2, 3, 4, 5],
        model="tie",
        riskset="active",
    )
    statistics = remstats(
        history, tie_effects="~ inertia() + reciprocity()", first=6, last=35
    )
    mle = remstimate(history, statistics, method="MLE")
    hmc = remstimate(
        history,
        statistics,
        method="HMC",
        nchains=1,
        nsim=5,
        burnin=2,
        L=3,
        epsilon=0.002,
        seed=23929,
    )

    assert mle.converged
    assert diagnostics(mle, history, statistics).residuals.size > 0
    assert hmc.draws is not None and hmc.draws.shape == (5, 3)
