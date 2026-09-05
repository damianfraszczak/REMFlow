"""Estimation for relational event models."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np
import pandas as pd

from remflow.backends import ArrayBackend, JaxBackend, resolve_backend
from remflow.history import EventHistory
from remflow.stats import (
    AomStats,
    Effect,
    Formula,
    RemStats,
    RemStatsDuration,
    RemStatsStackedDuration,
    remstats,
)


@dataclass(frozen=True)
class _EstimatorEngine:
    """Configuration for a supported optimization engine."""

    name: str
    optimizer: str


_ESTIMATOR_ENGINES = {
    "scipy": _EstimatorEngine(name="scipy", optimizer="BFGS"),
}


def _resolve_estimator_engine(value: str) -> _EstimatorEngine:
    """Resolve an explicit estimator engine without accepting arbitrary strings."""

    if not isinstance(value, str):
        raise TypeError("engine must be a string")
    selected = "scipy" if value == "auto" else value
    if selected not in _ESTIMATOR_ENGINES:
        choices = ", ".join(["auto", *_ESTIMATOR_ENGINES])
        raise ValueError(f"engine must be one of {choices}")
    return _ESTIMATOR_ENGINES[selected]


@dataclass(frozen=True)
class RemEstimate:
    coef: np.ndarray
    names: list[str]
    log_likelihood: float
    converged: bool
    covariance: np.ndarray | None
    metadata: dict[str, Any]
    event_probabilities: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)
    observed_indices: tuple[int, ...] = field(default_factory=tuple, repr=False)
    gradient: np.ndarray | None = field(default=None, repr=False)
    hessian: np.ndarray | None = field(default=None, repr=False)
    residual_deviance: float = float("nan")
    null_deviance: float = float("nan")
    model_deviance: float = float("nan")
    iterations: int = 0
    sampled: bool = False
    draws: np.ndarray | None = field(default=None, repr=False)
    log_posterior: np.ndarray | None = field(default=None, repr=False)
    posterior_mean: np.ndarray | None = field(default=None, repr=False)
    posterior_sd: np.ndarray | None = field(default=None, repr=False)

    @property
    def coefficients(self) -> np.ndarray:
        return self.coef

    @property
    def loglik(self) -> float:
        return self.log_likelihood

    @property
    def vcov(self) -> np.ndarray | None:
        return self.covariance

    @property
    def se(self) -> np.ndarray | None:
        if self.covariance is None:
            return None
        return np.asarray(np.sqrt(np.maximum(np.diag(self.covariance), 0.0)), dtype=float)

    @property
    def AIC(self) -> float:
        return float(-2.0 * self.log_likelihood + 2.0 * len(self.coef))

    @property
    def AICC(self) -> float:
        count = int(self.metadata.get("n_observations", 0))
        parameters = len(self.coef)
        if count <= parameters + 1:
            return float("inf")
        return float(self.AIC + (2.0 * parameters * (parameters + 1)) / (count - parameters - 1))

    @property
    def BIC(self) -> float:
        count = max(
            1,
            int(
                self.metadata.get(
                    "bic_n_observations", self.metadata.get("n_observations", 1)
                )
            ),
        )
        return float(-2.0 * self.log_likelihood + len(self.coef) * np.log(count))

    @property
    def df_null(self) -> int:
        return int(self.metadata.get("df.null", 0))

    @property
    def df_model(self) -> int:
        return int(self.metadata.get("df.model", len(self.coef)))

    @property
    def df_residual(self) -> int:
        return int(self.metadata.get("df.residual", self.df_null - self.df_model))

    @property
    def model(self) -> str:
        return str(self.metadata.get("model", "tie"))

    @property
    def ordinal(self) -> bool:
        return bool(self.metadata.get("ordinal", False))

    @property
    def method(self) -> str:
        return str(self.metadata.get("method", "MLE"))

    @property
    def approach(self) -> str:
        return str(self.metadata.get("approach", "Frequentist"))

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        if self.draws is not None:
            if self.metadata.get("component") in {"sender", "receiver"}:
                return {
                    "coefficients": dict(
                        zip(self.names, self.coef.tolist(), strict=True)
                    ),
                    "post.mean": (
                        None
                        if self.posterior_mean is None
                        else dict(
                            zip(
                                self.names,
                                self.posterior_mean.tolist(),
                                strict=True,
                            )
                        )
                    ),
                    "vcov": (
                        None
                        if self.covariance is None
                        else np.array(self.covariance, copy=True)
                    ),
                    "sd": (
                        None
                        if self.posterior_sd is None
                        else dict(
                            zip(self.names, self.posterior_sd.tolist(), strict=True)
                        )
                    ),
                    "loglik": self.log_likelihood,
                    "draws": np.array(self.draws, copy=True),
                    "df.null": self.df_null,
                    "df.model": self.df_model,
                    "df.residual": self.df_residual,
                }
            return {
                "draws": np.array(self.draws, copy=True),
                "log_posterior": (
                    None
                    if self.log_posterior is None
                    else np.array(self.log_posterior, copy=True)
                ),
                "coefficients": dict(zip(self.names, self.coef.tolist(), strict=True)),
                "post.mean": (
                    None
                    if self.posterior_mean is None
                    else dict(zip(self.names, self.posterior_mean.tolist(), strict=True))
                ),
                "vcov": (
                    None if self.covariance is None else np.array(self.covariance, copy=True)
                ),
                "sd": (
                    None
                    if self.posterior_sd is None
                    else dict(zip(self.names, self.posterior_sd.tolist(), strict=True))
                ),
                "loglik": self.log_likelihood,
                "sampled": self.sampled,
                "df.null": self.df_null,
            }
        common = {
            "coefficients": dict(zip(self.names, self.coef.tolist(), strict=True)),
            "loglik": self.log_likelihood,
            "gradient": (None if self.gradient is None else np.array(self.gradient, copy=True)),
            "hessian": (None if self.hessian is None else np.array(self.hessian, copy=True)),
            "vcov": None if self.covariance is None else np.array(self.covariance, copy=True),
            "se": None if self.se is None else np.array(self.se, copy=True),
        }
        if self.metadata.get("component") in {"sender", "receiver"}:
            component_result = {
                **common,
                "residual.deviance": self.residual_deviance,
                "AIC": self.AIC,
                "AICC": self.AICC,
                "BIC": self.BIC,
                "converged": self.converged,
                "iterations": self.iterations,
                "df.null": self.df_null,
                "df.model": self.df_model,
                "df.residual": self.df_residual,
                "null.deviance": self.null_deviance,
                "model.deviance": self.model_deviance,
            }
            if "WAIC" in self.metadata:
                component_result["WAIC"] = float(self.metadata["WAIC"])
            return component_result
        result = {
            **common,
            "residual.deviance": self.residual_deviance,
            "null.deviance": self.null_deviance,
            "model.deviance": self.model_deviance,
            "df.null": self.df_null,
            "df.model": self.df_model,
            "df.residual": self.df_residual,
            "AIC": self.AIC,
            "AICC": self.AICC,
            "BIC": self.BIC,
            "converged": self.converged,
            "iterations": self.iterations,
            "sampled": self.sampled,
        }
        if self.sampled:
            result["samp_num"] = self.metadata.get("samp_num")
            result["sampling_scheme"] = self.metadata.get("sampling_scheme")
        return result

    def summary(self) -> dict[str, Any]:
        if self.draws is not None:
            return {
                "coefficients": dict(zip(self.names, self.coef.tolist(), strict=True)),
                "posterior_mean": (
                    None
                    if self.posterior_mean is None
                    else dict(zip(self.names, self.posterior_mean.tolist(), strict=True))
                ),
                "posterior_sd": (
                    None
                    if self.posterior_sd is None
                    else dict(zip(self.names, self.posterior_sd.tolist(), strict=True))
                ),
                "draws": len(self.draws),
                "acceptance_rate": self.metadata.get("acceptance_rate"),
                "divergences": self.metadata.get("divergences"),
                "backend": self.metadata.get("backend"),
            }
        se = None
        if self.covariance is not None:
            se = np.sqrt(np.diag(self.covariance)).tolist()
        result = {
            "coefficients": dict(zip(self.names, self.coef.tolist(), strict=True)),
            "std_error": None if se is None else dict(zip(self.names, se, strict=True)),
            "log_likelihood": self.log_likelihood,
            "AIC": self.AIC,
            "AICC": self.AICC,
            "BIC": self.BIC,
            "null.deviance": self.null_deviance,
            "residual.deviance": self.residual_deviance,
            "model.deviance": self.model_deviance,
            "converged": self.converged,
            "backend": self.metadata.get("backend"),
            "device": self.metadata.get("device"),
        }
        if "WAIC" in self.metadata:
            result["WAIC"] = float(self.metadata["WAIC"])
        return result

    def plot(
        self,
        *,
        reh: EventHistory | None = None,
        stats: RemStats | AomStats | RemStatsDuration | None = None,
        diagnostics: Diagnostics | ActorDiagnostics | None = None,
        **kwargs: Any,
    ) -> Diagnostics | ActorDiagnostics | RemEstimate:
        """Support the legacy positional plotting convention."""

        if reh is None:
            if kwargs.get("which") == 0:
                return self
            raise ValueError("'reh' is required except for coefficient panel which=0")
        return _plot_fitted_result(self, reh, stats, diagnostics, **kwargs)


@dataclass(frozen=True)
class ActorRemEstimate:
    """Separate sender-rate and receiver-choice fits for an actor REM."""

    sender_model: RemEstimate | None
    receiver_model: RemEstimate | None
    metadata: dict[str, Any]

    @property
    def _components(self) -> tuple[RemEstimate, ...]:
        return tuple(
            component
            for component in (self.sender_model, self.receiver_model)
            if component is not None
        )

    @property
    def coef(self) -> np.ndarray:
        components = self._components
        return (
            np.concatenate([component.coef for component in components])
            if components
            else np.zeros(0, dtype=float)
        )

    @property
    def names(self) -> list[str]:
        names: list[str] = []
        if self.sender_model is not None:
            names.extend(f"sender::{name}" for name in self.sender_model.names)
        if self.receiver_model is not None:
            names.extend(f"receiver::{name}" for name in self.receiver_model.names)
        return names

    @property
    def log_likelihood(self) -> float:
        return float(sum(component.log_likelihood for component in self._components))

    @property
    def converged(self) -> bool:
        return bool(self._components) and all(
            component.converged for component in self._components
        )

    @property
    def covariance(self) -> np.ndarray | None:
        covariance = [component.covariance for component in self._components]
        if not covariance or any(values is None for values in covariance):
            return None
        from scipy.linalg import block_diag

        return np.asarray(block_diag(*covariance), dtype=float)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_model": self.sender_model,
            "receiver_model": self.receiver_model,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "sender_model": (
                None if self.sender_model is None else self.sender_model.summary()
            ),
            "receiver_model": (
                None if self.receiver_model is None else self.receiver_model.summary()
            ),
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
        }

    def plot(
        self,
        *,
        reh: EventHistory,
        stats: RemStats | AomStats | RemStatsDuration | None = None,
        diagnostics: Diagnostics | ActorDiagnostics | None = None,
        **kwargs: Any,
    ) -> Diagnostics | ActorDiagnostics:
        """Support the legacy positional plotting convention."""

        return _plot_fitted_result(self, reh, stats, diagnostics, **kwargs)


@dataclass(frozen=True)
class Diagnostics:
    fit: RemEstimate
    residuals: np.ndarray
    observed_probabilities: np.ndarray
    ranks: np.ndarray
    predicted_indices: np.ndarray
    rates: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)
    recall: dict[str, Any] = field(default_factory=dict)
    reh_processed: EventHistory | None = field(default=None, repr=False)
    effect_processes: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    ranef: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    use_ranef: bool = False

    def __getitem__(self, key: str) -> Any:
        """Expose the documented list-style diagnostics interface."""

        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit": self.fit,
            "residuals": np.array(self.residuals, copy=True),
            "observed_probabilities": np.array(self.observed_probabilities, copy=True),
            "ranks": np.array(self.ranks, copy=True),
            "predicted_indices": np.array(self.predicted_indices, copy=True),
            "rates": [np.array(values, copy=True) for values in self.rates],
            "recall": self.recall,
            ".reh.processed": self.reh_processed,
            "effect_processes": self.effect_processes.copy(),
            "ranef": {name: values.copy() for name, values in self.ranef.items()},
            "use_ranef": self.use_ranef,
        }

    def plot_data(
        self,
        *,
        which: int | Sequence[int] = (1, 2),
        effects: str | Sequence[str] | None = None,
        object: RemEstimate | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return stable data for residual, effect, posterior, and trace panels."""

        return _diagnostic_plot_data(
            self,
            which=which,
            effects=effects,
            fitted=object,
            warn_unavailable=False,
        )

    def plot(
        self,
        *,
        which: int | Sequence[int] = (1, 2),
        effects: str | Sequence[str] | None = None,
        object: RemEstimate | None = None,
    ) -> Diagnostics:
        """Validate and materialize diagnostic panels, returning ``self`` like R."""

        self.plot_data(which=which, effects=effects, object=object)
        return self


@dataclass(frozen=True)
class ActorDiagnostics:
    """Separate diagnostics for actor sender-rate and receiver-choice models."""

    fit: ActorRemEstimate
    sender_model: Diagnostics | None
    receiver_model: Diagnostics | None
    reh_processed: EventHistory | None = field(default=None, repr=False)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_model": self.sender_model,
            "receiver_model": self.receiver_model,
            ".reh.processed": self.reh_processed,
        }

    def plot_data(
        self,
        *,
        which: int | Sequence[int] = (1, 2),
        sender_effects: str | Sequence[str] | None = "all",
        receiver_effects: str | Sequence[str] | None = "all",
        object: ActorRemEstimate | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return component-labelled plot data for an actor-oriented model."""

        panels: dict[str, pd.DataFrame] = {}
        components = (
            (
                "sender",
                self.sender_model,
                sender_effects,
                None if object is None else object.sender_model,
            ),
            (
                "receiver",
                self.receiver_model,
                receiver_effects,
                None if object is None else object.receiver_model,
            ),
        )
        for label, component, selected, fitted in components:
            if component is None or selected is None:
                continue
            effects = None if selected == "all" else selected
            for panel, frame in _diagnostic_plot_data(
                component,
                which=which,
                effects=effects,
                fitted=fitted,
                warn_unavailable=False,
            ).items():
                panels[f"{label}.{panel}"] = frame
        requested = _normalize_diagnostic_panels(which)
        if any(panel in {3, 4} for panel in requested) and (
            object is None or object.metadata.get("method") != "HMC"
        ):
            warnings.warn(
                "posterior and trace panels require an HMC result; unavailable panels were skipped",
                UserWarning,
                stacklevel=2,
            )
        return panels

    def plot(
        self,
        *,
        which: int | Sequence[int] = (1, 2),
        sender_effects: str | Sequence[str] | None = "all",
        receiver_effects: str | Sequence[str] | None = "all",
        object: ActorRemEstimate | None = None,
    ) -> ActorDiagnostics:
        """Validate and materialize actor diagnostic panels, returning ``self``."""

        self.plot_data(
            which=which,
            sender_effects=sender_effects,
            receiver_effects=receiver_effects,
            object=object,
        )
        return self


@dataclass(frozen=True)
class RemEstimateDuration(RemEstimate):
    """Frequentist start/end duration REM result."""

    stacked_data: RemStatsStackedDuration | None = field(default=None, repr=False)
    backend_fit: dict[str, Any] = field(default_factory=dict, repr=False)
    fitted_values: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)
    random_effects: dict[str, pd.Series] = field(default_factory=dict)
    variance_components: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    @property
    def coefficients(self) -> np.ndarray:
        return self.coef

    @property
    def loglik(self) -> float:
        return self.log_likelihood

    @property
    def vcov(self) -> np.ndarray | None:
        return self.covariance

    @property
    def se(self) -> np.ndarray | None:
        if self.covariance is None:
            return None
        return np.asarray(np.sqrt(np.maximum(np.diag(self.covariance), 0.0)), dtype=float)

    @property
    def AIC(self) -> float:
        return float(-2.0 * self.log_likelihood + 2.0 * len(self.coef))

    @property
    def AICC(self) -> float:
        count = int(self.metadata.get("n_observations", 0))
        parameters = len(self.coef)
        denominator = max(count - parameters - 1, 1)
        return float(self.AIC + (2.0 * parameters * (parameters + 1)) / denominator)

    @property
    def BIC(self) -> float:
        count = max(
            1,
            int(
                self.metadata.get(
                    "bic_n_observations", self.metadata.get("n_observations", 1)
                )
            ),
        )
        return float(-2.0 * self.log_likelihood + len(self.coef) * np.log(count))

    @property
    def model(self) -> str:
        return str(self.metadata.get("model", "tie"))

    @property
    def method(self) -> str:
        return str(self.metadata.get("method", "MLE"))

    @property
    def engine(self) -> str:
        return str(self.metadata.get("engine", "glm"))

    @property
    def ordinal(self) -> bool:
        return bool(self.metadata.get("ordinal", False))

    @property
    def df_null(self) -> int:
        return int(self.metadata.get("df.null", 0))

    @property
    def df_model(self) -> int:
        return int(self.metadata.get("df.model", len(self.coef)))

    @property
    def df_residual(self) -> int:
        return int(self.metadata.get("df.residual", self.df_null - self.df_model))

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "coefficients": dict(zip(self.names, self.coef.tolist(), strict=True)),
                "loglik": self.log_likelihood,
                "gradient": (None if self.gradient is None else np.array(self.gradient, copy=True)),
                "hessian": (None if self.hessian is None else np.array(self.hessian, copy=True)),
                "vcov": (None if self.covariance is None else np.array(self.covariance, copy=True)),
                "se": None if self.se is None else np.array(self.se, copy=True),
                "residual.deviance": self.residual_deviance,
                "null.deviance": self.null_deviance,
                "model.deviance": self.model_deviance,
                "AIC": self.AIC,
                "AICC": self.AICC,
                "BIC": self.BIC,
                "df.null": self.df_null,
                "df.model": self.df_model,
                "df.residual": self.df_residual,
                "iterations": self.iterations,
                "stacked_data": self.stacked_data,
                "backend_fit": dict(self.backend_fit),
                "fitted_values": np.array(self.fitted_values, copy=True),
                "random_effects": {
                    name: values.copy() for name, values in self.random_effects.items()
                },
                "variance_components": self.variance_components.copy(),
            }
        )
        return result

    def summary(self) -> dict[str, Any]:
        standard_errors = self.se
        if standard_errors is None:
            standard_errors = np.full(len(self.coef), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_values = self.coef / standard_errors
        from scipy.stats import norm

        coefficient_table = pd.DataFrame(
            {
                "Estimate": self.coef,
                "Std. Error": standard_errors,
                "z value": z_values,
                "Pr(>|z|)": 2.0 * norm.sf(np.abs(z_values)),
            },
            index=self.names,
        )
        return {
            "coefsTab": coefficient_table,
            "coefficients": coefficient_table,
            "log_likelihood": self.log_likelihood,
            "deviance": self.residual_deviance,
            "converged": self.converged,
            "iterations": self.iterations,
            "backend": self.metadata.get("backend"),
            "engine": self.metadata.get("engine"),
            "AIC": self.AIC,
            "AICC": self.AICC,
            "BIC": self.BIC,
            "null.deviance": self.null_deviance,
            "residual.deviance": self.residual_deviance,
            "model.deviance": self.model_deviance,
            "loglik": self.log_likelihood,
        }

    def __str__(self) -> str:
        return (
            "Duration relational-event model\n"
            f"engine: {self.metadata.get('engine')}\n"
            f"coefficients: {len(self.coef)}\n"
            f"log-likelihood: {self.log_likelihood:.6g}\n"
            f"converged: {self.converged}"
        )


@dataclass(frozen=True)
class RemEstimateDurationGlmnet(RemEstimateDuration):
    """Elastic-net duration REM retaining duration-specific diagnostics data."""

    unpenalized: tuple[str, ...] = field(default_factory=tuple)
    penalty: dict[str, Any] = field(default_factory=dict)
    lambda_value: float = 0.0
    lambda_min: float | None = None
    lambda_1se: float | None = None
    lambda_select: str = "explicit"

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "unpenalized": list(self.unpenalized),
                "penalty": dict(self.penalty),
                "lambda": self.lambda_value,
                "lambda_min": self.lambda_min,
                "lambda_1se": self.lambda_1se,
                "lambda_sel": self.lambda_value,
                "lambda_select": self.lambda_select,
            }
        )
        return result


@dataclass(frozen=True)
class RemEstimateGLMM(RemEstimate):
    """Frequentist REM with Gaussian random effects and retained BLUPs."""

    backend_fit: dict[str, Any] = field(default_factory=dict, repr=False)
    random_effects: dict[str, pd.Series] = field(default_factory=dict)
    variance_components: pd.Series = field(
        default_factory=lambda: pd.Series(dtype=float)
    )

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "backend_fit": dict(self.backend_fit),
                "random_effects": {
                    name: values.copy() for name, values in self.random_effects.items()
                },
                "variance_components": self.variance_components.copy(),
            }
        )
        return result


@dataclass(frozen=True)
class _RandomEffectTerm:
    grouping: tuple[str, ...]
    slope: str | None

    @property
    def name(self) -> str:
        effect = "(Intercept)" if self.slope is None else self.slope
        return f"{':'.join(self.grouping)}::{effect}"


@dataclass(frozen=True)
class _MixtureRows:
    design: np.ndarray
    response: np.ndarray
    offset: np.ndarray
    group_values: np.ndarray
    event_rows: tuple[np.ndarray, ...]
    observed_groups: tuple[tuple[int, ...], ...]
    names: list[str]
    ordinal: bool
    model: str


@dataclass(frozen=True)
class RemEstimateGlmnet(RemEstimate):
    """Elastic-net REM result with explicit penalty metadata."""

    unpenalized: tuple[str, ...] = field(default_factory=tuple)
    penalty: dict[str, Any] = field(default_factory=dict)
    lambda_value: float = 0.0
    lambda_min: float | None = None
    lambda_1se: float | None = None
    lambda_select: str = "explicit"

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "unpenalized": list(self.unpenalized),
                "penalty": dict(self.penalty),
                "lambda": self.lambda_value,
                "lambda_min": self.lambda_min,
                "lambda_1se": self.lambda_1se,
                "lambda_sel": self.lambda_value,
                "lambda_select": self.lambda_select,
            }
        )
        return result


@dataclass(frozen=True)
class RemEstimateMixture(RemEstimate):
    """Finite-mixture REM with component coefficients and memberships."""

    prior_probs: np.ndarray = field(default_factory=lambda: np.empty(0))
    posterior: np.ndarray = field(default_factory=lambda: np.empty((0, 0)), repr=False)
    assignments: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    grouping: str = "dyad"
    group_levels: tuple[Any, ...] = field(default_factory=tuple)
    component_event_probabilities: tuple[tuple[np.ndarray, ...], ...] = field(
        default_factory=tuple,
        repr=False,
    )
    backend_fit: dict[str, Any] = field(default_factory=dict, repr=False)
    bic_value: float = float("nan")
    aic_value: float = float("nan")

    @property
    def k(self) -> int:
        return int(self.coef.shape[1]) if self.coef.ndim == 2 else 1

    @property
    def BIC(self) -> float:
        return self.bic_value

    @property
    def AIC(self) -> float:
        return self.aic_value

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "coefficients": np.array(self.coef, copy=True),
                "k": self.k,
                "prior_probs": np.array(self.prior_probs, copy=True),
                "posterior": np.array(self.posterior, copy=True),
                "assignments": self.assignments.copy(),
                "grouping": self.grouping,
                "group_levels": list(self.group_levels),
                "backend_fit": dict(self.backend_fit),
                "AIC": self.aic_value,
                "BIC": self.bic_value,
            }
        )
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "prior_probs": pd.Series(
                self.prior_probs,
                index=[f"Component.{index + 1}" for index in range(self.k)],
            ),
            "coefficients": pd.DataFrame(
                self.coef,
                index=self.names,
                columns=[f"Component.{index + 1}" for index in range(self.k)],
            ),
            "log_likelihood": self.log_likelihood,
            "AIC": self.aic_value,
            "BIC": self.bic_value,
            "converged": self.converged,
            "iterations": self.iterations,
        }

    def __str__(self) -> str:
        proportions = ", ".join(f"{value:.4f}" for value in self.prior_probs)
        return (
            f"Finite-mixture relational-event model (k={self.k})\n"
            f"mixing proportions: {proportions}\n"
            f"log-likelihood: {self.log_likelihood:.6g}; BIC: {self.bic_value:.6g}"
        )


@dataclass(frozen=True)
class MixtureDiagnostics(Diagnostics):
    """Posterior-weighted and per-component mixture recall diagnostics."""

    recall_by_component: dict[str, dict[str, Any]] = field(default_factory=dict)
    prior_probs: np.ndarray = field(default_factory=lambda: np.empty(0))
    k: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "recall_by_component": self.recall_by_component,
                "prior_probs": np.array(self.prior_probs, copy=True),
                "k": self.k,
            }
        )
        return result


@dataclass(frozen=True)
class RemEstimateShrinkage(RemEstimate):
    """Approximate-Bayesian regularized REM based on the MLE covariance."""

    shrinkage_type: str = "horseshoe"
    estimates: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))
    unpenalized: tuple[str, ...] = field(default_factory=tuple)
    backend_fit: dict[str, Any] = field(default_factory=dict, repr=False)
    stacked_data: RemStatsStackedDuration | None = field(default=None, repr=False)
    fitted_values: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "shrinkem_type": self.shrinkage_type,
                "estimates": self.estimates.copy(),
                "selected": np.array(self.selected, copy=True),
                "unpenalized": list(self.unpenalized),
                "backend_fit": dict(self.backend_fit),
                "stacked_data": self.stacked_data,
                "fitted_values": np.array(self.fitted_values, copy=True),
            }
        )
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "type": self.shrinkage_type,
            "estimates": self.estimates.copy(),
            "unpenalized": list(self.unpenalized),
            "log_likelihood": self.log_likelihood,
            "converged": self.converged,
        }

    def __str__(self) -> str:
        return (
            f"Approximate Bayesian REM regularization [{self.shrinkage_type}]\n"
            f"selected: {int(self.selected.sum())}/{len(self.selected)}; "
            f"log-likelihood: {self.log_likelihood:.6g}"
        )


@dataclass(frozen=True)
class RemEstimateWindow:
    """Repeated REM fits over contiguous event or duration-time windows.

    The object exposes stable tabular coefficient and plotting data for Python
    callers.
    """

    fits: tuple[RemEstimate | ActorRemEstimate | Exception, ...]
    windows: pd.DataFrame
    type: str
    mode: str
    metadata: dict[str, Any]
    source_stats: RemStats | AomStats | RemStatsDuration = field(repr=False)
    window_stats: tuple[RemStats | AomStats | RemStatsDuration, ...] = field(
        repr=False
    )

    @property
    def n_windows(self) -> int:
        return len(self.fits)

    @property
    def coef(self) -> dict[str, Any]:
        return self.coefficients()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fits": list(self.fits),
            "windows": self.windows.copy(),
            "type": self.type,
            "mode": self.mode,
            "n.windows": self.n_windows,
        }

    def coefficients(self, *, ci: float = 0.95, k: float = 10.0) -> dict[str, Any]:
        return _window_coefficient_blocks(self, ci=ci, k=k)

    def summary(self, *, k: float = 10.0) -> dict[str, Any]:
        return _window_summary(self, k=k)

    def plot_data(self, *, ci: float = 0.95, k: float = 4.0) -> pd.DataFrame:
        del k  # reserved for robust-limit control
        return _window_coefficient_plot_frame(self, ci=ci)

    def plot(self, *, ci: float = 0.95, k: float = 4.0) -> RemEstimateWindow:
        self.plot_data(ci=ci, k=k)
        return self

    def __str__(self) -> str:
        return (
            f"Moving-window relational event model ({self.type})\n"
            f"windows = {self.n_windows} (mode: {self.mode})"
        )


@dataclass(frozen=True)
class WindowDiagnostics:
    """Recall diagnostics evaluated with time-varying window coefficients."""

    fit: RemEstimateWindow
    windows: pd.DataFrame
    type: str
    recall: dict[str, Any] | None = None
    sender: dict[str, Any] | None = None
    receiver: dict[str, Any] | None = None
    start: dict[str, Any] | None = None
    end: dict[str, Any] | None = None

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": self.recall,
            "sender": self.sender,
            "receiver": self.receiver,
            "start": self.start,
            "end": self.end,
            "windows": self.windows.copy(),
            "type": self.type,
        }

    def plot_data(self) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []
        for component, value in (
            ("duration" if self.type == "duration" else "tie", self.recall),
            ("sender", self.sender),
            ("receiver", self.receiver),
            ("start", self.start),
            ("end", self.end),
        ):
            if value is None:
                continue
            recall = (
                value
                if component in {"tie", "duration", "start", "end"}
                else value.get("recall", {})
            )
            frame = recall.get("per_event")
            if isinstance(frame, pd.DataFrame):
                item = frame.copy()
                item.insert(0, "component", component)
                blocks.append(item)
        return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()

    def plot(self) -> WindowDiagnostics:
        self.plot_data()
        return self

    def __str__(self) -> str:
        return (
            f"Moving-window diagnostics ({self.type}, interpolated coefficients)\n"
            f"windows = {len(self.windows)}"
        )


@dataclass(frozen=True)
class RemTribute:
    """Conditional event-attribute model returned by :func:`remtribute`."""

    coefficients: pd.Series | pd.DataFrame
    covariance: pd.DataFrame | None
    log_likelihood: float
    backend_fit: dict[str, Any]
    attribute: str
    attribute_type: str
    n_events: int
    stat_names: list[str]
    formula: str
    data: pd.DataFrame = field(repr=False)
    levels: tuple[Any, ...] = field(default_factory=tuple)
    aic: float = float("nan")
    bic: float = float("nan")

    @property
    def coef(self) -> pd.Series | pd.DataFrame:
        return self.coefficients

    @property
    def vcov(self) -> pd.DataFrame | None:
        return self.covariance

    @property
    def loglik(self) -> float:
        return self.log_likelihood

    @property
    def fit(self) -> dict[str, Any]:
        return self.backend_fit

    @property
    def n_levels(self) -> int | None:
        return len(self.levels) if self.levels else None

    @property
    def AIC(self) -> float:
        return self.aic

    @property
    def BIC(self) -> float:
        return self.bic

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "coefficients": self.coefficients.copy(),
            "vcov": None if self.covariance is None else self.covariance.copy(),
            "loglik": self.log_likelihood,
            "fit": dict(self.backend_fit),
            "attribute": self.attribute,
            "attribute_type": self.attribute_type,
            "n_events": self.n_events,
            "stat_names": list(self.stat_names),
            "formula": self.formula,
            "data": self.data.copy(),
            "AIC": self.aic,
            "BIC": self.bic,
        }
        if self.levels:
            result["levels"] = list(self.levels)
            result["n_levels"] = len(self.levels)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "attribute_type": self.attribute_type,
            "n_events": self.n_events,
            "levels": list(self.levels) if self.levels else None,
            "statistics": list(self.stat_names),
            "coefficients": self.coefficients.copy(),
            "standard_errors": self.backend_fit.get("standard_errors"),
            "log_likelihood": self.log_likelihood,
            "AIC": self.aic,
            "BIC": self.bic,
            "converged": self.backend_fit.get("converged", True),
        }

    def __str__(self) -> str:
        levels = (
            ""
            if not self.levels
            else (
                f"\n  Levels:         {len(self.levels)} "
                f"({', '.join(map(str, self.levels))})"
            )
        )
        return (
            "Relational Event Attribute Model\n\n"
            f"  Attribute:      {self.attribute}\n"
            f"  Type:           {self.attribute_type}\n"
            f"  Events:         {self.n_events}"
            f"{levels}\n"
            f"  Statistics:     {', '.join(self.stat_names)}\n"
            f"  Log-likelihood: {self.log_likelihood:.3f}\n"
            f"  AIC:            {self.aic:.3f}\n"
            f"  BIC:            {self.bic:.3f}"
        )


@dataclass(frozen=True)
class DurationDiagnostics(Diagnostics):
    """Recall and residual diagnostics for a duration REM fit."""

    recall_joint: dict[str, Any] | None = None
    recall_start: dict[str, Any] | None = None
    recall_end: dict[str, Any] | None = None
    recall_by_type: dict[Any, dict[str, Any]] = field(default_factory=dict)
    recall_start_by_type: dict[Any, dict[str, Any]] = field(default_factory=dict)
    recall_end_by_type: dict[Any, dict[str, Any]] = field(default_factory=dict)
    surprises_joint: pd.DataFrame | None = None
    surprises_start: pd.DataFrame | None = None
    surprises_end: pd.DataFrame | None = None
    surprises_by_type: dict[Any, pd.DataFrame] = field(default_factory=dict)
    surprises_start_by_type: dict[Any, pd.DataFrame] = field(default_factory=dict)
    surprises_end_by_type: dict[Any, pd.DataFrame] = field(default_factory=dict)
    surprise_offenders_joint: pd.DataFrame | None = None
    surprise_offenders_start: pd.DataFrame | None = None
    surprise_offenders_end: pd.DataFrame | None = None
    surprise_threshold: float = 0.2
    deviance_residuals: np.ndarray = field(default_factory=lambda: np.empty(0))
    pearson_residuals: np.ndarray = field(default_factory=lambda: np.empty(0))
    residual_summary: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "recall_joint": self.recall_joint,
                "recall_start": self.recall_start,
                "recall_end": self.recall_end,
                "recall_by_type": self.recall_by_type,
                "recall_start_by_type": self.recall_start_by_type,
                "recall_end_by_type": self.recall_end_by_type,
                "surprises_joint": (
                    None if self.surprises_joint is None else self.surprises_joint.copy()
                ),
                "surprises_start": (
                    None if self.surprises_start is None else self.surprises_start.copy()
                ),
                "surprises_end": (
                    None if self.surprises_end is None else self.surprises_end.copy()
                ),
                "surprises_by_type": {
                    key: value.copy() for key, value in self.surprises_by_type.items()
                },
                "surprises_start_by_type": {
                    key: value.copy()
                    for key, value in self.surprises_start_by_type.items()
                },
                "surprises_end_by_type": {
                    key: value.copy() for key, value in self.surprises_end_by_type.items()
                },
                "surprise_offenders_joint": (
                    None
                    if self.surprise_offenders_joint is None
                    else self.surprise_offenders_joint.copy()
                ),
                "surprise_offenders_start": (
                    None
                    if self.surprise_offenders_start is None
                    else self.surprise_offenders_start.copy()
                ),
                "surprise_offenders_end": (
                    None
                    if self.surprise_offenders_end is None
                    else self.surprise_offenders_end.copy()
                ),
                "surprise_threshold": self.surprise_threshold,
                "deviance_residuals": np.array(self.deviance_residuals, copy=True),
                "pearson_residuals": np.array(self.pearson_residuals, copy=True),
                "residual_summary": (
                    None if self.residual_summary is None else self.residual_summary.copy()
                ),
            }
        )
        return result

    def __str__(self) -> str:
        return (
            "Duration REM diagnostics\n"
            f"Joint mean relative rank: {_recall_mean(self.recall_joint):.4f}\n"
            f"Start mean relative rank: {_recall_mean(self.recall_start):.4f}\n"
            f"End mean relative rank: {_recall_mean(self.recall_end):.4f}"
        )


def _estimation_histories_compatible(
    source: EventHistory, supplied: EventHistory
) -> bool:
    """Allow an ordinal/exact likelihood switch over one event design.

    Statistics built on an ordinal representation can be reused with the
    corresponding exact-time history. All structural
    event and risk-set identifiers must still be identical; only time values
    and the ordinal flag may differ.
    """

    if source is supplied:
        return True
    if (
        source.model != supplied.model
        or source.directed != supplied.directed
        or source.duration != supplied.duration
        or source.N != supplied.N
        or source.E != supplied.E
        or source.event_types != supplied.event_types
        or source.riskset_mode != supplied.riskset_mode
        or source.extend_riskset_by_type != supplied.extend_riskset_by_type
    ):
        return False
    event_columns = [
        name
        for name in (
            "event_id",
            "sender_id",
            "receiver_id",
            "dyad_id",
            "type_id",
            "event_type",
            "event_weight",
        )
        if name in source.events.columns and name in supplied.events.columns
    ]
    if not source.events[event_columns].reset_index(drop=True).equals(
        supplied.events[event_columns].reset_index(drop=True)
    ):
        return False
    if len(source.risksets) != len(supplied.risksets):
        return False
    for left, right in zip(source.risksets, supplied.risksets, strict=True):
        columns = [
            name
            for name in ("dyad_id", "sender_id", "receiver_id", "event_type", "type_id")
            if name in left.columns and name in right.columns
        ]
        if left[columns].reset_index(drop=True).equals(
            right[columns].reset_index(drop=True)
        ):
            continue
        return False
    return True


def remstimate(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    approach: Sequence[str] | str = ("frequentist", "Bayesian"),
    random: Any | None = None,
    penalty: Any | None = None,
    mixture: Any | None = None,
    engine: str = "auto",
    bayes: Any | None = None,
    seed: int | None = None,
    ncores: int = 1,
    WAIC: bool = False,
    backend: str | ArrayBackend = "numpy",
    method: str | None = None,
    riskset_chunk_size: int | None = None,
    **kwargs: Any,
) -> RemEstimate | ActorRemEstimate | dict[str, Any]:
    """Estimate an ordinal tie-choice relational event model.

    Parameters
    ----------
    history:
        Event history returned by :func:`remflow.remify`.
    stats:
        Risk-set statistics returned by :func:`remflow.remstats`.
    backend:
        `numpy`, `jax`, `jax:cpu`, or `jax:gpu`. Requested GPU execution never
        falls back to CPU silently.
    engine:
        `auto` or `scipy`. The automatic choice currently resolves to SciPy;
        the array backend independently controls NumPy/JAX objective evaluation.
    """

    if not isinstance(history, EventHistory):
        raise TypeError("history must be an EventHistory returned by remify")
    if not isinstance(stats, (RemStats, AomStats, RemStatsDuration)):
        raise TypeError("stats must be a RemStats, AomStats, or RemStatsDuration object")

    legacy_bayes_controls = {
        name: kwargs.pop(name)
        for name in (
            "nsim",
            "nchains",
            "burnin",
            "thin",
            "init",
            "L",
            "epsilon",
            "prior",
            "nsimWAIC",
        )
        if name in kwargs
    }
    glmm_controls: dict[str, Any] = {
        name: kwargs.pop(name)
        for name in ("verbose", "maxiter", "tol", "variance_floor", "variance_iterations")
        if name in kwargs and random is not None
    }
    if mixture is not None and not isinstance(mixture, Mapping):
        raise TypeError("mixture must be a mapping or None")
    mixture_controls = dict(mixture or {})
    for name in ("k", "concomitant", "nrep", "maxiter", "tol"):
        if name in kwargs and mixture is not None:
            warnings.warn(
                f"top-level {name!r} is deprecated; put it inside mixture",
                DeprecationWarning,
                stacklevel=2,
            )
            mixture_controls[name] = kwargs.pop(name)
    if penalty is not None and not isinstance(penalty, Mapping):
        raise TypeError("penalty must be a mapping or None")
    if penalty is not None:
        penalty = dict(penalty)
        for name in (
            "alpha",
            "lambda",
            "nfolds",
            "lambda_select",
            "foldid",
            "unpenalized",
            "penalized",
        ):
            if name in kwargs:
                warnings.warn(
                    f"top-level {name!r} is deprecated; put it inside penalty",
                    DeprecationWarning,
                    stacklevel=2,
                )
                penalty[name] = kwargs.pop(name)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported remstimate arguments: {names}")
    if not _estimation_histories_compatible(stats.history, history):
        raise ValueError("stats must have been computed from the supplied history")
    if history.duration and not isinstance(stats, RemStatsDuration):
        raise TypeError("duration histories require RemStatsDuration")
    if not history.duration and isinstance(stats, RemStatsDuration):
        raise TypeError("RemStatsDuration requires a duration history")
    if not history.duration and history.model == "actor" and not isinstance(stats, AomStats):
        raise TypeError("actor-oriented histories require AomStats")
    if not history.duration and history.model == "tie" and not isinstance(stats, RemStats):
        raise TypeError("tie-oriented histories require RemStats")
    if mixture is not None and (random is not None or penalty is not None):
        raise ValueError("mixture cannot be combined with random or penalty")
    if random is not None and penalty is not None:
        raise ValueError("random and penalty cannot be combined")
    engine_spec = _resolve_estimator_engine(engine)
    valid_ncores = (
        isinstance(ncores, (int, float, np.integer, np.floating))
        and not isinstance(ncores, (bool, np.bool_))
        and np.isfinite(float(ncores))
        and float(ncores) >= 1.0
    )
    if not valid_ncores:
        warnings.warn(
            "ncores must be a positive integer; using the default of 1",
            UserWarning,
            stacklevel=2,
        )
        ncores_value = 1
    else:
        ncores_value = int(ncores)
    if not isinstance(WAIC, bool):
        raise TypeError("WAIC must be a single boolean value")
    if bayes is not None and not isinstance(bayes, Mapping):
        raise TypeError("bayes must be a mapping or None")
    bayes_controls = {**legacy_bayes_controls, **dict(bayes or {})}
    raw_nsim_waic = bayes_controls.get("nsimWAIC")
    nsim_waic = _validate_nsim_waic(raw_nsim_waic)
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    method_value = method.upper() if isinstance(method, str) else method
    if method_value not in {None, "MLE", "BFGS", "HMC"}:
        raise ValueError("method must be None, 'MLE', 'BFGS', or 'HMC'")
    approach_value = _match_arg(approach, ("frequentist", "Bayesian"), "approach")
    if method_value == "HMC":
        approach_value = "Bayesian"
    elif method_value in {"MLE", "BFGS"}:
        approach_value = "frequentist"

    resolved = resolve_backend(backend)
    if riskset_chunk_size is not None:
        if not isinstance(riskset_chunk_size, int) or riskset_chunk_size <= 0:
            raise ValueError("riskset_chunk_size must be a positive integer or None")
        if not isinstance(resolved, JaxBackend):
            raise ValueError("riskset_chunk_size is available only for a JAX backend")
    optimizer = engine_spec.optimizer
    if mixture is not None:
        if approach_value != "frequentist":
            raise ValueError("finite-mixture estimation requires approach='frequentist'")
        if isinstance(resolved, JaxBackend):
            raise NotImplementedError(
                "finite-mixture estimation is currently available only on backend='numpy'"
            )
        if WAIC:
            raise NotImplementedError("WAIC for finite-mixture estimation is not implemented")
        if riskset_chunk_size is not None:
            raise NotImplementedError(
                "risk-set chunking for finite-mixture estimation is not implemented"
            )
        return _with_ncores(
            _remstimate_mixture(
                history,
                stats,
                controls=mixture_controls,
                seed=seed,
            ),
            ncores_value,
        )
    if random is not None:
        if approach_value == "Bayesian":
            raise NotImplementedError("Bayesian frailty estimation is not implemented yet")
        if isinstance(resolved, JaxBackend):
            raise NotImplementedError(
                "random-effects estimation is currently available only on backend='numpy'"
            )
        if WAIC:
            raise NotImplementedError("WAIC for random-effects estimation is not implemented yet")
        if riskset_chunk_size is not None:
            raise NotImplementedError(
                "risk-set chunking for random-effects estimation is not implemented yet"
            )
        selected_engine = _random_effect_solver(history, stats)
        if isinstance(stats, RemStatsDuration):
            return _with_ncores(
                _remstimate_duration_glmm(
                    stats,
                    random=random,
                    engine=selected_engine,
                    seed=seed,
                    controls=glmm_controls,
                ),
                ncores_value,
            )
        if isinstance(stats, AomStats):
            return _with_ncores(
                _remstimate_actor_glmm(
                    stats,
                    random=random,
                    engine=selected_engine,
                    seed=seed,
                    controls=glmm_controls,
                ),
                ncores_value,
            )
        return _with_ncores(
            _remstimate_tie_glmm(
                history,
                stats,
                random=random,
                engine=selected_engine,
                seed=seed,
                controls=glmm_controls,
            ),
            ncores_value,
        )
    if isinstance(stats, RemStatsDuration):
        if approach_value == "Bayesian":
            if penalty is not None:
                return _with_ncores(
                    _remstimate_duration_shrinkage(
                        stats,
                        resolved,
                        optimizer,
                        penalty=dict(penalty),
                        seed=seed,
                    ),
                    ncores_value,
                )
            raise NotImplementedError("Bayesian duration estimation is not implemented yet")
        if WAIC:
            raise NotImplementedError(
                "WAIC for duration estimation is not implemented yet"
            )
        if riskset_chunk_size is not None and (
            not stats.stacked.ordinal or penalty is not None
        ):
            raise NotImplementedError(
                "risk-set chunking for duration estimation currently requires an "
                "unpenalized ordinal model"
            )
        if penalty is not None:
            return _with_ncores(
                _remstimate_duration_penalized(
                    stats,
                    resolved,
                    optimizer,
                    penalty=dict(penalty),
                    seed=seed,
                ),
                ncores_value,
            )
        return _with_ncores(
            _remstimate_duration(
                stats,
                resolved,
                optimizer,
                seed=seed,
                riskset_chunk_size=riskset_chunk_size,
            ),
            ncores_value,
        )
    if isinstance(stats, AomStats):
        if approach_value == "Bayesian":
            if penalty is not None:
                return _with_ncores(
                    _remstimate_actor_shrinkage(
                        stats,
                        resolved,
                        optimizer,
                        penalty=dict(penalty),
                        seed=seed,
                    ),
                    ncores_value,
                )
            return _with_ncores(
                _remstimate_actor_hmc(
                    stats,
                    resolved,
                    optimizer,
                    controls=_hmc_controls(bayes_controls),
                    seed=seed,
                    compute_waic=WAIC,
                    riskset_chunk_size=riskset_chunk_size,
                ),
                ncores_value,
            )
        if penalty is not None:
            return _with_ncores(
                _remstimate_actor_penalized(
                    stats,
                    resolved,
                    optimizer,
                    penalty=dict(penalty),
                    seed=seed,
                ),
                ncores_value,
            )
        return _with_ncores(
            _remstimate_actor(
                stats,
                resolved,
                optimizer,
                seed=seed,
                compute_waic=WAIC,
                nsim_waic=nsim_waic,
                riskset_chunk_size=riskset_chunk_size,
            ),
            ncores_value,
        )
    if penalty is not None:
        if approach_value == "Bayesian":
            return _with_ncores(
                _remstimate_tie_shrinkage(
                    history,
                    stats,
                    resolved,
                    optimizer,
                    penalty=dict(penalty),
                    seed=seed,
                ),
                ncores_value,
            )
        return _with_ncores(
            _remstimate_tie_penalized(
                history,
                stats,
                resolved,
                optimizer,
                penalty=dict(penalty),
                seed=seed,
            ),
            ncores_value,
        )
    if approach_value == "Bayesian":
        return _with_ncores(
            _remstimate_tie_hmc(
                history,
                stats,
                resolved,
                optimizer,
                controls=_hmc_controls(bayes_controls),
                seed=seed,
                compute_waic=WAIC,
                riskset_chunk_size=riskset_chunk_size,
            ),
            ncores_value,
        )
    return _with_ncores(
        _remstimate_tie(
            history,
            stats,
            resolved,
            optimizer,
            seed=seed,
            riskset_chunk_size=riskset_chunk_size,
            compute_waic=WAIC,
            nsim_waic=nsim_waic,
        ),
        ncores_value,
    )


def _with_ncores(
    result: RemEstimate | ActorRemEstimate | dict[str, Any], ncores: int
) -> RemEstimate | ActorRemEstimate | dict[str, Any]:
    """Attach the normalized ``ncores`` value to every returned fit."""

    if isinstance(result, dict):
        for value in result.values():
            if isinstance(value, (RemEstimate, ActorRemEstimate)):
                value.metadata["ncores"] = ncores
                value.metadata["estimator_engine"] = "scipy"
    else:
        result.metadata["ncores"] = ncores
        result.metadata["estimator_engine"] = "scipy"
    return result


def _random_effect_solver(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
) -> str:
    if history.ordinal or (
        isinstance(stats, RemStatsDuration) and stats.stacked.ordinal
    ):
        return "scipy-conditional-laplace"
    return "scipy-laplace"


def _glmm_controls(values: Mapping[str, Any]) -> dict[str, Any]:
    controls = {
        "verbose": False,
        "maxiter": 500,
        "tol": 1e-7,
        "variance_floor": 1e-6,
        "variance_iterations": 25,
        **dict(values),
    }
    if not isinstance(controls["verbose"], bool):
        raise TypeError("verbose must be a boolean")
    for name in ("maxiter", "variance_iterations"):
        value = controls[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("tol", "variance_floor"):
        value = controls[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"{name} must be a positive finite number")
    return controls


def _parse_random_terms(
    value: Any,
    valid_names: Sequence[str],
) -> tuple[_RandomEffectTerm, ...]:
    import re

    if not isinstance(value, str):
        raise TypeError(
            "random must be a formula string such as '~ (1 | actor1)'"
        )
    matches = re.findall(r"\(([^|]+)\|([^)]+)\)", value)
    if not matches:
        raise ValueError("random must contain at least one '(effect | group)' term")
    terms: list[_RandomEffectTerm] = []
    for left, right in matches:
        grouping = tuple(part.strip() for part in right.split(":") if part.strip())
        if not grouping:
            raise ValueError("random grouping factor cannot be empty")
        tokens = [token.strip() for token in left.split("+") if token.strip()]
        has_no_intercept = any(token in {"0", "-1"} for token in tokens)
        if not has_no_intercept:
            terms.append(_RandomEffectTerm(grouping, None))
        for token in tokens:
            if token in {"0", "1", "-1"}:
                continue
            if token not in valid_names:
                raise ValueError(
                    f"random slope {token!r} is not a fitted statistic"
                )
            terms.append(_RandomEffectTerm(grouping, token))
    unique: list[_RandomEffectTerm] = []
    for term in terms:
        if term not in unique:
            unique.append(term)
    if not unique:
        raise ValueError("random formula contains neither an intercept nor a slope")
    return tuple(unique)


def _tie_random_contexts(
    history: EventHistory,
    stats: RemStats,
) -> list[dict[str, np.ndarray]]:
    contexts: list[dict[str, np.ndarray]] = []
    for position, event_index in enumerate(stats.event_indices):
        riskset = history.risksets[event_index]
        if stats.sample_map:
            riskset = riskset.iloc[stats.sample_map[position] - 1]
        sender = riskset["sender_id"].to_numpy(dtype=int)
        receiver = riskset["receiver_id"].to_numpy(dtype=int)
        context: dict[str, np.ndarray] = {
            "actor1": sender,
            "actor2": receiver,
            "sender": sender,
            "receiver": receiver,
            "dyad": riskset["dyad_id"].to_numpy(dtype=int),
        }
        if "event_type" in riskset:
            context["type"] = riskset["event_type"].to_numpy(copy=True)
        contexts.append(context)
    return contexts


def _actor_sender_random_contexts(stats: AomStats) -> list[dict[str, np.ndarray]]:
    actors = stats.history.sender_riskset.astype(int)
    return [
        {
            "actor": actors,
            "actor_label": actors,
            "actor1": actors,
            "sender": actors,
        }
        for _ in stats.sender_stats
    ]


def _actor_receiver_random_contexts(stats: AomStats) -> list[dict[str, np.ndarray]]:
    masks = stats.receiver_choice_masks or stats.receiver_masks
    contexts: list[dict[str, np.ndarray]] = []
    for mask in masks:
        actors = np.flatnonzero(mask).astype(int) + 1
        contexts.append(
            {
                "actor": actors,
                "actor_label": actors,
                "actor2": actors,
                "receiver": actors,
            }
        )
    return contexts


def _random_group_values(
    context: Mapping[str, np.ndarray],
    grouping: Sequence[str],
) -> list[Any]:
    arrays: list[np.ndarray] = []
    for name in grouping:
        if name not in context:
            raise ValueError(f"random grouping factor {name!r} is unavailable")
        arrays.append(np.asarray(context[name]))
    if not arrays:
        return []
    if any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("random grouping arrays do not align")
    if len(arrays) == 1:
        return [value.item() if isinstance(value, np.generic) else value for value in arrays[0]]
    return [
        tuple(value.item() if isinstance(value, np.generic) else value for value in row)
        for row in zip(*arrays, strict=True)
    ]


def _build_random_designs(
    designs: Sequence[np.ndarray],
    names: Sequence[str],
    contexts: Sequence[Mapping[str, np.ndarray]],
    terms: Sequence[_RandomEffectTerm],
) -> tuple[
    list[np.ndarray],
    list[tuple[_RandomEffectTerm, tuple[Any, ...], slice]],
    np.ndarray,
]:
    if len(designs) != len(contexts):
        raise ValueError("random-effect contexts do not align with model designs")
    descriptions: list[tuple[_RandomEffectTerm, tuple[Any, ...], slice]] = []
    term_columns: list[int] = []
    offset = 0
    for term_index, term in enumerate(terms):
        levels: list[Any] = []
        for context in contexts:
            for level in _random_group_values(context, term.grouping):
                if level not in levels:
                    levels.append(level)
        if not levels:
            raise ValueError(f"random term {term.name!r} has no grouping levels")
        selected = slice(offset, offset + len(levels))
        descriptions.append((term, tuple(levels), selected))
        term_columns.extend([term_index] * len(levels))
        offset += len(levels)

    output: list[np.ndarray] = []
    for design, context in zip(designs, contexts, strict=True):
        random_design = np.zeros((len(design), offset), dtype=float)
        for term, term_levels, selected in descriptions:
            level_index = {
                level: index for index, level in enumerate(term_levels)
            }
            groups = _random_group_values(context, term.grouping)
            slopes = (
                np.ones(len(design), dtype=float)
                if term.slope is None
                else np.asarray(design[:, list(names).index(term.slope)], dtype=float)
            )
            if len(groups) != len(design):
                raise ValueError("random grouping factor does not align with the risk set")
            for row, (level, slope) in enumerate(zip(groups, slopes, strict=True)):
                random_design[row, selected.start + level_index[level]] = slope
        output.append(random_design)
    return output, descriptions, np.asarray(term_columns, dtype=int)


def _fit_random_component(
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    names: list[str],
    contexts: list[dict[str, np.ndarray]],
    random: Any,
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
    engine: str,
    model: str,
    component: str | None,
    parent_ordinal: bool,
    seed: int | None,
    controls: Mapping[str, Any],
) -> RemEstimateGLMM:
    from scipy.optimize import minimize

    selected_names = list(names)
    selected_designs = [np.asarray(design, dtype=float) for design in designs]
    if exposures is None and "baseline" in selected_names:
        keep = [index for index, name in enumerate(selected_names) if name != "baseline"]
        selected_names = [selected_names[index] for index in keep]
        selected_designs = [design[:, keep] for design in selected_designs]
    terms = _parse_random_terms(random, selected_names)
    random_designs, descriptions, term_columns = _build_random_designs(
        selected_designs,
        selected_names,
        contexts,
        terms,
    )
    combined_designs = [
        np.column_stack([fixed, random_values])
        for fixed, random_values in zip(
            selected_designs, random_designs, strict=True
        )
    ]
    fixed_count = len(selected_names)
    random_count = random_designs[0].shape[1] if random_designs else 0
    settings = _glmm_controls(controls)
    variance_floor = float(settings["variance_floor"])
    variances = np.full(len(descriptions), 0.25, dtype=float)
    parameters = np.zeros(fixed_count + random_count, dtype=float)
    iterations = 0
    converged = False
    message = "variance iteration limit reached"
    covariance_all = np.eye(len(parameters), dtype=float)
    for _ in range(int(settings["variance_iterations"])):
        precision = 1.0 / np.maximum(variances[term_columns], variance_floor)

        def objective(
            value: np.ndarray,
            precision_values: np.ndarray = precision,
        ) -> tuple[float, np.ndarray]:
            loglik, gradient, _ = _tie_loglik_grad_hessian(
                value,
                combined_designs,
                observed_groups,
                exposures=exposures,
                sampling_weights=sampling_weights,
            )
            random_values = value[fixed_count:]
            penalty = 0.5 * float(
                np.dot(precision_values, random_values * random_values)
            )
            output_gradient = -gradient
            output_gradient[fixed_count:] += precision_values * random_values
            return -loglik + penalty, output_gradient

        result = minimize(
            fun=lambda value: objective(value)[0],
            jac=lambda value: objective(value)[1],
            x0=parameters,
            method="BFGS",
            options={
                "maxiter": int(settings["maxiter"]),
                "gtol": float(settings["tol"]),
                "disp": bool(settings["verbose"]),
            },
        )
        parameters = np.asarray(result.x, dtype=float)
        iterations += int(getattr(result, "nit", 0))
        _, _, likelihood_hessian = _tie_loglik_grad_hessian(
            parameters,
            combined_designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        information = -likelihood_hessian
        information[fixed_count:, fixed_count:] += np.diag(precision)
        covariance_all = np.linalg.pinv(information, hermitian=True)
        random_values = parameters[fixed_count:]
        updated = np.empty_like(variances)
        for term_index in range(len(descriptions)):
            columns = np.flatnonzero(term_columns == term_index) + fixed_count
            updated[term_index] = max(
                variance_floor,
                float(
                    np.mean(
                        random_values[columns - fixed_count] ** 2
                        + np.maximum(np.diag(covariance_all)[columns], 0.0)
                    )
                ),
            )
        relative_change = float(
            np.max(np.abs(updated - variances) / np.maximum(variances, variance_floor))
        )
        variances = np.minimum(updated, 1e4)
        gradient_norm = float(np.linalg.norm(objective(parameters)[1], ord=np.inf))
        if relative_change < 5e-4 and (
            bool(result.success) or gradient_norm < 1e-5
        ):
            converged = True
            message = str(result.message)
            break
        message = str(result.message)

    loglik, full_gradient, full_hessian = _tie_loglik_grad_hessian(
        parameters,
        combined_designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=sampling_weights,
    )
    probabilities, observed_indices = _tie_event_probabilities(
        parameters,
        combined_designs,
        observed_groups,
        sampling_weights=sampling_weights,
    )
    fixed_coefficients = parameters[:fixed_count]
    fixed_covariance = covariance_all[:fixed_count, :fixed_count]
    null_loglik = _tie_null_loglik(
        selected_designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=sampling_weights,
    )
    random_effects: dict[str, pd.Series] = {}
    for term, levels, selected in descriptions:
        values = parameters[fixed_count + selected.start : fixed_count + selected.stop]
        random_effects[term.name] = pd.Series(values, index=list(levels), dtype=float)
    variance_components = pd.Series(
        variances,
        index=[term.name for term, _, _ in descriptions],
        name="variance",
    )
    metadata = {
        "backend": "numpy",
        "device": "cpu",
        "precision": "float64",
        "optimizer": "BFGS",
        "n_events": len(designs),
        "n_observations": len(designs),
        "seed": seed,
        "timing": "ordinal" if exposures is None else "exact",
        "model": model,
        "component": component,
        "ordinal": exposures is None,
        "parent_ordinal": parent_ordinal,
        "method": "GLMM",
        "approach": "Frequentist",
        "engine": engine,
        "statistics": selected_names,
        "random_effects": random_effects,
        "variance_components": variance_components,
        "remstimate_converged": converged,
        "df.null": len(designs),
        "df.model": fixed_count + len(variances),
        "df.residual": len(designs) - fixed_count,
    }
    residual_deviance = -2.0 * loglik
    backend_fit = {
        "engine": engine,
        "converged": converged,
        "remstimate_converged": converged,
        "message": message,
        "random_effects": random_effects,
        "variance_components": variance_components,
        "joint_parameters": parameters,
        "joint_covariance": covariance_all,
        "linear_predictors": tuple(
            np.asarray(design @ parameters, dtype=float)
            for design in combined_designs
        ),
    }
    return RemEstimateGLMM(
        coef=fixed_coefficients,
        names=selected_names,
        log_likelihood=float(loglik),
        converged=converged,
        covariance=fixed_covariance,
        metadata=metadata,
        event_probabilities=probabilities,
        observed_indices=observed_indices,
        gradient=full_gradient[:fixed_count],
        hessian=full_hessian[:fixed_count, :fixed_count],
        residual_deviance=residual_deviance,
        null_deviance=-2.0 * null_loglik,
        model_deviance=-2.0 * null_loglik - residual_deviance,
        iterations=iterations,
        sampled=sampling_weights is not None,
        backend_fit=backend_fit,
        random_effects=random_effects,
        variance_components=variance_components,
    )


def _remstimate_tie_glmm(
    history: EventHistory,
    stats: RemStats,
    *,
    random: Any,
    engine: str,
    seed: int | None,
    controls: Mapping[str, Any],
) -> RemEstimateGLMM:
    terms = _parse_random_terms(random, stats.names)
    if not history.directed and any(
        set(term.grouping).intersection({"actor", "actor1", "actor2"})
        for term in terms
    ):
        raise ValueError(
            "actor-level random effects are not identified for an undirected "
            "tie-oriented model; use random='~ (1 | dyad)'"
        )
    exposures = None if history.ordinal else _event_exposures(history, stats.event_indices)
    return _fit_random_component(
        [np.asarray(values, dtype=float) for values in stats.stats],
        stats.observed_index_groups
        or [[index] for index in stats.observed_indices],
        list(stats.names),
        _tie_random_contexts(history, stats),
        random,
        exposures=exposures,
        sampling_weights=(
            [np.asarray(values, dtype=float) for values in stats.sampling_weights]
            if stats.sampling_weights
            else None
        ),
        engine=engine,
        model="tie",
        component=None,
        parent_ordinal=history.ordinal,
        seed=seed,
        controls=controls,
    )


def _remstimate_actor_glmm(
    stats: AomStats,
    *,
    random: Any,
    engine: str,
    seed: int | None,
    controls: Mapping[str, Any],
) -> ActorRemEstimate:
    if isinstance(random, Mapping):
        unknown = set(random).difference({"sender", "receiver"})
        if unknown:
            raise ValueError(f"unknown actor random components: {sorted(unknown)!r}")
        sender_random = random.get("sender")
        receiver_random = random.get("receiver")
    else:
        sender_random = random
        receiver_random = random
    sender_model: RemEstimateGLMM | None = None
    if stats.sender_names and sender_random is not None:
        sender_model = _fit_random_component(
            [np.asarray(values, dtype=float) for values in stats.sender_stats],
            stats.observed_sender_groups
            or [[index] for index in stats.observed_sender_indices],
            list(stats.sender_names),
            _actor_sender_random_contexts(stats),
            sender_random,
            exposures=(
                None
                if stats.history.ordinal
                else _event_exposures(stats.history, stats.event_indices)
            ),
            sampling_weights=None,
            engine=engine,
            model="actor",
            component="sender",
            parent_ordinal=stats.history.ordinal,
            seed=seed,
            controls=controls,
        )
    receiver_model: RemEstimateGLMM | None = None
    if stats.receiver_names and receiver_random is not None:
        receiver_designs, receiver_groups = _actor_receiver_diagnostic_designs(stats)
        receiver_model = _fit_random_component(
            receiver_designs,
            receiver_groups,
            list(stats.receiver_names),
            _actor_receiver_random_contexts(stats),
            receiver_random,
            exposures=None,
            sampling_weights=None,
            engine="scipy-conditional-laplace",
            model="actor",
            component="receiver",
            parent_ordinal=stats.history.ordinal,
            seed=seed,
            controls=controls,
        )
    if sender_model is None and receiver_model is None:
        raise ValueError("actor random specification does not select a fitted component")
    metadata = {
        "backend": "numpy",
        "device": "cpu",
        "precision": "float64",
        "optimizer": "BFGS",
        "n_events": len(stats.event_indices),
        "seed": seed,
        "timing": "ordinal" if stats.history.ordinal else "exact",
        "model": "actor",
        "ordinal": stats.history.ordinal,
        "method": "GLMM",
        "approach": "Frequentist",
        "engine": engine,
        "engine_receiver": "scipy-conditional-laplace",
    }
    return ActorRemEstimate(sender_model, receiver_model, metadata)


def _mixture_controls(values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"k", "random", "concomitant", "nrep", "maxiter", "tol"}
    unknown = set(values).difference(allowed)
    if unknown:
        raise TypeError(f"unsupported mixture controls: {sorted(unknown)!r}")
    if "random" not in values or values["random"] is None:
        raise ValueError(
            "mixture must specify clustering, for example random='~ (1 | dyad)'"
        )
    raw_k = values.get("k", 2)
    k_values: tuple[int, ...]
    if isinstance(raw_k, (int, np.integer)) and not isinstance(raw_k, bool):
        k_values = (int(raw_k),)
    elif isinstance(raw_k, Sequence) and not isinstance(raw_k, (str, bytes)):
        k_values = tuple(int(value) for value in raw_k)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in raw_k
        ):
            raise TypeError("mixture k values must be integers")
    else:
        raise TypeError("mixture k must be an integer or a sequence of integers")
    if not k_values or any(value < 1 for value in k_values):
        raise ValueError("mixture k values must be positive")
    if len(set(k_values)) != len(k_values):
        raise ValueError("mixture k values must be unique")
    nrep = values.get("nrep", 3)
    maxiter = values.get("maxiter", 200)
    for name, value in (("nrep", nrep), ("maxiter", maxiter)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"mixture {name} must be a positive integer")
    tol = values.get("tol", 1e-7)
    if (
        isinstance(tol, bool)
        or not isinstance(tol, (int, float))
        or not np.isfinite(float(tol))
        or float(tol) <= 0.0
    ):
        raise ValueError("mixture tol must be a positive finite number")
    concomitant = values.get("concomitant")
    if concomitant is not None:
        canonical = str(concomitant).replace(" ", "")
        if canonical not in {"~1", "1"}:
            raise NotImplementedError(
                "non-constant concomitant mixture formulas are not implemented yet"
            )
    return {
        "k": k_values,
        "random": values["random"],
        "concomitant": concomitant,
        "nrep": nrep,
        "maxiter": maxiter,
        "tol": float(tol),
    }


def _parse_mixture_group(random: Any) -> str:
    import re

    if not isinstance(random, str):
        raise TypeError(
            "mixture random must be a formula string such as '~ (1 | dyad)'"
        )
    match = re.search(r"\|\s*([A-Za-z_.][A-Za-z_.0-9]*)", random)
    if match is None:
        raise ValueError(
            "mixture random must contain a grouping variable after '|', "
            "for example '~ (1 | dyad)'"
        )
    return match.group(1)


def _object_array(values: Sequence[Any]) -> np.ndarray:
    output = np.empty(len(values), dtype=object)
    output[:] = list(values)
    return output


def _mixture_tie_rows(
    history: EventHistory,
    stats: RemStats,
    grouping: str,
) -> _MixtureRows:
    names = list(stats.names)
    designs = [np.asarray(values, dtype=float) for values in stats.stats]
    if history.ordinal and "baseline" in names:
        keep = [index for index, name in enumerate(names) if name != "baseline"]
        names = [names[index] for index in keep]
        designs = [design[:, keep] for design in designs]
    observed = stats.observed_index_groups or [
        [index] for index in stats.observed_indices
    ]
    contexts = _tie_random_contexts(history, stats)
    sampling = (
        [np.asarray(values, dtype=float) for values in stats.sampling_weights]
        if stats.sampling_weights
        else [np.ones(len(design), dtype=float) for design in designs]
    )
    exposures = (
        None if history.ordinal else _event_exposures(history, stats.event_indices)
    )
    row_designs: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    group_values: list[Any] = []
    event_rows: list[np.ndarray] = []
    observed_groups: list[tuple[int, ...]] = []
    position = 0
    for event, (design, cases, context, weights) in enumerate(
        zip(designs, observed, contexts, sampling, strict=True)
    ):
        if grouping not in context:
            raise ValueError(f"mixture grouping variable {grouping!r} is unavailable")
        count = len(design)
        response = np.zeros(count, dtype=float)
        response[np.asarray(cases, dtype=int)] = 1.0
        offset = np.log(weights)
        if exposures is not None:
            offset = offset + np.log(float(exposures[event]))
        row_designs.append(design)
        responses.append(response)
        offsets.append(offset)
        group_values.extend(_random_group_values(context, (grouping,)))
        event_rows.append(np.arange(position, position + count, dtype=int))
        observed_groups.append(tuple(int(value) for value in cases))
        position += count
    return _MixtureRows(
        design=np.vstack(row_designs),
        response=np.concatenate(responses),
        offset=np.concatenate(offsets),
        group_values=_object_array(group_values),
        event_rows=tuple(event_rows),
        observed_groups=tuple(observed_groups),
        names=names,
        ordinal=history.ordinal,
        model="tie",
    )


def _mixture_duration_rows(
    stats: RemStatsDuration,
    grouping: str,
) -> _MixtureRows:
    frame = stats.stacked.remstats_stack
    names = list(stats.stacked.stat_names)
    groups = _duration_groups(frame)
    contexts = _duration_random_contexts(stats, groups)
    group_values: list[Any] = []
    observed: list[tuple[int, ...]] = []
    for index, context in zip(groups, contexts, strict=True):
        if grouping not in context:
            raise ValueError(f"mixture grouping variable {grouping!r} is unavailable")
        group_values.extend(_random_group_values(context, (grouping,)))
        observed.append(
            tuple(
                int(value)
                for value in np.flatnonzero(
                    frame.iloc[index]["obs"].to_numpy(dtype=float) > 0.0
                )
            )
        )
    offset = (
        np.zeros(len(frame), dtype=float)
        if stats.stacked.ordinal
        else frame["log_interevent"].to_numpy(dtype=float)
    )
    return _MixtureRows(
        design=frame[names].to_numpy(dtype=float),
        response=frame["obs"].to_numpy(dtype=float),
        offset=offset,
        group_values=_object_array(group_values),
        event_rows=tuple(groups),
        observed_groups=tuple(observed),
        names=names,
        ordinal=stats.stacked.ordinal,
        model="duration",
    )


def _mixture_actor_rows(
    stats: AomStats,
    grouping: str,
    component: str,
) -> _MixtureRows | None:
    history = stats.history
    if component == "sender":
        if not stats.sender_names:
            return None
        names = list(stats.sender_names)
        designs = [np.asarray(values, dtype=float) for values in stats.sender_stats]
        observed = stats.observed_sender_groups or [
            [index] for index in stats.observed_sender_indices
        ]
        contexts = _actor_sender_random_contexts(stats)
        ordinal = history.ordinal
        exposures = None if ordinal else _event_exposures(history, stats.event_indices)
    else:
        if not stats.receiver_names:
            return None
        names = list(stats.receiver_names)
        designs, observed = _actor_receiver_diagnostic_designs(stats)
        contexts = _actor_receiver_random_contexts(stats)
        ordinal = True
        exposures = None
    if ordinal and "baseline" in names:
        keep = [index for index, name in enumerate(names) if name != "baseline"]
        names = [names[index] for index in keep]
        designs = [design[:, keep] for design in designs]
    row_designs: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    offsets: list[np.ndarray] = []
    group_values: list[Any] = []
    event_rows: list[np.ndarray] = []
    observed_groups: list[tuple[int, ...]] = []
    position = 0
    for event, (design, cases, context) in enumerate(
        zip(designs, observed, contexts, strict=True)
    ):
        if grouping not in context:
            raise ValueError(f"mixture grouping variable {grouping!r} is unavailable")
        count = len(design)
        response = np.zeros(count, dtype=float)
        response[np.asarray(cases, dtype=int)] = 1.0
        offset = np.zeros(count, dtype=float)
        if exposures is not None:
            offset += np.log(float(exposures[event]))
        row_designs.append(design)
        responses.append(response)
        offsets.append(offset)
        group_values.extend(_random_group_values(context, (grouping,)))
        event_rows.append(np.arange(position, position + count, dtype=int))
        observed_groups.append(tuple(int(value) for value in cases))
        position += count
    return _MixtureRows(
        design=np.vstack(row_designs),
        response=np.concatenate(responses),
        offset=np.concatenate(offsets),
        group_values=_object_array(group_values),
        event_rows=tuple(event_rows),
        observed_groups=tuple(observed_groups),
        names=names,
        ordinal=ordinal,
        model=f"actor_{component}",
    )


def _stable_group_codes(values: np.ndarray) -> tuple[tuple[Any, ...], np.ndarray]:
    levels: list[Any] = []
    codes = np.empty(len(values), dtype=int)
    lookup: dict[Any, int] = {}
    for index, value in enumerate(values):
        key = value.item() if isinstance(value, np.generic) else value
        if key not in lookup:
            lookup[key] = len(levels)
            levels.append(key)
        codes[index] = lookup[key]
    return tuple(levels), codes


def _mixture_augmented_design(rows: _MixtureRows) -> tuple[np.ndarray, int]:
    if not rows.ordinal:
        return rows.design, 0
    strata = np.zeros((len(rows.response), len(rows.event_rows)), dtype=float)
    for event, index in enumerate(rows.event_rows):
        strata[index, event] = 1.0
    return np.column_stack([strata, rows.design]), strata.shape[1]


def _weighted_glm_component(
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    weights: np.ndarray,
    *,
    ordinal: bool,
    initial: np.ndarray,
    maxiter: int,
) -> tuple[np.ndarray, bool, int]:
    from scipy.optimize import minimize
    from scipy.special import expit

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        eta = design @ parameters + offset
        if ordinal:
            probabilities = expit(eta)
            loglik = np.sum(
                weights * (response * eta - np.logaddexp(0.0, eta))
            )
            gradient = design.T @ (weights * (response - probabilities))
        else:
            means = np.exp(np.clip(eta, -745.0, 700.0))
            loglik = np.sum(weights * (response * eta - means))
            gradient = design.T @ (weights * (response - means))
        return -float(loglik), -np.asarray(gradient, dtype=float)

    result = minimize(
        fun=lambda value: objective(value)[0],
        jac=lambda value: objective(value)[1],
        x0=initial,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": 1e-7},
    )
    return (
        np.asarray(result.x, dtype=float),
        bool(result.success),
        int(getattr(result, "nit", 0)),
    )


def _component_row_loglik(
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    parameters: np.ndarray,
    *,
    ordinal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    eta = design @ parameters + offset
    if ordinal:
        values = response * eta - np.logaddexp(0.0, eta)
    else:
        values = response * eta - np.exp(np.clip(eta, -745.0, 700.0))
    return np.asarray(values, dtype=float), np.asarray(eta, dtype=float)


def _mixture_em_once(
    rows: _MixtureRows,
    codes: np.ndarray,
    group_count: int,
    k: int,
    *,
    rng: np.random.Generator,
    maxiter: int,
    tol: float,
) -> dict[str, Any]:
    from scipy.special import logsumexp

    design, stratum_count = _mixture_augmented_design(rows)
    parameter_count = design.shape[1]
    responsibilities = rng.dirichlet(np.ones(k), size=group_count)
    parameters = rng.normal(0.0, 0.02, size=(parameter_count, k))
    priors = np.full(k, 1.0 / k, dtype=float)
    previous = -np.inf
    converged = False
    optimizer_success = True
    optimizer_iterations = 0
    row_loglik = np.empty((len(rows.response), k), dtype=float)
    linear = np.empty_like(row_loglik)
    for _iteration in range(1, maxiter + 1):
        for component in range(k):
            parameters[:, component], success, used = _weighted_glm_component(
                design,
                rows.response,
                rows.offset,
                responsibilities[codes, component],
                ordinal=rows.ordinal,
                initial=parameters[:, component],
                maxiter=max(50, maxiter),
            )
            optimizer_success = optimizer_success and success
            optimizer_iterations += used
            row_loglik[:, component], linear[:, component] = _component_row_loglik(
                design,
                rows.response,
                rows.offset,
                parameters[:, component],
                ordinal=rows.ordinal,
            )
        group_loglik = np.zeros((group_count, k), dtype=float)
        for component in range(k):
            np.add.at(group_loglik[:, component], codes, row_loglik[:, component])
        joint = group_loglik + np.log(np.maximum(priors, 1e-15))[None, :]
        normalizer = logsumexp(joint, axis=1)
        loglik = float(normalizer.sum())
        responsibilities = np.exp(joint - normalizer[:, None])
        priors = np.maximum(responsibilities.mean(axis=0), 1e-12)
        priors /= priors.sum()
        if np.isfinite(previous) and abs(loglik - previous) <= tol * (1.0 + abs(previous)):
            converged = True
            break
        previous = loglik
    else:
        _iteration = maxiter
    order = np.argsort(-priors, kind="stable")
    return {
        "parameters": parameters[:, order],
        "coefficients": parameters[stratum_count:, order],
        "priors": priors[order],
        "responsibilities": responsibilities[:, order],
        "row_loglik": row_loglik[:, order],
        "linear": linear[:, order],
        "loglik": loglik,
        "converged": converged,
        "optimizer_success": optimizer_success,
        "iterations": _iteration,
        "optimizer_iterations": optimizer_iterations,
        "parameter_count": parameter_count * k + (k - 1),
        "stratum_count": stratum_count,
    }


def _mixture_event_probabilities(
    rows: _MixtureRows,
    coefficients: np.ndarray,
    row_posterior: np.ndarray,
) -> tuple[
    tuple[np.ndarray, ...],
    tuple[int, ...],
    tuple[tuple[np.ndarray, ...], ...],
]:
    from scipy.special import softmax

    component_linear = rows.design @ coefficients
    weighted_linear = np.sum(row_posterior * component_linear, axis=1)
    joint: list[np.ndarray] = []
    observed_indices: list[int] = []
    component_values: list[list[np.ndarray]] = [
        [] for _ in range(coefficients.shape[1])
    ]
    for index, observed in zip(rows.event_rows, rows.observed_groups, strict=True):
        joint_probabilities = np.asarray(softmax(weighted_linear[index]), dtype=float)
        by_component = [
            np.asarray(softmax(component_linear[index, component]), dtype=float)
            for component in range(coefficients.shape[1])
        ]
        for case in observed:
            joint.append(np.array(joint_probabilities, copy=True))
            observed_indices.append(case)
            for component, probabilities in enumerate(by_component):
                component_values[component].append(np.array(probabilities, copy=True))
    return (
        tuple(joint),
        tuple(observed_indices),
        tuple(tuple(values) for values in component_values),
    )


def _fit_mixture_rows(
    rows: _MixtureRows,
    *,
    grouping: str,
    k: int,
    nrep: int,
    maxiter: int,
    tol: float,
    seed: int | None,
) -> RemEstimateMixture:
    levels, codes = _stable_group_codes(rows.group_values)
    if k > len(levels):
        raise ValueError("mixture k cannot exceed the number of grouping levels")
    master = np.random.default_rng(seed)
    fits = [
        _mixture_em_once(
            rows,
            codes,
            len(levels),
            k,
            rng=np.random.default_rng(
                int(master.integers(0, np.iinfo(np.int64).max))
            ),
            maxiter=maxiter,
            tol=tol,
        )
        for _ in range(nrep)
    ]
    best = max(fits, key=lambda value: float(value["loglik"]))
    coefficients = np.asarray(best["coefficients"], dtype=float)
    responsibilities = np.asarray(best["responsibilities"], dtype=float)
    row_posterior = responsibilities[codes]
    probabilities, observed_indices, component_probabilities = (
        _mixture_event_probabilities(rows, coefficients, row_posterior)
    )
    parameter_count = int(best["parameter_count"])
    loglik = float(best["loglik"])
    aic_value = float(-2.0 * loglik + 2.0 * parameter_count)
    bic_value = float(-2.0 * loglik + np.log(max(1, len(rows.response))) * parameter_count)
    assignments = pd.Series(
        np.argmax(responsibilities, axis=1).astype(int) + 1,
        index=list(levels),
        name="component",
    )
    metadata = {
        "backend": "numpy",
        "device": "cpu",
        "precision": "float64",
        "model": "actor" if rows.model.startswith("actor_") else rows.model,
        "component": rows.model.removeprefix("actor_") if rows.model.startswith("actor_") else None,
        "approach": "Frequentist",
        "method": "MIXREM",
        "engine": "finite-mixture-em",
        "ordinal": rows.ordinal,
        "statistics": list(rows.names),
        "n_events": len(rows.event_rows),
        "n_observations": len(rows.response),
        "k": k,
        "grouping": grouping,
        "prior_probs": np.asarray(best["priors"], dtype=float),
        "BIC": bic_value,
        "AIC": aic_value,
        "seed": seed,
        "rng": "numpy.random.Generator",
        "nrep": nrep,
    }
    backend_fit = {
        "parameters": np.asarray(best["parameters"], dtype=float),
        "posterior": responsibilities,
        "row_posterior": row_posterior,
        "assignments": assignments,
        "group_levels": levels,
        "component_row_log_likelihood": np.asarray(best["row_loglik"], dtype=float),
        "converged": bool(best["converged"]),
        "optimizer_success": bool(best["optimizer_success"]),
        "optimizer_iterations": int(best["optimizer_iterations"]),
        "restart_log_likelihoods": np.asarray(
            [float(value["loglik"]) for value in fits], dtype=float
        ),
    }
    return RemEstimateMixture(
        coef=coefficients,
        names=list(rows.names),
        log_likelihood=loglik,
        converged=bool(best["converged"] and np.isfinite(coefficients).all()),
        covariance=None,
        metadata=metadata,
        event_probabilities=probabilities,
        observed_indices=observed_indices,
        residual_deviance=-2.0 * loglik,
        iterations=int(best["iterations"]),
        prior_probs=np.asarray(best["priors"], dtype=float),
        posterior=responsibilities,
        assignments=assignments,
        grouping=grouping,
        group_levels=levels,
        component_event_probabilities=component_probabilities,
        backend_fit=backend_fit,
        bic_value=bic_value,
        aic_value=aic_value,
    )


def _remstimate_mixture(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    controls: Mapping[str, Any],
    seed: int | None,
) -> RemEstimateMixture | ActorRemEstimate | dict[str, Any]:
    settings = _mixture_controls(controls)
    grouping = _parse_mixture_group(settings["random"])

    def fit_k(k: int) -> RemEstimateMixture | ActorRemEstimate:
        nrep = int(settings["nrep"])
        maxiter = int(settings["maxiter"])
        tol = float(settings["tol"])
        fit_seed = None if seed is None else seed + 10_000 * k
        if isinstance(stats, RemStatsDuration):
            return _fit_mixture_rows(
                _mixture_duration_rows(stats, grouping),
                grouping=grouping,
                k=k,
                nrep=nrep,
                maxiter=maxiter,
                tol=tol,
                seed=fit_seed,
            )
        if isinstance(stats, RemStats):
            return _fit_mixture_rows(
                _mixture_tie_rows(history, stats, grouping),
                grouping=grouping,
                k=k,
                nrep=nrep,
                maxiter=maxiter,
                tol=tol,
                seed=fit_seed,
            )
        sender_rows = _mixture_actor_rows(stats, grouping, "sender")
        receiver_rows = _mixture_actor_rows(stats, grouping, "receiver")
        sender = (
            None
            if sender_rows is None
            else _fit_mixture_rows(
                sender_rows,
                grouping=grouping,
                k=k,
                nrep=nrep,
                maxiter=maxiter,
                tol=tol,
                seed=fit_seed,
            )
        )
        receiver = (
            None
            if receiver_rows is None
            else _fit_mixture_rows(
                receiver_rows,
                grouping=grouping,
                k=k,
                nrep=nrep,
                maxiter=maxiter,
                tol=tol,
                seed=fit_seed,
            )
        )
        return ActorRemEstimate(
            sender,
            receiver,
            {
                "backend": "numpy",
                "model": "actor",
                "method": "MIXREM",
                "engine": "finite-mixture-em",
                "approach": "Frequentist",
                "k": k,
                "grouping": grouping,
                "seed": seed,
            },
        )

    k_values = settings["k"]
    fitted = {f"k{k}": fit_k(k) for k in k_values}
    return next(iter(fitted.values())) if len(fitted) == 1 else fitted


def _shrinkage_controls(penalty: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "prior",
        "type",
        "lambda",
        "scale",
        "maxiter",
        "tol",
        "threshold",
        "unpenalized",
        "penalized",
    }
    unknown = set(penalty).difference(allowed)
    if unknown:
        raise TypeError(f"unsupported Bayesian penalty controls: {sorted(unknown)!r}")
    prior = str(penalty.get("prior", penalty.get("type", "horseshoe"))).lower()
    if prior not in {"horseshoe", "lasso", "ridge"}:
        raise ValueError("Bayesian penalty prior must be 'horseshoe', 'lasso', or 'ridge'")
    lambda_value = penalty.get("lambda", penalty.get("scale", 1.0))
    tol = penalty.get("tol", 1e-9)
    threshold = penalty.get("threshold", 1e-6)
    for name, value in (
        ("lambda", lambda_value),
        ("tol", tol),
        ("threshold", threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"Bayesian penalty {name} must be a positive finite number")
    maxiter = penalty.get("maxiter", 1000)
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("Bayesian penalty maxiter must be a positive integer")
    return {
        "prior": prior,
        "lambda": float(lambda_value),
        "tol": float(tol),
        "threshold": float(threshold),
        "maxiter": maxiter,
        "unpenalized": penalty.get("unpenalized"),
        "penalized": penalty.get("penalized"),
    }


def _assert_estimable_design(design: np.ndarray, names: Sequence[str]) -> None:
    duplicated = [name for index, name in enumerate(names) if name in names[:index]]
    if duplicated:
        raise ValueError(
            "penalized REM has duplicated statistics and a rank-deficient design: "
            f"{sorted(set(duplicated))!r}"
        )
    varying = [
        index
        for index in range(design.shape[1])
        if len(design) > 1
        and np.isfinite(design[:, index]).all()
        and float(np.std(design[:, index], ddof=1)) > 0.0
    ]
    if len(varying) < 2:
        return
    selected = design[:, varying]
    scale = np.std(selected, axis=0, ddof=1)
    standardized = (selected - np.mean(selected, axis=0)) / scale
    _, _, pivot = _pivoted_qr(standardized)
    rank = int(np.linalg.matrix_rank(standardized, tol=1e-7))
    if rank < standardized.shape[1]:
        aliased = [names[varying[int(index)]] for index in pivot[rank:]]
        raise ValueError(
            "penalized REM has collinear statistics; the MLE covariance required "
            f"by Bayesian shrinkage is rank-deficient: {aliased!r}"
        )


def _pivoted_qr(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.linalg import qr

    q_value, r_value, pivot = qr(value, mode="economic", pivoting=True)
    return (
        np.asarray(q_value, dtype=float),
        np.asarray(r_value, dtype=float),
        np.asarray(pivot, dtype=int),
    )


def _approximate_bayesian_shrinkage(
    mle: np.ndarray,
    covariance: np.ndarray,
    penalized_mask: np.ndarray,
    *,
    prior: str,
    lambda_value: float,
    maxiter: int,
    tol: float,
    threshold: float,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    precision = np.linalg.pinv(
        0.5 * (covariance + covariance.T),
        hermitian=True,
    )
    target = precision @ mle
    beta = np.array(mle, dtype=float, copy=True)
    penalty_weights = np.zeros(len(beta), dtype=float)
    converged = False
    iterations = 0
    if prior == "ridge":
        penalty_weights[penalized_mask] = lambda_value
        beta = np.linalg.pinv(
            precision + np.diag(penalty_weights), hermitian=True
        ) @ target
        converged = True
        iterations = 1
    elif prior == "lasso":
        diagonal = np.maximum(np.diag(precision), np.finfo(float).eps)
        for _iterations in range(1, maxiter + 1):
            previous = beta.copy()
            for index in range(len(beta)):
                residual = target[index] - (
                    precision[index] @ beta - precision[index, index] * beta[index]
                )
                if penalized_mask[index]:
                    beta[index] = np.sign(residual) * max(
                        abs(residual) - lambda_value,
                        0.0,
                    ) / diagonal[index]
                else:
                    beta[index] = residual / diagonal[index]
            if np.max(np.abs(beta - previous), initial=0.0) <= tol * (
                1.0 + np.max(np.abs(previous), initial=0.0)
            ):
                converged = True
                break
        iterations = _iterations
        penalty_weights[penalized_mask] = lambda_value / np.maximum(
            np.abs(beta[penalized_mask]), threshold
        )
    else:
        global_scale = max(lambda_value, np.finfo(float).eps)
        for _iterations in range(1, maxiter + 1):
            previous = beta.copy()
            local_variance = beta * beta + np.diag(covariance)
            penalty_weights.fill(0.0)
            penalty_weights[penalized_mask] = 1.0 / np.maximum(
                local_variance[penalized_mask] + global_scale * global_scale,
                threshold * threshold,
            )
            beta = np.linalg.pinv(
                precision + np.diag(penalty_weights), hermitian=True
            ) @ target
            if np.max(np.abs(beta - previous), initial=0.0) <= tol * (
                1.0 + np.max(np.abs(previous), initial=0.0)
            ):
                converged = True
                break
        iterations = _iterations
    selected = (~penalized_mask) | (np.abs(beta) > threshold)
    posterior_covariance = np.linalg.pinv(
        precision + np.diag(penalty_weights), hermitian=True
    )
    posterior_sd = np.sqrt(np.maximum(np.diag(posterior_covariance), 0.0))
    estimates = pd.DataFrame(
        {
            "input.est": mle,
            "input.sd": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            "shrunk.mode": beta,
            "posterior.sd": posterior_sd,
            "nonzero": selected,
        }
    )
    backend = {
        "precision": precision,
        "posterior_covariance": posterior_covariance,
        "penalty_weights": penalty_weights,
        "converged": converged,
        "iterations": iterations,
        "prior": prior,
        "lambda": lambda_value,
    }
    return beta, estimates, selected, backend


def _shrink_reference_component(
    reference: RemEstimate,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
    controls: Mapping[str, Any],
    design_frame: pd.DataFrame,
    seed: int | None,
) -> RemEstimateShrinkage:
    if reference.covariance is None:
        raise ValueError("Bayesian shrinkage requires a finite MLE covariance")
    matrix = np.vstack(designs) if designs else np.empty((0, len(reference.names)))
    _assert_estimable_design(matrix, reference.names)
    _check_penalty_names(
        valid=reference.names,
        penalized=controls.get("penalized"),
        unpenalized=controls.get("unpenalized"),
    )
    unpenalized = _resolve_unpenalized(
        design_frame,
        reference.names,
        unpenalized=controls.get("unpenalized"),
        penalized=controls.get("penalized"),
    )
    penalized_mask = np.asarray(
        [name not in unpenalized for name in reference.names], dtype=bool
    )
    coefficients, estimates, selected, backend_fit = (
        _approximate_bayesian_shrinkage(
            np.asarray(reference.coef, dtype=float),
            np.asarray(reference.covariance, dtype=float),
            penalized_mask,
            prior=str(controls["prior"]),
            lambda_value=float(controls["lambda"]),
            maxiter=int(controls["maxiter"]),
            tol=float(controls["tol"]),
            threshold=float(controls["threshold"]),
        )
    )
    estimates.index = reference.names
    probabilities, observed_indices = _tie_event_probabilities(
        coefficients,
        designs,
        observed_groups,
        sampling_weights=sampling_weights,
    )
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "approach": "Bayesian",
            "method": "SHRINKEM",
            "engine": "empirical-bayes-shrinkage",
            "shrinkem_type": controls["prior"],
            "unpenalized": list(unpenalized),
            "selected": selected,
            "seed": seed,
        }
    )
    return RemEstimateShrinkage(
        coef=coefficients,
        names=list(reference.names),
        log_likelihood=reference.log_likelihood,
        converged=bool(reference.converged and backend_fit["converged"]),
        covariance=np.asarray(
            backend_fit["posterior_covariance"], dtype=float
        ),
        metadata=metadata,
        event_probabilities=probabilities,
        observed_indices=observed_indices,
        residual_deviance=reference.residual_deviance,
        null_deviance=reference.null_deviance,
        model_deviance=reference.model_deviance,
        iterations=int(backend_fit["iterations"]),
        posterior_mean=coefficients,
        posterior_sd=estimates["posterior.sd"].to_numpy(dtype=float),
        shrinkage_type=str(controls["prior"]),
        estimates=estimates,
        selected=selected,
        unpenalized=tuple(unpenalized),
        backend_fit=backend_fit,
    )


def _remstimate_tie_shrinkage(
    history: EventHistory,
    stats: RemStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    penalty: Mapping[str, Any],
    seed: int | None,
) -> RemEstimateShrinkage:
    if isinstance(backend, JaxBackend):
        raise NotImplementedError(
            "Bayesian shrinkage is currently available only on backend='numpy'"
        )
    controls = _shrinkage_controls(penalty)
    reference = _remstimate_tie(
        history,
        stats,
        backend,
        optimizer,
        seed=seed,
        riskset_chunk_size=None,
        compute_waic=False,
        nsim_waic=100,
    )
    designs = _select_design_columns(
        [np.asarray(values, dtype=float) for values in stats.stats],
        stats.names,
        reference.names,
    )
    frame = pd.DataFrame(np.vstack(designs), columns=reference.names)
    return _shrink_reference_component(
        reference,
        designs,
        stats.observed_index_groups
        or [[index] for index in stats.observed_indices],
        exposures=None if history.ordinal else _event_exposures(history, stats.event_indices),
        sampling_weights=(
            [np.asarray(values, dtype=float) for values in stats.sampling_weights]
            if stats.sampling_weights
            else None
        ),
        controls=controls,
        design_frame=frame,
        seed=seed,
    )


def _remstimate_actor_shrinkage(
    stats: AomStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    penalty: Mapping[str, Any],
    seed: int | None,
) -> ActorRemEstimate:
    if isinstance(backend, JaxBackend):
        raise NotImplementedError(
            "Bayesian shrinkage is currently available only on backend='numpy'"
        )
    controls = _shrinkage_controls(penalty)
    reference = _remstimate_actor(
        stats,
        backend,
        optimizer,
        seed=seed,
        compute_waic=False,
        nsim_waic=100,
    )
    sender: RemEstimateShrinkage | None = None
    if reference.sender_model is not None:
        designs = _select_design_columns(
            [np.asarray(values, dtype=float) for values in stats.sender_stats],
            stats.sender_names,
            reference.sender_model.names,
        )
        sender = _shrink_reference_component(
            reference.sender_model,
            designs,
            stats.observed_sender_groups
            or [[index] for index in stats.observed_sender_indices],
            exposures=(
                None
                if stats.history.ordinal
                else _event_exposures(stats.history, stats.event_indices)
            ),
            sampling_weights=None,
            controls=controls,
            design_frame=pd.DataFrame(
                np.vstack(designs), columns=reference.sender_model.names
            ),
            seed=seed,
        )
    receiver: RemEstimateShrinkage | None = None
    if reference.receiver_model is not None:
        designs, observed = _actor_receiver_diagnostic_designs(stats)
        designs = _select_design_columns(
            designs,
            stats.receiver_names,
            reference.receiver_model.names,
        )
        receiver = _shrink_reference_component(
            reference.receiver_model,
            designs,
            observed,
            exposures=None,
            sampling_weights=None,
            controls=controls,
            design_frame=pd.DataFrame(
                np.vstack(designs), columns=reference.receiver_model.names
            ),
            seed=seed,
        )
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "approach": "Bayesian",
            "method": "SHRINKEM",
            "engine": "empirical-bayes-shrinkage",
            "shrinkem_type": controls["prior"],
        }
    )
    return ActorRemEstimate(sender, receiver, metadata)


def _remstimate_duration_shrinkage(
    stats: RemStatsDuration,
    backend: ArrayBackend,
    optimizer: str,
    *,
    penalty: Mapping[str, Any],
    seed: int | None,
) -> RemEstimateShrinkage:
    if isinstance(backend, JaxBackend):
        raise NotImplementedError(
            "Bayesian shrinkage is currently available only on backend='numpy'"
        )
    controls = _shrinkage_controls(penalty)
    reference = _remstimate_duration(stats, backend, optimizer, seed=seed)
    if reference.covariance is None:
        raise ValueError("Bayesian shrinkage requires a finite MLE covariance")
    frame = stats.stacked.remstats_stack
    names = list(reference.names)
    design = frame[names].to_numpy(dtype=float)
    _check_penalty_names(
        valid=names,
        penalized=controls.get("penalized"),
        unpenalized=controls.get("unpenalized"),
    )
    unpenalized = _resolve_unpenalized(
        frame,
        names,
        unpenalized=controls.get("unpenalized"),
        penalized=controls.get("penalized"),
    )
    penalized_mask = np.asarray(
        [name not in unpenalized for name in names], dtype=bool
    )
    coefficients, estimates, selected, backend_fit = (
        _approximate_bayesian_shrinkage(
            reference.coef,
            reference.covariance,
            penalized_mask,
            prior=str(controls["prior"]),
            lambda_value=float(controls["lambda"]),
            maxiter=int(controls["maxiter"]),
            tol=float(controls["tol"]),
            threshold=float(controls["threshold"]),
        )
    )
    estimates.index = names
    groups = _duration_groups(frame)
    offset = (
        None
        if stats.stacked.ordinal
        else frame["log_interevent"].to_numpy(dtype=float)
    )
    fitted_values = _duration_fitted_values(
        coefficients,
        design,
        offset=offset,
        groups=groups,
    )
    probabilities, observed_indices = _duration_event_probabilities(
        fitted_values,
        frame["obs"].to_numpy(dtype=float),
        groups,
    )
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "model": "duration",
            "approach": "Bayesian",
            "method": "SHRINKEM",
            "engine": "empirical-bayes-shrinkage",
            "shrinkem_type": controls["prior"],
            "unpenalized": list(unpenalized),
            "selected": selected,
            "seed": seed,
        }
    )
    return RemEstimateShrinkage(
        coef=coefficients,
        names=names,
        log_likelihood=reference.log_likelihood,
        converged=bool(reference.converged and backend_fit["converged"]),
        covariance=np.asarray(
            backend_fit["posterior_covariance"], dtype=float
        ),
        metadata=metadata,
        event_probabilities=probabilities,
        observed_indices=observed_indices,
        residual_deviance=reference.residual_deviance,
        null_deviance=reference.null_deviance,
        model_deviance=reference.model_deviance,
        iterations=int(backend_fit["iterations"]),
        posterior_mean=coefficients,
        posterior_sd=estimates["posterior.sd"].to_numpy(dtype=float),
        shrinkage_type=str(controls["prior"]),
        estimates=estimates,
        selected=selected,
        unpenalized=tuple(unpenalized),
        backend_fit=backend_fit,
        stacked_data=stats.stacked,
        fitted_values=fitted_values,
    )


def _remstimate_tie(
    history: EventHistory,
    stats: RemStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    seed: int | None,
    riskset_chunk_size: int | None,
    compute_waic: bool,
    nsim_waic: int,
) -> RemEstimate:
    names = list(stats.names)
    designs = [np.asarray(values, dtype=float) for values in stats.stats]
    if history.ordinal and "baseline" in names:
        keep = [index for index, name in enumerate(names) if name != "baseline"]
        names = [names[index] for index in keep]
        designs = [design[:, keep] for design in designs]
    observed_groups = [
        [int(index) for index in group]
        for group in (stats.observed_index_groups or [[index] for index in stats.observed_indices])
    ]
    sampling_weights = (
        [np.asarray(values, dtype=float) for values in stats.sampling_weights]
        if stats.sampling_weights
        else None
    )
    exposures = None if history.ordinal else _event_exposures(history, stats.event_indices)
    parameter_count = len(names)
    if designs and designs[0].shape[1] != parameter_count:
        raise ValueError("tie statistic names do not align with the model matrix")

    if parameter_count == 0:
        coefficients = np.zeros(0, dtype=float)
        loglik, gradient, hessian = _tie_loglik_grad_hessian(
            coefficients,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        success = True
        message = "closed-form zero-parameter model"
        iterations = 0
    else:
        coefficients, success, iterations = _tie_trust_fit(
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        message = "trust region converged" if success else "trust iteration limit reached"
        loglik, gradient, hessian = _tie_loglik_grad_hessian(
            coefficients,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
    information = _tie_observed_information(
        coefficients,
        designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=sampling_weights,
    )
    covariance = (
        np.zeros((0, 0), dtype=float)
        if parameter_count == 0
        else np.linalg.pinv(information, hermitian=True)
    )
    null_loglik = _tie_null_loglik(
        designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=sampling_weights,
    )
    residual_deviance = -2.0 * loglik
    null_deviance = -2.0 * null_loglik
    event_probabilities, observed_indices = _tie_event_probabilities(
        coefficients,
        designs,
        observed_groups,
        sampling_weights=sampling_weights,
    )
    metadata = _metadata(
        backend,
        optimizer,
        len(designs),
        seed=seed,
        timing="ordinal" if history.ordinal else "exact",
    )
    metadata.update(
        {
            "message": message,
            "model": "tie",
            "ordinal": history.ordinal,
            "method": "MLE",
            "approach": "Frequentist",
            "statistics": names,
            "where_is_baseline": (names.index("baseline") + 1 if "baseline" in names else None),
            "ncores": 1,
            "sampled": sampling_weights is not None,
            "samp_num": len(designs[0]) if sampling_weights and designs else None,
            "sampling_scheme": (
                "case-control" if sampling_weights is not None else None
            ),
            "case_control_sampling": sampling_weights is not None,
            "riskset_chunk_size": riskset_chunk_size,
            "n_observations": len(designs),
            "df.null": len(designs),
            "df.model": parameter_count,
            "df.residual": len(designs) - parameter_count,
            "jax_batched": bool(
                isinstance(backend, JaxBackend)
                and riskset_chunk_size is None
                and all(len(group) == 1 for group in observed_groups)
                and len({design.shape for design in designs}) <= 1
            ),
        }
    )
    if compute_waic:
        parameter_draws = _mvn_parameter_draws(
            coefficients,
            covariance,
            nsim=nsim_waic,
            seed=seed,
        )
        event_loglik = _tie_event_loglik_draws(
            parameter_draws,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        metadata["WAIC"] = _waic_from_event_loglik(event_loglik)
        metadata["nsimWAIC"] = nsim_waic
        metadata["waic_rng"] = "numpy.random.Generator"
    converged = bool(
        success
        and np.isfinite(loglik)
        and np.isfinite(coefficients).all()
        and np.isfinite(gradient).all()
    )
    return RemEstimate(
        coef=coefficients,
        names=names,
        log_likelihood=float(loglik),
        converged=converged,
        covariance=covariance,
        metadata=metadata,
        event_probabilities=event_probabilities,
        observed_indices=observed_indices,
        gradient=gradient,
        hessian=information,
        residual_deviance=residual_deviance,
        null_deviance=null_deviance,
        model_deviance=null_deviance - residual_deviance,
        iterations=iterations,
        sampled=sampling_weights is not None,
    )


def _validate_penalty_controls(
    penalty: Mapping[str, Any],
) -> tuple[float, float | None, int, str, np.ndarray | None]:
    allowed = {
        "alpha",
        "lambda",
        "nfolds",
        "lambda_select",
        "foldid",
        "unpenalized",
        "penalized",
    }
    unknown = sorted(set(penalty).difference(allowed))
    if unknown:
        raise TypeError(f"unsupported penalty controls: {', '.join(unknown)}")
    alpha_value = penalty.get("alpha", 1.0)
    if isinstance(alpha_value, bool) or not isinstance(
        alpha_value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("penalty.alpha must be a number between 0 and 1")
    alpha = float(alpha_value)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("penalty.alpha must be between 0 and 1")
    lambda_raw = penalty.get("lambda")
    lambda_value: float | None = None
    if lambda_raw is not None:
        if isinstance(lambda_raw, bool) or not isinstance(
            lambda_raw, (int, float, np.integer, np.floating)
        ):
            raise TypeError("penalty.lambda must be a positive number")
        lambda_value = float(lambda_raw)
        if not np.isfinite(lambda_value) or lambda_value <= 0.0:
            raise ValueError("penalty.lambda must be a positive number")
    nfolds_raw = penalty.get("nfolds", 10)
    if isinstance(nfolds_raw, bool) or not isinstance(nfolds_raw, (int, np.integer)):
        raise TypeError("penalty.nfolds must be an integer")
    nfolds = int(nfolds_raw)
    if nfolds < 3:
        raise ValueError("penalty.nfolds must be at least 3")
    lambda_select = str(penalty.get("lambda_select", "1se"))
    if lambda_select not in {"1se", "min"}:
        raise ValueError("penalty.lambda_select must be '1se' or 'min'")
    foldid_raw = penalty.get("foldid")
    foldid = None if foldid_raw is None else np.asarray(foldid_raw, dtype=int)
    if foldid is not None and (foldid.ndim != 1 or len(foldid) == 0):
        raise ValueError("penalty.foldid must be a non-empty one-dimensional sequence")
    return alpha, lambda_value, nfolds, lambda_select, foldid


def _flatten_glmnet_inputs(
    designs: Sequence[np.ndarray],
    observed_groups: Sequence[Sequence[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: Sequence[np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the row-wise response and offset used by penalized GLMs."""

    matrices = [np.asarray(value, dtype=float) for value in designs]
    design = np.vstack(matrices)
    response_parts: list[np.ndarray] = []
    offset_parts: list[np.ndarray] = []
    for position, (matrix, observed) in enumerate(
        zip(matrices, observed_groups, strict=True)
    ):
        values = np.bincount(np.asarray(observed, dtype=int), minlength=len(matrix)).astype(float)
        response_parts.append(values)
        current = np.zeros(len(matrix), dtype=float)
        if exposures is not None:
            exposure = float(exposures[position])
            if exposure <= 0.0 or not np.isfinite(exposure):
                raise ValueError("glmnet exposure must be finite and positive")
            current += np.log(exposure)
        if sampling_weights is not None:
            weights = np.asarray(sampling_weights[position], dtype=float)
            if len(weights) != len(matrix) or np.any(weights <= 0.0):
                raise ValueError("sampling weights must be positive and align with the risk set")
            current += np.log(weights)
        offset_parts.append(current)
    return design, np.concatenate(response_parts), np.concatenate(offset_parts)


def _flat_glm_loss_gradient(
    beta: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    *,
    family: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    eta = design @ beta + offset
    if family == "poisson":
        mean = np.exp(np.clip(eta, -745.0, 700.0))
        loss = float(np.mean(mean - response * eta))
        gradient = design.T @ (mean - response) / len(response)
        curvature = mean
    elif family == "binomial":
        probability = np.empty_like(eta)
        positive = eta >= 0.0
        probability[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
        exponential = np.exp(eta[~positive])
        probability[~positive] = exponential / (1.0 + exponential)
        loss = float(np.mean(np.logaddexp(0.0, eta) - response * eta))
        gradient = design.T @ (probability - response) / len(response)
        curvature = probability * (1.0 - probability)
    else:  # pragma: no cover - internal contract
        raise ValueError(f"unknown glmnet family: {family}")
    hessian = design.T @ (curvature[:, None] * design) / len(response)
    return loss, np.asarray(gradient, dtype=float), np.asarray(hessian, dtype=float)


def _flat_glm_unpenalized_fit(
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    penalty_mask: np.ndarray,
    *,
    family: str,
) -> np.ndarray:
    from scipy.optimize import minimize

    beta = np.zeros(design.shape[1], dtype=float)
    free = np.flatnonzero(~penalty_mask)
    if not len(free):
        return beta

    def objective(value: np.ndarray) -> tuple[float, np.ndarray]:
        candidate = np.zeros(design.shape[1], dtype=float)
        candidate[free] = value
        loss, gradient, _ = _flat_glm_loss_gradient(
            candidate, design, response, offset, family=family
        )
        return loss, gradient[free]

    fitted = minimize(
        fun=lambda value: objective(value)[0],
        x0=np.zeros(len(free), dtype=float),
        jac=lambda value: objective(value)[1],
        method="BFGS",
        options={"gtol": 1e-12, "maxiter": 1000},
    )
    beta[free] = np.asarray(fitted.x, dtype=float)
    return beta


def _flat_elastic_net_optimize(
    initial: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    penalty_mask: np.ndarray,
    *,
    family: str,
    alpha: float,
    lambda_value: float,
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int, bool]:
    """Fit glmnet's normalized elastic-net objective by proximal Newton steps."""

    beta = np.asarray(initial, dtype=float).copy()

    def objective(value: np.ndarray) -> float:
        loss, _, _ = _flat_glm_loss_gradient(
            value, design, response, offset, family=family
        )
        selected = value[penalty_mask]
        return float(
            loss
            + lambda_value
            * (
                alpha * np.abs(selected).sum()
                + 0.5 * (1.0 - alpha) * float(selected @ selected)
            )
        )

    current = objective(beta)
    for iteration in range(1, max_iterations + 1):
        _, gradient, hessian = _flat_glm_loss_gradient(
            beta, design, response, offset, family=family
        )
        gradient[penalty_mask] += (
            lambda_value * (1.0 - alpha) * beta[penalty_mask]
        )
        lipschitz = max(
            np.finfo(float).eps,
            float(np.linalg.eigvalsh(0.5 * (hessian + hessian.T)).max(initial=0.0))
            + lambda_value * (1.0 - alpha),
        )
        step = 1.0 / lipschitz
        candidate = beta.copy()
        candidate_objective = current
        for _ in range(40):
            candidate = beta - step * gradient
            selected = candidate[penalty_mask]
            candidate[penalty_mask] = np.sign(selected) * np.maximum(
                np.abs(selected) - step * lambda_value * alpha, 0.0
            )
            candidate_objective = objective(candidate)
            difference = candidate - beta
            if candidate_objective <= current - 1e-10 * float(difference @ difference):
                break
            step *= 0.5
        scale = 1.0 + float(np.max(np.abs(beta), initial=0.0))
        beta = candidate
        if float(np.max(np.abs(difference), initial=0.0)) <= tolerance * scale:
            return beta, iteration, True
        current = candidate_objective
    return beta, max_iterations, False


def _glmnet_penalty_fit(
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    names: Sequence[str],
    penalty_mask: np.ndarray,
    *,
    ordinal: bool,
    alpha: float,
    lambda_value: float | None,
    nfolds: int,
    lambda_select: str,
    foldid: np.ndarray | None,
    seed: int | None,
) -> tuple[np.ndarray, int, bool, float, float | None, float | None, str]:
    """Fit a cross-validated row-wise penalized GLM."""

    if len(design) != len(response) or len(offset) != len(response):
        raise ValueError("glmnet design, response, and offset must align")
    constant = np.asarray(
        [len(np.unique(design[:, column])) <= 1 for column in range(design.shape[1])],
        dtype=bool,
    )
    ones = constant & np.all(design == 1.0, axis=0)
    intercept_candidates = np.flatnonzero(ones)
    intercept_index = int(intercept_candidates[0]) if len(intercept_candidates) else None
    varying = np.flatnonzero(~constant)
    if len(varying) < 2:
        raise ValueError(
            "glmnet needs at least two varying statistics to penalize; "
            f"this model has: {[names[index] for index in varying]}"
        )

    raw = design[:, varying]
    if intercept_index is not None:
        centers = np.mean(raw, axis=0)
        centered = raw - centers
    else:
        centers = np.zeros(len(varying), dtype=float)
        centered = raw
    scales = np.sqrt(np.mean(np.square(centered), axis=0))
    scales[scales == 0.0] = 1.0
    standardized = centered / scales
    if intercept_index is not None:
        fit_design = np.column_stack([np.ones(len(design)), standardized])
        fit_penalty_mask = np.concatenate(
            [[False], penalty_mask[varying]]
        ).astype(bool)
    else:
        fit_design = standardized
        fit_penalty_mask = penalty_mask[varying].astype(bool)
    family = "binomial" if ordinal else "poisson"

    initial = _flat_glm_unpenalized_fit(
        fit_design, response, offset, fit_penalty_mask, family=family
    )
    selected_label = "explicit"
    lambda_min: float | None = None
    lambda_1se: float | None = None
    if lambda_value is None:
        _, null_gradient, _ = _flat_glm_loss_gradient(
            initial, fit_design, response, offset, family=family
        )
        divisor = max(alpha, 1e-3)
        lambda_max = float(
            np.max(np.abs(null_gradient[fit_penalty_mask]), initial=0.0) / divisor
        )
        if not np.isfinite(lambda_max) or lambda_max <= 0.0:
            raise ValueError("glmnet could not construct a positive lambda path")
        ratio = 1e-4 if len(response) >= len(varying) else 1e-2
        lambda_path = lambda_max * ratio ** (np.arange(100, dtype=float) / 99.0)
        lambda_min, lambda_1se = _glmnet_cross_validated_lambdas(
            fit_design,
            response,
            offset,
            fit_penalty_mask,
            family=family,
            alpha=alpha,
            lambdas=lambda_path,
            nfolds=nfolds,
            foldid=foldid,
            seed=seed,
        )
        lambda_value = lambda_min if lambda_select == "min" else lambda_1se
        selected_label = lambda_select

    fitted, iterations, converged = _flat_elastic_net_optimize(
        initial,
        fit_design,
        response,
        offset,
        fit_penalty_mask,
        family=family,
        alpha=alpha,
        lambda_value=lambda_value,
    )
    coefficients = np.zeros(design.shape[1], dtype=float)
    standardized_coefficients = fitted[1:] if intercept_index is not None else fitted
    coefficients[varying] = standardized_coefficients / scales
    if intercept_index is not None:
        coefficients[intercept_index] = float(
            fitted[0] - centers @ coefficients[varying]
        )
    coefficients[np.abs(coefficients) < 1e-12] = 0.0
    return (
        coefficients,
        iterations,
        converged,
        float(lambda_value),
        lambda_min,
        lambda_1se,
        selected_label,
    )


def _glmnet_cross_validated_lambdas(
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    penalty_mask: np.ndarray,
    *,
    family: str,
    alpha: float,
    lambdas: np.ndarray,
    nfolds: int,
    foldid: np.ndarray | None,
    seed: int | None,
) -> tuple[float, float]:
    row_count = len(response)
    if nfolds > row_count:
        raise ValueError("penalty.nfolds cannot exceed the number of model rows")
    if foldid is None:
        folds = _balanced_cv_folds(row_count, nfolds, seed)
    else:
        if len(foldid) != row_count:
            raise ValueError("penalty.foldid must have one value per model row")
        _, folds = np.unique(foldid, return_inverse=True)
        nfolds = int(folds.max(initial=-1)) + 1
        if nfolds < 3:
            raise ValueError("penalty.foldid must define at least three folds")

    fold_errors = np.empty((nfolds, len(lambdas)), dtype=float)
    for fold in range(nfolds):
        validation = folds == fold
        training = ~validation
        train_design = np.asarray(design[training], dtype=float).copy()
        validation_design = np.asarray(design[validation], dtype=float).copy()
        has_intercept = bool(
            design.shape[1]
            and not penalty_mask[0]
            and np.all(design[:, 0] == 1.0)
        )
        if has_intercept and design.shape[1] > 1:
            centers = np.mean(train_design[:, 1:], axis=0)
            scales = np.sqrt(
                np.mean(np.square(train_design[:, 1:] - centers), axis=0)
            )
            scales[scales == 0.0] = 1.0
            train_design[:, 1:] = (train_design[:, 1:] - centers) / scales
            validation_design[:, 1:] = (
                validation_design[:, 1:] - centers
            ) / scales
        beta = _flat_glm_unpenalized_fit(
            train_design,
            response[training],
            offset[training],
            penalty_mask,
            family=family,
        )
        for position, current_lambda in enumerate(lambdas):
            beta, _, _ = _flat_elastic_net_optimize(
                beta,
                train_design,
                response[training],
                offset[training],
                penalty_mask,
                family=family,
                alpha=alpha,
                lambda_value=float(current_lambda),
                max_iterations=1000,
                tolerance=1e-9,
            )
            eta = validation_design @ beta + offset[validation]
            observed = response[validation]
            if family == "poisson":
                mean = np.exp(np.clip(eta, -745.0, 700.0))
                positive = observed > 0.0
                deviance = 2.0 * (mean - observed)
                deviance[positive] += 2.0 * observed[positive] * np.log(
                    observed[positive] / mean[positive]
                )
            else:
                probability = np.clip(
                    1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0))),
                    np.finfo(float).eps,
                    1.0 - np.finfo(float).eps,
                )
                deviance = -2.0 * (
                    observed * np.log(probability)
                    + (1.0 - observed) * np.log1p(-probability)
                )
            fold_errors[fold, position] = float(np.mean(deviance))

    means = np.mean(fold_errors, axis=0)
    standard_errors = np.std(fold_errors, axis=0, ddof=1) / np.sqrt(nfolds)
    minimum_index = int(np.nanargmin(means))
    threshold = means[minimum_index] + standard_errors[minimum_index]
    eligible = np.flatnonzero(means <= threshold)
    one_se_index = int(eligible[0])
    return float(lambdas[minimum_index]), float(lambdas[one_se_index])


def _balanced_cv_folds(row_count: int, nfolds: int, seed: int | None) -> np.ndarray:
    """Reproduce R's seeded ``sample(rep(seq(nfolds), length = n))`` folds."""

    values = np.resize(np.arange(nfolds, dtype=int), row_count)
    if seed is None:
        np.random.default_rng().shuffle(values)
        return values

    state_value = int(seed) & 0xFFFFFFFF
    for _ in range(51):
        state_value = (69069 * state_value + 1) & 0xFFFFFFFF
    state: list[int] = []
    for _ in range(624):
        state_value = (69069 * state_value + 1) & 0xFFFFFFFF
        state.append(state_value)
    bit_generator = np.random.MT19937()
    bit_state = bit_generator.state
    bit_state["state"]["key"] = np.asarray(state, dtype=np.uint32)
    bit_state["state"]["pos"] = 624
    bit_generator.state = bit_state

    def uniform_index(limit: int) -> int:
        bits = int(np.ceil(np.log2(limit)))
        while True:
            value = 0
            for _ in range(0, bits + 1, 16):
                value = 65536 * value + (int(bit_generator.random_raw()) >> 16)
            value &= (1 << bits) - 1
            if value < limit:
                return value

    available = list(range(row_count))
    permutation = np.empty(row_count, dtype=int)
    for position in range(row_count):
        selected = uniform_index(len(available))
        permutation[position] = available[selected]
        available[selected] = available[-1]
        available.pop()
    return np.asarray(values[permutation], dtype=int)


def _remstimate_tie_penalized(
    history: EventHistory,
    stats: RemStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    penalty: dict[str, Any],
    seed: int | None,
) -> RemEstimateGlmnet:
    if isinstance(backend, JaxBackend):
        raise NotImplementedError(
            "penalized estimation is currently available only on backend='numpy'"
        )
    alpha, requested_lambda, nfolds, lambda_select, foldid = (
        _validate_penalty_controls(penalty)
    )

    reference = _remstimate_tie(
        history,
        stats,
        backend,
        optimizer,
        seed=seed,
        riskset_chunk_size=None,
        compute_waic=False,
        nsim_waic=100,
    )
    names = list(reference.names)
    designs = [np.asarray(values, dtype=float) for values in stats.stats]
    designs = _select_design_columns(designs, stats.names, names)
    observed_groups = [
        [int(index) for index in group]
        for group in (
            stats.observed_index_groups
            or [[index] for index in stats.observed_indices]
        )
    ]
    sampling_weights = (
        [np.asarray(values, dtype=float) for values in stats.sampling_weights]
        if stats.sampling_weights
        else None
    )
    exposures = None if history.ordinal else _event_exposures(history, stats.event_indices)
    design_frame = pd.DataFrame(
        np.vstack(designs) if designs else np.empty((0, len(names))),
        columns=names,
    )
    unpenalized = _resolve_unpenalized(
        design_frame,
        names,
        unpenalized=penalty.get("unpenalized"),
        penalized=penalty.get("penalized"),
    )
    penalty_mask = np.asarray([name not in unpenalized for name in names], dtype=bool)
    flat_design, flat_response, flat_offset = _flatten_glmnet_inputs(
        designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=sampling_weights,
    )
    (
        coefficients,
        iterations,
        converged,
        lambda_value,
        lambda_min,
        lambda_1se,
        selected_label,
    ) = _glmnet_penalty_fit(
        flat_design,
        flat_response,
        flat_offset,
        names,
        penalty_mask=penalty_mask,
        ordinal=history.ordinal,
        alpha=alpha,
        lambda_value=requested_lambda,
        nfolds=nfolds,
        lambda_select=lambda_select,
        foldid=foldid,
        seed=seed,
    )
    loglik, gradient, hessian = _tie_loglik_grad_hessian(
        coefficients,
        designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=sampling_weights,
    )
    ridge_information = lambda_value * (1.0 - alpha) * np.diag(
        penalty_mask.astype(float)
    )
    covariance = (
        np.zeros((0, 0), dtype=float)
        if not names
        else np.linalg.pinv(-hessian + ridge_information, hermitian=True)
    )
    null_loglik = _tie_null_loglik(
        designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=sampling_weights,
    )
    event_probabilities, observed_indices = _tie_event_probabilities(
        coefficients,
        designs,
        observed_groups,
        sampling_weights=sampling_weights,
    )
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "engine": "elastic-net",
            "method": "glmnet",
            "alpha": alpha,
            "lambda": lambda_value,
            "lambda_min": lambda_min,
            "lambda_1se": lambda_1se,
            "lambda_select": selected_label,
            "unpenalized": list(unpenalized),
            "penalized": [
                name
                for name, selected in zip(names, penalty_mask, strict=True)
                if selected
            ],
            "message": "proximal-gradient elastic-net optimization",
        }
    )
    residual_deviance = -2.0 * loglik
    null_deviance = -2.0 * null_loglik
    return RemEstimateGlmnet(
        coef=coefficients,
        names=names,
        log_likelihood=float(loglik),
        converged=bool(converged and np.isfinite(coefficients).all()),
        covariance=covariance,
        metadata=metadata,
        event_probabilities=event_probabilities,
        observed_indices=observed_indices,
        gradient=gradient,
        hessian=hessian,
        residual_deviance=residual_deviance,
        null_deviance=null_deviance,
        model_deviance=null_deviance - residual_deviance,
        iterations=iterations,
        sampled=sampling_weights is not None,
        unpenalized=tuple(unpenalized),
        penalty={
            "alpha": alpha,
            "lambda": lambda_value,
            "unpenalized": list(unpenalized),
        },
        lambda_value=lambda_value,
        lambda_min=lambda_min,
        lambda_1se=lambda_1se,
        lambda_select=selected_label,
    )


def _elastic_net_optimize(
    initial: np.ndarray,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
    penalty_mask: np.ndarray,
    alpha: float,
    lambda_value: float,
    max_iterations: int = 2000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int, bool]:
    beta = np.array(initial, dtype=float, copy=True)
    if not len(beta) or not penalty_mask.any():
        return beta, 0, True

    def objective(value: np.ndarray) -> float:
        loglik, _, _ = _tie_loglik_grad_hessian(
            value,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        selected = value[penalty_mask]
        regularizer = lambda_value * (
            alpha * np.abs(selected).sum()
            + 0.5 * (1.0 - alpha) * float(selected @ selected)
        )
        return float(-loglik + regularizer)

    current_objective = objective(beta)
    for iteration in range(1, max_iterations + 1):
        _, score, hessian = _tie_loglik_grad_hessian(
            beta,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        smooth_gradient = -score
        smooth_gradient[penalty_mask] += (
            lambda_value * (1.0 - alpha) * beta[penalty_mask]
        )
        information = -0.5 * (hessian + hessian.T)
        lipschitz = max(
            1.0,
            float(np.linalg.eigvalsh(information).max(initial=0.0))
            + lambda_value * (1.0 - alpha),
        )
        step = 1.0 / lipschitz
        candidate = np.array(beta, copy=True)
        for _ in range(30):
            candidate = beta - step * smooth_gradient
            threshold = step * lambda_value * alpha
            selected = candidate[penalty_mask]
            candidate[penalty_mask] = np.sign(selected) * np.maximum(
                np.abs(selected) - threshold, 0.0
            )
            candidate_objective = objective(candidate)
            difference = candidate - beta
            if candidate_objective <= current_objective - 1e-8 * float(
                difference @ difference
            ):
                break
            step *= 0.5
        scale = 1.0 + float(np.max(np.abs(beta), initial=0.0))
        if float(np.max(np.abs(candidate - beta), initial=0.0)) <= tolerance * scale:
            return candidate, iteration, True
        beta = candidate
        current_objective = candidate_objective
    return beta, max_iterations, False


def _tie_trust_fit(
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
    max_iterations: int = 100,
) -> tuple[np.ndarray, bool, int]:
    """Fit the tie model with a deterministic trust-region loop."""

    from scipy.optimize import brentq

    dimension = designs[0].shape[1] if designs else 0
    theta = np.zeros(dimension, dtype=float)
    radius = 1.0
    radius_max = 100.0
    termination = float(np.sqrt(np.finfo(float).eps))
    accepted = True
    value = float("nan")
    gradient = np.zeros(dimension, dtype=float)
    hessian = np.zeros((dimension, dimension), dtype=float)
    terminated = False
    iterations = 0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        loglik, score, _ = _tie_loglik_grad_hessian(
            parameters,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        information = _tie_observed_information(
            parameters,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        return -loglik, -score, information

    current = objective(theta)
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        if accepted:
            value, gradient, hessian = current
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            rotated_gradient = eigenvectors.T @ gradient

        newton = False
        if np.all(eigenvalues > 0.0):
            trial_step = -eigenvectors @ (rotated_gradient / eigenvalues)
            if np.linalg.norm(trial_step) <= radius:
                newton = True

        if not newton:
            minimum = float(np.min(eigenvalues))
            shifted = eigenvalues - minimum
            at_minimum = shifted == 0.0
            c1 = float(np.sum(np.square(rotated_gradient[~at_minimum] / shifted[~at_minimum])))
            c2 = float(np.sum(np.square(rotated_gradient[at_minimum])))
            c3 = float(np.sum(np.square(rotated_gradient)))
            if c2 > 0.0 or c1 > radius**2:
                lower = np.sqrt(c2) / radius
                upper = np.sqrt(c3) / radius

                root_args = (radius, c1, c2, shifted, rotated_gradient)
                if _trust_root_equation(upper, *root_args) <= 0.0:
                    root = upper
                elif _trust_root_equation(lower, *root_args) >= 0.0:
                    root = lower
                else:
                    root = brentq(
                        _trust_root_equation,
                        lower,
                        upper,
                        args=root_args,
                        xtol=np.finfo(float).eps ** 0.25,
                        rtol=4.0 * np.finfo(float).eps,
                    )
                rotated_step = rotated_gradient / (shifted + root)
                trial_step = -eigenvectors @ rotated_step
            else:
                rotated_step = np.zeros_like(rotated_gradient)
                rotated_step[~at_minimum] = (
                    rotated_gradient[~at_minimum] / shifted[~at_minimum]
                )
                trial_step = -eigenvectors @ rotated_step
                remaining = radius**2 - float(trial_step @ trial_step)
                if remaining > 0.0:
                    trial_step = trial_step + np.sqrt(remaining) * eigenvectors[:, at_minimum][
                        :, 0
                    ]

        predicted_difference = float(
            trial_step @ (gradient + (hessian @ trial_step) / 2.0)
        )
        candidate = theta + trial_step
        candidate_output = objective(candidate)
        candidate_value = candidate_output[0]
        # R follows IEEE-754 semantics here: 0 / 0 becomes NaN and the
        # subsequent model-change termination check wins.  Python scalar
        # division raises instead, so use NumPy's equivalent semantics.
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = float(np.divide(candidate_value - value, predicted_difference))
        if candidate_value < float("inf"):
            terminated = bool(
                abs(candidate_value - value) < termination
                or abs(predicted_difference) < termination
            )
        else:
            terminated = False
            rho = -float("inf")
        if terminated:
            if candidate_value < value:
                accepted = True
                theta = candidate
        elif rho < 0.25:
            accepted = False
            radius /= 4.0
        else:
            accepted = True
            theta = candidate
            if rho > 0.75 and not newton:
                radius = min(2.0 * radius, radius_max)
        current = candidate_output
        if terminated:
            break

    return theta, terminated, iterations


def _trust_root_equation(
    value: float,
    radius: float,
    c1: float,
    c2: float,
    shifted: np.ndarray,
    rotated_gradient: np.ndarray,
) -> float:
    if value == 0.0:
        return -1.0 / radius if c2 > 0.0 else np.sqrt(1.0 / c1) - 1.0 / radius
    denominator = shifted + value
    return float(
        np.sqrt(1.0 / np.sum(np.square(rotated_gradient / denominator)))
        - 1.0 / radius
    )


def _tie_observed_information(
    beta: np.ndarray,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
) -> np.ndarray:
    """Return the observed information under the documented ordinal convention."""

    if exposures is not None or sampling_weights is not None:
        _, _, likelihood_hessian = _tie_loglik_grad_hessian(
            beta,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        return -likelihood_hessian

    information = np.zeros((len(beta), len(beta)), dtype=float)
    for design, observed in zip(designs, observed_groups, strict=True):
        linear = design @ beta
        intensities = np.exp(linear)
        denominator = float(intensities.sum())
        observed_sum = design[np.asarray(observed, dtype=int)].sum(axis=0)
        score = observed_sum - (intensities @ design) / denominator
        second_moment = design.T @ (intensities[:, None] * design)
        information += second_moment / denominator
        information -= np.outer(score, score) / denominator**2
    return np.asarray(0.5 * (information + information.T), dtype=float)


def _tie_loglik_grad_hessian(
    beta: np.ndarray,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    parameter_count = len(beta)
    loglik = 0.0
    gradient = np.zeros(parameter_count, dtype=float)
    hessian = np.zeros((parameter_count, parameter_count), dtype=float)
    for position, (design, observed) in enumerate(zip(designs, observed_groups, strict=True)):
        if any(index < 0 or index >= len(design) for index in observed):
            raise ValueError("observed tie index is outside its risk set")
        eta = design @ beta
        weights = (
            sampling_weights[position]
            if sampling_weights is not None
            else np.ones(len(design), dtype=float)
        )
        if len(weights) != len(design) or np.any(weights <= 0):
            raise ValueError("sampling weights must be positive and align with the risk set")
        observed_design = design[np.asarray(observed, dtype=int)]
        if exposures is None:
            adjusted_eta = eta + np.log(weights)
            denominator, mean, covariance = _exact_conditional_moments(
                adjusted_eta, design, len(observed)
            )
            loglik += float(eta[observed].sum() - denominator)
            gradient += observed_design.sum(axis=0) - mean
            hessian -= covariance
        else:
            intensities = weights * np.exp(np.clip(eta, -745.0, 700.0))
            exposure = float(exposures[position])
            loglik += float(eta[observed].sum() - exposure * intensities.sum())
            gradient += observed_design.sum(axis=0) - exposure * (intensities @ design)
            hessian -= exposure * (design.T @ (intensities[:, None] * design))
    return loglik, gradient, hessian


def _tie_numpy_objective(
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loglik, gradient, _ = _tie_loglik_grad_hessian(
            beta,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        return -loglik, -gradient

    return objective


def _tie_jax_objective(
    backend: JaxBackend,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
    riskset_chunk_size: int | None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    single_case = all(len(group) == 1 for group in observed_groups)
    same_shape = len({design.shape for design in designs}) <= 1
    if single_case and same_shape and riskset_chunk_size is None:
        stacked = backend.asarray(np.stack(designs))
        observed = backend.asarray(
            np.asarray([group[0] for group in observed_groups], dtype=np.int32),
            dtype=np.int32,
        )
        weight_array = (
            backend.asarray(np.stack(sampling_weights)) if sampling_weights is not None else None
        )
        exposure_array = backend.asarray(exposures) if exposures is not None else None

        def loglik(beta: Any) -> Any:
            eta = stacked @ beta
            observed_eta = _jax_take_observed(eta, observed)
            if exposure_array is None:
                adjusted = eta if weight_array is None else eta + _jax_log(weight_array)
                return (observed_eta - backend.logsumexp(adjusted, axis=1)).sum()
            intensity = _jax_safe_exp(eta)
            if weight_array is not None:
                intensity = weight_array * intensity
            return (observed_eta - exposure_array * intensity.sum(axis=1)).sum()

    else:
        arrays = [backend.asarray(design) for design in designs]
        weight_arrays = (
            [backend.asarray(weights) for weights in sampling_weights]
            if sampling_weights is not None
            else None
        )

        def loglik(beta: Any) -> Any:
            total = 0.0
            for position, (design, observed) in enumerate(
                zip(arrays, observed_groups, strict=True)
            ):
                observed_eta = (
                    design[np.asarray(observed, dtype=np.int32)] @ beta
                ).sum()
                weights = weight_arrays[position] if weight_arrays is not None else None
                if exposures is None:
                    denominator = _jax_chunked_exact_conditional_log_normalizer(
                        backend,
                        design,
                        beta,
                        weights,
                        len(observed),
                        riskset_chunk_size,
                    )
                    total = total + observed_eta - denominator
                else:
                    intensity_sum = _jax_chunked_intensity_sum(
                        design, beta, weights, riskset_chunk_size
                    )
                    total = total + observed_eta - exposures[position] * intensity_sum
            return total

    value_and_grad = backend.jit(backend.value_and_grad(loglik))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(backend.asarray(beta))
        return float(-value), -backend.to_numpy(gradient)

    return objective


def _tie_null_loglik(
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
) -> float:
    if exposures is not None:
        cases = sum(len(group) for group in observed_groups)
        total_exposure = sum(
            float(exposures[position])
            * (
                float(sampling_weights[position].sum())
                if sampling_weights is not None
                else len(design)
            )
            for position, design in enumerate(designs)
        )
        if cases == 0 or total_exposure <= 0.0:
            return -total_exposure
        rate = cases / total_exposure
        return float(cases * np.log(rate) - rate * total_exposure)

    loglik = 0.0
    for position, (design, observed) in enumerate(zip(designs, observed_groups, strict=True)):
        weights = (
            sampling_weights[position]
            if sampling_weights is not None
            else np.ones(len(design), dtype=float)
        )
        denominator, _, _ = _exact_conditional_moments(
            np.log(weights), np.empty((len(design), 0), dtype=float), len(observed)
        )
        loglik -= denominator
    return loglik


def _tie_event_probabilities(
    beta: np.ndarray,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    sampling_weights: list[np.ndarray] | None,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    from scipy.special import softmax

    probabilities: list[np.ndarray] = []
    observed_indices: list[int] = []
    for position, (design, observed) in enumerate(zip(designs, observed_groups, strict=True)):
        eta = design @ beta
        if sampling_weights is not None:
            eta = eta + np.log(sampling_weights[position])
        relative = np.asarray(softmax(eta), dtype=float)
        for index in observed:
            probabilities.append(np.array(relative, copy=True))
            observed_indices.append(index)
    return tuple(probabilities), tuple(observed_indices)


def _mvn_parameter_draws(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    nsim: int,
    seed: int | None,
) -> np.ndarray:
    if len(mean) == 0:
        return np.zeros((nsim, 0), dtype=float)
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(eigenvalues)))) * len(mean)
    if float(np.min(eigenvalues)) < -100.0 * tolerance:
        raise np.linalg.LinAlgError("coefficient covariance is not positive semidefinite")
    stable_covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    rng = np.random.default_rng(seed)
    return np.asarray(
        rng.multivariate_normal(mean, stable_covariance, size=nsim, check_valid="raise"),
        dtype=float,
    )


def _tie_event_loglik_draws(
    parameter_draws: np.ndarray,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
) -> np.ndarray:
    values = np.empty((len(parameter_draws), len(designs)), dtype=float)
    for draw_index, beta in enumerate(parameter_draws):
        for event_index, (design, observed) in enumerate(
            zip(designs, observed_groups, strict=True)
        ):
            eta = design @ beta
            weights = (
                sampling_weights[event_index]
                if sampling_weights is not None
                else np.ones(len(design), dtype=float)
            )
            if exposures is None:
                denominator, _, _ = _exact_conditional_moments(
                    eta + np.log(weights), design, len(observed)
                )
                values[draw_index, event_index] = float(eta[observed].sum() - denominator)
            else:
                intensity = weights * np.exp(np.clip(eta, -745.0, 700.0))
                values[draw_index, event_index] = float(
                    eta[observed].sum() - exposures[event_index] * intensity.sum()
                )
    return values


def _waic_from_event_loglik(event_loglik: np.ndarray) -> float:
    if event_loglik.ndim != 2 or event_loglik.shape[0] < 1:
        raise ValueError("event log-likelihood draws must have shape (draws, events)")
    maxima = np.max(event_loglik, axis=0)
    lppd = maxima + np.log(np.mean(np.exp(event_loglik - maxima), axis=0))
    if event_loglik.shape[0] == 1:
        return float("nan")
    penalty = np.var(event_loglik, axis=0, ddof=1)
    return float(-2.0 * np.sum(lppd - penalty))


def _hmc_controls(values: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "nsim",
        "nchains",
        "burnin",
        "thin",
        "init",
        "L",
        "epsilon",
        "prior",
        "nsimWAIC",
    }
    unknown = set(values).difference(allowed)
    if unknown:
        raise TypeError(f"unsupported bayes controls: {', '.join(sorted(unknown))}")

    def positive_integer(name: str, default: int, none_default: int) -> int:
        value = values.get(name, default)
        if value is None:
            return none_default
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"bayes.{name} must be a positive integer")
        if int(value) <= 0:
            raise ValueError(f"bayes.{name} must be a positive integer")
        return int(value)

    nsim = positive_integer("nsim", 2000, 1000)
    nchains = positive_integer("nchains", 2, 1)
    burnin_value = values.get("burnin", 1000)
    if burnin_value is None:
        burnin = 500
    elif isinstance(burnin_value, (bool, np.bool_)) or not isinstance(
        burnin_value, (int, np.integer)
    ):
        raise TypeError("bayes.burnin must be a non-negative integer")
    elif int(burnin_value) < 0:
        raise ValueError("bayes.burnin must be a non-negative integer")
    else:
        burnin = int(burnin_value)
    if "thin" in values and values["thin"] is None:
        thin = 10 if nsim >= 100 else 1
    else:
        thin = positive_integer("thin", 1, 1)

    leapfrog_value = values.get("L", 50)
    if (
        leapfrog_value is None
        or isinstance(leapfrog_value, (bool, np.bool_))
        or not isinstance(leapfrog_value, (int, float, np.integer, np.floating))
        or float(leapfrog_value) <= 1.0
    ):
        leapfrog_steps = 50
    else:
        leapfrog_steps = int(leapfrog_value)
    epsilon_value = values.get("epsilon", 0.002)
    if (
        epsilon_value is None
        or isinstance(epsilon_value, (bool, np.bool_))
        or not isinstance(epsilon_value, (int, float, np.integer, np.floating))
        or not np.isfinite(float(epsilon_value))
        or float(epsilon_value) <= 0.0
    ):
        epsilon = 0.1 / leapfrog_steps
    else:
        epsilon = float(epsilon_value)
    return {
        "nsim": nsim,
        "nchains": nchains,
        "burnin": burnin,
        "thin": thin,
        "init": values.get("init"),
        "L": leapfrog_steps,
        "epsilon": epsilon,
        "prior": values.get("prior"),
    }


def _remstimate_tie_hmc(
    history: EventHistory,
    stats: RemStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    controls: Mapping[str, Any],
    seed: int | None,
    compute_waic: bool,
    riskset_chunk_size: int | None,
) -> RemEstimate:
    mle = _remstimate_tie(
        history,
        stats,
        backend,
        optimizer,
        seed=seed,
        riskset_chunk_size=riskset_chunk_size,
        compute_waic=False,
        nsim_waic=100,
    )
    designs = [np.asarray(values, dtype=float) for values in stats.stats]
    designs = _select_design_columns(designs, stats.names, mle.names)
    observed_groups = [
        [int(index) for index in group]
        for group in (stats.observed_index_groups or [[index] for index in stats.observed_indices])
    ]
    sampling_weights = (
        [np.asarray(values, dtype=float) for values in stats.sampling_weights]
        if stats.sampling_weights
        else None
    )
    exposures = None if history.ordinal else _event_exposures(history, stats.event_indices)
    return _hmc_component(
        mle,
        designs,
        observed_groups,
        backend=backend,
        exposures=exposures,
        sampling_weights=sampling_weights,
        riskset_chunk_size=riskset_chunk_size,
        controls=controls,
        seed=seed,
        compute_waic=compute_waic,
        component="tie",
    )


def _remstimate_actor_hmc(
    stats: AomStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    controls: Mapping[str, Any],
    seed: int | None,
    compute_waic: bool,
    riskset_chunk_size: int | None,
) -> ActorRemEstimate:
    mle = _remstimate_actor(
        stats,
        backend,
        optimizer,
        seed=seed,
        compute_waic=False,
        nsim_waic=100,
        riskset_chunk_size=riskset_chunk_size,
    )
    sender_model: RemEstimate | None = None
    if mle.sender_model is not None:
        sender_designs = _select_design_columns(
            [np.asarray(values, dtype=float) for values in stats.sender_stats],
            stats.sender_names,
            mle.sender_model.names,
        )
        sender_model = _hmc_component(
            mle.sender_model,
            sender_designs,
            stats.observed_sender_groups
            or [[index] for index in stats.observed_sender_indices],
            backend=backend,
            exposures=(
                None
                if stats.history.ordinal
                else _event_exposures(stats.history, stats.event_indices)
            ),
            sampling_weights=None,
            riskset_chunk_size=riskset_chunk_size,
            controls=controls,
            seed=seed,
            compute_waic=compute_waic,
            component="sender",
        )

    receiver_model: RemEstimate | None = None
    if mle.receiver_model is not None:
        source = stats.receiver_choice_stats or stats.receiver_stats
        masks = stats.receiver_choice_masks or stats.receiver_masks
        observed_source = (
            stats.receiver_choice_observed_indices or stats.observed_receiver_indices
        )
        receiver_designs: list[np.ndarray] = []
        receiver_observed: list[list[int]] = []
        for matrix, mask, observed_actor in zip(
            source, masks, observed_source, strict=True
        ):
            allowed = np.flatnonzero(mask)
            matches = np.flatnonzero(allowed == observed_actor)
            if len(matches) != 1:
                raise ValueError("observed receiver must occur once in its choice set")
            receiver_designs.append(np.asarray(matrix[mask], dtype=float))
            receiver_observed.append([int(matches[0])])
        receiver_model = _hmc_component(
            mle.receiver_model,
            receiver_designs,
            receiver_observed,
            backend=backend,
            exposures=None,
            sampling_weights=None,
            riskset_chunk_size=riskset_chunk_size,
            controls=controls,
            seed=None if seed is None else seed + 1,
            compute_waic=compute_waic,
            component="receiver",
        )

    metadata = dict(mle.metadata)
    metadata.update(
        {
            "method": "HMC",
            "approach": "Bayesian",
            **{
                name: controls[name]
                for name in ("nsim", "nchains", "burnin", "thin", "L", "epsilon")
            },
        }
    )
    return ActorRemEstimate(sender_model, receiver_model, metadata)


def _select_design_columns(
    designs: list[np.ndarray], source_names: list[str], target_names: list[str]
) -> list[np.ndarray]:
    indexes = [source_names.index(name) for name in target_names]
    return [design[:, indexes] for design in designs]


def _hmc_component(
    mle: RemEstimate,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    backend: ArrayBackend,
    exposures: np.ndarray | None,
    sampling_weights: list[np.ndarray] | None,
    riskset_chunk_size: int | None,
    controls: Mapping[str, Any],
    seed: int | None,
    compute_waic: bool,
    component: str,
) -> RemEstimate:
    parameter_count = len(mle.coef)
    prior_mean, prior_covariance = _hmc_prior(controls.get("prior"), parameter_count)
    prior_precision = np.linalg.inv(prior_covariance)
    initial = _hmc_initial_values(
        controls.get("init"),
        mle.coef,
        parameter_count,
        int(controls["nchains"]),
        component,
    )
    jax_objective = (
        _tie_jax_objective(
            backend,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
            riskset_chunk_size=riskset_chunk_size,
        )
        if isinstance(backend, JaxBackend)
        else None
    )

    def potential_and_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        if jax_objective is None:
            loglik, score, _ = _tie_loglik_grad_hessian(
                beta,
                designs,
                observed_groups,
                exposures=exposures,
                sampling_weights=sampling_weights,
            )
            negative_loglik = -loglik
            negative_score = -score
        else:
            negative_loglik, negative_score = jax_objective(beta)
        centered = beta - prior_mean
        potential = negative_loglik + 0.5 * float(
            centered @ prior_precision @ centered
        )
        gradient = negative_score + prior_precision @ centered
        return float(potential), np.asarray(gradient, dtype=float)

    seed_sequence = np.random.SeedSequence(seed)
    chain_seeds = seed_sequence.spawn(int(controls["nchains"]))
    kept_draws: list[np.ndarray] = []
    kept_potentials: list[float] = []
    acceptance_rates: list[float] = []
    divergences: list[int] = []
    max_energy_errors: list[float] = []
    total_iterations = int(controls["nsim"]) + int(controls["burnin"])
    for chain_index, chain_seed in enumerate(chain_seeds):
        rng = np.random.default_rng(chain_seed)
        current = np.array(initial[chain_index], copy=True)
        current_potential, _ = potential_and_gradient(current)
        accepted = 0
        chain_divergences = 0
        maximum_energy_error = 0.0
        chain_draws: list[np.ndarray] = []
        chain_potentials: list[float] = []
        for iteration in range(total_iterations):
            momentum = rng.normal(size=parameter_count)
            proposed = np.array(current, copy=True)
            proposed_momentum = np.array(momentum, copy=True)
            _, gradient = potential_and_gradient(proposed)
            proposed_momentum -= 0.5 * float(controls["epsilon"]) * gradient
            for step in range(int(controls["L"])):
                proposed += float(controls["epsilon"]) * proposed_momentum
                if step + 1 < int(controls["L"]):
                    _, gradient = potential_and_gradient(proposed)
                    proposed_momentum -= float(controls["epsilon"]) * gradient
            proposed_potential, gradient = potential_and_gradient(proposed)
            proposed_momentum -= 0.5 * float(controls["epsilon"]) * gradient
            current_energy = current_potential + 0.5 * float(momentum @ momentum)
            proposed_energy = proposed_potential + 0.5 * float(
                proposed_momentum @ proposed_momentum
            )
            energy_error = proposed_energy - current_energy
            if np.isfinite(energy_error):
                maximum_energy_error = max(maximum_energy_error, abs(energy_error))
            if not np.isfinite(energy_error) or abs(energy_error) > 1000.0:
                chain_divergences += 1
            if np.isfinite(energy_error) and np.log(rng.uniform()) < -energy_error:
                current = proposed
                current_potential = proposed_potential
                accepted += 1
            if iteration >= int(controls["burnin"]) and (
                iteration - int(controls["burnin"])
            ) % int(controls["thin"]) == 0:
                chain_draws.append(np.array(current, copy=True))
                chain_potentials.append(float(current_potential))
        kept_draws.extend(chain_draws)
        kept_potentials.extend(chain_potentials)
        acceptance_rates.append(accepted / max(total_iterations, 1))
        divergences.append(chain_divergences)
        max_energy_errors.append(maximum_energy_error)

    draws = np.asarray(kept_draws, dtype=float).reshape(
        len(kept_draws), parameter_count
    )
    log_posterior = np.asarray(kept_potentials, dtype=float)
    if not len(draws):
        raise RuntimeError("HMC retained no draws; reduce thin or increase nsim")
    map_index = int(np.argmin(log_posterior))
    coefficients = np.asarray(draws[map_index], dtype=float)
    posterior_mean = np.asarray(np.mean(draws, axis=0), dtype=float)
    covariance = (
        np.zeros((parameter_count, parameter_count), dtype=float)
        if len(draws) <= 1 or parameter_count == 0
        else np.atleast_2d(np.cov(draws, rowvar=False, ddof=1)).astype(float)
    )
    posterior_sd = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    probabilities, observed_indices = _tie_event_probabilities(
        coefficients,
        designs,
        observed_groups,
        sampling_weights=sampling_weights,
    )
    metadata = dict(mle.metadata)
    metadata.update(
        {
            "method": "HMC",
            "approach": "Bayesian",
            "component": component,
            "nsim": int(controls["nsim"]),
            "nchains": int(controls["nchains"]),
            "burnin": int(controls["burnin"]),
            "thin": int(controls["thin"]),
            "L": int(controls["L"]),
            "epsilon": float(controls["epsilon"]),
            "acceptance_rate": float(np.mean(acceptance_rates)),
            "chain_acceptance_rate": acceptance_rates,
            "divergences": int(sum(divergences)),
            "chain_divergences": divergences,
            "max_energy_error": max_energy_errors,
            "rng": "numpy.random.Generator",
            "seed": seed,
            "hmc_gradient_backend": backend.name,
            "riskset_chunk_size": riskset_chunk_size,
        }
    )
    if compute_waic:
        event_loglik = _tie_event_loglik_draws(
            draws,
            designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=sampling_weights,
        )
        metadata["posterior_log_likelihood"] = event_loglik
        metadata["WAIC"] = _waic_from_event_loglik(event_loglik)
    return RemEstimate(
        coef=coefficients,
        names=list(mle.names),
        log_likelihood=float(-log_posterior[map_index]),
        converged=bool(np.isfinite(draws).all() and np.isfinite(log_posterior).all()),
        covariance=covariance,
        metadata=metadata,
        event_probabilities=probabilities,
        observed_indices=observed_indices,
        iterations=total_iterations,
        sampled=mle.sampled,
        draws=draws,
        log_posterior=log_posterior,
        posterior_mean=posterior_mean,
        posterior_sd=posterior_sd,
    )


def _hmc_prior(value: Any, parameter_count: int) -> tuple[np.ndarray, np.ndarray]:
    if value is None:
        return np.zeros(parameter_count, dtype=float), np.eye(parameter_count) * 100.0
    if not isinstance(value, Mapping):
        raise TypeError("bayes.prior must be a mapping with mean and vcov")
    mean = np.asarray(value.get("mean", np.zeros(parameter_count)), dtype=float)
    covariance = np.asarray(
        value.get("vcov", np.eye(parameter_count) * 100.0), dtype=float
    )
    if mean.shape != (parameter_count,):
        raise ValueError("bayes.prior mean does not match the parameter count")
    if covariance.shape != (parameter_count, parameter_count):
        raise ValueError("bayes.prior vcov does not match the parameter count")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("bayes.prior must contain finite values")
    np.linalg.cholesky(covariance)
    return mean, covariance


def _hmc_initial_values(
    value: Any,
    mle: np.ndarray,
    parameter_count: int,
    nchains: int,
    component: str,
) -> np.ndarray:
    if isinstance(value, Mapping):
        value = value.get(f"{component}_model", value.get(component))
    if value is None:
        return np.repeat(np.asarray(mle, dtype=float)[None, :], nchains, axis=0)
    initial = np.asarray(value, dtype=float)
    if initial.shape == (parameter_count,):
        return np.repeat(initial[None, :], nchains, axis=0)
    if initial.shape == (parameter_count, nchains):
        return initial.T.copy()
    if initial.shape == (nchains, parameter_count):
        return initial.copy()
    raise ValueError("bayes.init must be a parameter vector or parameter-by-chain matrix")


def _remstimate_duration(
    stats: RemStatsDuration,
    backend: ArrayBackend,
    optimizer: str,
    *,
    seed: int | None,
    riskset_chunk_size: int | None = None,
) -> RemEstimateDuration:
    frame = stats.stacked.remstats_stack
    names = list(stats.stacked.stat_names)
    if frame.empty:
        raise ValueError("duration statistics contain no risk-set rows to estimate")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"duration stack is missing statistic columns: {missing}")
    response = frame["obs"].to_numpy(dtype=float)
    if (
        np.any(response < 0)
        or not np.isfinite(response).all()
        or not np.equal(response, np.floor(response)).all()
    ):
        raise ValueError("duration observations must be finite non-negative integers")
    design = frame[names].to_numpy(dtype=float)
    if not np.isfinite(design).all():
        raise ValueError("duration statistic matrix contains non-finite values")
    ordinal = bool(stats.stacked.ordinal)
    offset = None
    if not ordinal:
        if "log_interevent" not in frame:
            raise ValueError("interval duration estimation requires log_interevent")
        offset = frame["log_interevent"].to_numpy(dtype=float)
        if not np.isfinite(offset).all():
            raise ValueError("log_interevent must contain only finite values")
    groups = _duration_groups(frame)
    if not any(float(response[index].sum()) > 0.0 for index in groups):
        raise ValueError("duration statistics contain no observed events")

    parameter_count = design.shape[1]
    if parameter_count == 0:
        coefficients = np.zeros(0, dtype=float)
        loglik, gradient, hessian = _duration_loglik_grad_hessian(
            coefficients, design, response, offset=offset, groups=groups
        )
        success = True
        message = "closed-form zero-parameter model"
        iterations = 0
        covariance = np.zeros((0, 0), dtype=float)
    elif ordinal:
        coefficients, iterations, covariance, message = _duration_cox_exact_fit(
            design, response, groups
        )
        success = True  # survival::clogit reports a fit whenever coefficients exist.
        if isinstance(backend, JaxBackend):
            loglik, gradient, hessian = _duration_jax_derivatives(
                backend,
                design,
                response,
                groups,
                riskset_chunk_size=riskset_chunk_size,
            )(coefficients)
        else:
            loglik, gradient, hessian = _duration_loglik_grad_hessian(
                coefficients, design, response, offset=None, groups=groups
            )
    elif not ordinal and np.linalg.matrix_rank(design) == parameter_count:
        if offset is None:  # pragma: no cover - guarded by the interval branch above
            raise RuntimeError("interval duration estimation requires an offset")
        coefficients, success, iterations, covariance = _poisson_glm_irls(
            design, response, offset
        )
        message = "IRLS converged" if success else "IRLS iteration limit reached"
        loglik, gradient, hessian = _duration_loglik_grad_hessian(
            coefficients, design, response, offset=offset, groups=groups
        )
    else:
        from scipy.optimize import minimize

        objective = (
            _duration_jax_objective(backend, design, response, offset=offset, groups=groups)
            if isinstance(backend, JaxBackend)
            else _duration_numpy_objective(design, response, offset=offset, groups=groups)
        )
        result = minimize(
            fun=lambda beta: objective(beta)[0],
            x0=np.zeros(parameter_count, dtype=float),
            jac=lambda beta: objective(beta)[1],
            method=optimizer,
        )
        coefficients = np.asarray(result.x, dtype=float)
        loglik, gradient, hessian = _duration_loglik_grad_hessian(
            coefficients, design, response, offset=offset, groups=groups
        )
        success = bool(result.success)
        message = str(result.message)
        iterations = int(getattr(result, "nit", 0))
        information = -hessian
        covariance = np.linalg.pinv(information, hermitian=True)
    fitted_values = _duration_fitted_values(coefficients, design, offset=offset, groups=groups)
    residual_deviance = -2.0 * loglik
    null_deviance = _duration_null_deviance(response, offset=offset, groups=groups)
    event_probabilities, observed_indices = _duration_event_probabilities(
        fitted_values, response, groups
    )
    metadata = _metadata(
        backend,
        optimizer,
        int(response.sum()),
        seed=seed,
        timing="ordinal" if ordinal else "exact",
    )
    metadata.update(
        {
            "model": "tie",
            "approach": "Frequentist",
            "method": "MLE",
            "engine": "clogit" if ordinal else "glm",
            "ordinal": ordinal,
            "statistics": names,
            "n_observations": stats.stacked.E,
            "bic_n_observations": int(response.sum()) if ordinal else stats.stacked.E,
            "n_rows": len(frame),
            "df.null": stats.stacked.E,
            "df.model": parameter_count,
            "df.residual": stats.stacked.E - parameter_count,
            "where_is_baseline": (
                names.index("baseline.start") + 1 if "baseline.start" in names else None
            ),
            "ncores": 1,
            "message": message,
            "optimizer_device": (
                "cpu-reference" if ordinal and isinstance(backend, JaxBackend) else backend.device
            ),
            "riskset_chunk_size": riskset_chunk_size,
        }
    )
    backend_fit = {
        "engine": metadata["engine"],
        "optimizer": optimizer,
        "success": success,
        "message": message,
        "iterations": iterations,
        "objective": -loglik,
    }
    converged = bool(
        success
        and np.isfinite(loglik)
        and np.isfinite(coefficients).all()
        and np.isfinite(gradient).all()
    )
    return RemEstimateDuration(
        coef=coefficients,
        names=names,
        log_likelihood=float(loglik),
        converged=converged,
        covariance=covariance,
        metadata=metadata,
        event_probabilities=event_probabilities,
        observed_indices=observed_indices,
        stacked_data=stats.stacked,
        backend_fit=backend_fit,
        # The public duration result leaves these two fields unset; analytic
        # score and Hessian remain available through the pure
        # internal objective used by numerical validation tests.
        gradient=None,
        hessian=None,
        fitted_values=fitted_values,
        residual_deviance=residual_deviance,
        null_deviance=null_deviance,
        model_deviance=null_deviance - residual_deviance,
        iterations=iterations,
    )


def _poisson_glm_irls(
    design: np.ndarray,
    response: np.ndarray,
    offset: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 25,
) -> tuple[np.ndarray, bool, int, np.ndarray]:
    """Fit the interval duration model using R ``stats::glm.fit`` semantics."""

    from scipy.linalg import lstsq

    eta = np.log(response + 0.1)
    means = np.exp(eta)
    deviance_old = _poisson_deviance(response, means)
    coefficients = np.zeros(design.shape[1], dtype=float)
    weighted_design = design.copy()
    converged = False
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        working_response = (eta - offset) + (response - means) / means
        working_weights = np.sqrt(means)
        weighted_design = design * working_weights[:, None]
        weighted_response = working_response * working_weights
        coefficients, _, rank, _ = lstsq(
            weighted_design,
            weighted_response,
            cond=1e-11,
            lapack_driver="gelsy",
        )
        if rank < design.shape[1] or not np.isfinite(coefficients).all():
            raise ValueError("duration statistic matrix is rank deficient")
        eta = design @ coefficients + offset
        means = np.exp(eta)
        deviance = _poisson_deviance(response, means)
        if abs(deviance - deviance_old) / (0.1 + abs(deviance)) < tolerance:
            converged = True
            break
        deviance_old = deviance

    information = weighted_design.T @ weighted_design
    covariance = np.linalg.inv(information)
    return np.asarray(coefficients, dtype=float), converged, iterations, covariance


def _duration_cox_exact_fit(
    design: np.ndarray,
    response: np.ndarray,
    groups: list[np.ndarray],
    *,
    tolerance: float = 1e-9,
    cholesky_tolerance: float = np.finfo(float).eps**0.75,
    max_iterations: int = 20,
) -> tuple[np.ndarray, int, np.ndarray, str]:
    """Fit ordinal duration REMs with ``survival::coxexact.fit`` semantics."""

    scaled_design, scales = _cox_exact_scaled_design(design)
    beta = np.zeros(scaled_design.shape[1], dtype=float)
    def evaluate(value: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return _duration_loglik_grad_hessian(
            value, scaled_design, response, offset=None, groups=groups
        )
    old_loglik, score, hessian = evaluate(beta)
    factor, _ = _survival_ldl(-hessian, cholesky_tolerance)
    old_beta = beta.copy()
    beta += _survival_ldl_solve(factor, score)
    halving = False
    iterations = 0
    converged = False
    final_hessian = hessian

    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        loglik, score, hessian = evaluate(beta)
        factor, _ = _survival_ldl(-hessian, cholesky_tolerance)
        finite = bool(
            np.isfinite(loglik)
            and np.isfinite(score).all()
            and np.isfinite(hessian).all()
        )
        relative_change = (
            abs(1.0 - old_loglik / loglik) if loglik != 0.0 else float("inf")
        )
        if finite and relative_change <= tolerance and not halving:
            converged = True
            final_hessian = hessian
            break
        if iteration == max_iterations:
            break
        if not finite or loglik < old_loglik:
            halving = True
            beta = (old_beta + beta) / 2.0
        else:
            halving = False
            old_loglik = loglik
            old_beta = beta.copy()
            beta += _survival_ldl_solve(factor, score)

    if not converged and max_iterations > 1:
        beta = old_beta
        _, _, final_hessian = evaluate(beta)
        warnings.warn(
            "Ran out of iterations and did not converge",
            RuntimeWarning,
            stacklevel=3,
        )

    if converged:
        final_factor, _ = _survival_ldl(-final_hessian, cholesky_tolerance)
        scaled_covariance = _survival_ldl_inverse(final_factor)
    else:
        # Pinned survival 3.8-3 calls chinv2 on the unfactorized information
        # matrix after exhausting iterations. Its observable result is a
        # diagonal inverse with all cross-covariances cleared.
        information_diagonal = np.diag(-final_hessian)
        scaled_covariance = np.zeros_like(final_hessian)
        positive = information_diagonal > 0.0
        scaled_covariance[positive, positive] = 1.0 / information_diagonal[positive]
    inverse_scale = np.diag(1.0 / scales)
    covariance = inverse_scale @ scaled_covariance @ inverse_scale
    coefficients = beta / scales
    message = "coxexact converged" if converged else "coxexact iteration limit reached"
    return coefficients, iterations, covariance, message


def _cox_exact_scaled_design(design: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the centering and ``nocenter`` rules used by ``coxexact.fit``."""

    scaled = np.empty_like(design, dtype=float)
    scales = np.empty(design.shape[1], dtype=float)
    for column in range(design.shape[1]):
        values = design[:, column]
        if np.isin(values, (-1.0, 0.0, 1.0)).all():
            scaled[:, column] = values
            scales[column] = 1.0
            continue
        center = float(np.mean(values))
        scale = float(
            np.sqrt(np.sum(np.square(values - center)) / max(len(values) - 1, 1))
        )
        scaled[:, column] = (values - center) / scale
        scales[column] = scale
    return scaled, scales


def _survival_ldl(matrix: np.ndarray, tolerance: float) -> tuple[np.ndarray, int]:
    """Port survival's ordered ``F D F'`` decomposition and rank detection."""

    factor = np.asarray(matrix, dtype=float).copy()
    size = len(factor)
    largest_diagonal = max(float(np.max(np.diag(factor), initial=0.0)), 0.0)
    epsilon = tolerance if largest_diagonal == 0.0 else largest_diagonal * tolerance
    rank = 0
    nonnegative = 1
    for row in range(size):
        for column in range(row + 1, size):
            factor[column, row] = factor[row, column]
    for pivot_index in range(size):
        pivot = float(factor[pivot_index, pivot_index])
        if not np.isfinite(pivot) or pivot < epsilon:
            factor[pivot_index, pivot_index] = 0.0
            if pivot < -8.0 * epsilon:
                nonnegative = -1
            continue
        rank += 1
        for row in range(pivot_index + 1, size):
            multiplier = factor[row, pivot_index] / pivot
            factor[row, pivot_index] = multiplier
            factor[row, row] -= multiplier * multiplier * pivot
            for column in range(row + 1, size):
                factor[column, row] -= multiplier * factor[column, pivot_index]
    return factor, rank * nonnegative


def _survival_ldl_solve(factor: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Solve a system from survival's ``F D F'`` representation."""

    result = np.asarray(value, dtype=float).copy()
    for row in range(len(factor)):
        result[row] -= float(result[:row] @ factor[row, :row])
    for row in range(len(factor) - 1, -1, -1):
        diagonal = float(factor[row, row])
        if diagonal == 0.0:
            result[row] = 0.0
        else:
            result[row] = result[row] / diagonal - float(
                result[row + 1 :] @ factor[row + 1 :, row]
            )
    return result


def _survival_ldl_inverse(factor: np.ndarray) -> np.ndarray:
    """Invert survival's decomposition, preserving singular-column zeros."""

    inverse = np.asarray(factor, dtype=float).copy()
    size = len(inverse)
    for pivot in range(size):
        if inverse[pivot, pivot] > 0.0:
            inverse[pivot, pivot] = 1.0 / inverse[pivot, pivot]
            for row in range(pivot + 1, size):
                inverse[row, pivot] = -inverse[row, pivot]
                for column in range(pivot):
                    inverse[row, column] += (
                        inverse[row, pivot] * inverse[pivot, column]
                    )
    for pivot in range(size):
        if inverse[pivot, pivot] == 0.0:
            inverse[:pivot, pivot] = 0.0
            inverse[pivot, pivot:] = 0.0
        else:
            for row in range(pivot + 1, size):
                value = inverse[row, pivot] * inverse[row, row]
                inverse[pivot, row] = value
                for column in range(pivot, row):
                    inverse[pivot, column] += value * inverse[row, column]
    for row in range(1, size):
        inverse[row, :row] = inverse[:row, row]
    return inverse


def _poisson_deviance(response: np.ndarray, means: np.ndarray) -> float:
    positive = response > 0.0
    contributions = np.asarray(means, dtype=float).copy()
    contributions[positive] = (
        response[positive] * np.log(response[positive] / means[positive])
        - (response[positive] - means[positive])
    )
    return float(2.0 * contributions.sum())


def _duration_random_contexts(
    stats: RemStatsDuration,
    groups: Sequence[np.ndarray],
) -> list[dict[str, np.ndarray]]:
    """Recover actor/dyad/type grouping values for each duration risk set."""

    from remflow.stats import _duration_end_riskset

    history = stats.history
    frame = stats.stacked.remstats_stack
    start = history.risksets[0].reset_index(drop=True)
    end = _duration_end_riskset(
        history,
        directed=bool(history.durem.get("dur_directed_end", False)),
    )
    lookups = {
        "start": {
            int(row.dyad_id): (int(row.sender_id), int(row.receiver_id))
            for row in start.itertuples()
        },
        "end": {
            int(row.dyad_id): (int(row.sender_id), int(row.receiver_id))
            for row in end.itertuples()
        },
    }
    contexts: list[dict[str, np.ndarray]] = []
    for index in groups:
        selected = frame.iloc[index]
        senders: list[int] = []
        receivers: list[int] = []
        for row in selected.itertuples():
            process = str(row.process)
            dyad = int(row.dyad)
            try:
                sender, receiver = lookups[process][dyad]
            except KeyError as error:
                raise ValueError(
                    "duration dyad identifiers do not match their process risk set"
                ) from error
            senders.append(sender)
            receivers.append(receiver)
        sender_array = np.asarray(senders, dtype=int)
        receiver_array = np.asarray(receivers, dtype=int)
        context: dict[str, np.ndarray] = {
            "actor1": sender_array,
            "actor2": receiver_array,
            "sender": sender_array,
            "receiver": receiver_array,
            "dyad": selected["dyad"].to_numpy(dtype=int),
            "process": selected["process"].to_numpy(copy=True),
        }
        if "type" in selected:
            context["type"] = selected["type"].to_numpy(copy=True)
        contexts.append(context)
    return contexts


def _remstimate_duration_glmm(
    stats: RemStatsDuration,
    *,
    random: Any,
    engine: str,
    seed: int | None,
    controls: Mapping[str, Any],
) -> RemEstimateDuration:
    """Fit a start/end duration REM with Gaussian random effects."""

    from scipy.special import softmax

    frame = stats.stacked.remstats_stack
    names = list(stats.stacked.stat_names)
    if frame.empty:
        raise ValueError("duration statistics contain no risk-set rows to estimate")
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"duration stack is missing statistic columns: {missing}")
    response = frame["obs"].to_numpy(dtype=float)
    design = frame[names].to_numpy(dtype=float)
    if not np.isfinite(design).all():
        raise ValueError("duration statistic matrix contains non-finite values")
    groups = _duration_groups(frame)
    if not any(float(response[index].sum()) > 0.0 for index in groups):
        raise ValueError("duration statistics contain no observed events")
    designs = [np.asarray(design[index], dtype=float) for index in groups]
    observed_groups = [
        np.flatnonzero(response[index] > 0.0).astype(int).tolist() for index in groups
    ]
    ordinal = bool(stats.stacked.ordinal)
    offset = None
    exposures = None
    if not ordinal:
        if "log_interevent" not in frame:
            raise ValueError("interval duration estimation requires log_interevent")
        offset = frame["log_interevent"].to_numpy(dtype=float)
        if not np.isfinite(offset).all():
            raise ValueError("log_interevent must contain only finite values")
        exposures = np.asarray(
            [float(np.exp(offset[index[0]])) for index in groups],
            dtype=float,
        )
        if any(not np.allclose(offset[index], offset[index[0]]) for index in groups):
            raise ValueError("duration offsets must be constant within a time point")
    fitted = _fit_random_component(
        designs,
        observed_groups,
        names,
        _duration_random_contexts(stats, groups),
        random,
        exposures=exposures,
        sampling_weights=None,
        engine=engine,
        model="tie",
        component="duration",
        parent_ordinal=ordinal,
        seed=seed,
        controls=controls,
    )
    predictors = fitted.backend_fit.get("linear_predictors")
    if not isinstance(predictors, tuple) or len(predictors) != len(groups):
        raise RuntimeError("duration GLMM did not retain aligned linear predictors")
    fitted_values = np.zeros(len(frame), dtype=float)
    for index, eta in zip(groups, predictors, strict=True):
        linear = np.asarray(eta, dtype=float)
        if ordinal:
            fitted_values[index] = softmax(linear)
        else:
            assert offset is not None
            fitted_values[index] = np.exp(
                np.clip(linear + offset[index], -745.0, 700.0)
            )
    null_deviance = _duration_null_deviance(
        response,
        offset=offset,
        groups=groups,
    )
    residual_deviance = -2.0 * fitted.log_likelihood
    metadata = dict(fitted.metadata)
    metadata.update(
        {
            "component": None,
            "engine": engine,
            "statistics": names,
            "n_events": int(response.sum()),
            "n_observations": stats.stacked.E,
            "n_rows": len(frame),
            "where_is_baseline": (
                names.index("baseline.start") + 1
                if "baseline.start" in names
                else None
            ),
            "message": fitted.backend_fit.get("message"),
        }
    )
    return RemEstimateDuration(
        coef=np.array(fitted.coef, copy=True),
        names=list(fitted.names),
        log_likelihood=fitted.log_likelihood,
        converged=fitted.converged,
        covariance=(
            None if fitted.covariance is None else np.array(fitted.covariance, copy=True)
        ),
        metadata=metadata,
        event_probabilities=tuple(
            np.array(values, copy=True) for values in fitted.event_probabilities
        ),
        observed_indices=tuple(fitted.observed_indices),
        gradient=None if fitted.gradient is None else np.array(fitted.gradient, copy=True),
        hessian=None if fitted.hessian is None else np.array(fitted.hessian, copy=True),
        residual_deviance=residual_deviance,
        null_deviance=null_deviance,
        model_deviance=null_deviance - residual_deviance,
        iterations=fitted.iterations,
        stacked_data=stats.stacked,
        backend_fit=dict(fitted.backend_fit),
        fitted_values=fitted_values,
        random_effects={
            name: values.copy() for name, values in fitted.random_effects.items()
        },
        variance_components=fitted.variance_components.copy(),
    )


def _remstimate_duration_penalized(
    stats: RemStatsDuration,
    backend: ArrayBackend,
    optimizer: str,
    *,
    penalty: dict[str, Any],
    seed: int | None,
) -> RemEstimateDurationGlmnet:
    if isinstance(backend, JaxBackend):
        raise NotImplementedError(
            "penalized duration estimation is currently available only on backend='numpy'"
        )
    alpha, requested_lambda, nfolds, lambda_select, foldid = (
        _validate_penalty_controls(penalty)
    )
    reference = _remstimate_duration(stats, backend, optimizer, seed=seed)
    frame = stats.stacked.remstats_stack
    names = list(stats.stacked.stat_names)
    design = frame[names].to_numpy(dtype=float)
    response = frame["obs"].to_numpy(dtype=float)
    offset = (
        None
        if stats.stacked.ordinal
        else frame["log_interevent"].to_numpy(dtype=float)
    )
    groups = _duration_groups(frame)
    unpenalized = _resolve_unpenalized(
        frame,
        names,
        unpenalized=penalty.get("unpenalized"),
        penalized=penalty.get("penalized"),
    )
    penalty_mask = np.asarray([name not in unpenalized for name in names], dtype=bool)
    if int(penalty_mask.sum()) < 2:
        raise ValueError("penalized duration estimation requires at least two penalized effects")
    (
        coefficients,
        iterations,
        converged,
        lambda_value,
        lambda_min,
        lambda_1se,
        selected_label,
    ) = _glmnet_penalty_fit(
        design,
        response,
        np.zeros(len(response), dtype=float) if offset is None else offset,
        names,
        penalty_mask=penalty_mask,
        ordinal=stats.stacked.ordinal,
        alpha=alpha,
        lambda_value=requested_lambda,
        nfolds=nfolds,
        lambda_select=lambda_select,
        foldid=foldid,
        seed=seed,
    )
    loglik, _, hessian = _duration_loglik_grad_hessian(
        coefficients,
        design,
        response,
        offset=offset,
        groups=groups,
    )
    ridge_information = lambda_value * (1.0 - alpha) * np.diag(
        penalty_mask.astype(float)
    )
    covariance = np.linalg.pinv(-hessian + ridge_information, hermitian=True)
    fitted_values = _duration_fitted_values(
        coefficients, design, offset=offset, groups=groups
    )
    event_probabilities, observed_indices = _duration_event_probabilities(
        fitted_values, response, groups
    )
    residual_deviance = -2.0 * loglik
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "method": "glmnet",
            "engine": "elastic-net",
            "alpha": alpha,
            "lambda": lambda_value,
            "lambda_min": lambda_min,
            "lambda_1se": lambda_1se,
            "lambda_select": selected_label,
            "unpenalized": list(unpenalized),
        }
    )
    backend_fit = dict(reference.backend_fit)
    backend_fit.update(
        {
            "engine": "elastic-net",
            "iterations": iterations,
            "success": converged,
            "objective": -loglik,
        }
    )
    return RemEstimateDurationGlmnet(
        coef=coefficients,
        names=names,
        log_likelihood=float(loglik),
        converged=converged,
        covariance=covariance,
        metadata=metadata,
        event_probabilities=event_probabilities,
        observed_indices=observed_indices,
        residual_deviance=residual_deviance,
        null_deviance=reference.null_deviance,
        model_deviance=reference.null_deviance - residual_deviance,
        iterations=iterations,
        stacked_data=stats.stacked,
        backend_fit=backend_fit,
        fitted_values=fitted_values,
        unpenalized=tuple(unpenalized),
        penalty={"alpha": alpha, "lambda": lambda_value},
        lambda_value=lambda_value,
        lambda_min=lambda_min,
        lambda_1se=lambda_1se,
        lambda_select=selected_label,
    )


def _elastic_net_optimize_duration(
    initial: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
    *,
    offset: np.ndarray | None,
    groups: list[np.ndarray],
    penalty_mask: np.ndarray,
    alpha: float,
    lambda_value: float,
    max_iterations: int = 2000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, int, bool]:
    beta = np.array(initial, dtype=float, copy=True)

    def objective(value: np.ndarray) -> float:
        loglik, _, _ = _duration_loglik_grad_hessian(
            value, design, response, offset=offset, groups=groups
        )
        selected = value[penalty_mask]
        regularizer = lambda_value * (
            alpha * np.abs(selected).sum()
            + 0.5 * (1.0 - alpha) * float(selected @ selected)
        )
        return float(-loglik + regularizer)

    current_objective = objective(beta)
    for iteration in range(1, max_iterations + 1):
        _, score, hessian = _duration_loglik_grad_hessian(
            beta, design, response, offset=offset, groups=groups
        )
        smooth_gradient = -score
        smooth_gradient[penalty_mask] += (
            lambda_value * (1.0 - alpha) * beta[penalty_mask]
        )
        information = -0.5 * (hessian + hessian.T)
        lipschitz = max(
            1.0,
            float(np.linalg.eigvalsh(information).max(initial=0.0))
            + lambda_value * (1.0 - alpha),
        )
        step = 1.0 / lipschitz
        candidate = np.array(beta, copy=True)
        for _ in range(30):
            candidate = beta - step * smooth_gradient
            selected = candidate[penalty_mask]
            candidate[penalty_mask] = np.sign(selected) * np.maximum(
                np.abs(selected) - step * lambda_value * alpha, 0.0
            )
            candidate_objective = objective(candidate)
            difference = candidate - beta
            if candidate_objective <= current_objective - 1e-8 * float(
                difference @ difference
            ):
                break
            step *= 0.5
        scale = 1.0 + float(np.max(np.abs(beta), initial=0.0))
        if float(np.max(np.abs(candidate - beta), initial=0.0)) <= tolerance * scale:
            return candidate, iteration, True
        beta = candidate
        current_objective = candidate_objective
    return beta, max_iterations, False


def _duration_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(index, dtype=int)
        for index in frame.groupby("time_index", sort=False).indices.values()
    ]


def _duration_loglik_grad_hessian(
    beta: np.ndarray,
    design: np.ndarray,
    response: np.ndarray,
    *,
    offset: np.ndarray | None,
    groups: list[np.ndarray],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return duration log-likelihood, score, and Hessian."""

    from scipy.special import gammaln

    parameter_count = design.shape[1]
    gradient = np.zeros(parameter_count, dtype=float)
    hessian = np.zeros((parameter_count, parameter_count), dtype=float)
    linear = design @ beta
    if offset is not None:
        eta = linear + offset
        means = np.exp(np.clip(eta, -745.0, 700.0))
        # Remove the observed-event offset constant from the reported Poisson
        # GLM log likelihood.
        loglik = float(np.sum(response * linear - means - gammaln(response + 1.0)))
        gradient = design.T @ (response - means)
        hessian = -(design.T @ (means[:, None] * design))
        return loglik, gradient, hessian

    loglik = 0.0
    for index in groups:
        cases = float(response[index].sum())
        if cases <= 0.0:
            continue
        group_design = design[index]
        eta = linear[index]
        case_count = int(cases)
        if case_count > len(index):
            raise ValueError("observed duration cases exceed their risk-set size")
        denominator, mean, covariance = _exact_conditional_moments(eta, group_design, case_count)
        loglik += float(response[index] @ eta - denominator)
        gradient += group_design.T @ response[index] - mean
        hessian -= covariance
    return loglik, gradient, hessian


def _exact_conditional_moments(
    eta: np.ndarray, design: np.ndarray, cases: int
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return exact tied-choice normalizer and its feature moments."""

    parameter_count = design.shape[1]
    if cases == 0:
        return (
            0.0,
            np.zeros(parameter_count, dtype=float),
            np.zeros((parameter_count, parameter_count), dtype=float),
        )
    shift = float(np.max(eta))
    weights = np.exp(np.clip(eta - shift, -745.0, 0.0))
    partition = np.zeros(cases + 1, dtype=float)
    first = np.zeros((cases + 1, parameter_count), dtype=float)
    second = np.zeros((cases + 1, parameter_count, parameter_count), dtype=float)
    partition[0] = 1.0
    for position, (weight, values) in enumerate(zip(weights, design, strict=True), start=1):
        for subset_size in range(min(cases, position), 0, -1):
            previous_partition = partition[subset_size - 1]
            previous_first = first[subset_size - 1]
            partition[subset_size] += weight * previous_partition
            first[subset_size] += weight * (previous_first + previous_partition * values)
            second[subset_size] += weight * (
                second[subset_size - 1]
                + np.outer(previous_first, values)
                + np.outer(values, previous_first)
                + previous_partition * np.outer(values, values)
            )
    normalizer = partition[cases]
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise FloatingPointError("exact conditional duration normalizer is non-finite")
    mean = first[cases] / normalizer
    covariance = second[cases] / normalizer - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return float(np.log(normalizer) + cases * shift), mean, covariance


def _duration_numpy_objective(
    design: np.ndarray,
    response: np.ndarray,
    *,
    offset: np.ndarray | None,
    groups: list[np.ndarray],
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loglik, gradient, _ = _duration_loglik_grad_hessian(
            beta, design, response, offset=offset, groups=groups
        )
        return -loglik, -gradient

    return objective


def _duration_jax_objective(
    backend: JaxBackend,
    design: np.ndarray,
    response: np.ndarray,
    *,
    offset: np.ndarray | None,
    groups: list[np.ndarray],
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    x = backend.asarray(design)
    y = backend.asarray(response)
    offset_array = backend.asarray(offset) if offset is not None else None
    group_indices = [backend.asarray(index, dtype=np.int32) for index in groups]

    def loglik(beta: Any) -> Any:
        eta = x @ beta
        if offset_array is not None:
            means = _jax_safe_exp(eta + offset_array)
            return (y * eta - means - _jax_gammaln(y + 1.0)).sum()
        total = 0.0
        for index, numpy_index in zip(group_indices, groups, strict=True):
            group_y = y[index]
            cases = int(response[numpy_index].sum())
            if cases <= 0:
                continue
            group_eta = eta[index]
            denominator = _jax_exact_conditional_log_normalizer(backend, group_eta, cases)
            total = total + group_y @ group_eta - denominator
        return total

    value_and_grad = backend.jit(backend.value_and_grad(loglik))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        value, gradient = value_and_grad(backend.asarray(beta))
        return float(-value), -backend.to_numpy(gradient)

    return objective


def _duration_jax_derivatives(
    backend: JaxBackend,
    design: np.ndarray,
    response: np.ndarray,
    groups: list[np.ndarray],
    *,
    riskset_chunk_size: int | None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]:
    """Evaluate the exact tied duration objective and derivatives with JAX."""

    x = backend.asarray(design)
    y = backend.asarray(response)
    group_indices = [backend.asarray(index, dtype=np.int32) for index in groups]

    def loglik(beta: Any) -> Any:
        total = 0.0
        for index, numpy_index in zip(group_indices, groups, strict=True):
            cases = int(response[numpy_index].sum())
            if cases <= 0:
                continue
            group_design = x[index]
            group_y = y[index]
            group_eta = group_design @ beta
            denominator = _jax_chunked_exact_conditional_log_normalizer(
                backend,
                group_design,
                beta,
                None,
                cases,
                riskset_chunk_size,
            )
            total = total + group_y @ group_eta - denominator
        return total

    value_and_grad = backend.jit(backend.value_and_grad(loglik))
    hessian_fn = backend.jit(backend.hessian(loglik))

    def evaluate(beta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        device_beta = backend.asarray(beta)
        value, gradient = value_and_grad(device_beta)
        hessian = hessian_fn(device_beta)
        return (
            float(value),
            backend.to_numpy(gradient),
            backend.to_numpy(hessian),
        )

    return evaluate


def _jax_exact_conditional_log_normalizer(backend: JaxBackend, eta: Any, cases: int) -> Any:
    import jax.numpy as jnp

    shift = jnp.max(eta)
    weights = jnp.exp(jnp.clip(eta - shift, -745.0, 0.0))
    initial = jnp.zeros(cases + 1, dtype=eta.dtype).at[0].set(1.0)

    def update(coefficients: Any, weight: Any) -> tuple[Any, Any]:
        added = jnp.concatenate([jnp.zeros(1, dtype=eta.dtype), weight * coefficients[:-1]])
        result = coefficients + added
        return result, result

    coefficients, _ = backend.scan(update, initial, weights)
    return jnp.log(coefficients[cases]) + cases * shift


def _jax_chunked_exact_conditional_log_normalizer(
    backend: JaxBackend,
    design: Any,
    beta: Any,
    weights: Any | None,
    cases: int,
    chunk_size: int | None,
) -> Any:
    """Compute the exact tied-case normalizer without a full risk-set ``eta``."""

    import jax.numpy as jnp

    size = int(design.shape[0])
    width = size if chunk_size is None else chunk_size
    shift = jnp.asarray(-jnp.inf, dtype=design.dtype)
    for start in range(0, size, width):
        stop = min(start + width, size)
        eta = design[start:stop] @ beta
        if weights is not None:
            eta = eta + jnp.log(weights[start:stop])
        shift = jnp.maximum(shift, jnp.max(eta))

    coefficients = jnp.zeros(cases + 1, dtype=design.dtype).at[0].set(1.0)

    def update(current: Any, weight: Any) -> tuple[Any, Any]:
        added = jnp.concatenate(
            [jnp.zeros(1, dtype=design.dtype), weight * current[:-1]]
        )
        result = current + added
        return result, result

    for start in range(0, size, width):
        stop = min(start + width, size)
        eta = design[start:stop] @ beta
        if weights is not None:
            eta = eta + jnp.log(weights[start:stop])
        chunk_weights = jnp.exp(jnp.clip(eta - shift, -745.0, 0.0))
        coefficients, _ = backend.scan(update, coefficients, chunk_weights)
    return jnp.log(coefficients[cases]) + cases * shift


def _duration_fitted_values(
    beta: np.ndarray,
    design: np.ndarray,
    *,
    offset: np.ndarray | None,
    groups: list[np.ndarray],
) -> np.ndarray:
    from scipy.special import softmax

    eta = design @ beta
    if offset is not None:
        return np.asarray(np.exp(np.clip(eta + offset, -745.0, 700.0)), dtype=float)
    fitted = np.zeros(len(design), dtype=float)
    for index in groups:
        fitted[index] = softmax(eta[index])
    return fitted


def _duration_event_probabilities(
    fitted: np.ndarray,
    response: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    probabilities: list[np.ndarray] = []
    observed: list[int] = []
    for index in groups:
        group_values = fitted[index]
        total = float(group_values.sum())
        relative = (
            group_values / total
            if total > 0.0
            else np.full(len(index), 1.0 / len(index), dtype=float)
        )
        for local_index in np.flatnonzero(response[index] > 0.0):
            probabilities.append(np.array(relative, copy=True))
            observed.append(int(local_index))
    return tuple(probabilities), tuple(observed)


def _duration_null_deviance(
    response: np.ndarray,
    *,
    offset: np.ndarray | None,
    groups: list[np.ndarray],
) -> float:
    if offset is None:
        from scipy.special import gammaln

        null_loglik = 0.0
        for index in groups:
            cases = int(response[index].sum())
            if cases > 0:
                null_loglik -= float(
                    gammaln(len(index) + 1) - gammaln(cases + 1) - gammaln(len(index) - cases + 1)
                )
        return -2.0 * null_loglik

    exposures = np.exp(offset)
    exposure_sum = float(exposures.sum())
    rate = float(response.sum()) / exposure_sum if exposure_sum > 0.0 else 0.0
    null_fitted = rate * exposures
    null_loglik = _duration_poisson_rem_loglik(response, null_fitted, offset)
    return -2.0 * null_loglik


def _duration_poisson_rem_loglik(
    response: np.ndarray, fitted: np.ndarray, offset: np.ndarray
) -> float:
    from scipy.special import gammaln

    linear = np.log(np.maximum(fitted, np.finfo(float).tiny)) - offset
    return float(np.sum(response * linear - fitted - gammaln(response + 1.0)))


def _remstimate_actor(
    stats: AomStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    seed: int | None,
    compute_waic: bool,
    nsim_waic: int,
    riskset_chunk_size: int | None = None,
) -> ActorRemEstimate:
    sender_designs = [np.asarray(values, dtype=float) for values in stats.sender_stats]
    receiver_designs: list[np.ndarray] = []
    receiver_observed: list[int] = []
    receiver_source = stats.receiver_choice_stats or stats.receiver_stats
    receiver_masks = stats.receiver_choice_masks or stats.receiver_masks
    receiver_observed_source = (
        stats.receiver_choice_observed_indices or stats.observed_receiver_indices
    )
    for matrix, mask, observed_actor in zip(
        receiver_source,
        receiver_masks,
        receiver_observed_source,
        strict=True,
    ):
        allowed_actor_ids = np.flatnonzero(mask)
        matches = np.flatnonzero(allowed_actor_ids == observed_actor)
        if len(matches) != 1:
            raise ValueError("observed receiver must occur exactly once in its active risk set")
        receiver_designs.append(np.asarray(matrix[mask], dtype=float))
        receiver_observed.append(int(matches[0]))
    sender_exposures = (
        None
        if stats.history.ordinal
        else _event_exposures(stats.history, stats.event_indices)
    )
    sender_model = (
        _fit_actor_component(
            sender_designs,
            (
                stats.observed_sender_groups
                or [[index] for index in stats.observed_sender_indices]
            ),
            stats.sender_names,
            backend,
            optimizer,
            seed=seed,
            component="sender",
            exposures=sender_exposures,
            parent_ordinal=stats.history.ordinal,
            compute_waic=compute_waic,
            nsim_waic=nsim_waic,
            riskset_chunk_size=riskset_chunk_size,
        )
        if stats.sender_names
        else None
    )
    receiver_model = (
        _fit_actor_component(
            receiver_designs,
            [[index] for index in receiver_observed],
            stats.receiver_names,
            backend,
            optimizer,
            seed=seed,
            component="receiver",
            exposures=None,
            parent_ordinal=stats.history.ordinal,
            compute_waic=compute_waic,
            nsim_waic=nsim_waic,
            riskset_chunk_size=riskset_chunk_size,
        )
        if stats.receiver_names
        else None
    )
    metadata = {
        **backend.runtime_metadata,
        "optimizer": optimizer,
        "n_events": len(stats.event_indices),
        "seed": seed,
        "timing": "ordinal" if stats.history.ordinal else "exact",
        "model": "actor",
        "ordinal": stats.history.ordinal,
        "method": "MLE",
        "approach": "Frequentist",
        "nsimWAIC": nsim_waic if compute_waic else None,
        "riskset_chunk_size": riskset_chunk_size,
    }
    return ActorRemEstimate(sender_model, receiver_model, metadata)


def _remstimate_actor_penalized(
    stats: AomStats,
    backend: ArrayBackend,
    optimizer: str,
    *,
    penalty: dict[str, Any],
    seed: int | None,
) -> ActorRemEstimate:
    if isinstance(backend, JaxBackend):
        raise NotImplementedError(
            "penalized estimation is currently available only on backend='numpy'"
        )
    alpha, requested_lambda, nfolds, lambda_select, foldid = (
        _validate_penalty_controls(penalty)
    )
    reference = _remstimate_actor(
        stats,
        backend,
        optimizer,
        seed=seed,
        compute_waic=False,
        nsim_waic=100,
    )
    valid_names = [
        *(reference.sender_model.names if reference.sender_model is not None else []),
        *(reference.receiver_model.names if reference.receiver_model is not None else []),
    ]
    _check_penalty_names(
        valid=valid_names,
        penalized=penalty.get("penalized"),
        unpenalized=penalty.get("unpenalized"),
    )
    requested_unpenalized = _normalize_penalty_names(penalty.get("unpenalized"))
    requested_penalized = _normalize_penalty_names(penalty.get("penalized"))

    sender_model: RemEstimateGlmnet | None = None
    if reference.sender_model is not None:
        sender_designs = _select_design_columns(
            [np.asarray(values, dtype=float) for values in stats.sender_stats],
            stats.sender_names,
            reference.sender_model.names,
        )
        sender_model = _penalize_reference_component(
            reference.sender_model,
            sender_designs,
            stats.observed_sender_groups
            or [[index] for index in stats.observed_sender_indices],
            exposures=(
                None
                if stats.history.ordinal
                else _event_exposures(stats.history, stats.event_indices)
            ),
            ordinal=stats.history.ordinal,
            alpha=alpha,
            lambda_value=requested_lambda,
            nfolds=nfolds,
            lambda_select=lambda_select,
            foldid=foldid,
            seed=seed,
            unpenalized=[
                name
                for name in requested_unpenalized
                if name in reference.sender_model.names
            ],
            penalized=[
                name
                for name in requested_penalized
                if name in reference.sender_model.names
            ],
        )

    receiver_model: RemEstimateGlmnet | None = None
    if reference.receiver_model is not None:
        receiver_designs, receiver_groups = _actor_receiver_diagnostic_designs(stats)
        receiver_designs = _select_design_columns(
            receiver_designs,
            stats.receiver_names,
            reference.receiver_model.names,
        )
        receiver_model = _penalize_reference_component(
            reference.receiver_model,
            receiver_designs,
            receiver_groups,
            exposures=None,
            ordinal=stats.history.ordinal,
            alpha=alpha,
            lambda_value=requested_lambda,
            nfolds=nfolds,
            lambda_select=lambda_select,
            foldid=foldid,
            seed=seed,
            unpenalized=[
                name
                for name in requested_unpenalized
                if name in reference.receiver_model.names
            ],
            penalized=[
                name
                for name in requested_penalized
                if name in reference.receiver_model.names
            ],
        )

    metadata = dict(reference.metadata)
    metadata.update(
        {
            "method": "glmnet",
            "engine": "elastic-net",
            "alpha": alpha,
            "lambda": requested_lambda,
            "lambda_select": "explicit" if requested_lambda is not None else lambda_select,
        }
    )
    return ActorRemEstimate(sender_model, receiver_model, metadata)


def _penalize_reference_component(
    reference: RemEstimate,
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    *,
    exposures: np.ndarray | None,
    ordinal: bool,
    alpha: float,
    lambda_value: float | None,
    nfolds: int,
    lambda_select: str,
    foldid: np.ndarray | None,
    seed: int | None,
    unpenalized: Sequence[str],
    penalized: Sequence[str],
) -> RemEstimateGlmnet:
    design_frame = pd.DataFrame(
        np.vstack(designs) if designs else np.empty((0, len(reference.names))),
        columns=reference.names,
    )
    exemptions = _resolve_unpenalized(
        design_frame,
        reference.names,
        unpenalized=unpenalized,
        penalized=penalized,
    )
    penalty_mask = np.asarray(
        [name not in exemptions for name in reference.names], dtype=bool
    )
    flat_design, flat_response, flat_offset = _flatten_glmnet_inputs(
        designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=None,
    )
    (
        coefficients,
        iterations,
        converged,
        selected_lambda,
        lambda_min,
        lambda_1se,
        selected_label,
    ) = _glmnet_penalty_fit(
        flat_design,
        flat_response,
        flat_offset,
        reference.names,
        penalty_mask=penalty_mask,
        ordinal=ordinal,
        alpha=alpha,
        lambda_value=lambda_value,
        nfolds=nfolds,
        lambda_select=lambda_select,
        foldid=foldid,
        seed=seed,
    )
    loglik, gradient, hessian = _tie_loglik_grad_hessian(
        coefficients,
        designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=None,
    )
    ridge_information = selected_lambda * (1.0 - alpha) * np.diag(
        penalty_mask.astype(float)
    )
    covariance = np.linalg.pinv(-hessian + ridge_information, hermitian=True)
    probabilities, observed_indices = _tie_event_probabilities(
        coefficients,
        designs,
        observed_groups,
        sampling_weights=None,
    )
    metadata = dict(reference.metadata)
    metadata.update(
        {
            "method": "glmnet",
            "engine": "elastic-net",
            "alpha": alpha,
            "lambda": selected_lambda,
            "lambda_min": lambda_min,
            "lambda_1se": lambda_1se,
            "lambda_select": selected_label,
            "unpenalized": list(exemptions),
        }
    )
    residual_deviance = -2.0 * loglik
    return RemEstimateGlmnet(
        coef=coefficients,
        names=list(reference.names),
        log_likelihood=float(loglik),
        converged=converged,
        covariance=covariance,
        metadata=metadata,
        event_probabilities=probabilities,
        observed_indices=observed_indices,
        gradient=gradient,
        hessian=hessian,
        residual_deviance=residual_deviance,
        null_deviance=reference.null_deviance,
        model_deviance=reference.null_deviance - residual_deviance,
        iterations=iterations,
        sampled=reference.sampled,
        unpenalized=tuple(exemptions),
        penalty={"alpha": alpha, "lambda": selected_lambda},
        lambda_value=selected_lambda,
        lambda_min=lambda_min,
        lambda_1se=lambda_1se,
        lambda_select=selected_label,
    )


def _fit_actor_component(
    designs: list[np.ndarray],
    observed_groups: list[list[int]],
    names: list[str],
    backend: ArrayBackend,
    optimizer: str,
    *,
    seed: int | None,
    component: str,
    exposures: np.ndarray | None,
    parent_ordinal: bool,
    compute_waic: bool,
    nsim_waic: int,
    riskset_chunk_size: int | None,
) -> RemEstimate:
    component_names = list(names)
    component_designs = designs
    if exposures is None and "baseline" in component_names:
        keep = [index for index, name in enumerate(component_names) if name != "baseline"]
        component_names = [component_names[index] for index in keep]
        component_designs = [design[:, keep] for design in component_designs]
    parameter_count = (
        component_designs[0].shape[1] if component_designs else len(component_names)
    )
    observed_groups = [
        [int(index) for index in group] for group in observed_groups
    ]
    metadata = _metadata(
        backend,
        optimizer,
        len(component_designs),
        seed=seed,
        timing="ordinal" if exposures is None else "exact",
    )
    metadata.update(
        {
            "component": component,
            "model": "actor",
            "ordinal": exposures is None,
            "parent_ordinal": parent_ordinal,
            "method": "MLE",
            "approach": "Frequentist",
            "statistics": component_names,
            "where_is_baseline": (
                component_names.index("baseline") + 1
                if "baseline" in component_names
                else None
            ),
            "ncores": 1,
            "sampled": False,
            "n_observations": len(component_designs),
            "df.null": len(component_designs),
            "df.model": parameter_count,
            "df.residual": len(component_designs) - parameter_count,
            "engine": "clogit" if exposures is None else "glm",
            "riskset_chunk_size": riskset_chunk_size,
        }
    )
    if parameter_count == 0:
        coefficients = np.zeros(0, dtype=float)
        loglik, gradient, hessian = _tie_loglik_grad_hessian(
            coefficients,
            component_designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=None,
        )
        success = True
        message = "closed-form zero-parameter model"
        iterations = 0
    else:
        from scipy.optimize import minimize

        objective = (
            _tie_jax_objective(
                backend,
                component_designs,
                observed_groups,
                exposures=exposures,
                sampling_weights=None,
                riskset_chunk_size=riskset_chunk_size,
            )
            if isinstance(backend, JaxBackend)
            else _tie_numpy_objective(
                component_designs,
                observed_groups,
                exposures=exposures,
                sampling_weights=None,
            )
        )
        result = minimize(
            fun=lambda beta: objective(beta)[0],
            x0=np.zeros(parameter_count, dtype=float),
            jac=lambda beta: objective(beta)[1],
            method=optimizer,
        )
        coefficients = np.asarray(result.x, dtype=float)
        loglik, gradient, hessian = _tie_loglik_grad_hessian(
            coefficients,
            component_designs,
            observed_groups,
            exposures=exposures,
            sampling_weights=None,
        )
        success = bool(result.success)
        message = str(result.message)
        iterations = int(getattr(result, "nit", 0))
    covariance = (
        np.zeros((0, 0), dtype=float)
        if parameter_count == 0
        else np.linalg.pinv(-hessian, hermitian=True)
    )
    null_loglik = _tie_null_loglik(
        component_designs,
        observed_groups,
        exposures=exposures,
        sampling_weights=None,
    )
    event_probabilities, observed_indices = _tie_event_probabilities(
        coefficients,
        component_designs,
        observed_groups,
        sampling_weights=None,
    )
    metadata["message"] = message
    if compute_waic:
        parameter_draws = _mvn_parameter_draws(
            coefficients,
            covariance,
            nsim=nsim_waic,
            seed=seed,
        )
        metadata["WAIC"] = _waic_from_event_loglik(
            _tie_event_loglik_draws(
                parameter_draws,
                component_designs,
                observed_groups,
                exposures=exposures,
                sampling_weights=None,
            )
        )
        metadata["nsimWAIC"] = nsim_waic
        metadata["waic_rng"] = "numpy.random.Generator"
    return RemEstimate(
        coef=coefficients,
        names=component_names,
        log_likelihood=float(loglik),
        converged=bool(
            success
            and np.isfinite(loglik)
            and np.isfinite(coefficients).all()
            and np.isfinite(gradient).all()
        ),
        covariance=covariance,
        metadata=metadata,
        event_probabilities=event_probabilities,
        observed_indices=observed_indices,
        gradient=gradient,
        hessian=hessian,
        residual_deviance=-2.0 * loglik,
        null_deviance=-2.0 * null_loglik,
        model_deviance=-2.0 * null_loglik + 2.0 * loglik,
        iterations=iterations,
        sampled=False,
    )


def fit_rem(
    edgelist: Any,
    effects: str = "~ inertia() + reciprocity()",
    *,
    backend: str | ArrayBackend = "numpy",
    **kwargs: Any,
) -> RemEstimate:
    """Convenience pipeline: normalize history, compute stats, estimate model."""

    from remflow.history import remify
    from remflow.stats import remstats

    history_keys = {
        "directed",
        "ordinal",
        "model",
        "actors",
        "riskset",
        "event_type",
        "event_weight",
    }
    history_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in history_keys}
    history_kwargs.setdefault("ordinal", True)
    history = remify(edgelist, **history_kwargs)
    stats = remstats(history, tie_effects=effects, **kwargs)
    if not isinstance(stats, RemStats):
        raise ValueError("fit_rem supports tie-oriented histories; use remstimate for actor models")
    fitted = remstimate(history, stats, backend=backend)
    if not isinstance(fitted, RemEstimate):
        raise RuntimeError("tie-oriented fit returned an unexpected actor result")
    return fitted


def diagnostics(
    fit: RemEstimate | ActorRemEstimate | RemEstimateWindow,
    history: EventHistory | None = None,
    stats: RemStats | AomStats | RemStatsDuration | None = None,
    *,
    top_pct: float = 0.05,
    surprise_threshold: float = 0.2,
    k: float = 10.0,
) -> Diagnostics | ActorDiagnostics | MixtureDiagnostics | WindowDiagnostics:
    if not isinstance(fit, (RemEstimate, ActorRemEstimate, RemEstimateWindow)):
        raise TypeError(
            "fit must be a RemEstimate, ActorRemEstimate, or RemEstimateWindow object"
        )
    if history is not None and not isinstance(history, EventHistory):
        raise TypeError("history must be an EventHistory returned by remify")
    if isinstance(fit, RemEstimateWindow):
        if history is None or stats is None:
            raise ValueError("window diagnostics require history and stats")
        if not isinstance(stats, (RemStats, AomStats, RemStatsDuration)):
            raise TypeError(
                "window diagnostics require RemStats, AomStats, or RemStatsDuration"
            )
        return _window_diagnostics(
            fit,
            history,
            stats,
            top_pct=top_pct,
            surprise_threshold=surprise_threshold,
            k=k,
        )
    if history is not None and fit.metadata.get("n_events") is None:
        raise ValueError("fit does not contain event metadata for the supplied history")
    if stats is not None and isinstance(fit, RemEstimateDuration):
        if not isinstance(stats, RemStatsDuration):
            raise TypeError("duration diagnostics require RemStatsDuration")
        if fit.stacked_data is not stats.stacked:
            raise ValueError("stats must be the object used to estimate fit")
    if isinstance(fit, ActorRemEstimate):
        if stats is not None and not isinstance(stats, AomStats):
            raise TypeError("actor diagnostics require AomStats")
        if (
            isinstance(stats, AomStats)
            and history is not None
            and not _estimation_histories_compatible(stats.history, history)
        ):
            raise ValueError("stats must have been computed from the supplied history")
        sender_processes = pd.DataFrame()
        receiver_processes = pd.DataFrame()
        if isinstance(stats, AomStats):
            if fit.sender_model is not None:
                sender_processes = _effect_process_frame(
                    fit.sender_model,
                    [np.asarray(values, dtype=float) for values in stats.sender_stats],
                    stats.sender_names,
                    stats.observed_sender_groups
                    or [[index] for index in stats.observed_sender_indices],
                )
            if fit.receiver_model is not None:
                receiver_designs, receiver_groups = _actor_receiver_diagnostic_designs(stats)
                receiver_processes = _effect_process_frame(
                    fit.receiver_model,
                    receiver_designs,
                    stats.receiver_names,
                    receiver_groups,
                )
        return ActorDiagnostics(
            fit=fit,
            sender_model=(
                None
                if fit.sender_model is None
                else _component_diagnostics(
                    fit.sender_model,
                    reh_processed=history,
                    effect_processes=sender_processes,
                )
            ),
            receiver_model=(
                None
                if fit.receiver_model is None
                else _component_diagnostics(
                    fit.receiver_model,
                    reh_processed=history,
                    effect_processes=receiver_processes,
                )
            ),
            reh_processed=history,
        )
    if (
        isinstance(fit, RemEstimateShrinkage)
        and fit.metadata.get("model") == "duration"
    ):
        return _duration_diagnostics(
            fit,
            history,
            top_pct=top_pct,
            surprise_threshold=surprise_threshold,
        )
    if isinstance(fit, RemEstimateMixture):
        return _mixture_diagnostics(fit, history)
    if isinstance(fit, RemEstimateDuration):
        return _duration_diagnostics(
            fit,
            history,
            top_pct=top_pct,
            surprise_threshold=surprise_threshold,
        )
    if (
        isinstance(stats, RemStats)
        and history is not None
        and not _estimation_histories_compatible(stats.history, history)
    ):
        raise ValueError("stats must have been computed from the supplied history")
    if not fit.event_probabilities:
        return Diagnostics(
            fit=fit,
            residuals=np.array([], dtype=float),
            observed_probabilities=np.array([], dtype=float),
            ranks=np.array([], dtype=int),
            predicted_indices=np.array([], dtype=int),
            reh_processed=history,
        )
    effect_processes = pd.DataFrame()
    if isinstance(stats, RemStats):
        effect_processes = _effect_process_frame(
            fit,
            [np.asarray(values, dtype=float) for values in stats.stats],
            stats.names,
            stats.observed_index_groups
            or [[index] for index in stats.observed_indices],
        )
    return _component_diagnostics(
        fit,
        reh_processed=history,
        effect_processes=effect_processes,
    )


def _component_diagnostics(
    fit: RemEstimate,
    *,
    reh_processed: EventHistory | None = None,
    effect_processes: pd.DataFrame | None = None,
) -> Diagnostics:
    observed_probabilities = np.asarray(
        [
            probabilities[index]
            for probabilities, index in zip(
                fit.event_probabilities, fit.observed_indices, strict=True
            )
        ],
        dtype=float,
    )
    recall_ranks = [
        _recall_ranks(probabilities, index + 1)
        for probabilities, index in zip(
            fit.event_probabilities, fit.observed_indices, strict=True
        )
    ]
    ranks = np.asarray([value["rank"] for value in recall_ranks], dtype=float)
    predicted = np.asarray(
        [int(np.argmax(probabilities)) for probabilities in fit.event_probabilities],
        dtype=int,
    )
    relative_ranks = np.asarray(
        [
            0.0 if len(probabilities) <= 1 else float((rank - 1) / (len(probabilities) - 1))
            for probabilities, rank in zip(fit.event_probabilities, ranks, strict=True)
        ],
        dtype=float,
    )
    cumulative_probabilities = np.asarray(
        [value["cum"] for value in recall_ranks], dtype=float
    )
    top_pct = 10.0
    recall = {
        "per_event": pd.DataFrame(
            {
                "event": np.arange(1, len(ranks) + 1, dtype=int),
                "rel_rank": relative_ranks,
                "cum_prob": cumulative_probabilities,
            }
        ),
        "summary": {
            "mean_rel_rank": float(np.mean(relative_ranks)),
            "median_rel_rank": float(np.median(relative_ranks)),
            "mean_cum_prob": float(np.mean(cumulative_probabilities)),
            "top_pct": top_pct,
            "top_pct_prop": float(np.mean(relative_ranks <= top_pct / 100.0)),
        },
    }
    random_effects = (
        {name: values.copy() for name, values in fit.random_effects.items()}
        if isinstance(fit, RemEstimateGLMM)
        else {}
    )
    return Diagnostics(
        fit=fit,
        residuals=1.0 - observed_probabilities,
        observed_probabilities=observed_probabilities,
        ranks=ranks,
        predicted_indices=predicted,
        rates=tuple(np.array(values, copy=True) for values in fit.event_probabilities),
        recall=recall,
        reh_processed=reh_processed,
        effect_processes=(
            pd.DataFrame() if effect_processes is None else effect_processes.copy()
        ),
        ranef=random_effects,
        use_ranef=bool(random_effects),
    )


def _recall_from_probabilities(
    probabilities: Sequence[np.ndarray],
    observed_indices: Sequence[int],
) -> dict[str, Any]:
    ranks = [
        _recall_ranks(values, index + 1)
        for values, index in zip(probabilities, observed_indices, strict=True)
    ]
    average_ranks = np.asarray([value["rank"] for value in ranks], dtype=float)
    relative = np.asarray(
        [
            0.0 if len(values) <= 1 else float((rank - 1.0) / (len(values) - 1.0))
            for values, rank in zip(probabilities, average_ranks, strict=True)
        ],
        dtype=float,
    )
    cumulative = np.asarray([value["cum"] for value in ranks], dtype=float)
    return {
        "per_event": pd.DataFrame(
            {
                "event": np.arange(1, len(relative) + 1, dtype=int),
                "rel_rank": relative,
                "cum_prob": cumulative,
            }
        ),
        "summary": {
            "mean_rel_rank": float(np.mean(relative)),
            "median_rel_rank": float(np.median(relative)),
            "mean_cum_prob": float(np.mean(cumulative)),
            "top_pct": 10.0,
            "top_pct_prop": float(np.mean(relative <= 0.1)),
        },
    }


def _mixture_diagnostics(
    fit: RemEstimateMixture,
    history: EventHistory | None,
) -> MixtureDiagnostics:
    base = _component_diagnostics(fit, reh_processed=history)
    per_component = {
        f"Component.{component + 1}": _recall_from_probabilities(
            probabilities,
            fit.observed_indices,
        )
        for component, probabilities in enumerate(
            fit.component_event_probabilities
        )
    }
    return MixtureDiagnostics(
        fit=fit,
        residuals=np.array(base.residuals, copy=True),
        observed_probabilities=np.array(base.observed_probabilities, copy=True),
        ranks=np.array(base.ranks, copy=True),
        predicted_indices=np.array(base.predicted_indices, copy=True),
        rates=tuple(np.array(values, copy=True) for values in base.rates),
        recall=base.recall,
        reh_processed=history,
        effect_processes=base.effect_processes.copy(),
        recall_by_component=per_component,
        prior_probs=np.array(fit.prior_probs, copy=True),
        k=fit.k,
    )


def _recall_ranks(probabilities: Any, position: int) -> dict[str, float]:
    """Return average rank and cumulative mass through an observed tie block.

    ``position`` follows the public one-based convention. Equal-probability
    alternatives receive their shared average rank rather than a stack-order
    dependent best rank.
    """

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("probabilities must be a non-empty one-dimensional sequence")
    if position < 1 or position > len(values):
        raise ValueError("position is outside the probability vector")
    observed = float(values[position - 1])
    greater = int(np.sum(values > observed))
    tied = int(np.sum(values == observed))
    average_rank = greater + (tied + 1.0) / 2.0
    cumulative = float(values[values > observed].sum() + observed)
    return {"rank": float(average_rank), "cum": cumulative}


def _normalize_penalty_names(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    names = [value] if isinstance(value, str) else list(value)
    if not all(isinstance(name, str) for name in names):
        raise TypeError("penalty statistic names must be strings")
    return names


def _intercept_like_stats(
    design: pd.DataFrame, statistic_names: Sequence[str]
) -> list[str]:
    """Identify present model columns whose finite values are structurally 0/1."""

    result: list[str] = []
    for name in statistic_names:
        if name not in design:
            continue
        values = pd.to_numeric(design[name], errors="coerce").to_numpy(dtype=float)
        if np.isnan(values).any():
            continue
        if set(np.unique(values)).issubset({0.0, 1.0}):
            result.append(name)
    return result


def _check_penalty_names(
    *,
    valid: Sequence[str],
    penalized: str | Sequence[str] | None = None,
    unpenalized: str | Sequence[str] | None = None,
) -> None:
    valid_names = set(valid)
    unknown = [
        name
        for name in [
            *_normalize_penalty_names(penalized),
            *_normalize_penalty_names(unpenalized),
        ]
        if name not in valid_names
    ]
    if unknown:
        warnings.warn(
            f"penalty statistic names not found among the model statistics: {unknown!r}",
            UserWarning,
            stacklevel=2,
        )


def _resolve_unpenalized(
    design: pd.DataFrame,
    statistic_names: Sequence[str],
    *,
    unpenalized: str | Sequence[str] | None = None,
    penalized: str | Sequence[str] | None = None,
) -> list[str]:
    _check_penalty_names(
        valid=statistic_names,
        penalized=penalized,
        unpenalized=unpenalized,
    )
    exemptions = set(_intercept_like_stats(design, statistic_names))
    exemptions.update(_normalize_penalty_names(unpenalized))
    exemptions.difference_update(_normalize_penalty_names(penalized))
    return [name for name in statistic_names if name in exemptions]


def _actor_receiver_diagnostic_designs(
    stats: AomStats,
) -> tuple[list[np.ndarray], list[list[int]]]:
    source = stats.receiver_choice_stats or stats.receiver_stats
    masks = stats.receiver_choice_masks or stats.receiver_masks
    observed = stats.receiver_choice_observed_indices or stats.observed_receiver_indices
    designs: list[np.ndarray] = []
    groups: list[list[int]] = []
    for matrix, mask, observed_actor in zip(source, masks, observed, strict=True):
        allowed = np.flatnonzero(mask)
        match = np.flatnonzero(allowed == observed_actor)
        if len(match) != 1:
            raise ValueError("observed receiver must occur once in its diagnostic choice set")
        designs.append(np.asarray(matrix[mask], dtype=float))
        groups.append([int(match[0])])
    return designs, groups


def _effect_process_frame(
    fit: RemEstimate,
    designs: list[np.ndarray],
    source_names: list[str],
    observed_groups: list[list[int]],
) -> pd.DataFrame:
    columns = ["event", "effect", "residual", "observed", "expected"]
    if not fit.names:
        return pd.DataFrame(columns=columns)
    try:
        indexes = [source_names.index(name) for name in fit.names]
    except ValueError as error:
        raise ValueError("diagnostic statistics do not match fitted effect names") from error
    rows: list[dict[str, Any]] = []
    probability_position = 0
    event = 1
    for design, observed in zip(designs, observed_groups, strict=True):
        selected = np.asarray(design[:, indexes], dtype=float)
        for observed_index in observed:
            if probability_position >= len(fit.event_probabilities):
                raise ValueError("diagnostic probabilities do not align with event statistics")
            probabilities = fit.event_probabilities[probability_position]
            if len(probabilities) != len(selected):
                raise ValueError("diagnostic probability and risk-set sizes do not align")
            expected = probabilities @ selected
            actual = selected[int(observed_index)]
            for name, observed_value, expected_value in zip(
                fit.names, actual, expected, strict=True
            ):
                rows.append(
                    {
                        "event": event,
                        "effect": name,
                        "residual": float(observed_value - expected_value),
                        "observed": float(observed_value),
                        "expected": float(expected_value),
                    }
                )
            probability_position += 1
            event += 1
    if probability_position != len(fit.event_probabilities):
        raise ValueError("diagnostic event statistics do not cover every fitted probability")
    return pd.DataFrame(rows, columns=columns)


def _normalize_diagnostic_panels(which: int | Sequence[int]) -> list[int]:
    raw = [which] if isinstance(which, int) and not isinstance(which, bool) else which
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError("which must be an integer or a sequence of integers")
    panels: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError("which must contain only integers")
        panel = int(value)
        if panel not in {1, 2, 3, 4, 5, 6, 8, 9, 10}:
            raise ValueError(
                "which panels must be selected from 1, 2, 3, 4, 5, 6, 8, 9, and 10"
            )
        if panel not in panels:
            panels.append(panel)
    if not panels:
        raise ValueError("which must select at least one diagnostic panel")
    return panels


def _diagnostic_effect_names(
    fit: RemEstimate, effects: str | Sequence[str] | None
) -> list[str]:
    if effects is None:
        selected = list(fit.names)
    elif isinstance(effects, str):
        selected = [effects]
    else:
        selected = list(effects)
    if not all(isinstance(name, str) for name in selected):
        raise TypeError("effects must contain effect names")
    unknown = [name for name in selected if name not in fit.names]
    if unknown:
        raise ValueError(f"effects not found inside diagnostics: {unknown!r}")
    return selected


def _posterior_plot_frame(fit: RemEstimate, effects: list[str]) -> pd.DataFrame:
    if fit.draws is None:
        return pd.DataFrame(columns=["chain", "draw", "effect", "value"])
    indexes = [fit.names.index(name) for name in effects]
    draws = np.asarray(fit.draws[:, indexes], dtype=float)
    chains = max(1, int(fit.metadata.get("nchains", 1)))
    per_chain = len(draws) // chains
    if per_chain * chains != len(draws):
        chains = 1
        per_chain = len(draws)
    chain_ids = np.repeat(np.arange(1, chains + 1, dtype=int), per_chain)
    draw_ids = np.tile(np.arange(1, per_chain + 1, dtype=int), chains)
    frames = [
        pd.DataFrame(
            {
                "chain": chain_ids,
                "draw": draw_ids,
                "effect": name,
                "value": draws[:, column],
            }
        )
        for column, name in enumerate(effects)
    ]
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["chain", "draw", "effect", "value"])
    )


def _random_effect_qq_frame(value: Diagnostics) -> pd.DataFrame:
    """Return deterministic normal Q-Q coordinates for fitted random effects."""

    from scipy.stats import norm

    frames: list[pd.DataFrame] = []
    for term, effects in value.ranef.items():
        ordered = effects.sort_values(kind="stable")
        count = len(ordered)
        if not count:
            continue
        probabilities = (np.arange(count, dtype=float) + 0.5) / count
        frames.append(
            pd.DataFrame(
                {
                    "term": term,
                    "level": ordered.index.to_list(),
                    "theoretical_quantile": norm.ppf(probabilities),
                    "random_effect": ordered.to_numpy(dtype=float),
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=["term", "level", "theoretical_quantile", "random_effect"]
        )
    )


def _diagnostic_plot_data(
    value: Diagnostics,
    *,
    which: int | Sequence[int],
    effects: str | Sequence[str] | None,
    fitted: RemEstimate | None,
    warn_unavailable: bool = True,
) -> dict[str, pd.DataFrame]:
    panels = _normalize_diagnostic_panels(which)
    selected = _diagnostic_effect_names(value.fit, effects)
    result: dict[str, pd.DataFrame] = {}
    if isinstance(value, DurationDiagnostics):
        def recall_frame(
            recall: dict[str, Any] | None, process: str
        ) -> pd.DataFrame:
            if recall is None:
                return pd.DataFrame()
            frame = recall.get("per_event", pd.DataFrame()).copy()
            if not frame.empty:
                frame.insert(0, "process", process)
            return frame

        joint = recall_frame(value.recall_joint, "joint")
        start = recall_frame(value.recall_start, "start")
        end = recall_frame(value.recall_end, "end")
        if 1 in panels:
            result["panel1"] = joint
        if 2 in panels:
            count = len(value.deviance_residuals)
            pearson = (
                value.pearson_residuals
                if len(value.pearson_residuals) == count
                else np.full(count, np.nan, dtype=float)
            )
            result["panel2"] = pd.DataFrame(
                {
                    "row": np.arange(1, count + 1, dtype=int),
                    "deviance_residual": value.deviance_residuals,
                    "pearson_residual": pearson,
                }
            )
        if 3 in panels:
            result["panel3"] = start
        if 4 in panels:
            result["panel4"] = end
        if 5 in panels:
            result["panel5"] = pd.concat([joint, start, end], ignore_index=True)
        if 6 in panels:
            typed: list[pd.DataFrame] = []
            for event_type, recall in value.recall_by_type.items():
                frame = recall_frame(recall, "joint")
                if not frame.empty:
                    frame.insert(0, "event_type", event_type)
                    typed.append(frame)
            result["panel6"] = (
                pd.concat(typed, ignore_index=True) if typed else pd.DataFrame()
            )
        if 9 in panels:
            result["panel9"] = joint
        if 10 in panels:
            result["panel10"] = pd.concat([start, end], ignore_index=True)
        return result
    if 1 in panels:
        result["panel1"] = pd.DataFrame(
            {
                "event": np.arange(1, len(value.residuals) + 1, dtype=int),
                "residual": value.residuals,
                "observed_probability": value.observed_probabilities,
                "rank": value.ranks,
                "predicted_index": value.predicted_indices,
            }
        )
    if 2 in panels:
        processes = value.effect_processes
        result["panel2"] = processes[
            processes.get("effect", pd.Series(dtype=object)).isin(selected)
        ].reset_index(drop=True)
    if isinstance(value, MixtureDiagnostics):
        joint = value.recall.get("per_event", pd.DataFrame()).copy()
        if 3 in panels:
            result["panel3"] = joint
        if 6 in panels:
            result["panel6"] = joint
        if 9 in panels:
            component_frames: list[pd.DataFrame] = []
            for component, recall in value.recall_by_component.items():
                frame = recall.get("per_event", pd.DataFrame()).copy()
                frame.insert(0, "component", component)
                component_frames.append(frame)
            result["panel9"] = (
                pd.concat(component_frames, ignore_index=True)
                if component_frames
                else pd.DataFrame()
            )
        if 8 in panels and warn_unavailable:
            warnings.warn(
                "per-type MIXREM diagnostics are unavailable without retained type rows",
                UserWarning,
                stacklevel=2,
            )
        return result
    if isinstance(value.fit, RemEstimateShrinkage):
        if 3 in panels:
            result["panel3"] = value.recall.get("per_event", pd.DataFrame()).copy()
        if 6 in panels:
            estimates = value.fit.estimates.copy().reset_index(names="effect")
            result["panel6"] = estimates
        return result
    posterior_requested = any(panel in {3, 4} for panel in panels)
    posterior_available = (
        fitted is not None
        and fitted.metadata.get("method") == "HMC"
        and fitted.draws is not None
    )
    if posterior_requested and not posterior_available:
        if warn_unavailable:
            warnings.warn(
                "posterior and trace panels require an HMC result; unavailable panels were skipped",
                UserWarning,
                stacklevel=2,
            )
    elif posterior_requested and fitted is not None:
        posterior = _posterior_plot_frame(fitted, selected)
        if 3 in panels:
            result["panel3"] = posterior.copy()
        if 4 in panels:
            result["panel4"] = posterior.copy()
    if 6 in panels:
        if value.use_ranef and value.ranef:
            result["panel6"] = _random_effect_qq_frame(value)
        elif warn_unavailable:
            warnings.warn(
                "random-effects Q-Q panel requires a GLMM result; panel 6 was skipped",
                UserWarning,
                stacklevel=2,
            )
    return result


def _plot_fitted_result(
    fit: RemEstimate | ActorRemEstimate,
    reh: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration | None,
    diagnostic_result: Diagnostics | ActorDiagnostics | None,
    **kwargs: Any,
) -> Diagnostics | ActorDiagnostics:
    if diagnostic_result is None:
        if stats is None:
            raise ValueError("'stats' must be provided if argument 'diagnostics' is None")
        computed_diagnostics = diagnostics(fit, reh, stats)
        if isinstance(computed_diagnostics, WindowDiagnostics):
            raise RuntimeError("a non-window estimate returned window diagnostics")
        diagnostic_result = computed_diagnostics
    if not isinstance(diagnostic_result, (Diagnostics, ActorDiagnostics)):
        raise TypeError("diagnostics must be a Diagnostics or ActorDiagnostics object")
    if isinstance(fit, ActorRemEstimate):
        if not isinstance(diagnostic_result, ActorDiagnostics):
            raise TypeError("actor estimates require ActorDiagnostics")
        return diagnostic_result.plot(object=fit, **kwargs)
    if isinstance(diagnostic_result, ActorDiagnostics):
        raise TypeError("tie estimates require Diagnostics")
    return diagnostic_result.plot(object=fit, **kwargs)


def _duration_diagnostics(
    fit: RemEstimateDuration | RemEstimateShrinkage,
    history: EventHistory | None = None,
    *,
    top_pct: float = 0.05,
    surprise_threshold: float = 0.2,
) -> DurationDiagnostics:
    if fit.stacked_data is None:
        raise ValueError("duration fit does not retain stacked_data")
    frame = fit.stacked_data.remstats_stack.reset_index(drop=True)
    if len(fit.fitted_values) != len(frame):
        raise ValueError("duration fitted values do not align with stacked_data")
    response = frame["obs"].to_numpy(dtype=float)
    fitted = np.maximum(fit.fitted_values, np.finfo(float).tiny)
    if fit.stacked_data.ordinal:
        pearson = np.empty(0, dtype=float)
        deviance = np.sign(response - fitted) * np.sqrt(
            -2.0
            * np.where(
                response > 0.0,
                np.log(np.minimum(fitted, 1.0)),
                np.log(np.maximum(1.0 - fitted, np.finfo(float).tiny)),
            )
        )
    else:
        pearson = (response - fitted) / np.sqrt(fitted)
        contributions = fitted - response
        positive = response > 0.0
        contributions[positive] += response[positive] * np.log(
            response[positive] / fitted[positive]
        )
        deviance = np.sign(response - fitted) * np.sqrt(np.maximum(2.0 * contributions, 0.0))
    recall_tables = _duration_recall_tables(
        frame,
        fitted,
        history=history,
        start_names=fit.stacked_data.stat_names_start,
        end_names=fit.stacked_data.stat_names_end,
        top_pct=top_pct,
        surprise_threshold=surprise_threshold,
    )
    recall_joint = recall_tables["recall_joint"]
    joint_frame = (
        pd.DataFrame()
        if recall_joint is None
        else recall_joint["per_event"]
    )
    observed_probabilities = (
        joint_frame["obs_prob"].to_numpy(dtype=float)
        if "obs_prob" in joint_frame
        else np.empty(0, dtype=float)
    )
    ranks = (
        1.0
        + (1.0 - joint_frame["rel_rank"].to_numpy(dtype=float))
        * (joint_frame["D_t"].to_numpy(dtype=float) - 1.0)
        if not joint_frame.empty
        else np.empty(0, dtype=float)
    )
    predicted = np.asarray(
        [
            int(index[int(np.argmax(fitted[index]))])
            for index in _duration_groups(frame)
            if np.any(response[index] == 1.0)
        ],
        dtype=int,
    )
    random_effects = (
        {name: values.copy() for name, values in fit.random_effects.items()}
        if isinstance(fit, RemEstimateDuration)
        else {}
    )
    residual_summary = (
        pd.DataFrame(
            [
                {
                    "min": float(np.min(deviance)),
                    "q1": float(np.quantile(deviance, 0.25)),
                    "median": float(np.median(deviance)),
                    "q3": float(np.quantile(deviance, 0.75)),
                    "max": float(np.max(deviance)),
                }
            ]
        )
        if len(deviance)
        else None
    )
    return DurationDiagnostics(
        fit=fit,
        residuals=deviance,
        observed_probabilities=observed_probabilities,
        ranks=ranks,
        predicted_indices=predicted,
        recall_joint=recall_joint,
        recall_start=recall_tables["recall_start"],
        recall_end=recall_tables["recall_end"],
        recall_by_type=recall_tables["recall_by_type"],
        recall_start_by_type=recall_tables["recall_start_by_type"],
        recall_end_by_type=recall_tables["recall_end_by_type"],
        surprises_joint=recall_tables["surprises_joint"],
        surprises_start=recall_tables["surprises_start"],
        surprises_end=recall_tables["surprises_end"],
        surprises_by_type=recall_tables["surprises_by_type"],
        surprises_start_by_type=recall_tables["surprises_start_by_type"],
        surprises_end_by_type=recall_tables["surprises_end_by_type"],
        surprise_offenders_joint=recall_tables["surprise_offenders_joint"],
        surprise_offenders_start=recall_tables["surprise_offenders_start"],
        surprise_offenders_end=recall_tables["surprise_offenders_end"],
        surprise_threshold=surprise_threshold,
        deviance_residuals=deviance,
        pearson_residuals=pearson,
        residual_summary=residual_summary,
        reh_processed=history,
        ranef=random_effects,
        use_ranef=bool(random_effects),
    )


def _duration_recall_tables(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    history: EventHistory | None,
    start_names: Sequence[str],
    end_names: Sequence[str],
    top_pct: float,
    surprise_threshold: float,
) -> dict[str, Any]:
    response = frame["obs"].to_numpy(dtype=float)
    observed = np.flatnonzero(response == 1.0)
    event_ids = frame["time_index"].to_numpy(dtype=int)
    eidx = _duration_event_source_ids(frame, history)

    present_start = [name for name in start_names if name in frame]
    present_end = [name for name in end_names if name in frame]
    is_start = np.zeros(len(frame), dtype=bool)
    is_end = np.zeros(len(frame), dtype=bool)
    if present_start:
        is_start = (
            np.abs(frame[present_start].to_numpy(dtype=float)).sum(axis=1) > 0.0
        )
    if present_end:
        is_end = np.abs(frame[present_end].to_numpy(dtype=float)).sum(axis=1) > 0.0

    recall_joint = _duration_recall_block(
        scores,
        observed,
        event_ids,
        eidx,
        top_pct=top_pct,
    )
    start_observed = np.intersect1d(observed, np.flatnonzero(is_start))
    recall_start = None
    if len(start_observed):
        start_events = np.unique(event_ids[start_observed])
        start_mask = np.flatnonzero(is_start & np.isin(event_ids, start_events))
        recall_start = _duration_recall_block(
            scores[start_mask],
            np.flatnonzero(np.isin(start_mask, start_observed)),
            event_ids[start_mask],
            eidx[start_mask],
            top_pct=top_pct,
        )
    end_observed = np.intersect1d(observed, np.flatnonzero(is_end))
    recall_end = None
    if len(end_observed):
        end_events = np.unique(event_ids[end_observed])
        end_mask = np.flatnonzero(is_end & np.isin(event_ids, end_events))
        recall_end = _duration_recall_block(
            scores[end_mask],
            np.flatnonzero(np.isin(end_mask, end_observed)),
            event_ids[end_mask],
            eidx[end_mask],
            top_pct=top_pct,
            min_riskset_size=2,
        )

    recall_by_type: dict[Any, dict[str, Any]] = {}
    recall_start_by_type: dict[Any, dict[str, Any]] = {}
    recall_end_by_type: dict[Any, dict[str, Any]] = {}
    if "type" in frame and not frame["type"].isna().all():
        type_values = frame["type"].dropna().unique().tolist()
        try:
            type_values = sorted(type_values)
        except TypeError:
            type_values = sorted(type_values, key=lambda value: (type(value).__name__, str(value)))
        if len(type_values) > 1:
            for event_type in type_values:
                type_mask = np.flatnonzero(frame["type"].to_numpy() == event_type)
                type_observed = np.intersect1d(observed, type_mask)
                if len(type_observed):
                    value = _duration_recall_block(
                        scores[type_mask],
                        np.flatnonzero(np.isin(type_mask, type_observed)),
                        event_ids[type_mask],
                        eidx[type_mask],
                        top_pct=top_pct,
                    )
                    if value is not None:
                        recall_by_type[event_type] = value

                type_start = np.flatnonzero(
                    is_start & (frame["type"].to_numpy() == event_type)
                )
                type_start_observed = np.intersect1d(observed, type_start)
                if len(type_start_observed):
                    selected_events = np.unique(event_ids[type_start_observed])
                    mask = np.flatnonzero(
                        is_start
                        & (frame["type"].to_numpy() == event_type)
                        & np.isin(event_ids, selected_events)
                    )
                    value = _duration_recall_block(
                        scores[mask],
                        np.flatnonzero(np.isin(mask, type_start_observed)),
                        event_ids[mask],
                        eidx[mask],
                        top_pct=top_pct,
                    )
                    if value is not None:
                        recall_start_by_type[event_type] = value

                type_end = np.flatnonzero(
                    is_end & (frame["type"].to_numpy() == event_type)
                )
                type_end_observed = np.intersect1d(observed, type_end)
                if len(type_end_observed):
                    selected_events = np.unique(event_ids[type_end_observed])
                    mask = np.flatnonzero(
                        is_end
                        & (frame["type"].to_numpy() == event_type)
                        & np.isin(event_ids, selected_events)
                    )
                    value = _duration_recall_block(
                        scores[mask],
                        np.flatnonzero(np.isin(mask, type_end_observed)),
                        event_ids[mask],
                        eidx[mask],
                        top_pct=top_pct,
                        min_riskset_size=2,
                    )
                    if value is not None:
                        recall_end_by_type[event_type] = value

    surprises_joint = _surprises_from_recall(recall_joint, surprise_threshold)
    surprises_start = _surprises_from_recall(recall_start, surprise_threshold)
    surprises_end = _surprises_from_recall(recall_end, surprise_threshold)
    surprises_by_type = {
        key: value
        for key, recall in recall_by_type.items()
        if (value := _surprises_from_recall(recall, surprise_threshold)) is not None
    }
    surprises_start_by_type = {
        key: value
        for key, recall in recall_start_by_type.items()
        if (value := _surprises_from_recall(recall, surprise_threshold)) is not None
    }
    surprises_end_by_type = {
        key: value
        for key, recall in recall_end_by_type.items()
        if (value := _surprises_from_recall(recall, surprise_threshold)) is not None
    }

    def labels(value: pd.DataFrame | None) -> list[str]:
        if value is None or "eidx" not in value:
            return []
        return _duration_dyad_labels(history, value["eidx"].to_numpy())

    return {
        "recall_joint": recall_joint,
        "recall_start": recall_start,
        "recall_end": recall_end,
        "recall_by_type": recall_by_type,
        "recall_start_by_type": recall_start_by_type,
        "recall_end_by_type": recall_end_by_type,
        "surprises_joint": surprises_joint,
        "surprises_start": surprises_start,
        "surprises_end": surprises_end,
        "surprises_by_type": surprises_by_type,
        "surprises_start_by_type": surprises_start_by_type,
        "surprises_end_by_type": surprises_end_by_type,
        "surprise_offenders_joint": _offender_table(
            labels(surprises_joint),
            labels(None if recall_joint is None else recall_joint["per_event"]),
        ),
        "surprise_offenders_start": _offender_table(
            labels(surprises_start),
            labels(None if recall_start is None else recall_start["per_event"]),
        ),
        "surprise_offenders_end": _offender_table(
            labels(surprises_end), labels(None if recall_end is None else recall_end["per_event"])
        ),
    }


def _duration_recall_block(
    scores: np.ndarray,
    observed_indices: np.ndarray,
    event_ids: np.ndarray,
    eidx: np.ndarray,
    *,
    top_pct: float,
    min_riskset_size: int = 1,
) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for event_id in pd.unique(event_ids):
        mask = np.flatnonzero(event_ids == event_id)
        observed = np.intersect1d(observed_indices, mask)
        if not len(observed) or len(mask) < min_riskset_size:
            continue
        values = np.asarray(scores[mask], dtype=float)
        total = float(values.sum())
        probabilities = (
            values / total
            if np.isfinite(total) and total > 0.0
            else np.full(len(mask), 1.0 / len(mask), dtype=float)
        )
        for observed_index in observed:
            position = int(np.flatnonzero(mask == observed_index)[0])
            probability = float(probabilities[position])
            greater = int(np.sum(probabilities > probability))
            tied = int(np.sum(probabilities == probability))
            rank = greater + (tied + 1.0) / 2.0
            relative_rank = (
                1.0 - (rank - 1.0) / (len(mask) - 1.0)
                if len(mask) > 1
                else 0.5
            )
            rows.append(
                {
                    "time": int(event_id),
                    "rel_rank": float(relative_rank),
                    "cum_prob": float(
                        probabilities[probabilities > probability].sum()
                        + probability
                    ),
                    "obs_prob": probability,
                    "prob_ratio": probability * len(mask),
                    "log_loss": -float(np.log(probability)),
                    "D_t": int(len(mask)),
                    "eidx": eidx[observed_index],
                }
            )
    if not rows:
        return None
    per_event = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "mean_rel_rank": float(per_event["rel_rank"].mean()),
                "median_rel_rank": float(per_event["rel_rank"].median()),
                "mean_cum_prob": float(per_event["cum_prob"].mean()),
                "mean_prob_ratio": float(per_event["prob_ratio"].mean()),
                "mean_log_loss": float(per_event["log_loss"].mean()),
                "top_pct": float(top_pct),
                "top_pct_prop": float(
                    (per_event["rel_rank"] >= 1.0 - top_pct).mean()
                ),
            }
        ]
    )
    return {"per_event": per_event, "summary": summary}


def _duration_event_source_ids(
    frame: pd.DataFrame,
    history: EventHistory | None,
) -> np.ndarray:
    positions = frame["time_index"].to_numpy(dtype=int) - 1
    if history is None:
        return np.asarray(positions, dtype=int) + 1
    starts = [
        (float(row.time), int(row.event_id))
        for row in history.events.itertuples()
    ]
    ends = [
        (float(row.end), int(row.event_id))
        for row in history.events.itertuples()
        if not pd.isna(row.end)
    ]
    dual = sorted([*starts, *ends], key=lambda item: item[0])
    result = np.full(len(frame), np.nan)
    valid = (positions >= 0) & (positions < len(dual))
    result[valid] = np.asarray([dual[index][1] for index in positions[valid]], dtype=float)
    return result


def _surprises_from_recall(
    recall: dict[str, Any] | None,
    threshold: float,
) -> pd.DataFrame | None:
    if recall is None:
        return None
    per_event = recall["per_event"]
    return (
        per_event.loc[per_event["rel_rank"] <= threshold]
        .sort_values("rel_rank", kind="stable")
        .reset_index(drop=True)
    )


def _duration_dyad_labels(
    history: EventHistory | None,
    source_ids: np.ndarray,
) -> list[str]:
    if history is None:
        return [str(int(value)) for value in source_ids if np.isfinite(value)]
    labels: list[str] = []
    for value in source_ids:
        if not np.isfinite(value):
            continue
        position = int(value) - 1
        if 0 <= position < len(history.events):
            row = history.events.iloc[position]
            labels.append(f"{row['sender']} -> {row['receiver']}")
    return labels


def _offender_table(
    surprise_ids: Sequence[str],
    all_ids: Sequence[str],
) -> pd.DataFrame | None:
    if not all_ids:
        return None
    ids = sorted(set(all_ids))
    all_counts = {
        value: sum(item == value for item in all_ids) for value in ids
    }
    surprise_counts = {
        value: sum(item == value for item in surprise_ids) for value in ids
    }
    result = pd.DataFrame(
        {
            "id": ids,
            "n_surprises": [surprise_counts[value] for value in ids],
            "n_total": [all_counts[value] for value in ids],
        }
    )
    result["prop"] = result["n_surprises"] / result["n_total"]
    return result.sort_values(
        ["n_surprises", "prop"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)


def _recall_mean(recall: dict[str, Any] | None) -> float:
    if recall is None:
        return float("nan")
    summary = recall.get("summary")
    if isinstance(summary, pd.DataFrame):
        return (
            float(summary.iloc[0]["mean_rel_rank"])
            if not summary.empty
            else float("nan")
        )
    if isinstance(summary, Mapping):
        return float(summary.get("mean_rel_rank", float("nan")))
    return float("nan")


def AIC(fit: RemEstimate | ActorRemEstimate) -> float:
    if str(fit.metadata.get("approach", "Frequentist")) != "Frequentist":
        raise ValueError("'approach' must be 'Frequentist'")
    if isinstance(fit, RemEstimateDuration):
        return fit.AIC
    if isinstance(fit, RemEstimateMixture):
        return fit.AIC
    return float(-2 * fit.log_likelihood + 2 * len(fit.coef))


def BIC(fit: RemEstimate | ActorRemEstimate, n: int | None = None) -> float:
    if str(fit.metadata.get("approach", "Frequentist")) != "Frequentist":
        raise ValueError("'approach' must be 'Frequentist'")
    if isinstance(fit, RemEstimateDuration) and n is None:
        return fit.BIC
    if isinstance(fit, RemEstimateMixture) and n is None:
        return fit.BIC
    nobs = n or int(
        fit.metadata.get("n_observations", fit.metadata.get("n_events", max(1, len(fit.coef))))
    )
    return float(-2 * fit.log_likelihood + len(fit.coef) * np.log(max(1, nobs)))


def AICC(fit: RemEstimate | ActorRemEstimate, n: int | None = None) -> float:
    if str(fit.metadata.get("approach", "Frequentist")) != "Frequentist":
        raise ValueError("'approach' must be 'Frequentist'")
    if isinstance(fit, RemEstimateDuration) and n is None:
        return fit.AICC
    k = len(fit.coef)
    nobs = n or int(fit.metadata.get("n_observations", fit.metadata.get("n_events", 0)))
    aic = AIC(fit)
    if nobs <= k + 1:
        return float("inf")
    return float(aic + (2 * k * (k + 1)) / (nobs - k - 1))


def bic_table(*fits: Any) -> list[dict[str, Any]]:
    """Return a BIC comparison, including multi-``k`` MIXREM collections."""

    values: list[RemEstimate | ActorRemEstimate]
    if len(fits) == 1 and isinstance(fits[0], Mapping):
        values = list(fits[0].values())
    elif len(fits) == 1 and isinstance(fits[0], Sequence) and not isinstance(
        fits[0], (str, bytes)
    ):
        values = list(fits[0])
    else:
        values = list(fits)
    rows: list[dict[str, Any]] = []
    for index, fit in enumerate(values):
        if isinstance(fit, RemEstimateMixture):
            rows.append({"k": fit.k, "BIC": fit.BIC})
        else:
            rows.append({"model": index + 1, "BIC": BIC(fit)})
    if rows and all("k" in row for row in rows):
        rows.sort(key=lambda row: float(row["BIC"]))
        minimum = min(float(row["BIC"]) for row in rows)
        for row in rows:
            row["delta_BIC"] = float(row["BIC"]) - minimum
    return rows


def WAIC(fit: RemEstimate | ActorRemEstimate) -> float:
    """Compute WAIC from posterior event log-likelihood draws.

    Bayesian estimators must store a two-dimensional ``draws x events`` array
    under ``metadata['posterior_log_likelihood']``. The current frequentist
    estimator does not manufacture pseudo-draws and therefore fails clearly.
    """

    if isinstance(fit, ActorRemEstimate):
        components = fit._components
        if not components or any("WAIC" not in component.metadata for component in components):
            raise ValueError("WAIC was not computed for every actor-model component")
        return float(sum(float(component.metadata["WAIC"]) for component in components))
    if "WAIC" in fit.metadata:
        return float(fit.metadata["WAIC"])
    values = fit.metadata.get("posterior_log_likelihood")
    if values is None:
        raise ValueError("WAIC requires posterior_log_likelihood draws from a Bayesian fit")
    log_likelihood = np.asarray(values, dtype=float)
    if log_likelihood.ndim != 2 or log_likelihood.shape[0] < 2:
        raise ValueError("posterior_log_likelihood must have shape (draws, events) with >=2 draws")
    maxima = np.max(log_likelihood, axis=0)
    lppd = np.sum(maxima + np.log(np.mean(np.exp(log_likelihood - maxima), axis=0)))
    effective_parameters = np.sum(np.var(log_likelihood, axis=0, ddof=1))
    return float(-2.0 * (lppd - effective_parameters))


def frailty_rem(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    **kwargs: Any,
) -> RemEstimateGLMM | RemEstimateDuration | ActorRemEstimate:
    """Deprecated alias of :func:`remfrailty`."""

    warnings.warn(
        "frailty_rem is deprecated; use remfrailty",
        DeprecationWarning,
        stacklevel=2,
    )
    return remfrailty(history, stats, **kwargs)


def remfrailty(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    approach: Sequence[str] | str = ("frequentist", "Bayesian"),
    engine: str = "auto",
    **kwargs: Any,
) -> RemEstimateGLMM | RemEstimateDuration | ActorRemEstimate:
    """Fit the default sender/receiver or dyad frailty structure."""

    if not isinstance(history, EventHistory):
        raise TypeError("history must be an EventHistory returned by remify")
    if not isinstance(stats, (RemStats, AomStats, RemStatsDuration)):
        raise TypeError("stats must be a relational-event statistic object")
    selected_approach = _match_arg(
        approach, ("frequentist", "Bayesian"), "approach"
    )
    if selected_approach == "Bayesian":
        raise NotImplementedError("Bayesian frailty estimation is not implemented yet")
    if isinstance(stats, AomStats):
        random: Any = {
            "sender": "~ (1 | actor)",
            "receiver": "~ (1 | actor)",
        }
    elif not history.directed:
        random = "~ (1 | dyad)"
        warnings.warn(
            "undirected tie model uses dyad-level frailty because actor-level "
            "frailty is not identified",
            UserWarning,
            stacklevel=2,
        )
    else:
        random = "~ (1 | actor1) + (1 | actor2)"
    fitted = remstimate(
        history,
        stats,
        approach=selected_approach,
        random=random,
        engine=engine,
        **kwargs,
    )
    if not isinstance(
        fitted,
        (RemEstimateGLMM, RemEstimateDuration, ActorRemEstimate),
    ):
        raise RuntimeError("remfrailty did not return a random-effects result")
    return fitted


def rempenalty(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    approach: Sequence[str] | str = ("frequentist", "Bayesian"),
    alpha: float = 1.0,
    prior: str = "horseshoe",
    nfolds: int = 10,
    lambda_select: Sequence[str] | str = ("1se", "min"),
    penalty: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> (
    RemEstimateGlmnet
    | RemEstimateDurationGlmnet
    | RemEstimateShrinkage
    | ActorRemEstimate
):
    """Fit elastic-net or approximate-Bayesian shrinkage regularization."""

    selected_approach = _match_arg(
        approach, ("frequentist", "Bayesian"), "approach"
    )
    selected_lambda = _match_arg(lambda_select, ("1se", "min"), "lambda_select")
    supplied = dict(penalty or {})
    controls = (
        {"prior": prior, **supplied}
        if selected_approach == "Bayesian"
        else {
            "alpha": alpha,
            "nfolds": nfolds,
            "lambda_select": selected_lambda,
            **supplied,
        }
    )
    fitted = remstimate(
        history,
        stats,
        approach=selected_approach,
        penalty=controls,
        **kwargs,
    )
    if not isinstance(
        fitted,
        (
            RemEstimateGlmnet,
            RemEstimateDurationGlmnet,
            RemEstimateShrinkage,
            ActorRemEstimate,
        ),
    ):
        raise RuntimeError("rempenalty did not return a penalized REM result")
    return fitted


def remixture(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    random: str,
    k: int | Sequence[int] = 2,
    concomitant: str | None = None,
    nrep: int = 3,
    **kwargs: Any,
) -> RemEstimateMixture | ActorRemEstimate | dict[str, Any]:
    """Fit a finite-mixture REM with a user-selected clustering unit."""

    fitted = remstimate(
        history,
        stats,
        mixture={
            "k": k,
            "random": random,
            "concomitant": concomitant,
            "nrep": nrep,
        },
        **kwargs,
    )
    if isinstance(fitted, (RemEstimateMixture, ActorRemEstimate, dict)):
        return fitted
    raise RuntimeError("remixture did not return a finite-mixture result")


def dlcrem(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    k: int | Sequence[int] = 2,
    nrep: int = 3,
    **kwargs: Any,
) -> RemEstimateMixture | ActorRemEstimate | dict[str, Any]:
    """Fit the dyadic-latent-class special case of :func:`remixture`."""

    return remixture(
        history,
        stats,
        random="~ (1 | dyad)",
        k=k,
        nrep=nrep,
        **kwargs,
    )


def remwindow(
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    n_windows: int = 5,
    window_width: int | None = None,
    step_size_window: int | None = None,
    start_point: int = 1,
    min_events: int = 50,
    approach: str = "frequentist",
    parallel: bool = False,
    ncores_window: int = 1,
    **kwargs: Any,
) -> RemEstimateWindow:
    """Fit one REM repeatedly over contiguous event or duration-time slices.

    Auto mode partitions all selected events into approximately equal windows.
    Manual mode advances a fixed-width window by ``step_size_window`` and lets
    the final window absorb the remainder. Duration windows use complete
    start/end time strata, so simultaneous process rows are never split.
    """

    if not isinstance(history, EventHistory):
        raise TypeError("history must be an EventHistory returned by remify")
    if not isinstance(stats, (RemStats, AomStats, RemStatsDuration)):
        raise TypeError("stats must be a RemStats, AomStats, or RemStatsDuration object")
    if not _estimation_histories_compatible(stats.history, history):
        raise ValueError("stats must have been computed from the supplied history")
    if isinstance(stats, RemStats) and history.model != "tie":
        raise TypeError("RemStats requires a tie-oriented history")
    if isinstance(stats, AomStats) and history.model != "actor":
        raise TypeError("AomStats requires an actor-oriented history")
    if isinstance(stats, RemStatsDuration) and not history.duration:
        raise TypeError("RemStatsDuration requires a duration history")
    for name, value in (
        ("n_windows", n_windows),
        ("start_point", start_point),
        ("min_events", min_events),
        ("ncores_window", ncores_window),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, optional_value in (
        ("window_width", window_width),
        ("step_size_window", step_size_window),
    ):
        if optional_value is not None and (
            isinstance(optional_value, bool)
            or not isinstance(optional_value, int)
            or optional_value < 1
        ):
            raise ValueError(f"{name} must be a positive integer or None")
    if not isinstance(parallel, bool):
        raise TypeError("parallel must be a boolean")
    if window_width is None and step_size_window is not None:
        raise ValueError("step_size_window requires window_width")

    event_count = (
        len(_duration_window_time_indices(stats))
        if isinstance(stats, RemStatsDuration)
        else len(stats.event_indices)
    )
    if event_count == 0:
        raise ValueError("stats contains no events to window")
    if start_point > event_count:
        raise ValueError("start_point is outside the statistic event range")
    if isinstance(stats, RemStatsDuration):
        parameter_count = len(stats.stacked.stat_names)
    elif isinstance(stats, RemStats):
        parameter_count = len(stats.names)
    else:
        parameter_count = max(len(stats.sender_names), len(stats.receiver_names))
    parameter_floor = 20 * parameter_count
    minimum_width = max(min_events, parameter_floor)
    total = event_count - start_point + 1
    auto_mode = window_width is None
    if auto_mode:
        selected_windows = n_windows
        while selected_windows > 1 and total // selected_windows < minimum_width:
            selected_windows -= 1
        if total // selected_windows < minimum_width:
            raise ValueError(
                f"not enough events ({total}) for one window of the required "
                f"minimum width ({minimum_width})"
            )
        if selected_windows < n_windows:
            warnings.warn(
                f"reduced n_windows from {n_windows} to {selected_windows} to keep "
                f"each window >= {minimum_width} events",
                UserWarning,
                stacklevel=2,
            )
        edges = np.floor(
            np.linspace(start_point - 1, event_count, selected_windows + 1)
        ).astype(int)
        starts = (edges[:-1] + 1).tolist()
        ends = edges[1:].tolist()
    else:
        assert window_width is not None
        if window_width < minimum_width:
            warnings.warn(
                f"window_width ({window_width}) is below the recommended EPV "
                f"floor ({minimum_width}); estimates may be unstable",
                UserWarning,
                stacklevel=2,
            )
        step = window_width if step_size_window is None else step_size_window
        starts = list(range(start_point, event_count + 1, step))
        ends = [min(start + window_width - 1, event_count) for start in starts]
        ends[-1] = event_count

    selected_stats = tuple(
        _window_subset_stats(stats, start, end)
        for start, end in zip(starts, ends, strict=True)
    )

    def fit_one(
        window_stats: RemStats | AomStats | RemStatsDuration,
    ) -> RemEstimate | ActorRemEstimate | Exception:
        try:
            fitted = remstimate(
                history,
                window_stats,
                approach=approach,
                **kwargs,
            )
            if isinstance(fitted, dict):
                raise ValueError("remwindow requires a single fitted model per window")
            return fitted
        except Exception as error:  # retain failed windows for diagnosis
            return error

    use_parallel = parallel and ncores_window > 1
    if use_parallel:
        import os

        if os.name == "nt":
            warnings.warn(
                "parallel window fitting is unavailable on Windows; using sequential fitting",
                UserWarning,
                stacklevel=2,
            )
            use_parallel = False
    if use_parallel:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=ncores_window) as executor:
            fits = tuple(executor.map(fit_one, selected_stats))
    else:
        fits = tuple(fit_one(window_stats) for window_stats in selected_stats)

    rows: list[dict[str, Any]] = []
    for number, (start, end, window_stats, fitted) in enumerate(
        zip(starts, ends, selected_stats, fits, strict=True),
        start=1,
    ):
        if isinstance(window_stats, RemStatsDuration):
            time_indices = _duration_window_time_indices(window_stats)
            timeline = _duration_history_timeline(history)
            start_time = timeline[time_indices[0] - 1]
            end_time = timeline[time_indices[-1] - 1]
        else:
            first_event = window_stats.event_indices[0]
            last_event = window_stats.event_indices[-1]
            start_time = history.events.iloc[first_event]["time"]
            end_time = history.events.iloc[last_event]["time"]
        converged: bool | None
        if isinstance(fitted, (RemEstimate, ActorRemEstimate)):
            converged = fitted.converged
        else:
            converged = None
        row = {
            "window": number,
            "start_event": start,
            "end_event": end,
            "start_time": start_time,
            "end_time": end_time,
            "n_events": end - start + 1,
            "converged": converged,
        }
        if isinstance(window_stats, RemStatsDuration):
            row["n_strata"] = end - start + 1
        rows.append(row)
    windows = pd.DataFrame(rows)
    model_type = (
        "duration"
        if isinstance(stats, RemStatsDuration)
        else "tie" if isinstance(stats, RemStats) else "actor"
    )
    return RemEstimateWindow(
        fits=fits,
        windows=windows,
        type=model_type,
        mode="auto" if auto_mode else "manual",
        metadata={
            "model": model_type,
            "approach": _match_arg(approach, ("frequentist", "Bayesian"), "approach"),
            "n_events": event_count,
            "n_windows": len(starts),
            "window_unit": (
                "duration_time_strata"
                if isinstance(stats, RemStatsDuration)
                else "events"
            ),
        },
        source_stats=stats,
        window_stats=selected_stats,
    )


def _window_subset_stats(
    stats: RemStats | AomStats | RemStatsDuration,
    start: int,
    end: int,
) -> RemStats | AomStats | RemStatsDuration:
    """Slice one-based inclusive event or duration-stratum positions."""

    if isinstance(stats, RemStatsDuration):
        time_indices = _duration_window_time_indices(stats)
        selected = time_indices[start - 1 : end]
        if not selected:
            raise ValueError("duration window contains no start/end time strata")
        frame = stats.stacked.remstats_stack
        window_frame = frame.loc[frame["time_index"].isin(selected)].reset_index(drop=True)
        stacked = RemStatsStackedDuration(
            remstats_stack=window_frame,
            subset=(selected[0], selected[-1]),
            D_start=stats.stacked.D_start,
            D_end=stats.stacked.D_end,
            E=len(selected),
            stat_names=list(stats.stacked.stat_names),
            stat_names_start=list(stats.stacked.stat_names_start),
            stat_names_end=list(stats.stacked.stat_names_end),
            ordinal=stats.stacked.ordinal,
            model=stats.stacked.model,
        )
        return RemStatsDuration(
            history=stats.history,
            stacked=stacked,
            start_formula=stats.start_formula,
            end_formula=stats.end_formula,
            psi_start=stats.psi_start,
            psi_end=stats.psi_end,
        )

    positions = list(range(start - 1, end))
    if isinstance(stats, RemStats):
        return type(stats)(
            history=stats.history,
            stats=[stats.stats[index] for index in positions],
            names=list(stats.names),
            formula=stats.formula,
            observed_indices=[stats.observed_indices[index] for index in positions],
            event_indices=[stats.event_indices[index] for index in positions],
            observed_index_groups=(
                [stats.observed_index_groups[index] for index in positions]
                if stats.observed_index_groups
                else []
            ),
            sample_map=(
                [stats.sample_map[index] for index in positions]
                if stats.sample_map
                else []
            ),
            sampling_weights=(
                [stats.sampling_weights[index] for index in positions]
                if stats.sampling_weights
                else []
            ),
        )

    def subset(values: Sequence[Any]) -> list[Any]:
        return [values[index] for index in positions] if values else []

    return AomStats(
        history=stats.history,
        sender_stats=subset(stats.sender_stats),
        receiver_stats=subset(stats.receiver_stats),
        sender_names=list(stats.sender_names),
        receiver_names=list(stats.receiver_names),
        observed_sender_indices=subset(stats.observed_sender_indices),
        observed_receiver_indices=subset(stats.observed_receiver_indices),
        receiver_masks=subset(stats.receiver_masks),
        event_indices=subset(stats.event_indices),
        observed_sender_groups=subset(stats.observed_sender_groups),
        receiver_choice_stats=subset(stats.receiver_choice_stats),
        receiver_choice_observed_indices=subset(
            stats.receiver_choice_observed_indices
        ),
        receiver_choice_masks=subset(stats.receiver_choice_masks),
        receiver_choice_event_indices=subset(stats.receiver_choice_event_indices),
    )


def _duration_window_time_indices(stats: RemStatsDuration) -> list[int]:
    """Return ordered, non-empty duration strata represented in a statistic object."""

    return [
        int(value)
        for value in stats.stacked.remstats_stack["time_index"].drop_duplicates()
    ]


def _duration_history_timeline(history: EventHistory) -> list[float]:
    """Return the ordered start/end timeline used by duration statistics."""

    starts = history.events["time"].astype(float).to_list()
    ends = (
        history.events.loc[history.events["end"].notna(), "end"]
        .astype(float)
        .to_list()
    )
    return sorted(set([*starts, *ends]))


def _window_fit_component(
    fitted: RemEstimate | ActorRemEstimate | Exception,
    component: str,
) -> RemEstimate | None:
    if isinstance(fitted, RemEstimate):
        return fitted if component == "tie" else None
    if isinstance(fitted, ActorRemEstimate):
        if component == "sender":
            return fitted.sender_model
        if component == "receiver":
            return fitted.receiver_model
    return None


def _window_coefficient_block(
    result: RemEstimateWindow,
    component: str,
    *,
    ci: float,
    k: float,
) -> dict[str, Any] | None:
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be between zero and one")
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError("k must be positive")
    components = [
        _window_fit_component(fitted, component) for fitted in result.fits
    ]
    anchor = next((fitted for fitted in components if fitted is not None), None)
    if anchor is None:
        return None
    names = list(anchor.names)
    coefficients = np.full((result.n_windows, len(names)), np.nan, dtype=float)
    standard_errors = np.full_like(coefficients, np.nan)
    for row, fitted in enumerate(components):
        if fitted is None or not fitted.converged:
            continue
        by_name = dict(zip(fitted.names, fitted.coef, strict=True))
        se_values = fitted.se
        se_by_name = (
            {}
            if se_values is None
            else dict(zip(fitted.names, se_values, strict=True))
        )
        for column, name in enumerate(names):
            if name in by_name:
                coefficients[row, column] = float(by_name[name])
            if name in se_by_name:
                standard_errors[row, column] = float(se_by_name[name])
    separated = np.zeros_like(coefficients, dtype=bool)
    for column in range(len(names)):
        values = standard_errors[:, column]
        finite = values[np.isfinite(values)]
        if finite.size:
            median = float(np.median(finite))
            if median > 0.0:
                separated[:, column] = values > k * median
    from scipy.stats import norm

    critical = float(norm.ppf(1.0 - (1.0 - ci) / 2.0))
    labels = pd.Index(names, name="effect")
    return {
        "coefficients": pd.DataFrame(coefficients, columns=labels),
        "se": pd.DataFrame(standard_errors, columns=labels),
        "separated": pd.DataFrame(separated, columns=labels),
        "lower": pd.DataFrame(
            coefficients - critical * standard_errors, columns=labels
        ),
        "upper": pd.DataFrame(
            coefficients + critical * standard_errors, columns=labels
        ),
    }


def _window_coefficient_blocks(
    result: RemEstimateWindow,
    *,
    ci: float,
    k: float,
) -> dict[str, Any]:
    if result.type in {"tie", "duration"}:
        block = _window_coefficient_block(result, "tie", ci=ci, k=k)
        if block is None:
            raise ValueError("no successful fits in this RemEstimateWindow")
        return {**block, "windows": result.windows.copy()}
    return {
        "sender": _window_coefficient_block(result, "sender", ci=ci, k=k),
        "receiver": _window_coefficient_block(result, "receiver", ci=ci, k=k),
        "windows": result.windows.copy(),
    }


def _window_coefficient_plot_frame(
    result: RemEstimateWindow,
    *,
    ci: float,
) -> pd.DataFrame:
    blocks = result.coefficients(ci=ci)
    output: list[pd.DataFrame] = []
    selected = (
        ((result.type, blocks),)
        if result.type in {"tie", "duration"}
        else (("sender", blocks["sender"]), ("receiver", blocks["receiver"]))
    )
    midpoint = (
        result.windows[["start_time", "end_time"]].astype(float).mean(axis=1)
    )
    for component, block in selected:
        if block is None:
            continue
        coefficients = block["coefficients"]
        lower = block["lower"]
        upper = block["upper"]
        for effect in coefficients.columns:
            output.append(
                pd.DataFrame(
                    {
                        "window": result.windows["window"].to_numpy(dtype=int),
                        "mid_time": midpoint.to_numpy(dtype=float),
                        "component": component,
                        "effect": effect,
                        "coefficient": coefficients[effect].to_numpy(dtype=float),
                        "lower": lower[effect].to_numpy(dtype=float),
                        "upper": upper[effect].to_numpy(dtype=float),
                    }
                )
            )
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _window_summary(result: RemEstimateWindow, *, k: float) -> dict[str, Any]:
    fit_table = result.windows.copy()
    loglik: list[float] = []
    aic: list[float] = []
    bic: list[float] = []
    for fitted in result.fits:
        if isinstance(fitted, (RemEstimate, ActorRemEstimate)) and fitted.converged:
            loglik.append(fitted.log_likelihood)
            aic.append(AIC(fitted))
            bic.append(BIC(fitted))
        else:
            loglik.append(float("nan"))
            aic.append(float("nan"))
            bic.append(float("nan"))
    fit_table["loglik"] = loglik
    fit_table["AIC"] = aic
    fit_table["BIC"] = bic
    blocks = result.coefficients(k=k)

    def stability(block: dict[str, Any] | None) -> pd.DataFrame | None:
        if block is None:
            return None
        coefficients = block["coefficients"].mask(block["separated"])
        errors = block["se"].mask(block["separated"])
        rows: list[dict[str, Any]] = []
        for effect in coefficients.columns:
            values = coefficients[effect]
            excluded = np.flatnonzero(block["separated"][effect].to_numpy()) + 1
            rows.append(
                {
                    "effect": effect,
                    "n": int(values.notna().sum()),
                    "mean": float(values.mean()),
                    "sd": float(values.std()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean_se": float(errors[effect].mean()),
                    "sd_over_se": float(values.std() / errors[effect].mean()),
                    "excluded": "-" if not len(excluded) else ", ".join(
                        f"W{index}" for index in excluded
                    ),
                }
            )
        return pd.DataFrame(rows)

    if result.type in {"tie", "duration"}:
        return {
            "fit": fit_table,
            "stability": stability(blocks),
            "type": result.type,
        }
    return {
        "fit": fit_table,
        "stability_sender": stability(blocks["sender"]),
        "stability_receiver": stability(blocks["receiver"]),
        "type": "actor",
    }


def _window_interpolated_parameters(
    block: dict[str, Any],
    names: Sequence[str],
    mid_times: np.ndarray,
    event_times: np.ndarray,
) -> np.ndarray:
    coefficients = block["coefficients"].mask(block["separated"])
    output = np.zeros((len(event_times), len(names)), dtype=float)
    for column, name in enumerate(names):
        if name not in coefficients:
            continue
        values = coefficients[name].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(mid_times)
        if not np.any(valid):
            raise ValueError(f"no converged window coefficient is available for {name!r}")
        selected_times = mid_times[valid]
        selected_values = values[valid]
        order = np.argsort(selected_times, kind="stable")
        output[:, column] = np.interp(
            event_times,
            selected_times[order],
            selected_values[order],
        )
    return output


def _window_probabilities(
    designs: Sequence[np.ndarray],
    parameters: np.ndarray,
    *,
    sampling_weights: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, ...]:
    from scipy.special import softmax

    output: list[np.ndarray] = []
    for position, design in enumerate(designs):
        eta = np.asarray(design, dtype=float) @ parameters[position]
        if sampling_weights is not None:
            eta = eta + np.log(np.asarray(sampling_weights[position], dtype=float))
        output.append(np.asarray(softmax(eta), dtype=float))
    return tuple(output)


def _window_recall(
    probabilities: Sequence[np.ndarray],
    observed_groups: Sequence[Sequence[int]],
    event_indices: Sequence[int],
    history: EventHistory,
    *,
    top_pct: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    tiny = np.finfo(float).tiny
    for event_number, (values, observed, event_index) in enumerate(
        zip(probabilities, observed_groups, event_indices, strict=True),
        start=1,
    ):
        size = len(values)
        for sub_index, position in enumerate(observed, start=1):
            ranked = _recall_ranks(values, int(position) + 1)
            relative_rank = (
                0.0 if size <= 1 else float((ranked["rank"] - 1.0) / (size - 1))
            )
            probability = float(values[position])
            rows.append(
                {
                    "event": event_number,
                    "sub_index": sub_index,
                    "obs_id": int(position) + 1,
                    "rel_rank": relative_rank,
                    "cum_prob": float(ranked["cum"]),
                    "obs_prob": probability,
                    "prob_ratio": probability * size,
                    "log_loss": -float(np.log(max(probability, tiny))),
                    "time": history.events.iloc[int(event_index)]["time"],
                }
            )
    per_event = pd.DataFrame(rows)
    if per_event.empty:
        summary = {
            "mean_rel_rank": float("nan"),
            "median_rel_rank": float("nan"),
            "mean_cum_prob": float("nan"),
            "mean_prob_ratio": float("nan"),
            "mean_log_loss": float("nan"),
            "top_pct": top_pct,
            "top_pct_prop": float("nan"),
        }
    else:
        summary = {
            "mean_rel_rank": float(per_event["rel_rank"].mean()),
            "median_rel_rank": float(per_event["rel_rank"].median()),
            "mean_cum_prob": float(per_event["cum_prob"].mean()),
            "mean_prob_ratio": float(per_event["prob_ratio"].mean()),
            "mean_log_loss": float(per_event["log_loss"].mean()),
            "top_pct": top_pct,
            "top_pct_prop": float((per_event["rel_rank"] <= top_pct).mean()),
        }
    return {"per_event": per_event, "summary": summary}


def _window_event_times(history: EventHistory, indices: Sequence[int]) -> np.ndarray:
    return np.asarray(
        [history.events.iloc[int(index)]["time"] for index in indices],
        dtype=float,
    )


def _window_diagnostics(
    result: RemEstimateWindow,
    history: EventHistory,
    stats: RemStats | AomStats | RemStatsDuration,
    *,
    top_pct: float,
    surprise_threshold: float,
    k: float,
) -> WindowDiagnostics:
    if not np.isfinite(top_pct) or not 0.0 < top_pct < 1.0:
        raise ValueError("top_pct must be between zero and one")
    if not _estimation_histories_compatible(stats.history, history):
        raise ValueError("stats must have been computed from the supplied history")
    if result.type == "tie" and not isinstance(stats, RemStats):
        raise TypeError("tie window diagnostics require RemStats")
    if result.type == "actor" and not isinstance(stats, AomStats):
        raise TypeError("actor window diagnostics require AomStats")
    if result.type == "duration" and not isinstance(stats, RemStatsDuration):
        raise TypeError("duration window diagnostics require RemStatsDuration")
    blocks = result.coefficients(k=k)
    mid_times = (
        result.windows[["start_time", "end_time"]]
        .astype(float)
        .mean(axis=1)
        .to_numpy(dtype=float)
    )
    if isinstance(stats, RemStatsDuration):
        frame = stats.stacked.remstats_stack.reset_index(drop=True)
        names = list(stats.stacked.stat_names)
        groups = _duration_groups(frame)
        timeline = _duration_history_timeline(history)
        group_times = np.asarray(
            [timeline[int(frame.iloc[index[0]]["time_index"]) - 1] for index in groups],
            dtype=float,
        )
        parameters = _window_interpolated_parameters(
            blocks,
            names,
            mid_times,
            group_times,
        )
        design = frame[names].to_numpy(dtype=float)
        offset = (
            None
            if stats.stacked.ordinal
            else frame["log_interevent"].to_numpy(dtype=float)
        )
        scores = np.zeros(len(frame), dtype=float)
        from scipy.special import softmax

        for position, index in enumerate(groups):
            eta = design[index] @ parameters[position]
            if offset is not None:
                scores[index] = np.exp(np.clip(eta + offset[index], -745.0, 700.0))
            else:
                scores[index] = softmax(eta)
        recall = _duration_recall_tables(
            frame,
            scores,
            history=history,
            start_names=stats.stacked.stat_names_start,
            end_names=stats.stacked.stat_names_end,
            top_pct=top_pct,
            surprise_threshold=surprise_threshold,
        )
        return WindowDiagnostics(
            fit=result,
            windows=result.windows.copy(),
            type="duration",
            recall=recall["recall_joint"],
            start=recall["recall_start"],
            end=recall["recall_end"],
        )

    if isinstance(stats, RemStats):
        event_times = _window_event_times(history, stats.event_indices)
        parameters = _window_interpolated_parameters(
            blocks,
            stats.names,
            mid_times,
            event_times,
        )
        probabilities = _window_probabilities(
            stats.stats,
            parameters,
            sampling_weights=(stats.sampling_weights or None),
        )
        observed = stats.observed_index_groups or [
            [index] for index in stats.observed_indices
        ]
        return WindowDiagnostics(
            fit=result,
            windows=result.windows.copy(),
            type="tie",
            recall=_window_recall(
                probabilities,
                observed,
                stats.event_indices,
                history,
                top_pct=top_pct,
            ),
        )

    sender: dict[str, Any] | None = None
    sender_block = blocks["sender"]
    if sender_block is not None and stats.sender_names:
        event_times = _window_event_times(history, stats.event_indices)
        parameters = _window_interpolated_parameters(
            sender_block,
            stats.sender_names,
            mid_times,
            event_times,
        )
        probabilities = _window_probabilities(stats.sender_stats, parameters)
        observed = stats.observed_sender_groups or [
            [index] for index in stats.observed_sender_indices
        ]
        sender = {
            "recall": _window_recall(
                probabilities,
                observed,
                stats.event_indices,
                history,
                top_pct=top_pct,
            )
        }

    receiver: dict[str, Any] | None = None
    receiver_block = blocks["receiver"]
    if receiver_block is not None and stats.receiver_names:
        designs, observed = _actor_receiver_diagnostic_designs(stats)
        event_indices = stats.receiver_choice_event_indices or stats.event_indices
        event_times = _window_event_times(history, event_indices)
        parameters = _window_interpolated_parameters(
            receiver_block,
            stats.receiver_names,
            mid_times,
            event_times,
        )
        probabilities = _window_probabilities(designs, parameters)
        receiver = {
            "recall": _window_recall(
                probabilities,
                observed,
                event_indices,
                history,
                top_pct=top_pct,
            )
        }
    return WindowDiagnostics(
        fit=result,
        windows=result.windows.copy(),
        type="actor",
        sender=sender,
        receiver=receiver,
    )


def remtribute(
    history: EventHistory,
    stats: RemStats | None = None,
    *,
    effects: str | Effect | Formula | None = None,
    attribute: str = "type",
    attribute_type: Sequence[str] | str = ("nominal", "ordinal", "numeric"),
    attr_actors: pd.DataFrame | None = None,
    memory: Sequence[str] | str = "full",
    memory_value: float | Sequence[float] | None = float("inf"),
    **kwargs: Any,
) -> RemTribute:
    """Model an event attribute conditional on the observed dyad sequence.

    Tie-level statistics are evaluated only at the observed dyad. Nominal,
    ordinal, and numeric attributes use multinomial-logit, proportional-odds,
    and Gaussian models, respectively.
    """

    if not isinstance(history, EventHistory):
        raise TypeError("history must be an EventHistory returned by remify")
    if history.duration:
        raise TypeError("remtribute currently supports non-duration event histories")
    if stats is None and effects is None:
        raise ValueError("provide either stats or effects")
    if stats is not None and not isinstance(stats, RemStats):
        raise TypeError("stats must be a tie-oriented RemStats object")
    if effects is not None and not isinstance(effects, (str, Effect, Formula)):
        raise TypeError("effects must be a one-sided formula, Effect, or Formula")
    if not isinstance(attribute, str) or not attribute:
        raise TypeError("attribute must be a non-empty column name")
    selected_type = _match_arg(
        attribute_type,
        ("nominal", "ordinal", "numeric"),
        "attribute_type",
    )
    outcome = _tribute_attribute_values(history, attribute)

    statistics = stats
    if effects is not None:
        tie_history = _tribute_tie_history(history, attribute)
        computed = remstats(
            tie_history,
            tie_effects=effects,
            memory=memory,
            memory_value=memory_value,
            attr_actors=attr_actors,
        )
        if not isinstance(computed, RemStats):
            raise RuntimeError("tie-oriented remtribute statistics were not produced")
        statistics = computed
    assert statistics is not None
    if not _tribute_histories_compatible(statistics.history, history):
        raise ValueError("stats and history describe different event histories")

    design, selected_outcome = _tribute_observed_design(
        history,
        statistics,
        outcome,
    )
    design, selected_outcome, names = _tribute_prepare_design(
        design,
        selected_outcome,
        statistics.names,
        attribute=attribute,
        attribute_type=selected_type,
    )
    controls = _tribute_controls(kwargs)
    if selected_type == "numeric":
        backend = _tribute_fit_numeric(design, selected_outcome, names)
        levels: tuple[Any, ...] = ()
    elif selected_type == "nominal":
        levels = _tribute_levels(selected_outcome, ordered=False)
        backend = _tribute_fit_nominal(
            design,
            selected_outcome,
            names,
            levels,
            controls=controls,
        )
    else:
        levels = _tribute_levels(selected_outcome, ordered=True)
        backend = _tribute_fit_ordinal(
            design,
            selected_outcome,
            names,
            levels,
            controls=controls,
        )

    data = pd.DataFrame(design, columns=names)
    data[".y"] = selected_outcome.reset_index(drop=True)
    formula_value = ".y ~ " + " + ".join(names)
    return RemTribute(
        coefficients=backend["coefficients"],
        covariance=backend["covariance"],
        log_likelihood=float(backend["log_likelihood"]),
        backend_fit=backend,
        attribute=attribute,
        attribute_type=selected_type,
        n_events=len(data),
        stat_names=names,
        formula=formula_value,
        data=data,
        levels=levels,
        aic=float(backend["AIC"]),
        bic=float(backend["BIC"]),
    )


def _tribute_attribute_values(history: EventHistory, attribute: str) -> pd.Series:
    column = attribute
    if column not in history.events:
        aliases = {"type": "event_type", "weight": "event_weight"}
        column = aliases.get(attribute, attribute)
    if column not in history.events:
        raise ValueError(
            f"column {attribute!r} was not retained; use event_type or "
            "event_attributes in remify"
        )
    return history.events[column].reset_index(drop=True).copy()


def _tribute_tie_history(history: EventHistory, attribute: str) -> EventHistory:
    if history.model == "tie":
        return history
    from remflow.history import remify

    frame = pd.DataFrame(
        {
            "time": history.events["time"].to_numpy(copy=True),
            "actor1": history.events["sender"].to_numpy(copy=True),
            "actor2": history.events["receiver"].to_numpy(copy=True),
        }
    )
    has_types = bool(history.event_types)
    if has_types:
        frame["type"] = history.events["event_type"].to_numpy(copy=True)
    custom_attributes: list[str] = []
    if attribute not in {"type", "weight"}:
        frame[attribute] = _tribute_attribute_values(history, attribute).to_numpy(
            copy=True
        )
        custom_attributes.append(attribute)
    frame["__event_weight"] = history.events["event_weight"].to_numpy(copy=True)
    manual_riskset: pd.DataFrame | None = None
    if history.riskset_mode == "manual" and history.risksets:
        columns = ["sender", "receiver"]
        if "event_type" in history.risksets[0]:
            columns.append("event_type")
        manual_riskset = history.risksets[0][columns].copy()
    return remify(
        frame,
        directed=history.directed,
        ordinal=history.ordinal,
        model="tie",
        actors=history.actors["actor"].to_list(),
        riskset=history.riskset_mode,
        manual_riskset=manual_riskset,
        event_type="type" if has_types else None,
        event_weight="__event_weight",
        event_attributes=custom_attributes or None,
    )


def _tribute_histories_compatible(source: EventHistory, supplied: EventHistory) -> bool:
    if source.E != supplied.E or source.directed != supplied.directed:
        return False
    columns = ["sender", "receiver", "event_type"]
    if not source.events[columns].reset_index(drop=True).equals(
        supplied.events[columns].reset_index(drop=True)
    ):
        return False
    source_groups = source.events.groupby("time", sort=False, dropna=False).size().to_list()
    supplied_groups = (
        supplied.events.groupby("time", sort=False, dropna=False).size().to_list()
    )
    return bool(source_groups == supplied_groups)


def _tribute_observed_design(
    history: EventHistory,
    statistics: RemStats,
    outcome: pd.Series,
) -> tuple[np.ndarray, pd.Series]:
    groups = {
        int(indexes[0]): [int(index) for index in indexes]
        for indexes in statistics.history.events.groupby(
            "time", sort=False, dropna=False
        ).groups.values()
    }
    rows: list[np.ndarray] = []
    values: list[Any] = []
    for row, (event_index, matrix) in enumerate(
        zip(statistics.event_indices, statistics.stats, strict=True)
    ):
        event_group = groups.get(int(event_index))
        if event_group is None:
            raise ValueError("stats event indices do not align with their history")
        observed = (
            statistics.observed_index_groups[row]
            if statistics.observed_index_groups
            else [statistics.observed_indices[row]]
        )
        if len(event_group) != len(observed):
            raise ValueError(
                "cannot align simultaneous events with observed statistic rows"
            )
        for source_index, risk_index in zip(event_group, observed, strict=True):
            if risk_index < 0 or risk_index >= len(matrix):
                raise ValueError("observed dyad index is outside the statistic risk set")
            rows.append(np.asarray(matrix[risk_index], dtype=float))
            values.append(outcome.iloc[source_index])
    if not rows:
        raise ValueError("no events fall inside the statistics window")
    return np.vstack(rows), pd.Series(values, name=".y")


def _tribute_prepare_design(
    design: np.ndarray,
    outcome: pd.Series,
    names: Sequence[str],
    *,
    attribute: str,
    attribute_type: str,
) -> tuple[np.ndarray, pd.Series, list[str]]:
    selected_names = list(names)
    if design.shape[1] != len(selected_names):
        raise ValueError("statistic names do not align with the observed design")
    numeric_outcome: pd.Series | None = None
    if attribute_type == "numeric":
        numeric_outcome = pd.to_numeric(outcome, errors="coerce")
        outcome_valid = numeric_outcome.notna().to_numpy()
    else:
        outcome_valid = outcome.notna().to_numpy()
    design_valid = np.isfinite(design).all(axis=1)
    keep = outcome_valid & design_valid
    if not np.any(keep):
        if attribute_type == "numeric":
            raise ValueError(f"all values in {attribute!r} are missing or non-numeric")
        raise ValueError(f"all values in {attribute!r} are missing")
    design = design[keep]
    selected_outcome = (
        numeric_outcome.loc[keep].reset_index(drop=True)
        if numeric_outcome is not None
        else outcome.loc[keep].reset_index(drop=True)
    )
    constant = np.asarray(
        [np.unique(design[:, column]).size == 1 for column in range(design.shape[1])],
        dtype=bool,
    )
    if np.any(constant):
        nonbaseline = [
            name
            for name, is_constant in zip(selected_names, constant, strict=True)
            if is_constant and name.lower() != "baseline"
        ]
        if nonbaseline:
            warnings.warn(
                f"dropping constant statistics: {', '.join(nonbaseline)}",
                UserWarning,
                stacklevel=2,
            )
        design = design[:, ~constant]
        selected_names = [
            name
            for name, is_constant in zip(selected_names, constant, strict=True)
            if not is_constant
        ]
    if not selected_names:
        raise ValueError(
            "no non-constant statistics remain after subsetting to observed dyads"
        )
    if attribute_type != "numeric" and selected_outcome.nunique(dropna=True) < 2:
        raise ValueError(f"{attribute_type!r} requires at least two categories")
    return design, selected_outcome, selected_names


def _tribute_controls(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    controls = dict(kwargs)
    maxiter = controls.pop("maxiter", controls.pop("maxit", 1000))
    tol = controls.pop("tol", 1e-9)
    method = controls.pop("method", "BFGS")
    trace = controls.pop("trace", False)
    if controls:
        names = ", ".join(sorted(controls))
        raise TypeError(f"unsupported remtribute backend arguments: {names}")
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be a positive number")
    if method not in {"BFGS", "L-BFGS-B"}:
        raise ValueError("method must be 'BFGS' or 'L-BFGS-B'")
    if not isinstance(trace, bool):
        raise TypeError("trace must be a boolean")
    return {
        "maxiter": maxiter,
        "tol": float(tol),
        "method": method,
        "trace": trace,
    }


def _tribute_levels(outcome: pd.Series, *, ordered: bool) -> tuple[Any, ...]:
    if isinstance(outcome.dtype, pd.CategoricalDtype):
        categories = outcome.dtype.categories.to_list()
        if ordered and outcome.dtype.ordered:
            return tuple(categories)
    values = outcome.dropna().unique().tolist()
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=lambda value: (type(value).__name__, str(value))))


def _tribute_fit_numeric(
    design: np.ndarray,
    outcome: pd.Series,
    names: Sequence[str],
) -> dict[str, Any]:
    response = outcome.to_numpy(dtype=float)
    model = np.column_stack([np.ones(len(design), dtype=float), design])
    labels = ["(Intercept)", *names]
    coefficients, _, rank, _ = np.linalg.lstsq(model, response, rcond=None)
    fitted = model @ coefficients
    residuals = response - fitted
    rss = float(residuals @ residuals)
    nobs = len(response)
    sigma2_ml = max(rss / nobs, np.finfo(float).tiny)
    loglik = float(
        -0.5 * nobs * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2_ml))
    )
    residual_df = max(nobs - int(rank), 1)
    covariance_values = (rss / residual_df) * np.linalg.pinv(model.T @ model)
    covariance = pd.DataFrame(covariance_values, index=labels, columns=labels)
    coefficient_series = pd.Series(coefficients, index=labels, name="Estimate")
    degrees = int(rank) + 1  # Gaussian dispersion is a fitted parameter in R logLik
    return {
        "engine": "gaussian",
        "coefficients": coefficient_series,
        "covariance": covariance,
        "standard_errors": pd.Series(
            np.sqrt(np.maximum(np.diag(covariance_values), 0.0)), index=labels
        ),
        "log_likelihood": loglik,
        "AIC": -2.0 * loglik + 2.0 * degrees,
        "BIC": -2.0 * loglik + np.log(nobs) * degrees,
        "fitted_values": fitted,
        "residuals": residuals,
        "dispersion": rss / residual_df,
        "rank": int(rank),
        "converged": True,
    }


def _tribute_fit_nominal(
    design: np.ndarray,
    outcome: pd.Series,
    names: Sequence[str],
    levels: Sequence[Any],
    *,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    from scipy.optimize import minimize
    from scipy.special import logsumexp, softmax

    model = np.column_stack([np.ones(len(design), dtype=float), design])
    labels = ["(Intercept)", *names]
    level_index = {value: index for index, value in enumerate(levels)}
    response = np.asarray([level_index[value] for value in outcome], dtype=int)
    classes = len(levels)
    width = model.shape[1]

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        beta = flat.reshape(classes - 1, width)
        logits = np.column_stack([np.zeros(len(model)), model @ beta.T])
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        probabilities = np.exp(log_probs)
        target = np.eye(classes, dtype=float)[response, 1:]
        gradient = (probabilities[:, 1:] - target).T @ model
        return -float(log_probs[np.arange(len(response)), response].sum()), gradient.ravel()

    result = minimize(
        fun=lambda value: objective(value)[0],
        jac=lambda value: objective(value)[1],
        x0=np.zeros((classes - 1) * width, dtype=float),
        method=str(controls["method"]),
        tol=float(controls["tol"]),
        options={"maxiter": int(controls["maxiter"]), "disp": bool(controls["trace"])},
    )
    beta = np.asarray(result.x, dtype=float).reshape(classes - 1, width)
    logits = np.column_stack([np.zeros(len(model)), model @ beta.T])
    probabilities = np.asarray(softmax(logits, axis=1), dtype=float)
    information = np.zeros(((classes - 1) * width, (classes - 1) * width))
    for left in range(classes - 1):
        for right in range(classes - 1):
            weights = probabilities[:, left + 1] * (
                float(left == right) - probabilities[:, right + 1]
            )
            information[
                left * width : (left + 1) * width,
                right * width : (right + 1) * width,
            ] = model.T @ (weights[:, None] * model)
    covariance_values = np.linalg.pinv(information)
    parameter_index = pd.MultiIndex.from_product(
        [list(levels[1:]), labels], names=["level", "term"]
    )
    covariance = pd.DataFrame(
        covariance_values, index=parameter_index, columns=parameter_index
    )
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance_values), 0.0)
    ).reshape(classes - 1, width)
    if classes == 2:
        coefficients: pd.Series | pd.DataFrame = pd.Series(
            beta[0], index=labels, name=levels[1]
        )
        errors: pd.Series | pd.DataFrame = pd.Series(
            standard_errors[0], index=labels, name=levels[1]
        )
    else:
        coefficients = pd.DataFrame(beta, index=list(levels[1:]), columns=labels)
        errors = pd.DataFrame(
            standard_errors, index=list(levels[1:]), columns=labels
        )
    loglik = -float(result.fun)
    degrees = (classes - 1) * width
    return {
        "engine": "multinomial_logit",
        "coefficients": coefficients,
        "covariance": covariance,
        "standard_errors": errors,
        "log_likelihood": loglik,
        "AIC": -2.0 * loglik + 2.0 * degrees,
        "BIC": -2.0 * loglik + np.log(len(response)) * degrees,
        "fitted_values": probabilities,
        "classes": list(levels),
        "converged": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
    }


def _tribute_ordinal_thresholds(raw: np.ndarray) -> np.ndarray:
    thresholds = np.empty_like(raw)
    thresholds[0] = raw[0]
    if len(raw) > 1:
        thresholds[1:] = raw[0] + np.cumsum(np.exp(np.clip(raw[1:], -30.0, 30.0)))
    return thresholds


def _tribute_numerical_hessian(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
) -> np.ndarray:
    width = len(point)
    step = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(point))
    hessian = np.empty((width, width), dtype=float)
    for column in range(width):
        plus = point.copy()
        minus = point.copy()
        plus[column] += step[column]
        minus[column] -= step[column]
        gradient_plus = np.empty(width, dtype=float)
        gradient_minus = np.empty(width, dtype=float)
        for row in range(width):
            row_step = step[row]
            pp = plus.copy()
            pm = plus.copy()
            mp = minus.copy()
            mm = minus.copy()
            pp[row] += row_step
            pm[row] -= row_step
            mp[row] += row_step
            mm[row] -= row_step
            gradient_plus[row] = (function(pp) - function(pm)) / (2.0 * row_step)
            gradient_minus[row] = (function(mp) - function(mm)) / (2.0 * row_step)
        hessian[:, column] = (gradient_plus - gradient_minus) / (2.0 * step[column])
    return np.asarray(0.5 * (hessian + hessian.T), dtype=float)


def _tribute_fit_ordinal(
    design: np.ndarray,
    outcome: pd.Series,
    names: Sequence[str],
    levels: Sequence[Any],
    *,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    from scipy.optimize import minimize
    from scipy.special import expit, logit

    level_index = {value: index for index, value in enumerate(levels)}
    response = np.asarray([level_index[value] for value in outcome], dtype=int)
    categories = len(levels)
    predictor_count = design.shape[1]
    cumulative = np.asarray(
        [np.mean(response <= index) for index in range(categories - 1)], dtype=float
    )
    initial_thresholds = logit(np.clip(cumulative, 1e-4, 1.0 - 1e-4))
    initial_thresholds = np.maximum.accumulate(
        initial_thresholds + np.arange(len(initial_thresholds)) * 1e-4
    )
    raw_thresholds = initial_thresholds.copy()
    if len(raw_thresholds) > 1:
        raw_thresholds[1:] = np.log(
            np.maximum(np.diff(initial_thresholds), 1e-4)
        )
    initial = np.concatenate([np.zeros(predictor_count), raw_thresholds])

    def objective(parameters: np.ndarray) -> float:
        beta = parameters[:predictor_count]
        thresholds = _tribute_ordinal_thresholds(parameters[predictor_count:])
        eta = design @ beta
        cdf = expit(thresholds[None, :] - eta[:, None])
        probabilities = np.column_stack(
            [cdf[:, 0], np.diff(cdf, axis=1), 1.0 - cdf[:, -1]]
        )
        selected = probabilities[np.arange(len(response)), response]
        if np.any(selected <= 0.0) or not np.isfinite(selected).all():
            return 1e100
        return -float(np.log(selected).sum())

    result = minimize(
        objective,
        initial,
        method=str(controls["method"]),
        tol=float(controls["tol"]),
        options={"maxiter": int(controls["maxiter"]), "disp": bool(controls["trace"])},
    )
    raw_parameters = np.asarray(result.x, dtype=float)
    beta = raw_parameters[:predictor_count]
    raw_thresholds = raw_parameters[predictor_count:]
    thresholds = _tribute_ordinal_thresholds(raw_thresholds)
    raw_information = _tribute_numerical_hessian(objective, raw_parameters)
    raw_covariance = np.linalg.pinv(raw_information)
    transform = np.eye(len(raw_parameters), dtype=float)
    transform[predictor_count:, predictor_count:] = 0.0
    transform[predictor_count:, predictor_count] = 1.0
    for threshold in range(1, len(raw_thresholds)):
        transform[
            predictor_count + threshold :, predictor_count + threshold
        ] = np.exp(np.clip(raw_thresholds[threshold], -30.0, 30.0))
    covariance_values = transform @ raw_covariance @ transform.T
    threshold_labels = [
        f"{levels[index]}|{levels[index + 1]}"
        for index in range(categories - 1)
    ]
    labels = [*names, *threshold_labels]
    covariance = pd.DataFrame(covariance_values, index=labels, columns=labels)
    coefficients = pd.Series(beta, index=list(names), name="Estimate")
    eta = design @ beta
    cdf = expit(thresholds[None, :] - eta[:, None])
    probabilities = np.column_stack(
        [cdf[:, 0], np.diff(cdf, axis=1), 1.0 - cdf[:, -1]]
    )
    loglik = -float(result.fun)
    degrees = predictor_count + categories - 1
    return {
        "engine": "proportional_odds",
        "coefficients": coefficients,
        "covariance": covariance,
        "standard_errors": pd.Series(
            np.sqrt(np.maximum(np.diag(covariance_values)[:predictor_count], 0.0)),
            index=list(names),
        ),
        "thresholds": pd.Series(thresholds, index=threshold_labels),
        "log_likelihood": loglik,
        "AIC": -2.0 * loglik + 2.0 * degrees,
        "BIC": -2.0 * loglik + np.log(len(response)) * degrees,
        "fitted_values": probabilities,
        "classes": list(levels),
        "converged": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
        "message": str(result.message),
    }


def _unsupported_extended_model(name: str) -> NoReturn:
    raise NotImplementedError(
        f"{name} is part of the public API but its implementation is not complete"
    )


def _numpy_objective(
    designs: list[np.ndarray],
    observed: list[int],
    *,
    sampling_weights: list[np.ndarray] | None = None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        ll, grad = _loglik_and_grad(beta, designs, observed, sampling_weights=sampling_weights)
        return -ll, -grad

    return objective


def _numpy_exact_objective(
    designs: list[np.ndarray],
    observed: list[int],
    exposures: np.ndarray,
    *,
    sampling_weights: list[np.ndarray] | None = None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        loglik = 0.0
        gradient = np.zeros_like(beta)
        for position, (design, index, exposure) in enumerate(
            zip(designs, observed, exposures, strict=True)
        ):
            eta = design @ beta
            intensities = np.exp(np.clip(eta, -745.0, 700.0))
            weights = (
                sampling_weights[position]
                if sampling_weights is not None
                else np.ones(len(design), dtype=float)
            )
            weighted_intensities = weights * intensities
            loglik += float(eta[index] - exposure * weighted_intensities.sum())
            gradient += design[index] - exposure * (weighted_intensities @ design)
        return -loglik, -gradient

    return objective


def _jax_objective(
    backend: JaxBackend,
    designs: list[np.ndarray],
    observed: list[int],
    *,
    exposures: np.ndarray | None = None,
    sampling_weights: list[np.ndarray] | None = None,
    riskset_chunk_size: int | None = None,
) -> Callable[[np.ndarray], tuple[float, np.ndarray]]:
    obs = [int(value) for value in observed]
    same_shape = len({design.shape for design in designs}) <= 1
    if same_shape and riskset_chunk_size is None:
        stacked = backend.asarray(np.stack(designs))
        obs_array = backend.asarray(np.asarray(obs, dtype=np.int32), dtype=np.int32)
        weight_array = (
            backend.asarray(np.stack(sampling_weights)) if sampling_weights is not None else None
        )
        exposure_array = backend.asarray(exposures) if exposures is not None else None

        def loglik(beta: Any) -> Any:
            eta = stacked @ beta
            observed_eta = _jax_take_observed(eta, obs_array)
            if exposure_array is None:
                weighted_eta = eta if weight_array is None else eta + _jax_log(weight_array)
                return (observed_eta - backend.logsumexp(weighted_eta, axis=1)).sum()
            intensity = _jax_safe_exp(eta)
            if weight_array is not None:
                intensity = weight_array * intensity
            return (observed_eta - exposure_array * intensity.sum(axis=1)).sum()

    else:
        arrays = [backend.asarray(design) for design in designs]
        weight_arrays = (
            [backend.asarray(weights) for weights in sampling_weights]
            if sampling_weights is not None
            else None
        )

        def loglik(beta: Any) -> Any:
            total = 0.0
            for event_position, (x, index) in enumerate(zip(arrays, obs, strict=True)):
                eta_observed = x[index] @ beta
                weights = weight_arrays[event_position] if weight_arrays is not None else None
                if exposures is None:
                    denominator = _jax_chunked_logsumexp(
                        backend, x, beta, weights, riskset_chunk_size
                    )
                    total = total + eta_observed - denominator
                else:
                    intensity_sum = _jax_chunked_intensity_sum(x, beta, weights, riskset_chunk_size)
                    total = total + eta_observed - exposures[event_position] * intensity_sum
            return total

    value_and_grad = backend.jit(backend.value_and_grad(loglik))

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = value_and_grad(backend.asarray(beta))
        return float(-value), -backend.to_numpy(grad)

    return objective


def _jax_exp(value: Any) -> Any:
    import jax.numpy as jnp

    return jnp.exp(value)


def _jax_safe_exp(value: Any) -> Any:
    import jax.numpy as jnp

    return jnp.exp(jnp.clip(value, -745.0, 700.0))


def _jax_log(value: Any) -> Any:
    import jax.numpy as jnp

    return jnp.log(value)


def _jax_gammaln(value: Any) -> Any:
    from jax.scipy.special import gammaln

    return gammaln(value)


def _jax_take_observed(eta: Any, observed: Any) -> Any:
    import jax.numpy as jnp

    return eta[jnp.arange(eta.shape[0]), observed]


def _jax_chunked_logsumexp(
    backend: JaxBackend,
    design: Any,
    beta: Any,
    weights: Any | None,
    chunk_size: int | None,
) -> Any:
    import jax.numpy as jnp

    size = int(design.shape[0])
    width = size if chunk_size is None else chunk_size
    total = jnp.asarray(-jnp.inf, dtype=design.dtype)
    for start in range(0, size, width):
        stop = min(start + width, size)
        eta = design[start:stop] @ beta
        if weights is not None:
            eta = eta + jnp.log(weights[start:stop])
        total = jnp.logaddexp(total, backend.logsumexp(eta))
    return total


def _jax_chunked_intensity_sum(
    design: Any,
    beta: Any,
    weights: Any | None,
    chunk_size: int | None,
) -> Any:
    import jax.numpy as jnp

    size = int(design.shape[0])
    width = size if chunk_size is None else chunk_size
    total = jnp.asarray(0.0, dtype=design.dtype)
    for start in range(0, size, width):
        stop = min(start + width, size)
        intensity = _jax_safe_exp(design[start:stop] @ beta)
        if weights is not None:
            intensity = weights[start:stop] * intensity
        total = total + intensity.sum()
    return total


def _loglik_and_grad(
    beta: np.ndarray,
    designs: list[np.ndarray],
    observed: list[int],
    *,
    sampling_weights: list[np.ndarray] | None = None,
) -> tuple[float, np.ndarray]:
    loglik = 0.0
    grad = np.zeros_like(beta)
    for position, (x, obs) in enumerate(zip(designs, observed, strict=True)):
        eta = x @ beta
        max_eta = float(np.max(eta))
        weights = (
            sampling_weights[position]
            if sampling_weights is not None
            else np.ones(len(x), dtype=float)
        )
        exp_eta = weights * np.exp(eta - max_eta)
        probs = exp_eta / exp_eta.sum()
        loglik += float(eta[obs] - (max_eta + np.log(exp_eta.sum())))
        grad += x[obs] - probs @ x
    return loglik, grad


def _event_probabilities(
    beta: np.ndarray,
    designs: list[np.ndarray],
    *,
    sampling_weights: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    probabilities: list[np.ndarray] = []
    for position, design in enumerate(designs):
        eta = design @ beta
        eta = eta - float(np.max(eta))
        exp_eta = np.exp(eta)
        if sampling_weights is not None:
            exp_eta = sampling_weights[position] * exp_eta
        probabilities.append(exp_eta / exp_eta.sum())
    return probabilities


def _event_exposures(history: EventHistory, event_indices: list[int]) -> np.ndarray:
    try:
        times = history.events["time"].astype(float).to_numpy()
    except (TypeError, ValueError) as error:  # pragma: no cover - remify normalizes supported times
        raise TypeError("exact-time estimation requires numeric normalized event times") from error
    exposures: list[float] = []
    for event_index in event_indices:
        previous_time = 0.0 if event_index == 0 else float(times[event_index - 1])
        exposure = float(times[event_index] - previous_time)
        if exposure < 0:
            raise ValueError("event times must be nondecreasing for exact-time estimation")
        exposures.append(exposure)
    return np.asarray(exposures, dtype=float)


def _metadata(
    backend: ArrayBackend,
    optimizer: str,
    n_events: int,
    *,
    seed: int | None,
    timing: str,
) -> dict[str, Any]:
    return {
        **backend.runtime_metadata,
        "optimizer": optimizer,
        "n_events": n_events,
        "seed": seed,
        "timing": timing,
    }


def _match_arg(value: Sequence[str] | str, choices: Sequence[str], name: str) -> str:
    selected = value[0] if isinstance(value, Sequence) and not isinstance(value, str) else value
    if selected not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return str(selected)


def _validate_nsim_waic(value: Any) -> int:
    if value is None:
        return 100
    valid_numeric = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )
    if not valid_numeric or not np.isfinite(float(value)) or float(value) < 1.0:
        warnings.warn(
            "nsimWAIC must be a positive integer; using the default of 100",
            UserWarning,
            stacklevel=3,
        )
        return 100
    return int(value)


__all__ = [
    "RemEstimate",
    "RemEstimateDuration",
    "RemEstimateDurationGlmnet",
    "RemEstimateGLMM",
    "RemEstimateGlmnet",
    "RemEstimateMixture",
    "RemEstimateShrinkage",
    "RemEstimateWindow",
    "RemTribute",
    "Diagnostics",
    "DurationDiagnostics",
    "WindowDiagnostics",
    "remstimate",
    "fit_rem",
    "diagnostics",
    "AIC",
    "BIC",
    "AICC",
    "WAIC",
    "bic_table",
    "frailty_rem",
    "remfrailty",
    "rempenalty",
    "remixture",
    "dlcrem",
    "remwindow",
    "remtribute",
]
