"""Readable tabular summaries for fitted REMFlow objects."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from remflow.estimate import ActorDiagnostics, ActorRemEstimate, Diagnostics, RemEstimate


def coefficient_table(
    fit: RemEstimate | ActorRemEstimate,
    *,
    digits: int | None = 4,
) -> pd.DataFrame:
    """Return coefficients and uncertainty in a tidy table.

    Actor-oriented fits are labelled by their sender or receiver component.
    Posterior standard deviations are used for HMC results; otherwise the
    standard errors derived from the covariance matrix are reported.
    """

    rows: list[dict[str, Any]] = []
    if isinstance(fit, ActorRemEstimate):
        components: Iterable[tuple[str, RemEstimate | None]] = (
            ("sender", fit.sender_model),
            ("receiver", fit.receiver_model),
        )
    elif isinstance(fit, RemEstimate):
        components = ((str(fit.metadata.get("component", "tie")), fit),)
    else:
        raise TypeError("fit must be a RemEstimate or ActorRemEstimate")

    for component, model in components:
        if model is None:
            continue
        uncertainty = model.posterior_sd if model.draws is not None else model.se
        for position, (name, estimate) in enumerate(
            zip(model.names, model.coef, strict=True)
        ):
            spread = (
                float(uncertainty[position])
                if uncertainty is not None and position < len(uncertainty)
                else np.nan
            )
            rows.append(
                {
                    "component": component,
                    "effect": name,
                    "estimate": float(estimate),
                    "std_error": spread,
                }
            )
    result = pd.DataFrame(rows, columns=["component", "effect", "estimate", "std_error"])
    if digits is not None and not result.empty:
        result[["estimate", "std_error"]] = result[["estimate", "std_error"]].round(digits)
    return result


def fit_table(
    fit: RemEstimate | ActorRemEstimate,
    *,
    digits: int | None = 4,
) -> pd.DataFrame:
    """Return the main fit measures as one row per model component."""

    if isinstance(fit, ActorRemEstimate):
        components: Iterable[tuple[str, RemEstimate | None]] = (
            ("sender", fit.sender_model),
            ("receiver", fit.receiver_model),
        )
    elif isinstance(fit, RemEstimate):
        components = ((str(fit.metadata.get("component", "tie")), fit),)
    else:
        raise TypeError("fit must be a RemEstimate or ActorRemEstimate")

    rows = []
    for component, model in components:
        if model is None:
            continue
        rows.append(
            {
                "component": component,
                "events": int(model.metadata.get("n_observations", 0)),
                "parameters": len(model.coef),
                "log_likelihood": float(model.log_likelihood),
                "AIC": float(model.AIC),
                "BIC": float(model.BIC),
                "converged": bool(model.converged),
                "backend": model.metadata.get("backend"),
            }
        )
    result = pd.DataFrame(rows)
    numeric = ["log_likelihood", "AIC", "BIC"]
    if digits is not None and not result.empty:
        result[numeric] = result[numeric].round(digits)
    return result


def diagnostic_table(
    value: Diagnostics | ActorDiagnostics,
    *,
    digits: int | None = 4,
) -> pd.DataFrame:
    """Return recall diagnostics in a compact, component-labelled table."""

    if isinstance(value, ActorDiagnostics):
        components: Iterable[tuple[str, Diagnostics | None]] = (
            ("sender", value.sender_model),
            ("receiver", value.receiver_model),
        )
    elif isinstance(value, Diagnostics):
        components = ((str(value.fit.metadata.get("component", "tie")), value),)
    else:
        raise TypeError("value must be Diagnostics or ActorDiagnostics")

    rows = []
    for component, diagnostic in components:
        if diagnostic is None:
            continue
        summary = diagnostic.recall.get("summary", {})
        rows.append(
            {
                "component": component,
                "events": len(diagnostic.observed_probabilities),
                "mean_observed_probability": float(
                    np.mean(diagnostic.observed_probabilities)
                ),
                "mean_relative_rank": summary.get("mean_rel_rank", np.nan),
                "median_relative_rank": summary.get("median_rel_rank", np.nan),
                "top_fraction": summary.get("top_pct", np.nan),
                "observed_in_top_fraction": summary.get("top_pct_prop", np.nan),
            }
        )
    result = pd.DataFrame(rows)
    numeric = [
        "mean_observed_probability",
        "mean_relative_rank",
        "median_relative_rank",
        "top_fraction",
        "observed_in_top_fraction",
    ]
    if digits is not None and not result.empty:
        result[numeric] = result[numeric].astype(float).round(digits)
    return result
