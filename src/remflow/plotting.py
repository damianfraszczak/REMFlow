"""Optional Matplotlib visualizations for REMFlow results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from remflow.estimate import ActorDiagnostics, ActorRemEstimate, Diagnostics, RemEstimate
from remflow.history import EventHistory
from remflow.reporting import coefficient_table
from remflow.stats import RemStats


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ImportError(
            "Plotting requires Matplotlib. Install it with `pip install remflow[plot]`."
        ) from exc
    return plt


def plot_event_history(
    history: EventHistory,
    *,
    actors: Sequence[Any] | None = None,
    figsize: tuple[float, float] = (9.0, 4.8),
) -> tuple[Any, Any]:
    """Plot a directed event sequence as sender-to-receiver arrows."""

    plt = _pyplot()
    data = history.plot(actors=actors)
    events = data["events"].reset_index(drop=True)
    labels = data["actors"]["actor"].to_list()
    positions = {actor: index for index, actor in enumerate(labels)}
    if history.ordinal:
        x = events["event_id"].to_numpy(dtype=float)
        x_label = "Event order"
    else:
        numeric = pd.to_numeric(events["time"], errors="coerce")
        if numeric.isna().any():
            x = events["event_id"].to_numpy(dtype=float)
            x_label = "Event order"
        else:
            x = numeric.to_numpy(dtype=float)
            x_label = "Time"

    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    for when, event in zip(x, events.itertuples(index=False), strict=True):
        sender = positions[event.sender]
        receiver = positions[event.receiver]
        axis.annotate(
            "",
            xy=(when, receiver),
            xytext=(when, sender),
            arrowprops={"arrowstyle": "-|>", "color": "#2F6B8A", "lw": 1.1},
        )
        axis.scatter([when], [sender], s=16, color="#D55E00", zorder=3)
    axis.set_yticks(range(len(labels)), [str(label) for label in labels])
    axis.set_xlabel(x_label)
    axis.set_ylabel("Actor")
    axis.set_title("Relational event history")
    axis.grid(axis="x", color="#D9E2E8", linewidth=0.7)
    return figure, axis


def plot_statistic(
    statistics: RemStats,
    effect: str | int,
    *,
    subset: int | Sequence[int] | None = None,
    figsize: tuple[float, float] = (9.0, 4.8),
) -> tuple[Any, Any]:
    """Plot selected risk-set statistic trajectories over event order."""

    plt = _pyplot()
    payload = statistics.plot(effect, subset=subset)
    frame = payload["data"]
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    for risk_id, group in frame.groupby("risk_id", sort=False):
        axis.plot(group["event_id"], group["value"], marker="o", ms=3, label=str(risk_id))
    axis.set_xlabel("Event order")
    axis.set_ylabel(str(payload["effect"]))
    axis.set_title("Statistic trajectory")
    axis.grid(color="#D9E2E8", linewidth=0.7)
    if frame["risk_id"].nunique() <= 10:
        axis.legend(title="Risk-set ID", frameon=False)
    return figure, axis


def plot_coefficients(
    fit: RemEstimate | ActorRemEstimate,
    *,
    confidence: float = 0.95,
    figsize: tuple[float, float] = (8.0, 4.8),
) -> tuple[Any, Any]:
    """Plot coefficient estimates with normal-approximation intervals."""

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    from scipy.stats import norm

    plt = _pyplot()
    frame = coefficient_table(fit, digits=None)
    if frame.empty:
        raise ValueError("fit has no coefficients to plot")
    labels = [
        f"{row.component}: {row.effect}" if row.component != "tie" else str(row.effect)
        for row in frame.itertuples(index=False)
    ]
    positions = np.arange(len(frame), dtype=float)
    errors = norm.ppf(0.5 + confidence / 2.0) * frame["std_error"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=figsize, constrained_layout=True)
    finite = np.isfinite(errors)
    axis.errorbar(
        frame.loc[finite, "estimate"],
        positions[finite],
        xerr=errors[finite],
        fmt="o",
        color="#2F6B8A",
        ecolor="#6B8797",
        capsize=3,
    )
    axis.scatter(
        frame.loc[~finite, "estimate"],
        positions[~finite],
        color="#2F6B8A",
        marker="o",
    )
    axis.axvline(0.0, color="#6B7280", linewidth=1.0, linestyle="--")
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Coefficient estimate")
    axis.set_title(f"Model coefficients ({confidence:.0%} intervals)")
    axis.grid(axis="x", color="#D9E2E8", linewidth=0.7)
    return figure, axis


def plot_diagnostics(
    value: Diagnostics | ActorDiagnostics,
    *,
    figsize: tuple[float, float] = (10.0, 4.2),
) -> tuple[Any, Any]:
    """Plot observed-event probabilities and relative ranks."""

    plt = _pyplot()
    if isinstance(value, ActorDiagnostics):
        components = [
            ("sender", value.sender_model),
            ("receiver", value.receiver_model),
        ]
        available = [(name, item) for name, item in components if item is not None]
        if not available:
            raise ValueError("actor diagnostics contain no fitted component")
    elif isinstance(value, Diagnostics):
        available = [(str(value.fit.metadata.get("component", "tie")), value)]
    else:
        raise TypeError("value must be Diagnostics or ActorDiagnostics")

    figure, axes = plt.subplots(
        len(available),
        2,
        figsize=(figsize[0], figsize[1] * len(available)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, (component, diagnostic) in enumerate(available):
        assert diagnostic is not None
        event = np.arange(1, len(diagnostic.observed_probabilities) + 1)
        probabilities = diagnostic.observed_probabilities
        denominators = np.asarray([max(len(rates) - 1, 1) for rates in diagnostic.rates])
        relative_rank = (diagnostic.ranks - 1) / denominators

        axes[row, 0].plot(event, probabilities, color="#0072B2", linewidth=1.2)
        axes[row, 0].set_ylabel("Observed-event probability")
        axes[row, 0].set_ylim(0.0, 1.0)
        axes[row, 0].set_title(f"{component.capitalize()} probabilities")

        axes[row, 1].plot(event, relative_rank, color="#D55E00", linewidth=1.2)
        axes[row, 1].set_ylabel("Relative rank (lower is better)")
        axes[row, 1].set_ylim(0.0, 1.0)
        axes[row, 1].set_title(f"{component.capitalize()} ranks")
        for axis in axes[row]:
            axis.set_xlabel("Event order")
            axis.grid(color="#D9E2E8", linewidth=0.7)
    return figure, axes
