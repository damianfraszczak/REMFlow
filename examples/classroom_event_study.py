"""Reproducible classroom relational-event study.

Source provenance, licensing, and the comparison boundary are documented in
``docs/CLASSROOM_EVENT_STUDY.md``. This example runs exact-time and ordinal
REMFlow models over the documented classroom histories. Pass ``--plot`` to
save a figure of full-model coefficients and in-sample event rankings.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from remflow import (
    Diagnostics,
    RemEstimate,
    RemStats,
    coefficient_table,
    diagnostic_table,
    diagnostics,
    fit_table,
    remify,
    remstimate,
)
from remflow.stats import Effect, Formula, observed_risk_index

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "classroom_events"
PLOTTED_TERMS = (
    "Recency_ji",
    "Recency_ij",
    "Receiver_teacher",
    "Seating",
    "Friendship",
    "PSAB_BA",
    "PSAB_BY",
    "PSAB_XB",
)


@dataclass(frozen=True)
class ClassroomData:
    date_label: str
    interactions: pd.DataFrame
    attributes: pd.DataFrame
    seating: np.ndarray
    friendship: np.ndarray
    actors: list[int]


def load_classroom_data(date_label: str = "date1") -> ClassroomData:
    """Load and prepare the classroom study data."""

    if date_label not in {"date1", "date2"}:
        raise ValueError("date_label must be 'date1' or 'date2'")

    interactions = _read_table(DATA_DIR / f"class_interactions_{date_label}.txt")
    interactions = interactions[
        (interactions["to_all_col"].astype(int) == 0)
        & (interactions["from_all_col"].astype(int) == 0)
    ].copy()
    interactions = interactions.sort_values("time_estimate_col").reset_index(drop=True)

    attributes = _read_table(DATA_DIR / "class_attributes.txt")
    attributes["id"] = attributes["id"].astype(int)
    attributes["intercept"] = 1
    attributes["male"] = (attributes["gnd"].astype(int) == 1).astype(int)
    attributes["teacher"] = (attributes["grd"].astype(int) == 16).astype(int)
    actors = attributes["id"].astype(int).to_list()

    seating_edges = _read_table(DATA_DIR / f"class_seating_{date_label}.txt")
    seating = _adjacency_matrix(
        seating_edges,
        actors=actors,
        sender_col="ego_id",
        receiver_col="alter_id",
        symmetrize=True,
    )

    friendship_edges = _read_table(DATA_DIR / "class_edgelist_sem2.txt")
    friendship = _adjacency_matrix(
        friendship_edges,
        actors=actors,
        sender_col="sender",
        receiver_col="receiver",
        symmetrize=False,
    )

    return ClassroomData(date_label, interactions, attributes, seating, friendship, actors)


def build_history(data: ClassroomData, *, ordinal: bool = True):
    """Build a REMFlow event history from the source edgelist columns."""

    events = data.interactions.rename(
        columns={
            "time_estimate_col": "time",
            "send_col": "sender",
            "receive_col": "receiver",
        }
    )[["time", "sender", "receiver"]]
    return remify(events, actors=data.actors, riskset="full", ordinal=ordinal)


def build_classroom_stats(
    history,
    data: ClassroomData,
    *,
    terms: Iterable[str],
    first: int = 1,
) -> RemStats:
    """Compute study covariates, recency, and participation-shift statistics."""

    term_list = list(terms)
    arrays: list[np.ndarray] = []
    observed: list[int] = []
    for event_index in range(first - 1, len(history.events)):
        riskset = history.risksets[event_index]
        previous = history.events.iloc[:event_index]
        matrix = np.column_stack(
            [
                _term_values(
                    term, riskset, previous, data.attributes, data.seating, data.friendship
                )
                for term in term_list
            ]
        )
        arrays.append(matrix.astype(float))
        observed.append(observed_risk_index(history, event_index))

    parsed = Formula(tuple(Effect(term) for term in term_list))
    event_indices = list(range(first - 1, len(history.events)))
    return RemStats(
        history,
        arrays,
        term_list,
        parsed,
        observed,
        event_indices=event_indices,
        observed_index_groups=[[index] for index in observed],
    )


def fit_study_model(
    date_label: str,
    terms: Iterable[str],
    *,
    first: int = 1,
    backend: str = "numpy",
    ordinal: bool = True,
) -> tuple[ClassroomData, RemEstimate]:
    """Load data, build stats, and fit one appendix model."""

    data = load_classroom_data(date_label)
    history = build_history(data, ordinal=ordinal)
    stats = build_classroom_stats(history, data, terms=terms, first=first)
    fit = remstimate(history, stats, backend=backend)
    return data, fit


def study_model_terms(model: str) -> list[str]:
    """Return term sets matching the documented study model families."""

    models = {
        "mod1": ["Intercept"],
        "mod2a": ["Intercept", "Sender_male", "Receiver_male"],
        "mod2b": [
            "Intercept",
            "Sender_male",
            "Sender_teacher",
            "Receiver_male",
            "Receiver_teacher",
        ],
        "mod3b": ["Intercept", "Sender_male", "Receiver_male", "Seating", "Friendship"],
        "mod4f": [
            "Recency_ji",
            "Recency_ij",
            "Intercept",
            "Sender_male",
            "Sender_teacher",
            "Receiver_male",
            "Receiver_teacher",
            "Seating",
            "Friendship",
            "PSAB_BA",
            "PSAB_BY",
            "PSAB_XA",
            "PSAB_XB",
            "PSAB_AY",
        ],
    }
    if model not in models:
        raise ValueError(f"unknown study model: {model}")
    return models[model]


def run_appendix(
    *,
    ordinal: bool = True,
    backend: str = "numpy",
    plot_path: Path | None = None,
    full_model_outputs: dict[str, tuple[pd.DataFrame, Diagnostics]] | None = None,
) -> pd.DataFrame:
    """Fit selected model families for both classroom dates."""

    rows = []
    plot_data: dict[str, tuple[pd.DataFrame, Diagnostics]] = {}
    for date_label in ("date1", "date2"):
        for model in ("mod1", "mod2a", "mod3b", "mod4f"):
            data, fit = fit_study_model(
                date_label,
                study_model_terms(model),
                ordinal=ordinal,
                backend=backend,
            )
            if model == "mod4f" and (plot_path is not None or full_model_outputs is not None):
                history = build_history(data, ordinal=ordinal)
                stats = build_classroom_stats(history, data, terms=fit.names)
                plot_data[date_label] = (
                    coefficient_table(fit, digits=None),
                    diagnostics(fit, history, stats),
                )
            fit_summary = fit_table(fit, digits=None).iloc[0]
            rows.append(
                {
                    "date": date_label,
                    "model": model,
                    "events": len(data.interactions),
                    "actors": len(data.actors),
                    "terms": len(fit.names),
                    "timing": "ordinal" if ordinal else "exact",
                    "log_likelihood": fit.log_likelihood,
                    "BIC": fit_summary["BIC"],
                    "converged": fit.converged,
                }
            )
    if full_model_outputs is not None:
        full_model_outputs.update(plot_data)
    if plot_path is not None:
        save_classroom_plot(plot_data, plot_path, ordinal=ordinal)
    return pd.DataFrame(rows)


def save_classroom_plot(
    plot_data: dict[str, tuple[pd.DataFrame, Diagnostics]],
    output: Path,
    *,
    ordinal: bool,
) -> None:
    """Save full-model coefficient and in-sample rank panels."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Plotting requires Matplotlib. Install it with `pip install remflow[plot]`."
        ) from exc

    colors = {"date1": "#0072B2", "date2": "#D55E00"}
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.5),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
        constrained_layout=True,
    )
    try:
        positions = np.arange(len(PLOTTED_TERMS), dtype=float)
        for date_label, label, offset in (("date1", "Day 1", -0.12), ("date2", "Day 2", 0.12)):
            frame, _ = plot_data[date_label]
            selected = frame.set_index("effect").loc[list(PLOTTED_TERMS)]
            errors = 1.96 * selected["std_error"].to_numpy(dtype=float)
            axes[0].errorbar(
                selected["estimate"],
                positions + offset,
                xerr=errors,
                fmt="o",
                color=colors[date_label],
                ecolor=colors[date_label],
                capsize=3,
                label=label,
            )
        axes[0].axvline(0.0, color="#6B7280", linewidth=1.0, linestyle="--")
        axes[0].set_yticks(positions, PLOTTED_TERMS)
        axes[0].invert_yaxis()
        scale = "log-odds" if ordinal else "log-hazard"
        axes[0].set_xlabel(f"Conditional {scale} coefficient (95% interval)")
        axes[0].set_title("a) Full-model estimates")
        axes[0].grid(axis="x", color="#D9E2E8", linewidth=0.7)
        axes[0].legend(frameon=False)

        positions = np.arange(2, dtype=float)
        width = 0.34
        for index, (date_label, label) in enumerate((("date1", "Day 1"), ("date2", "Day 2"))):
            _, report = plot_data[date_label]
            values = [
                float(np.mean(report.ranks == 1)),
                float(report.recall["summary"]["top_pct_prop"]),
            ]
            bars = axes[1].bar(
                positions + (index - 0.5) * width,
                values,
                width,
                color=colors[date_label],
                label=label,
            )
            axes[1].bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
        axes[1].set_xticks(positions, ["Rank 1", "Top 10%"])
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("Share of observed events")
        axes[1].set_title("b) In-sample event ranking")
        axes[1].grid(axis="y", color="#D9E2E8", linewidth=0.7)
        axes[1].legend(frameon=False)
        figure.suptitle("Classroom example: estimates and diagnostic output", fontsize=14)

        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=300, facecolor="white")
    finally:
        plt.close(figure)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", quotechar='"')


def _adjacency_matrix(
    edges: pd.DataFrame,
    *,
    actors: list[int],
    sender_col: str,
    receiver_col: str,
    symmetrize: bool,
) -> np.ndarray:
    index = {actor: pos for pos, actor in enumerate(actors)}
    matrix = np.zeros((len(actors), len(actors)), dtype=float)
    for row in edges.itertuples(index=False):
        sender = int(getattr(row, sender_col))
        receiver = int(getattr(row, receiver_col))
        if sender not in index or receiver not in index:
            continue
        matrix[index[sender], index[receiver]] = 1.0
        if symmetrize:
            matrix[index[receiver], index[sender]] = 1.0
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _actor_lookup(attributes: pd.DataFrame, column: str) -> dict[int, float]:
    return dict(zip(attributes["id"].astype(int), attributes[column].astype(float), strict=True))


def _term_values(
    term: str,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    attributes: pd.DataFrame,
    seating: np.ndarray,
    friendship: np.ndarray,
) -> np.ndarray:
    sender_ids = riskset["sender_id"].astype(int).to_numpy()
    receiver_ids = riskset["receiver_id"].astype(int).to_numpy()
    if term == "Intercept":
        return np.ones(len(riskset), dtype=float)
    if term.startswith("Sender_"):
        values = _actor_lookup(attributes, term.removeprefix("Sender_"))
        return np.array([values[int(actor)] for actor in sender_ids], dtype=float)
    if term.startswith("Receiver_"):
        values = _actor_lookup(attributes, term.removeprefix("Receiver_"))
        return np.array([values[int(actor)] for actor in receiver_ids], dtype=float)
    if term == "Seating":
        return seating[sender_ids - 1, receiver_ids - 1]
    if term == "Friendship":
        return friendship[sender_ids - 1, receiver_ids - 1]
    if term == "Recency_ij":
        return np.array(
            [
                ((previous["sender_id"] == sender) & (previous["receiver_id"] == receiver)).sum()
                for sender, receiver in zip(sender_ids, receiver_ids, strict=True)
            ],
            dtype=float,
        )
    if term == "Recency_ji":
        return np.array(
            [
                ((previous["sender_id"] == receiver) & (previous["receiver_id"] == sender)).sum()
                for sender, receiver in zip(sender_ids, receiver_ids, strict=True)
            ],
            dtype=float,
        )
    if term.startswith("PSAB_"):
        return _pshift(term, sender_ids, receiver_ids, previous)
    raise ValueError(f"unsupported classroom appendix term: {term}")


def _pshift(
    term: str, sender_ids: np.ndarray, receiver_ids: np.ndarray, previous: pd.DataFrame
) -> np.ndarray:
    if previous.empty:
        return np.zeros(len(sender_ids), dtype=float)
    last = previous.iloc[-1]
    a = int(last["sender_id"])
    b = int(last["receiver_id"])
    if term == "PSAB_BA":
        mask = (sender_ids == b) & (receiver_ids == a)
    elif term == "PSAB_BY":
        mask = (sender_ids == b) & (receiver_ids != a)
    elif term == "PSAB_AY":
        mask = (sender_ids == a) & (receiver_ids != b)
    elif term == "PSAB_XA":
        mask = (sender_ids != a) & (sender_ids != b) & (receiver_ids == a)
    elif term == "PSAB_XB":
        mask = (sender_ids != a) & (sender_ids != b) & (receiver_ids == b)
    else:
        raise ValueError(f"unsupported p-shift term: {term}")
    return mask.astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", choices=("ordinal", "exact"), default="ordinal")
    parser.add_argument("--backend", default="numpy")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="classroom_diagnostics.png",
        metavar="PATH",
        help="save the coefficient and ranking figure (default: classroom_diagnostics.png)",
    )
    args = parser.parse_args()
    plot_path = Path(args.plot) if args.plot is not None else None
    full_models: dict[str, tuple[pd.DataFrame, Diagnostics]] = {}
    summary = run_appendix(
        ordinal=args.timing == "ordinal",
        backend=args.backend,
        plot_path=plot_path,
        full_model_outputs=full_models,
    )
    print(summary.to_string(index=False))
    for date_label, (coefficients, report) in full_models.items():
        print(f"\n{date_label} full-model coefficients:")
        print(coefficients.to_string(index=False))
        print(f"\n{date_label} in-sample diagnostics:")
        print(diagnostic_table(report, digits=None).to_string(index=False))
    if plot_path is not None:
        print(f"Saved classroom plot to {plot_path}")


if __name__ == "__main__":
    main()
