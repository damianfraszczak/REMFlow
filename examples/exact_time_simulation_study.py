"""Run a reproducible de novo exact-time simulation study.

Source provenance and the comparison boundary are documented in this module
and the project README. The simulation uses REMFlow statistics and
an explicit NumPy random generator before re-estimating the generating model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from remflow import Diagnostics, RemEstimate, diagnostics, remify, remstats, remstimate


@dataclass(frozen=True)
class SimulationStudy:
    """Inputs and outputs of the simulation study."""

    events: pd.DataFrame
    fit: RemEstimate
    diagnostics: Diagnostics
    truth: dict[str, float]


def simulate_study_process(
    *,
    actors: int = 25,
    events: int = 100,
    seed: int = 1331,
    baseline: float = 0.25,
    reciprocity: float = -1.0,
    sender_recency: float = 4.0,
) -> pd.DataFrame:
    """Simulate the documented three-effect exact-time process.

    The constant sender covariate in the source study is the exact-time
    baseline here. ``psABBA`` represents AB-BA, and ``recencySendSender``
    represents recent activity of the candidate sender.
    """

    if actors < 2:
        raise ValueError("actors must be at least 2")
    if events < 1:
        raise ValueError("events must be positive")

    rng = np.random.default_rng(seed)
    sender = np.repeat(np.arange(1, actors + 1, dtype=int), actors)
    receiver = np.tile(np.arange(1, actors + 1, dtype=int), actors)
    valid = sender != receiver
    sender = sender[valid]
    receiver = receiver[valid]

    rows: list[dict[str, float | int]] = []
    current_time = 0.0
    for _ in range(events):
        reverse = np.zeros(len(sender), dtype=float)
        if rows:
            last_sender = int(rows[-1]["sender"])
            last_receiver = int(rows[-1]["receiver"])
            reverse = ((sender == last_receiver) & (receiver == last_sender)).astype(float)

        recency_by_sender = np.zeros(actors + 1, dtype=float)
        for actor in range(1, actors + 1):
            for distance, prior in enumerate(reversed(rows), start=1):
                if int(prior["sender"]) == actor:
                    recency_by_sender[actor] = 1.0 / (distance + 1.0)
                    break

        linear = baseline + reciprocity * reverse + sender_recency * recency_by_sender[sender]
        rates = np.exp(linear)
        current_time += float(rng.exponential(1.0 / rates.sum()))
        selected = int(rng.choice(len(rates), p=rates / rates.sum()))
        rows.append(
            {
                "time": current_time,
                "sender": int(sender[selected]),
                "receiver": int(receiver[selected]),
            }
        )

    return pd.DataFrame(rows)


def run_study(
    *, actors: int = 25, events: int = 100, seed: int = 1331, backend: str = "numpy"
) -> SimulationStudy:
    """Simulate and re-estimate the exact-time model."""

    truth = {"baseline": 0.25, "psABBA": -1.0, "recencySendSender": 4.0}
    event_table = simulate_study_process(
        actors=actors,
        events=events,
        seed=seed,
        baseline=truth["baseline"],
        reciprocity=truth["psABBA"],
        sender_recency=truth["recencySendSender"],
    )
    history = remify(
        event_table,
        actors=list(range(1, actors + 1)),
        riskset="full",
        ordinal=False,
    )
    statistics = remstats(
        history,
        tie_effects="~ psABBA() + recencySendSender()",
        first=1,
    )
    fit = remstimate(history, statistics, backend=backend)
    return SimulationStudy(event_table, fit, diagnostics(fit, history, statistics), truth)


def result_table(study: SimulationStudy) -> pd.DataFrame:
    """Return parameter truth, estimate, and uncertainty as a tidy table."""

    standard_errors = (
        np.full(len(study.fit.names), np.nan)
        if study.fit.se is None
        else np.asarray(study.fit.se, dtype=float)
    )
    return pd.DataFrame(
        {
            "parameter": study.fit.names,
            "truth": [study.truth[name] for name in study.fit.names],
            "estimate": study.fit.coef,
            "standard_error": standard_errors,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actors", type=int, default=25)
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1331)
    parser.add_argument("--backend", default="numpy")
    args = parser.parse_args()

    study = run_study(
        actors=args.actors,
        events=args.events,
        seed=args.seed,
        backend=args.backend,
    )
    print(result_table(study).to_string(index=False))
    print(
        f"events={len(study.events)} converged={study.fit.converged} "
        f"logLik={study.fit.log_likelihood:.6f} BIC={study.fit.BIC:.6f} "
        f"mean_rank={float(np.mean(study.diagnostics.ranks)):.6f}"
    )


if __name__ == "__main__":
    main()
