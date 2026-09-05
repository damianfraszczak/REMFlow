"""Benchmark REMFlow backends on synthetic relational event histories."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from remflow import BackendUnavailable, available_backends, remify, remstats, remstimate


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    actors: int
    events: int
    effects: str
    riskset: str
    history_seconds: float
    stats_seconds: float
    estimate_seconds: float
    total_seconds: float
    converged: bool
    log_likelihood: float
    metadata: dict[str, Any]
    platform: dict[str, Any]
    backend_inventory: dict[str, Any]


def synthetic_events(actor_count: int, event_count: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    actors = np.array([f"A{i}" for i in range(actor_count)])
    senders = rng.integers(0, actor_count, size=event_count)
    receivers = rng.integers(0, actor_count - 1, size=event_count)
    receivers = np.where(receivers >= senders, receivers + 1, receivers)
    return pd.DataFrame(
        {
            "time": np.arange(1, event_count + 1),
            "sender": actors[senders],
            "receiver": actors[receivers],
        }
    )


def run(args: argparse.Namespace) -> BenchmarkResult:
    events = synthetic_events(args.actors, args.events, args.seed)
    actors = [f"A{i}" for i in range(args.actors)]

    start = time.perf_counter()
    history = remify(events, actors=actors, riskset=args.riskset, ordinal=True)
    after_history = time.perf_counter()
    stats = remstats(history, tie_effects=args.effects, first=args.first)
    after_stats = time.perf_counter()
    fit = remstimate(history, stats, backend=args.backend)
    after_fit = time.perf_counter()

    return BenchmarkResult(
        backend=args.backend,
        actors=args.actors,
        events=args.events,
        effects=args.effects,
        riskset=args.riskset,
        history_seconds=after_history - start,
        stats_seconds=after_stats - after_history,
        estimate_seconds=after_fit - after_stats,
        total_seconds=after_fit - start,
        converged=fit.converged,
        log_likelihood=fit.log_likelihood,
        metadata=fit.metadata,
        platform={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        backend_inventory=available_backends(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="numpy", choices=["numpy", "jax", "jax:cpu", "jax:gpu"]
    )
    parser.add_argument("--actors", type=int, default=25)
    parser.add_argument("--events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--riskset", default="full", choices=["full", "active", "active_saturated"])
    parser.add_argument("--effects", default="~ inertia() + reciprocity() + send() + receive()")
    parser.add_argument("--first", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run(args)
    except BackendUnavailable as exc:
        print(
            json.dumps({"backend": args.backend, "available": False, "error": str(exc)}, indent=2)
        )
        return 2
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
