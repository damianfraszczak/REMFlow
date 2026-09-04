from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from remflow import remify, tomstats

DIRECTED_BASE_FORMULAS = [
    "~ inertia() + reciprocity()",
    "~ indegreeSender() + outdegreeSender() + indegreeReceiver() + outdegreeReceiver()",
    "~ otp() + itp() + isp() + osp()",
    "~ recencySendReceiver() + recencyReceiveReceiver() + recencyContinue()",
    "~ psABBA() + psABXY() + psABAY()",
    '~ send("extraversion") + receive("extraversion")',
    '~ same("sex") + difference("age")',
    '~ average("extraversion") + minimum("age") + maximum("age")',
]

DIRECTED_PROPORTIONAL_FORMULAS = [
    '~ inertia(scaling="prop") + reciprocity(scaling="prop")',
    '~ indegreeSender(scaling="prop") + outdegreeSender(scaling="prop") + '
    'indegreeReceiver(scaling="prop") + outdegreeReceiver(scaling="prop")',
]

UNDIRECTED_BASE_FORMULAS = [
    "~ inertia()",
    "~ totaldegreeDyad() + degreeMin() + degreeMax() + degreeDiff()",
    "~ sp()",
    "~ recencyContinue()",
    "~ psABAY() + psABAB()",
    '~ same("sex") + difference("age") + average("extraversion") + '
    'minimum("age") + maximum("age")',
]

DIRECTED_FORMULAS = [
    "~ inertia(consider_type=True)",
    "~ inertia(consider_type=False)",
    "~ reciprocity(consider_type=True)",
    "~ reciprocity(consider_type=False)",
    "~ indegreeSender(consider_type=True) + outdegreeSender(consider_type=True) + "
    "indegreeReceiver(consider_type=True) + outdegreeReceiver(consider_type=True)",
    "~ indegreeSender(consider_type=False) + outdegreeSender(consider_type=False) + "
    "indegreeReceiver(consider_type=False) + outdegreeReceiver(consider_type=False)",
    '~ indegreeSender(scaling="prop", consider_type=True) + '
    'outdegreeSender(scaling="prop", consider_type=True)',
    "~ otp(consider_type=True) + itp(consider_type=True) + "
    "isp(consider_type=True) + osp(consider_type=True)",
    "~ otp(consider_type=False) + itp(consider_type=False) + "
    "isp(consider_type=False) + osp(consider_type=False)",
    "~ psABBA(consider_type=True) + psABXY(consider_type=True) + "
    "psABAY(consider_type=True)",
    "~ recencySendReceiver(consider_type=True) + "
    "recencyReceiveReceiver(consider_type=True) + recencyContinue(consider_type=True)",
    "~ rrankSend(consider_type=True) + rrankReceive(consider_type=True)",
    '~ send("extraversion") + receive("extraversion") + '
    'same("sex") + difference("age")',
]

UNDIRECTED_FORMULAS = [
    "~ inertia(consider_type=True)",
    "~ inertia(consider_type=False)",
    "~ totaldegreeDyad(consider_type=True) + degreeMin(consider_type=True) + "
    "degreeMax(consider_type=True) + degreeDiff(consider_type=True)",
    "~ totaldegreeDyad(consider_type=False) + degreeMin(consider_type=False) + "
    "degreeMax(consider_type=False) + degreeDiff(consider_type=False)",
    "~ sp(consider_type=True)",
    "~ sp(consider_type=False)",
    "~ psABAY(consider_type=True) + psABAB(consider_type=True)",
    "~ recencyContinue(consider_type=True)",
    '~ same("sex") + difference("age") + average("extraversion") + '
    'minimum("age") + maximum("age")',
]


def actor_attributes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [1, 2, 3, 4],
            "time": [0, 0, 0, 0],
            "extraversion": [0.2, 0.7, -0.1, 1.1],
            "age": [20, 31, 27, 45],
            "sex": ["x", "y", "x", "y"],
        }
    )


def typed_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [1, 2, 4, 4, 8, 11, 15, 16, 22, 25, 31, 34, 40, 47],
            "actor1": [1, 2, 1, 3, 4, 2, 3, 1, 4, 2, 3, 1, 4, 2],
            "actor2": [2, 1, 3, 1, 1, 4, 2, 4, 3, 3, 4, 2, 1, 4],
            "type": ["social", "work", "social", "work", "work", "social",
                     "work", "social", "social", "work", "social", "work",
                     "social", "work"],
        }
    )


def make_history(
    *, riskset: str, directed: bool, ordinal: bool, extend_riskset_by_type: bool
):
    events = typed_events()
    kwargs: dict[str, Any] = {}
    if riskset == "manual":
        kwargs["manual_riskset"] = events[["actor1", "actor2"]]
    return remify(
        events,
        actors=[1, 2, 3, 4],
        model="tie",
        riskset=riskset,
        directed=directed,
        ordinal=ordinal,
        extend_riskset_by_type=extend_riskset_by_type,
        **kwargs,
    )


def assert_sample_matches_full(
    history,
    formula: Any,
    *,
    memory: str,
    memory_value: float | None,
    sample_size: int = 5,
    attr_dyads: pd.DataFrame | np.ndarray | None = None,
) -> None:
    common = {
        "reh": history,
        "attr_actors": actor_attributes(),
        "memory": memory,
        "memory_value": memory_value,
        "first": 1,
        "attr_dyads": attr_dyads,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        full = tomstats(formula, sampling=False, **common)
        first = tomstats(formula, sampling=True, samp_num=sample_size, seed=1, **common)
        repeated = tomstats(
            formula,
            sampling=True,
            samp_num=sample_size,
            seed=1,
            **common,
        )
        second = tomstats(formula, sampling=True, samp_num=sample_size, seed=42, **common)

    assert first.names == full.names == repeated.names == second.names
    assert first.sample_map and second.sample_map
    for left, right in zip(first.sample_map, repeated.sample_map, strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(first.stats, repeated.stats, strict=True):
        np.testing.assert_array_equal(left, right)
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(first.sample_map, second.sample_map, strict=True)
    )
    for sampled in (first, second):
        for event, indexes in enumerate(sampled.sample_map):
            zero_based = indexes - 1
            np.testing.assert_allclose(
                sampled.stats[event], full.stats[event][zero_based], rtol=0, atol=0
            )
            assert set(sampled.observed_index_groups[event]).issubset(
                set(range(len(indexes)))
            )

    if history.extend_riskset_by_type and "interact" in formula:
        for event, indexes in enumerate(first.sample_map):
            sampled_riskset = history.risksets[first.event_indices[event]].iloc[indexes - 1]
            for column, name in enumerate(first.names):
                if name.count(".") < 2:
                    continue
                candidate_type = name.rsplit(".", 1)[-1]
                mismatched = sampled_riskset["event_type"].to_numpy() != candidate_type
                np.testing.assert_array_equal(first.stats[event][mismatched, column], 0.0)
