"""Statistic formulas and kernels for relational event models."""

from __future__ import annotations

import ast
import io
import json
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from remflow.history import EventHistory


@dataclass(frozen=True)
class Effect:
    name: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    components: tuple[Effect, ...] = ()

    def __add__(self, other: Effect | Formula) -> Formula:
        return Formula((self,)) + other

    def __mul__(self, other: Effect | Formula) -> Formula:
        return Formula((self,)) * other

    @property
    def statistic_name(self) -> str:
        if len(self.components) == 2:
            return ":".join(component.statistic_name for component in self.components)
        kwargs = dict(self.kwargs)
        if self.name in {"sp", "isp", "itp", "osp", "otp"} and kwargs.get("unique", False):
            return f"{self.name}.unique"
        covariate_effects = {
            "send",
            "receive",
            "same",
            "difference",
            "average",
            "minimum",
            "maximum",
            "tie",
            "dyad",
            "event",
        }
        if self.name in covariate_effects:
            variable = self.args[0] if self.args else kwargs.get("variable")
            if isinstance(variable, str):
                return f"{self.name}_{variable}"
        endogenous_options = {"scaling", "consider_type", "unique"}
        if not self.args and set(kwargs).issubset(endogenous_options):
            return self.name
        return self.canonical_call

    @property
    def canonical_call(self) -> str:
        if len(self.components) == 2:
            return ":".join(component.canonical_call for component in self.components)
        if not self.args and not self.kwargs:
            return self.name
        args = [repr(arg) for arg in self.args]
        hidden = {"attr_actors", "attr_dyads", "event_attr", "x"}
        args.extend(f"{key}={value!r}" for key, value in self.kwargs if key not in hidden)
        return f"{self.name}({', '.join(args)})"


@dataclass(frozen=True)
class Formula:
    terms: tuple[Effect, ...]
    intercept: bool | None = None

    def __add__(self, other: Effect | Formula) -> Formula:
        rhs = other.terms if isinstance(other, Formula) else (other,)
        return Formula((*self.terms, *rhs), self.intercept)

    def __mul__(self, other: Effect | Formula) -> Formula:
        rhs = other.terms if isinstance(other, Formula) else (other,)
        interactions = tuple(
            _interaction_effect(left, right) for left in self.terms for right in rhs
        )
        return Formula((*self.terms, *rhs, *interactions), self.intercept)

    def canonical(self) -> dict[str, Any]:
        return {"intercept": self.intercept, "terms": [term.canonical_call for term in self.terms]}


@dataclass(frozen=True)
class RemStats:
    history: EventHistory
    stats: list[np.ndarray]
    names: list[str]
    formula: Formula
    observed_indices: list[int]
    event_indices: list[int] = field(default_factory=list)
    observed_index_groups: list[list[int]] = field(default_factory=list)
    sample_map: list[np.ndarray] = field(default_factory=list)
    sampling_weights: list[np.ndarray] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "statistics": [np.array(values, copy=True) for values in self.stats],
            "names": list(self.names),
            "formula": self.formula.canonical(),
            "observed_indices": list(self.observed_indices),
            "event_indices": list(self.event_indices),
            "observed_index_groups": [list(group) for group in self.observed_index_groups],
            "sample_map": [np.array(indexes, copy=True) for indexes in self.sample_map],
            "sampling_weights": [np.array(weights, copy=True) for weights in self.sampling_weights],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "events": len(self.stats),
            "terms": list(self.names),
            "riskset_sizes": [int(values.shape[0]) for values in self.stats],
        }

    def plot(
        self,
        effect: str | int,
        *,
        subset: int | Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Return trajectory data for a selected statistic.

        Integer effects use a one-based index over non-baseline
        statistics. ``subset`` contains public risk-set IDs; by default the
        first five are retained.
        """

        name = self._resolve_effect(effect)
        frame = self._effect_frame(name)
        selected = self._normalize_plot_subset(
            subset,
            frame["risk_id"].drop_duplicates().to_list()[:5],
            "risk_id",
        )
        return {
            "data": frame[frame["risk_id"].isin(selected)].reset_index(drop=True),
            "effect": name,
            "by": "dyads",
        }

    def boxplot(
        self,
        effect: str | int,
        *,
        by: str = "timepoints",
        subset: int | Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Return grouped distribution data for a selected statistic."""

        if by not in {"timepoints", "dyads"}:
            raise ValueError("by must be 'timepoints' or 'dyads'")
        name = self._resolve_effect(effect)
        frame = self._effect_frame(name)
        column = "event_id" if by == "timepoints" else "risk_id"
        defaults = frame[column].drop_duplicates().to_list()[:20]
        selected = self._normalize_plot_subset(subset, defaults, column)
        return {
            "data": frame[frame[column].isin(selected)].reset_index(drop=True),
            "effect": name,
            "by": by,
        }

    def _resolve_effect(self, effect: str | int) -> str:
        available = [name for name in self.names if name != "baseline"]
        if isinstance(effect, bool):
            raise TypeError("effect must be a statistic name or one-based integer")
        if isinstance(effect, int):
            if effect < 1 or effect > len(available):
                raise ValueError("effect index is outside the available statistics")
            return available[effect - 1]
        if not isinstance(effect, str):
            raise TypeError("effect must be a statistic name or one-based integer")
        if effect not in available:
            raise ValueError(f"effect {effect!r} is not present in this RemStats object")
        return effect

    def _effect_frame(self, name: str) -> pd.DataFrame:
        column = self.names.index(name)
        frames: list[pd.DataFrame] = []
        for output_index, (event_index, values) in enumerate(
            zip(self.event_indices, self.stats, strict=True)
        ):
            riskset = self.history.risksets[event_index]
            if self.sample_map:
                riskset = riskset.iloc[self.sample_map[output_index] - 1]
            risk_ids = (
                riskset["risk_id"].to_numpy(dtype=int)
                if "risk_id" in riskset
                else np.arange(1, len(values) + 1, dtype=int)
            )
            observed = set(
                self.observed_index_groups[output_index]
                if self.observed_index_groups
                else [self.observed_indices[output_index]]
            )
            frames.append(
                pd.DataFrame(
                    {
                        "event_id": event_index + 1,
                        "risk_id": risk_ids,
                        "value": values[:, column],
                        "observed": [index in observed for index in range(len(values))],
                    }
                )
            )
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["event_id", "risk_id", "value", "observed"])
        )

    @staticmethod
    def _normalize_plot_subset(
        subset: int | Sequence[int] | None,
        default: list[int],
        label: str,
    ) -> list[int]:
        if subset is None:
            return [int(value) for value in default]
        if isinstance(subset, bool):
            raise TypeError(f"{label} subset must contain positive integers")
        values = [subset] if isinstance(subset, int) else list(subset)
        if not values or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
            for value in values
        ):
            raise ValueError(f"{label} subset must contain positive integers")
        return [int(value) for value in values]

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize statistic tensors and their event history to JSON."""

        payload = {
            "schema": "remflow.remstats",
            "schema_version": 1,
            "class": type(self).__name__,
            "history": json.loads(self.history.to_json()),
            "statistics": [values.tolist() for values in self.stats],
            "names": self.names,
            "formula": self.formula.canonical(),
            "observed_indices": self.observed_indices,
            "event_indices": self.event_indices,
            "observed_index_groups": self.observed_index_groups,
            "sample_map": [values.tolist() for values in self.sample_map],
            "sampling_weights": [values.tolist() for values in self.sampling_weights],
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if path is not None:
            Path(path).write_text(serialized, encoding="utf-8")
        return serialized

    @classmethod
    def from_json(cls, value: str | Path) -> RemStats:
        """Restore statistic tensors produced by :meth:`to_json`."""

        payload = _read_json_payload(value, "remflow.remstats")
        history = EventHistory.from_json(json.dumps(payload["history"]))
        formula_value = payload["formula"]
        parsed = Formula(
            tuple(Effect(str(name)) for name in formula_value["terms"]),
            formula_value["intercept"],
        )
        result_class = TomStatsSampled if payload["sample_map"] else TomStats
        return result_class(
            history=history,
            stats=[np.asarray(values, dtype=float) for values in payload["statistics"]],
            names=[str(name) for name in payload["names"]],
            formula=parsed,
            observed_indices=[int(value) for value in payload["observed_indices"]],
            event_indices=[int(value) for value in payload["event_indices"]],
            observed_index_groups=[
                [int(value) for value in group] for group in payload["observed_index_groups"]
            ],
            sample_map=[np.asarray(values, dtype=int) for values in payload["sample_map"]],
            sampling_weights=[
                np.asarray(values, dtype=float) for values in payload["sampling_weights"]
            ],
        )


class TomStats(RemStats):
    """Tie-oriented statistic result."""


class TomStatsSampled(TomStats):
    """Tie-oriented statistic result with case-control sampled risk sets."""


@dataclass(frozen=True)
class AomStats:
    """Actor-oriented sender-rate and receiver-choice statistics."""

    history: EventHistory
    sender_stats: list[np.ndarray]
    receiver_stats: list[np.ndarray]
    sender_names: list[str]
    receiver_names: list[str]
    observed_sender_indices: list[int]
    observed_receiver_indices: list[int]
    receiver_masks: list[np.ndarray]
    event_indices: list[int]
    observed_sender_groups: list[list[int]] = field(default_factory=list)
    receiver_choice_stats: list[np.ndarray] = field(default_factory=list, repr=False)
    receiver_choice_observed_indices: list[int] = field(default_factory=list)
    receiver_choice_masks: list[np.ndarray] = field(default_factory=list, repr=False)
    receiver_choice_event_indices: list[int] = field(default_factory=list)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "sender_stats": [np.array(values, copy=True) for values in self.sender_stats],
            "receiver_stats": [np.array(values, copy=True) for values in self.receiver_stats],
            "sender_names": list(self.sender_names),
            "receiver_names": list(self.receiver_names),
            "observed_sender_indices": list(self.observed_sender_indices),
            "observed_receiver_indices": list(self.observed_receiver_indices),
            "receiver_masks": [np.array(mask, copy=True) for mask in self.receiver_masks],
            "event_indices": list(self.event_indices),
            "observed_sender_groups": [list(group) for group in self.observed_sender_groups],
            "receiver_choice_stats": [
                np.array(values, copy=True) for values in self.receiver_choice_stats
            ],
            "receiver_choice_observed_indices": list(
                self.receiver_choice_observed_indices
            ),
            "receiver_choice_masks": [
                np.array(mask, copy=True) for mask in self.receiver_choice_masks
            ],
            "receiver_choice_event_indices": list(self.receiver_choice_event_indices),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "events": len(self.event_indices),
            "sender_terms": list(self.sender_names),
            "receiver_terms": list(self.receiver_names),
            "sender_riskset_size": len(self.history.sender_riskset),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        payload = {
            "schema": "remflow.aomstats",
            "schema_version": 1,
            "history": json.loads(self.history.to_json()),
            "sender_stats": [values.tolist() for values in self.sender_stats],
            "receiver_stats": [values.tolist() for values in self.receiver_stats],
            "sender_names": self.sender_names,
            "receiver_names": self.receiver_names,
            "observed_sender_indices": self.observed_sender_indices,
            "observed_receiver_indices": self.observed_receiver_indices,
            "receiver_masks": [mask.tolist() for mask in self.receiver_masks],
            "event_indices": self.event_indices,
            "observed_sender_groups": self.observed_sender_groups,
            "receiver_choice_stats": [
                values.tolist() for values in self.receiver_choice_stats
            ],
            "receiver_choice_observed_indices": self.receiver_choice_observed_indices,
            "receiver_choice_masks": [
                mask.tolist() for mask in self.receiver_choice_masks
            ],
            "receiver_choice_event_indices": self.receiver_choice_event_indices,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if path is not None:
            Path(path).write_text(serialized, encoding="utf-8")
        return serialized

    @classmethod
    def from_json(cls, value: str | Path) -> AomStats:
        payload = _read_json_payload(value, "remflow.aomstats")
        return cls(
            history=EventHistory.from_json(json.dumps(payload["history"])),
            sender_stats=[np.asarray(values, dtype=float) for values in payload["sender_stats"]],
            receiver_stats=[
                np.asarray(values, dtype=float) for values in payload["receiver_stats"]
            ],
            sender_names=[str(name) for name in payload["sender_names"]],
            receiver_names=[str(name) for name in payload["receiver_names"]],
            observed_sender_indices=[int(value) for value in payload["observed_sender_indices"]],
            observed_receiver_indices=[
                int(value) for value in payload["observed_receiver_indices"]
            ],
            receiver_masks=[np.asarray(values, dtype=bool) for values in payload["receiver_masks"]],
            event_indices=[int(value) for value in payload["event_indices"]],
            observed_sender_groups=[
                [int(value) for value in group]
                for group in payload.get("observed_sender_groups", [])
            ],
            receiver_choice_stats=[
                np.asarray(values, dtype=float)
                for values in payload.get("receiver_choice_stats", [])
            ],
            receiver_choice_observed_indices=[
                int(value)
                for value in payload.get("receiver_choice_observed_indices", [])
            ],
            receiver_choice_masks=[
                np.asarray(values, dtype=bool)
                for values in payload.get("receiver_choice_masks", [])
            ],
            receiver_choice_event_indices=[
                int(value) for value in payload.get("receiver_choice_event_indices", [])
            ],
        )


@dataclass(frozen=True)
class AomStatsStacked:
    """Long-form sender-rate and receiver-choice actor model matrices."""

    sender_stack: pd.DataFrame | None
    receiver_stack: pd.DataFrame | None
    subset: tuple[int, int]
    E: int
    ordinal: bool

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sender_stack": None if self.sender_stack is None else self.sender_stack.copy(),
            "receiver_stack": (
                None if self.receiver_stack is None else self.receiver_stack.copy()
            ),
            "subset": list(self.subset),
            "E": self.E,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True)
class RemStatsStackedDuration:
    """Fit-ready start/end design for a duration relational event model."""

    remstats_stack: pd.DataFrame
    subset: tuple[int, int]
    D_start: int
    D_end: int
    E: int
    stat_names: list[str]
    stat_names_start: list[str]
    stat_names_end: list[str]
    ordinal: bool = False
    model: str = "durem"

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "remstats_stack": self.remstats_stack.copy(),
            "subset": list(self.subset),
            "D_start": self.D_start,
            "D_end": self.D_end,
            "E": self.E,
            "ordinal": self.ordinal,
            "model": self.model,
            "stat_names": list(self.stat_names),
            "stat_names_start": list(self.stat_names_start),
            "stat_names_end": list(self.stat_names_end),
        }


@dataclass(frozen=True)
class RemStatsDuration:
    """Duration REM statistics with separate start and end processes."""

    history: EventHistory
    stacked: RemStatsStackedDuration
    start_formula: Formula | None
    end_formula: Formula | None
    psi_start: float
    psi_end: float

    @property
    def start_stats(self) -> None:
        """Raw arrays are intentionally discarded after duration stacking."""

        return None

    @property
    def end_stats(self) -> None:
        """Raw arrays are intentionally discarded after duration stacking."""

        return None

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "stacked": self.stacked,
            "start_stats": None,
            "end_stats": None,
            "psi_start": self.psi_start,
            "psi_end": self.psi_end,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "events": self.stacked.E,
            "start_terms": list(self.stacked.stat_names_start),
            "end_terms": list(self.stacked.stat_names_end),
            "psi_start": self.psi_start,
            "psi_end": self.psi_end,
        }

    def __str__(self) -> str:
        return (
            "Duration relational-event statistics\n"
            f"time points: {self.stacked.E}\n"
            f"start terms: {len(self.stacked.stat_names_start)}\n"
            f"end terms: {len(self.stacked.stat_names_end)}"
        )

    def to_json(self, path: str | Path | None = None) -> str:
        payload = {
            "schema": "remflow.remstats_duration",
            "schema_version": 1,
            "history": json.loads(self.history.to_json()),
            "stacked": json.loads(
                self.stacked.remstats_stack.to_json(orient="table", date_format="iso", index=False)
            ),
            "subset": list(self.stacked.subset),
            "D_start": self.stacked.D_start,
            "D_end": self.stacked.D_end,
            "E": self.stacked.E,
            "ordinal": self.stacked.ordinal,
            "model": self.stacked.model,
            "stat_names": self.stacked.stat_names,
            "stat_names_start": self.stacked.stat_names_start,
            "stat_names_end": self.stacked.stat_names_end,
            "start_formula": (
                None if self.start_formula is None else self.start_formula.canonical()
            ),
            "end_formula": (None if self.end_formula is None else self.end_formula.canonical()),
            "psi_start": self.psi_start,
            "psi_end": self.psi_end,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if path is not None:
            Path(path).write_text(serialized, encoding="utf-8")
        return serialized

    @classmethod
    def from_json(cls, value: str | Path) -> RemStatsDuration:
        payload = _read_json_payload(value, "remflow.remstats_duration")
        history = EventHistory.from_json(json.dumps(payload["history"]))
        frame = pd.read_json(io.StringIO(json.dumps(payload["stacked"])), orient="table")
        stacked = RemStatsStackedDuration(
            remstats_stack=frame,
            subset=(int(payload["subset"][0]), int(payload["subset"][1])),
            D_start=int(payload["D_start"]),
            D_end=int(payload["D_end"]),
            E=int(payload["E"]),
            stat_names=[str(name) for name in payload["stat_names"]],
            stat_names_start=[str(name) for name in payload["stat_names_start"]],
            stat_names_end=[str(name) for name in payload["stat_names_end"]],
            ordinal=bool(payload.get("ordinal", history.ordinal)),
            model=str(payload.get("model", "durem")),
        )
        return cls(
            history=history,
            stacked=stacked,
            start_formula=_formula_from_canonical(payload["start_formula"]),
            end_formula=_formula_from_canonical(payload["end_formula"]),
            psi_start=float(payload["psi_start"]),
            psi_end=float(payload["psi_end"]),
        )


@dataclass(frozen=True)
class RemStatsStacked:
    """Long-form model matrix produced by :func:`stack_stats`."""

    remstats_stack: pd.DataFrame
    subset: tuple[int, int]
    D: int
    E: int
    ordinal: bool
    S: int | None = None

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "remstats_stack": self.remstats_stack.copy(),
            "subset": list(self.subset),
            "E": self.E,
            "ordinal": self.ordinal,
        }
        if self.S is None:
            result["D"] = self.D
        else:
            result["S"] = self.S
        return result


_EFFECT_NAMES = [
    "FEtype",
    "inertia",
    "reciprocity",
    "send",
    "receive",
    "tie",
    "dyad",
    "event",
    "userStat",
    "indegreeReceiver",
    "indegreeSender",
    "outdegreeReceiver",
    "outdegreeSender",
    "totaldegreeDyad",
    "totaldegreeReceiver",
    "totaldegreeSender",
    "degreeDiff",
    "degreeMax",
    "degreeMin",
    "same",
    "difference",
    "average",
    "minimum",
    "maximum",
    "isp",
    "itp",
    "osp",
    "otp",
    "sp",
    "spUnique",
    "psABA",
    "psABAB",
    "psABAY",
    "psABB",
    "psABBA",
    "psABBY",
    "psABX",
    "psABXA",
    "psABXB",
    "psABXY",
    "recencyContinue",
    "recencyReceiveReceiver",
    "recencyReceiveSender",
    "recencySendReceiver",
    "recencySendSender",
    "rrankReceive",
    "rrankSend",
    "activeTie",
    "activeReciprocalTie",
    "activeDegreeDyad",
    "activeDegreeMin",
    "activeDegreeMax",
    "activeIndegreeReceiver",
    "activeOutdegreeSender",
    "activeTotaldegreeDyad",
    "activeTotaldegreeReceiver",
    "activeTotaldegreeSender",
    "activeSharedPartners",
    "activeSharedPartners_isp",
    "activeSharedPartners_itp",
    "activeSharedPartners_osp",
    "activeSharedPartners_otp",
]


def formula(value: str | Effect | Formula) -> Formula:
    if isinstance(value, Formula):
        return value
    if isinstance(value, Effect):
        return Formula((value,))
    if not isinstance(value, str):
        raise TypeError("formula must be a string, Effect, or Formula")
    return _parse_formula(value)


def remstats(
    history: EventHistory,
    tie_effects: str | Effect | Formula | None = None,
    *,
    sender_effects: str | Effect | Formula | None = None,
    receiver_effects: str | Effect | Formula | None = None,
    start_effects: str | Effect | Formula | None = None,
    end_effects: str | Effect | Formula | None = None,
    memory: Sequence[str] | str = ("full", "window", "decay", "interval"),
    memory_value: float | Sequence[float] | None = None,
    psi_start: float = 1,
    psi_end: float = 1,
    first: int = 2,
    last: float = float("inf"),
    display_progress: bool = False,
    sampling: bool = False,
    samp_num: int = 10,
    seed: int | None = None,
    attr_actors: pd.DataFrame | None = None,
    attr_dyads: pd.DataFrame | np.ndarray | None = None,
) -> RemStats | AomStats | RemStatsDuration:
    """Compute risk-set statistics for a relational event history."""

    if not isinstance(history, EventHistory):
        raise TypeError("history must be an object of class EventHistory returned by remify")
    if not isinstance(first, int) or first < 1:
        raise ValueError("first must be 1 or a larger integer")
    if last != float("inf") and (not isinstance(last, Real) or int(last) < first):
        raise ValueError("last cannot be smaller than first")
    memory_mode = _match_arg(memory, ("full", "window", "decay", "interval"), "memory")
    memory_parameter = _validate_memory(memory_mode, memory_value)
    if attr_actors is not None:
        warnings.warn(
            "'attr_actors' is deprecated. Supply attributes directly in the stat functions "
            "(e.g., send()). It is used only when the effect omits attr_actors.",
            DeprecationWarning,
            stacklevel=2,
        )
    if attr_dyads is not None:
        warnings.warn(
            "'attr_dyads' is deprecated. Supply attributes directly in the stat functions "
            "(e.g., tie()). It is used only when the effect omits attr_dyads.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not isinstance(display_progress, (bool, np.bool_)):
        raise TypeError("display_progress must be a boolean value")
    if display_progress:
        print("Calculating relational event statistics")
    if samp_num <= 0:
        raise ValueError("samp_num must be a positive integer")
    if seed is not None and not isinstance(seed, int):
        raise TypeError("seed must be an integer or None")
    for name, value in (("psi_start", psi_start), ("psi_end", psi_end)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"{name} must be a finite non-negative number")
    if history.duration:
        if tie_effects is not None:
            raise ValueError(
                "Use start_effects and/or end_effects for duration histories, not tie_effects"
            )
        if sender_effects is not None or receiver_effects is not None:
            raise NotImplementedError(
                "sender_effects and receiver_effects are not yet supported for duration "
                "histories; use start_effects and/or end_effects"
            )
        if start_effects is None and end_effects is None:
            raise ValueError("at least one of start_effects or end_effects is required")
        if sampling:
            warnings.warn(
                "sampling is not supported for duration statistics and is ignored",
                UserWarning,
                stacklevel=2,
            )
        return _duration_remstats(
            history,
            start_effects=start_effects,
            end_effects=end_effects,
            psi_start=float(psi_start),
            psi_end=float(psi_end),
            first=first,
            last=last,
            attr_actors=attr_actors,
            attr_dyads=attr_dyads,
        )
    if start_effects is not None or end_effects is not None:
        raise ValueError("start_effects and end_effects require a duration history")
    if history.model == "actor":
        if tie_effects is not None:
            raise ValueError("tie_effects cannot be used for an actor-oriented history")
        if sender_effects is None and receiver_effects is None:
            raise ValueError("sender_effects or receiver_effects is required for an actor model")
        if sampling:
            raise NotImplementedError("actor-oriented case-control sampling is not implemented yet")
        return _actor_remstats(
            history,
            sender_effects=sender_effects,
            receiver_effects=receiver_effects,
            memory=memory_mode,
            memory_value=memory_parameter,
            first=first,
            last=last,
            attr_actors=attr_actors,
            attr_dyads=attr_dyads,
        )
    if sender_effects is not None or receiver_effects is not None:
        raise ValueError("sender_effects and receiver_effects require model='actor'")
    if not history.risksets:
        raise ValueError("history has no attached risk sets; call remify(..., attach_riskset=True)")
    parsed = formula(tie_effects or Formula(()))
    if attr_actors is not None:
        _validate_deprecated_actor_attributes(parsed, attr_actors, history)
    parsed = _attach_deprecated_attributes(parsed, attr_actors, attr_dyads)
    effect_names = [
        name for effect in parsed.terms for name in _effect_statistic_names(effect, history)
    ]
    stop = None if last == float("inf") else int(last)
    time_groups = [
        [int(index) for index in indexes]
        for indexes in history.events.groupby("time", sort=False, dropna=False).groups.values()
    ]
    selected_groups = time_groups[max(first - 1, 0) : stop]
    stats: list[np.ndarray] = []
    observed: list[int] = []
    event_indices: list[int] = []
    observed_groups: list[list[int]] = []
    sample_map: list[np.ndarray] = []
    sampling_weights: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    for group in selected_groups:
        event_index = group[0]
        matrix = _stats_for_event(history, event_index, parsed.terms, memory_mode, memory_parameter)
        if parsed.intercept is not False:
            matrix = np.column_stack([np.ones(len(matrix), dtype=float), matrix])
        group_observed_full = [observed_risk_index(history, index) for index in group]
        if sampling:
            indexes, weights = _sample_riskset(len(matrix), group_observed_full, samp_num, rng)
            matrix = matrix[indexes]
            positions = {full_index: position for position, full_index in enumerate(indexes)}
            group_observed = [positions[index] for index in group_observed_full]
            sample_map.append(indexes.astype(int) + 1)
            sampling_weights.append(weights)
        else:
            group_observed = group_observed_full
        stats.append(matrix)
        observed.append(group_observed[0])
        observed_groups.append(group_observed)
        event_indices.append(event_index)
    result_class = TomStatsSampled if sampling else TomStats
    return result_class(
        history,
        stats,
        (["baseline"] if parsed.intercept is not False else []) + effect_names,
        parsed,
        observed,
        event_indices,
        observed_groups,
        sample_map,
        sampling_weights,
    )


def tomstats(
    effects: str | Effect | Formula | EventHistory | None = None,
    *,
    reh: EventHistory | None = None,
    **kwargs: Any,
) -> RemStats:
    """Tie-oriented model statistics.

    This is the public entry point for tie-oriented statistic construction.
    """

    if isinstance(effects, EventHistory):
        if reh is not None:
            raise TypeError("reh was supplied twice")
        reh = effects
        selected_effects: str | Effect | Formula | None = kwargs.pop("tie_effects", None)
    else:
        selected_effects = effects
    if reh is None:
        raise TypeError("tomstats requires reh")
    if "tie_effects" in kwargs:
        if selected_effects is not None:
            raise TypeError("effects and tie_effects cannot both be supplied")
        selected_effects = kwargs.pop("tie_effects")
    result = remstats(reh, tie_effects=selected_effects, **kwargs)
    if not isinstance(result, RemStats):
        raise ValueError("tomstats requires an EventHistory with model='tie'")
    return result


def aomstats(
    effects: str | Effect | Formula | EventHistory | None = None,
    *,
    reh: EventHistory | None = None,
    sender_effects: str | Effect | Formula | None = None,
    receiver_effects: str | Effect | Formula | None = None,
    **kwargs: Any,
) -> AomStats:
    """Compute actor-oriented sender-rate and receiver-choice statistics."""

    if isinstance(effects, EventHistory):
        if reh is not None:
            raise TypeError("reh was supplied twice")
        reh = effects
    elif effects is not None:
        if receiver_effects is not None:
            raise TypeError("effects and receiver_effects cannot both be supplied")
        receiver_effects = effects
    if reh is None:
        raise TypeError("aomstats requires reh")
    result = remstats(
        reh,
        sender_effects=sender_effects,
        receiver_effects=receiver_effects,
        **kwargs,
    )
    if not isinstance(result, AomStats):
        raise ValueError("aomstats requires an EventHistory with model='actor'")
    return result


def stack_stats(
    stats: RemStats | AomStats | RemStatsDuration,
    history: EventHistory | None = None,
    *,
    add_actors: bool = False,
) -> RemStatsStacked | AomStatsStacked | RemStatsStackedDuration:
    """Stack event-by-risk-set tensors into an estimation data frame."""

    if not isinstance(add_actors, (bool, np.bool_)):
        raise TypeError("add_actors must be a boolean value")

    if isinstance(stats, AomStats):
        reh = stats.history if history is None else history
        if reh is not stats.history:
            raise ValueError("history must be the EventHistory used to compute stats")
        return _stack_aomstats(stats, reh, add_actors=bool(add_actors))
    if isinstance(stats, RemStatsDuration):
        reh = stats.history if history is None else history
        if reh is not stats.history:
            raise ValueError("history must be the EventHistory used to compute stats")
        return stats.stacked
    if not isinstance(stats, RemStats):
        raise TypeError("stats must be a RemStats or RemStatsDuration object")
    reh = stats.history if history is None else history
    if reh is not stats.history:
        raise ValueError("history must be the EventHistory used to compute stats")
    if not stats.stats:
        empty = pd.DataFrame(columns=["time_index", "obs", "dyad", *stats.names])
        return RemStatsStacked(empty, (0, 0), 0, 0, reh.ordinal, 0 if stats.sample_map else None)
    risk_sizes = {matrix.shape[0] for matrix in stats.stats}
    if len(risk_sizes) != 1:
        raise ValueError("stack_stats requires a stable risk-set size")
    d = risk_sizes.pop()
    frames: list[pd.DataFrame] = []
    times = reh.events["time"].astype(float).to_numpy()
    observed_groups = stats.observed_index_groups or [[index] for index in stats.observed_indices]
    for position, (matrix, event_index) in enumerate(
        zip(stats.stats, stats.event_indices, strict=True)
    ):
        observed = np.bincount(observed_groups[position], minlength=d).astype(int)
        sampled = bool(stats.sample_map)
        data: dict[str, Any] = {
            "time_index": np.full(d, _time_point_position(reh, event_index), dtype=int),
            "obs": observed,
            "dyad": stats.sample_map[position] if sampled else np.arange(1, d + 1, dtype=int),
        }
        if sampled:
            data["weight"] = stats.sampling_weights[position]
        if not reh.ordinal:
            previous_time = 0.0 if event_index == 0 else float(times[event_index - 1])
            interevent = float(times[event_index] - previous_time)
            data["log_interevent"] = np.full(
                d, -np.inf if interevent == 0 else np.log(interevent), dtype=float
            )
        for column, name in enumerate(stats.names):
            data[name] = matrix[:, column]
        frames.append(pd.DataFrame(data))
    stacked = pd.concat(frames, ignore_index=True)
    subset = (
        _time_point_position(reh, stats.event_indices[0]),
        _time_point_position(reh, stats.event_indices[-1]),
    )
    full_d = len(reh.risksets[stats.event_indices[0]])
    return RemStatsStacked(
        stacked,
        subset,
        full_d,
        len(stats.stats),
        reh.ordinal,
        d if stats.sample_map else None,
    )


def _stack_aomstats(
    stats: AomStats,
    history: EventHistory,
    *,
    add_actors: bool,
) -> AomStatsStacked:
    if not stats.event_indices:
        return AomStatsStacked(None, None, (0, 0), 0, history.ordinal)
    times = history.events["time"].astype(float).to_numpy()
    sender_frames: list[pd.DataFrame] = []
    receiver_frames: list[pd.DataFrame] = []
    for position, event_index in enumerate(stats.event_indices):
        time_index = _time_point_position(history, event_index)
        if stats.sender_names:
            sender_matrix = stats.sender_stats[position]
            sender_data: dict[str, Any] = {
                "time_index": np.full(len(sender_matrix), time_index, dtype=int),
                "obs": np.bincount(
                    (
                        stats.observed_sender_groups[position]
                        if stats.observed_sender_groups
                        else [stats.observed_sender_indices[position]]
                    ),
                    minlength=len(sender_matrix),
                ).astype(int),
            }
            if add_actors:
                sender_data["actor"] = history.sender_riskset.astype(int)
            if not history.ordinal:
                previous_time = 0.0 if event_index == 0 else float(times[event_index - 1])
                interevent = float(times[event_index] - previous_time)
                sender_data["log_interevent"] = np.full(
                    len(sender_matrix),
                    -np.inf if interevent == 0 else np.log(interevent),
                    dtype=float,
                )
            for column, name in enumerate(stats.sender_names):
                sender_data[name] = sender_matrix[:, column]
            sender_frames.append(pd.DataFrame(sender_data))

    if stats.receiver_names:
        choice_stats = stats.receiver_choice_stats or stats.receiver_stats
        choice_observed = (
            stats.receiver_choice_observed_indices or stats.observed_receiver_indices
        )
        choice_masks = stats.receiver_choice_masks or stats.receiver_masks
        choice_event_indices = stats.receiver_choice_event_indices or stats.event_indices
        for choice_index, (matrix, observed, mask, event_index) in enumerate(
            zip(
                choice_stats,
                choice_observed,
                choice_masks,
                choice_event_indices,
                strict=True,
            ),
            start=1,
        ):
            receiver_matrix = matrix[mask]
            receiver_actor_ids = np.flatnonzero(mask) + 1
            receiver_data: dict[str, Any] = {
                "time_index": np.full(
                    len(receiver_matrix), _time_point_position(history, event_index), dtype=int
                ),
                "choice_index": np.full(len(receiver_matrix), choice_index, dtype=int),
                "obs": np.equal(receiver_actor_ids - 1, observed).astype(int),
            }
            if add_actors:
                receiver_data["actor"] = receiver_actor_ids.astype(int)
            for column, name in enumerate(stats.receiver_names):
                receiver_data[name] = receiver_matrix[:, column]
            receiver_frames.append(pd.DataFrame(receiver_data))

    subset = (
        _time_point_position(history, stats.event_indices[0]),
        _time_point_position(history, stats.event_indices[-1]),
    )
    return AomStatsStacked(
        sender_stack=(pd.concat(sender_frames, ignore_index=True) if sender_frames else None),
        receiver_stack=(
            pd.concat(receiver_frames, ignore_index=True) if receiver_frames else None
        ),
        subset=subset,
        E=len(stats.event_indices),
        ordinal=history.ordinal,
    )


def bind_remstats(*stats: RemStats) -> RemStats:
    """Combine compatible statistic objects, dropping duplicate terms."""

    if not stats or any(not isinstance(value, RemStats) for value in stats):
        raise TypeError("all objects must be of class RemStats")
    first = stats[0]
    for value in stats[1:]:
        if value.history is not first.history:
            raise ValueError("all RemStats objects must use the same EventHistory")
        if value.event_indices != first.event_indices:
            raise ValueError("all RemStats objects must cover the same event subset")
        if value.observed_index_groups != first.observed_index_groups:
            raise ValueError("all RemStats objects must have matching observed events")
        if [matrix.shape[0] for matrix in value.stats] != [
            matrix.shape[0] for matrix in first.stats
        ]:
            raise ValueError("all RemStats objects must have matching risk sets")
    names: list[str] = []
    sources: list[tuple[int, int]] = []
    duplicates: list[str] = []
    for object_index, value in enumerate(stats):
        for column, name in enumerate(value.names):
            if name in names:
                duplicates.append(name)
                continue
            names.append(name)
            sources.append((object_index, column))
    if duplicates:
        warnings.warn(
            f"duplicate statistics were retained once: {', '.join(dict.fromkeys(duplicates))}",
            UserWarning,
            stacklevel=2,
        )
    matrices = [
        np.column_stack(
            [stats[object_index].stats[event][:, column] for object_index, column in sources]
        )
        for event in range(len(first.stats))
    ]
    return RemStats(
        history=first.history,
        stats=matrices,
        names=names,
        formula=Formula(tuple(Effect(name) for name in names if name != "baseline")),
        observed_indices=list(first.observed_indices),
        event_indices=list(first.event_indices),
        observed_index_groups=[list(group) for group in first.observed_index_groups],
        sample_map=[np.array(indexes, copy=True) for indexes in first.sample_map],
        sampling_weights=[np.array(weights, copy=True) for weights in first.sampling_weights],
    )


def _time_point_position(history: EventHistory, event_index: int) -> int:
    unique_times = list(dict.fromkeys(history.events["time"].to_list()))
    return unique_times.index(history.events.iloc[event_index]["time"]) + 1


def _sample_riskset(
    riskset_size: int,
    observed: Sequence[int],
    sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    cases = np.asarray(list(dict.fromkeys(observed)), dtype=int)
    if len(cases) > sample_size:
        raise ValueError("samp_num cannot be smaller than the number of simultaneous cases")
    if sample_size >= riskset_size:
        indexes = np.arange(riskset_size, dtype=int)
    else:
        controls = np.setdiff1d(np.arange(riskset_size, dtype=int), cases, assume_unique=True)
        selected_controls = rng.choice(
            controls, size=sample_size - len(cases), replace=False
        ).astype(int)
        indexes = np.sort(np.concatenate([cases, selected_controls]))
    case_set = set(cases.tolist())
    control_count = len(indexes) - len(cases)
    available_controls = riskset_size - len(cases)
    control_weight = 1.0 if control_count == 0 else float(available_controls) / float(control_count)
    weights = np.asarray(
        [1.0 if int(index) in case_set else control_weight for index in indexes],
        dtype=float,
    )
    return indexes, weights


def _read_json_payload(value: str | Path, expected_schema: str) -> dict[str, Any]:
    source = str(value)
    if isinstance(value, Path) or (not source.lstrip().startswith("{") and Path(source).exists()):
        source = Path(source).read_text(encoding="utf-8")
    payload: dict[str, Any] = json.loads(source)
    if payload.get("schema") != expected_schema:
        raise ValueError(f"JSON does not contain {expected_schema}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported {expected_schema} JSON schema version")
    return payload


def _formula_from_canonical(value: Any) -> Formula | None:
    if value is None:
        return None
    return Formula(
        tuple(_parse_call(str(term)) for term in value["terms"]),
        value["intercept"],
    )


def _duration_remstats(
    history: EventHistory,
    *,
    start_effects: str | Effect | Formula | None,
    end_effects: str | Effect | Formula | None,
    psi_start: float,
    psi_end: float,
    first: int,
    last: float,
    attr_actors: pd.DataFrame | None,
    attr_dyads: pd.DataFrame | np.ndarray | None,
) -> RemStatsDuration:
    if not history.risksets:
        raise ValueError("duration history has no attached risk set")
    start_formula = (
        _attach_deprecated_attributes(formula(start_effects), attr_actors, attr_dyads)
        if start_effects is not None
        else None
    )
    end_formula = (
        _attach_deprecated_attributes(formula(end_effects), attr_actors, attr_dyads)
        if end_effects is not None
        else None
    )
    start_names = _duration_stat_names(start_formula, history, "start")
    end_names = _duration_stat_names(end_formula, history, "end")
    all_names = [*start_names, *end_names]
    start_base = history.risksets[0].reset_index(drop=True).copy()
    end_directed = bool(history.durem.get("dur_directed_end", False))
    end_base = _duration_end_riskset(history, directed=end_directed)
    timeline = _duration_timeline(history)
    stop = len(timeline) if last == float("inf") else min(int(last), len(timeline))
    positions = list(range(max(first - 1, 0), stop))
    frames: list[pd.DataFrame] = []
    for position in positions:
        time = timeline[position]
        previous_time = 0.0 if position == 0 else timeline[position - 1]
        interevent = float(time - previous_time)
        if interevent <= 0:
            raise ValueError("duration timeline must be strictly increasing")
        active = _duration_active_events(history.events, time)
        if end_formula is not None:
            end_riskset, end_observed = _duration_end_process_rows(
                history, end_base, time, directed=end_directed
            )
            end_previous = _duration_completed_history(history.events, time, psi_end)
            end_active = _duration_active_history(active)
            if not end_directed:
                end_previous = _canonicalize_duration_history(end_previous, history)
                end_active = _canonicalize_duration_history(end_active, history)
            end_design = _duration_design_matrix(
                end_formula,
                history,
                end_riskset,
                end_previous,
                end_active,
                time,
                directed=end_directed,
            )
            frames.append(
                _duration_process_frame(
                    process="end",
                    riskset=end_riskset,
                    design=end_design,
                    own_names=end_names,
                    all_names=all_names,
                    observed=None,
                    observed_flags=end_observed,
                    timeline_position=position + 1,
                    log_interevent=(None if history.ordinal else float(np.log(interevent))),
                    directed=end_directed,
                    typed=history.extend_riskset_by_type and bool(history.event_types),
                )
            )
        if start_formula is not None:
            start_riskset, start_observed = _duration_start_process_rows(
                history, start_base, active, time
            )
            start_previous = _duration_completed_history(history.events, time, psi_start)
            start_active = _duration_active_history(active)
            start_design = _duration_design_matrix(
                start_formula,
                history,
                start_riskset,
                start_previous,
                start_active,
                time,
                directed=history.directed,
            )
            frames.append(
                _duration_process_frame(
                    process="start",
                    riskset=start_riskset,
                    design=start_design,
                    own_names=start_names,
                    all_names=all_names,
                    observed=None,
                    observed_flags=start_observed,
                    timeline_position=position + 1,
                    log_interevent=(None if history.ordinal else float(np.log(interevent))),
                    directed=history.directed,
                    typed=history.extend_riskset_by_type and bool(history.event_types),
                )
            )
    columns = ["obs"]
    if not history.ordinal:
        columns.append("log_interevent")
    columns.extend(all_names)
    columns.extend(["time_index", "time", "dyad", "process"])
    if history.extend_riskset_by_type and history.event_types:
        columns.append("type")
    nonempty = [frame for frame in frames if not frame.empty]
    stacked_frame = (
        pd.concat(nonempty, ignore_index=True)[columns]
        if nonempty
        else pd.DataFrame(columns=columns)
    )
    subset = (positions[0] + 1, positions[-1] + 1) if positions else (0, 0)
    stacked = RemStatsStackedDuration(
        remstats_stack=stacked_frame,
        subset=subset,
        D_start=len(start_base),
        D_end=(len(end_base) if end_formula is not None else 0),
        E=len(positions),
        stat_names=all_names,
        stat_names_start=start_names,
        stat_names_end=end_names,
        ordinal=history.ordinal,
    )
    return RemStatsDuration(
        history=history,
        stacked=stacked,
        start_formula=start_formula,
        end_formula=end_formula,
        psi_start=psi_start,
        psi_end=psi_end,
    )


def _duration_stat_names(parsed: Formula | None, history: EventHistory, suffix: str) -> list[str]:
    if parsed is None:
        return []
    base = (
        ["baseline"] if parsed.intercept is not False and not history.ordinal else []
    ) + [
        name for effect in parsed.terms for name in _effect_statistic_names(effect, history)
    ]
    return [f"{name}.{suffix}" for name in base]


def _duration_timeline(history: EventHistory) -> list[float]:
    starts = history.events["time"].astype(float).to_list()
    ends = history.events.loc[history.events["end"].notna(), "end"].astype(float).to_list()
    return sorted(set([*starts, *ends]))


def _duration_active_events(events: pd.DataFrame, time: float) -> pd.DataFrame:
    return events[
        (events["time"].astype(float) < time)
        & (events["end"].isna() | (events["end"].astype(float) >= time))
    ].copy()


def _duration_completed_history(events: pd.DataFrame, time: float, psi: float) -> pd.DataFrame:
    result = events[events["end"].notna() & (events["end"].astype(float) < time)].copy()
    if result.empty:
        result["__effective_weight"] = pd.Series(dtype=float)
        return result
    duration = (
        result["end"].astype(float).to_numpy() - result["time"].astype(float).to_numpy() + 1.0
    )
    duration_weight = np.power(np.maximum(duration, 0.0), psi)
    result["__effective_weight"] = (
        result.get("event_weight", pd.Series(1.0, index=result.index)).astype(float).to_numpy()
        * duration_weight
    )
    return result


def _duration_active_history(active: pd.DataFrame) -> pd.DataFrame:
    result = active.copy()
    result["__effective_weight"] = np.ones(len(result), dtype=float)
    return result


def _duration_start_process_rows(
    history: EventHistory,
    base: pd.DataFrame,
    active: pd.DataFrame,
    time: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return observed starts first, followed by inactive start-risk rows."""

    type_sensitive = bool(
        history.extend_riskset_by_type
        and history.event_types
        and not history.durem.get("dur_type_exclusive", False)
    )
    active_keys = {
        _duration_row_key(row, directed=history.directed, typed=type_sensitive)
        for row in active.itertuples()
    }
    observed_events = history.events[history.events["time"].astype(float) == time]
    observed_ids = _stable_integer_unique(observed_events["dyad_id"])
    observed_set = set(observed_ids)
    by_id = {int(row.dyad_id): index for index, row in enumerate(base.itertuples())}
    missing = [dyad_id for dyad_id in observed_ids if dyad_id not in by_id]
    if missing:
        raise ValueError(f"observed duration starts are outside the risk set: {missing}")
    inactive_ids = [
        int(row.dyad_id)
        for row in base.itertuples()
        if int(row.dyad_id) not in observed_set
        and _duration_row_key(
            row, directed=history.directed, typed=type_sensitive
        ) not in active_keys
    ]
    selected = [*observed_ids, *inactive_ids]
    rows = base.iloc[[by_id[dyad_id] for dyad_id in selected]].reset_index(drop=True)
    flags = np.asarray(
        [1] * len(observed_ids) + [0] * len(inactive_ids), dtype=int
    )
    return rows, flags


def _duration_end_riskset(history: EventHistory, *, directed: bool) -> pd.DataFrame:
    labels = dict(zip(history.actors["actor_id"], history.actors["actor"], strict=True))
    types: list[Any | None] = (
        list(history.event_types)
        if history.extend_riskset_by_type and history.event_types
        else [None]
    )
    rows: list[dict[str, Any]] = []
    dyad_id = 1
    for event_type in types:
        for sender_id in range(1, history.N + 1):
            receiver_values = (
                range(1, history.N + 1) if directed else range(sender_id + 1, history.N + 1)
            )
            for receiver_id in receiver_values:
                if sender_id == receiver_id:
                    continue
                row: dict[str, Any] = {
                    "dyad_id": dyad_id,
                    "sender_id": sender_id,
                    "receiver_id": receiver_id,
                    "sender": labels[sender_id],
                    "receiver": labels[receiver_id],
                }
                if event_type is not None:
                    row["event_type"] = event_type
                rows.append(row)
                dyad_id += 1
    return pd.DataFrame(rows)


def _duration_end_process_rows(
    history: EventHistory,
    base: pd.DataFrame,
    current_time: float,
    *,
    directed: bool,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ending rows first and ongoing rows second, preserving duplicates."""

    typed = history.extend_riskset_by_type and bool(history.event_types)
    by_key = {
        _duration_row_key(row, directed=directed, typed=typed): index
        for index, row in enumerate(base.itertuples())
    }
    events = history.events
    ending = events[
        events["end"].notna()
        & (events["end"].astype(float) == current_time)
        & (events["time"].astype(float) < current_time)
    ]
    ongoing = events[
        (events["time"].astype(float) < current_time)
        & (events["end"].isna() | (events["end"].astype(float) > current_time))
    ]

    def mapped_ids(frame: pd.DataFrame) -> list[int]:
        indexes: list[int] = []
        for row in frame.itertuples():
            key = _duration_row_key(row, directed=directed, typed=typed)
            if key not in by_key:
                raise ValueError(f"duration end is outside the risk set: {key}")
            indexes.append(int(base.iloc[by_key[key]]["dyad_id"]))
        return _stable_integer_unique(indexes)

    ending_ids = mapped_ids(ending)
    ongoing_ids = mapped_ids(ongoing)
    by_id = {int(row.dyad_id): index for index, row in enumerate(base.itertuples())}
    selected = [*ending_ids, *ongoing_ids]
    rows = base.iloc[[by_id[dyad_id] for dyad_id in selected]].reset_index(drop=True)
    flags = np.asarray([1] * len(ending_ids) + [0] * len(ongoing_ids), dtype=int)
    return rows, flags


def _stable_integer_unique(values: Any) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        integer = int(value)
        if integer not in seen:
            result.append(integer)
            seen.add(integer)
    return result


def _canonicalize_duration_history(previous: pd.DataFrame, history: EventHistory) -> pd.DataFrame:
    result = previous.copy()
    if result.empty:
        return result
    reverse = result["sender_id"] > result["receiver_id"]
    if reverse.any():
        sender_ids = result.loc[reverse, "sender_id"].copy()
        senders = result.loc[reverse, "sender"].copy()
        result.loc[reverse, "sender_id"] = result.loc[reverse, "receiver_id"].to_numpy()
        result.loc[reverse, "receiver_id"] = sender_ids.to_numpy()
        result.loc[reverse, "sender"] = result.loc[reverse, "receiver"].to_numpy()
        result.loc[reverse, "receiver"] = senders.to_numpy()
    return result


def _duration_row_key(row: Any, *, directed: bool, typed: bool) -> tuple[Any, ...]:
    sender_id = int(row.sender_id)
    receiver_id = int(row.receiver_id)
    if not directed and sender_id > receiver_id:
        sender_id, receiver_id = receiver_id, sender_id
    key: tuple[Any, ...] = (sender_id, receiver_id)
    if typed:
        key = (*key, getattr(row, "event_type", None))
    return key


def _duration_design_matrix(
    parsed: Formula,
    history: EventHistory,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    active: pd.DataFrame,
    current_time: float,
    *,
    directed: bool,
) -> np.ndarray:
    columns = [
        column
        for effect in parsed.terms
        for column in _duration_effect_columns(
            effect,
            history,
            riskset,
            active if _is_duration_active_effect(effect) else previous,
            current_time,
            directed=directed,
        )
    ]
    matrix = np.column_stack(columns) if columns else np.empty((len(riskset), 0), dtype=float)
    if parsed.intercept is not False and not history.ordinal:
        matrix = np.column_stack([np.ones(len(riskset), dtype=float), matrix])
    return matrix


def _is_duration_active_effect(effect: Effect) -> bool:
    if effect.components:
        return all(_is_duration_active_effect(component) for component in effect.components)
    return effect.name.startswith("active")


def _duration_effect_columns(
    effect: Effect,
    history: EventHistory,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    current_time: float,
    *,
    directed: bool,
) -> list[np.ndarray]:
    _validate_effect_for_direction(effect, directed)
    mode = _consider_type_mode(effect)
    base_effect = _without_effect_kwarg(effect, "consider_type")
    if mode == "ignore":
        return [
            _compute_effect(
                base_effect,
                riskset,
                previous,
                directed,
                event_index=0,
                current_time=current_time,
            )
        ]
    if not history.event_types:
        raise ValueError(f"consider_type={mode!r} requires a typed event history")
    by_history_type = [
        _compute_effect(
            base_effect,
            riskset,
            previous[previous["event_type"] == event_type].copy(),
            directed,
            event_index=0,
            current_time=current_time,
        )
        for event_type in history.event_types
    ]
    if mode == "separate" or "event_type" not in riskset.columns:
        return by_history_type
    columns: list[np.ndarray] = []
    for values in by_history_type:
        for candidate_type in history.event_types:
            mask = (riskset["event_type"] == candidate_type).to_numpy(dtype=float)
            columns.append(values * mask)
    return columns


def _duration_process_frame(
    *,
    process: str,
    riskset: pd.DataFrame,
    design: np.ndarray,
    own_names: list[str],
    all_names: list[str],
    observed: pd.DataFrame | None,
    observed_flags: np.ndarray | None = None,
    timeline_position: int,
    log_interevent: float | None,
    directed: bool,
    typed: bool,
) -> pd.DataFrame:
    if riskset.empty:
        return pd.DataFrame()
    if observed_flags is None:
        if observed is None:
            raise ValueError("duration process rows require observations or explicit flags")
        observed_keys = {
            _duration_row_key(row, directed=directed, typed=typed)
            for row in observed.itertuples()
        }
        flags = np.asarray(
            [
                int(_duration_row_key(row, directed=directed, typed=typed) in observed_keys)
                for row in riskset.itertuples()
            ],
            dtype=int,
        )
    else:
        flags = np.asarray(observed_flags, dtype=int)
        if flags.shape != (len(riskset),):
            raise ValueError("duration observation flags do not align with the risk set")
    data: dict[str, Any] = {
        "obs": flags,
        "time_index": np.full(len(riskset), timeline_position, dtype=int),
        "time": np.full(len(riskset), timeline_position, dtype=int),
        "dyad": riskset[
            "risk_id" if "risk_id" in riskset.columns else "dyad_id"
        ].to_numpy(dtype=int),
        "process": np.full(len(riskset), process, dtype=object),
    }
    if log_interevent is not None:
        data["log_interevent"] = np.full(len(riskset), log_interevent, dtype=float)
    if "event_type" in riskset.columns:
        data["type"] = riskset["event_type"].to_numpy(copy=True)
    frame = pd.DataFrame(data)
    for name in all_names:
        frame[name] = 0.0
    for column, name in enumerate(own_names):
        frame[name] = design[:, column]
    return frame


def _actor_remstats(
    history: EventHistory,
    *,
    sender_effects: str | Effect | Formula | None,
    receiver_effects: str | Effect | Formula | None,
    memory: str,
    memory_value: float | tuple[float, float] | None,
    first: int,
    last: float,
    attr_actors: pd.DataFrame | None,
    attr_dyads: pd.DataFrame | np.ndarray | None,
) -> AomStats:
    sender_formula = (
        _attach_deprecated_attributes(formula(sender_effects), attr_actors, attr_dyads)
        if sender_effects is not None
        else Formula((), intercept=False)
    )
    receiver_formula = (
        _attach_deprecated_attributes(formula(receiver_effects), attr_actors, attr_dyads)
        if receiver_effects is not None
        else Formula((), intercept=False)
    )
    # A receiver intercept is constant within every sender-conditioned choice
    # set and is therefore not an estimable actor-oriented statistic.  The R
    # aomstats contract omits it even when the ordinary formula intercept is
    # present; only the sender-rate component carries a baseline.
    receiver_formula = Formula(receiver_formula.terms, intercept=False)
    sender_names = (["baseline"] if sender_formula.intercept is not False else []) + [
        name for effect in sender_formula.terms for name in _effect_statistic_names(effect, history)
    ]
    receiver_names = [
        name
        for effect in receiver_formula.terms
        for name in _effect_statistic_names(effect, history)
    ]
    time_groups = [
        [int(index) for index in indexes]
        for indexes in history.events.groupby("time", sort=False, dropna=False).groups.values()
    ]
    stop = None if last == float("inf") else int(last)
    selected_groups = time_groups[max(first - 1, 0) : stop]
    sender_matrices: list[np.ndarray] = []
    receiver_matrices: list[np.ndarray] = []
    observed_senders: list[int] = []
    observed_sender_groups: list[list[int]] = []
    observed_receivers: list[int] = []
    receiver_masks: list[np.ndarray] = []
    receiver_choice_stats: list[np.ndarray] = []
    receiver_choice_observed: list[int] = []
    receiver_choice_masks: list[np.ndarray] = []
    receiver_choice_event_indices: list[int] = []
    event_indices: list[int] = []
    labels = dict(zip(history.actors["actor_id"], history.actors["actor"], strict=True))
    sender_ids = history.sender_riskset.astype(int)
    for group in selected_groups:
        event_index = group[0]
        previous = _previous_events(history.events, event_index, memory, memory_value)
        sender_riskset = _sender_actor_riskset(history, labels)
        sender_matrix = _actor_design_matrix(
            sender_formula,
            history,
            sender_riskset,
            previous,
            event_index,
        )
        group_sender_positions: list[int] = []
        group_receiver_matrices: list[np.ndarray] = []
        group_observed_receivers: list[int] = []
        group_receiver_masks: list[np.ndarray] = []
        for choice_event_index in group:
            event_row = history.events.iloc[choice_event_index]
            observed_sender_id = int(event_row["sender_id"])
            matches = np.flatnonzero(sender_ids == observed_sender_id)
            if len(matches) != 1:
                raise ValueError(
                    "observed sender must occur exactly once in the sender risk set"
                )
            group_sender_positions.append(int(matches[0]))

            allowed_receivers = history.receiver_riskset[event_row["sender"]].astype(int)
            receiver_riskset = _receiver_actor_riskset(
                observed_sender_id, allowed_receivers, labels, history.N
            )
            compact_receiver = _actor_design_matrix(
                receiver_formula,
                history,
                receiver_riskset,
                previous,
                choice_event_index,
            )
            receiver_matrix = np.zeros((history.N, len(receiver_names)), dtype=float)
            receiver_matrix[allowed_receivers - 1] = compact_receiver
            receiver_mask = np.zeros(history.N, dtype=bool)
            receiver_mask[allowed_receivers - 1] = True
            observed_receiver = int(event_row["receiver_id"]) - 1
            if not receiver_mask[observed_receiver]:
                raise ValueError("observed receiver is not in the receiver risk set")
            group_receiver_matrices.append(receiver_matrix)
            group_observed_receivers.append(observed_receiver)
            group_receiver_masks.append(receiver_mask)
            receiver_choice_stats.append(receiver_matrix)
            receiver_choice_observed.append(observed_receiver)
            receiver_choice_masks.append(receiver_mask)
            receiver_choice_event_indices.append(choice_event_index)

        sender_matrices.append(sender_matrix)
        receiver_matrices.append(group_receiver_matrices[0])
        observed_senders.append(group_sender_positions[0])
        observed_sender_groups.append(group_sender_positions)
        observed_receivers.append(group_observed_receivers[0])
        receiver_masks.append(group_receiver_masks[0])
        event_indices.append(event_index)
    return AomStats(
        history=history,
        sender_stats=sender_matrices,
        receiver_stats=receiver_matrices,
        sender_names=sender_names,
        receiver_names=receiver_names,
        observed_sender_indices=observed_senders,
        observed_receiver_indices=observed_receivers,
        receiver_masks=receiver_masks,
        event_indices=event_indices,
        observed_sender_groups=observed_sender_groups,
        receiver_choice_stats=receiver_choice_stats,
        receiver_choice_observed_indices=receiver_choice_observed,
        receiver_choice_masks=receiver_choice_masks,
        receiver_choice_event_indices=receiver_choice_event_indices,
    )


def _actor_design_matrix(
    parsed: Formula,
    history: EventHistory,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    event_index: int,
) -> np.ndarray:
    columns = [
        column
        for effect in parsed.terms
        for column in _compute_effect_columns(effect, history, riskset, previous, event_index)
    ]
    matrix = np.column_stack(columns) if columns else np.empty((len(riskset), 0), dtype=float)
    if parsed.intercept is not False:
        matrix = np.column_stack([np.ones(len(riskset), dtype=float), matrix])
    return matrix


def _sender_actor_riskset(
    history: EventHistory,
    labels: Mapping[int, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sender_id in history.sender_riskset.astype(int):
        sender = labels[int(sender_id)]
        receivers = history.receiver_riskset[sender].astype(int)
        if len(receivers) == 0:
            raise ValueError(f"sender {sender!r} has an empty receiver risk set")
        receiver_id = int(receivers[0])
        rows.append(
            {
                "sender_id": int(sender_id),
                "receiver_id": receiver_id,
                "sender": sender,
                "receiver": labels[receiver_id],
                "dyad_id": (int(sender_id) - 1) * (history.N - 1)
                + receiver_id
                - int(receiver_id > sender_id),
            }
        )
    return pd.DataFrame(rows)


def _receiver_actor_riskset(
    sender_id: int,
    receiver_ids: np.ndarray,
    labels: Mapping[int, Any],
    actor_count: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sender_id": sender_id,
                "receiver_id": int(receiver_id),
                "sender": labels[sender_id],
                "receiver": labels[int(receiver_id)],
                "dyad_id": (sender_id - 1) * (actor_count - 1)
                + int(receiver_id)
                - int(receiver_id > sender_id),
            }
            for receiver_id in receiver_ids
        ]
    )


def select_stats(stats: RemStats, names: Sequence[str]) -> RemStats:
    indexes = [stats.names.index(name) for name in names]
    return RemStats(
        history=stats.history,
        stats=[matrix[:, indexes] for matrix in stats.stats],
        names=list(names),
        formula=Formula(tuple(Effect(name) for name in names), stats.formula.intercept),
        observed_indices=list(stats.observed_indices),
        event_indices=list(stats.event_indices),
        observed_index_groups=[list(group) for group in stats.observed_index_groups],
        sample_map=[np.array(indexes, copy=True) for indexes in stats.sample_map],
        sampling_weights=[np.array(weights, copy=True) for weights in stats.sampling_weights],
    )


def tie_effects(value: str | Effect | Formula) -> Formula:
    return formula(value)


def actor_effects(value: str | Effect | Formula) -> Formula:
    return formula(value)


def is_remstats_durem(value: object) -> bool:
    return isinstance(value, RemStatsDuration)


def observed_risk_index(history: EventHistory, event_index: int) -> int:
    event = history.events.iloc[event_index]
    riskset = history.risksets[event_index]
    mask = (riskset["sender_id"] == event["sender_id"]) & (
        riskset["receiver_id"] == event["receiver_id"]
    )
    if "event_type" in riskset.columns and riskset["event_type"].nunique(dropna=False) > 1:
        if pd.isna(event["event_type"]):
            mask &= riskset["event_type"].isna()
        else:
            mask &= riskset["event_type"] == event["event_type"]
    matches = np.flatnonzero(mask.to_numpy())
    if len(matches) != 1:
        raise ValueError("observed event must appear exactly once in each risk set")
    return int(matches[0])


def _make_effect(name: str) -> Callable[..., Effect]:
    def factory(*args: Any, **kwargs: Any) -> Effect:
        effect = Effect(name, tuple(args), tuple(kwargs.items()))
        _validate_effect_constructor(effect)
        return effect

    factory.__name__ = name
    factory.__qualname__ = name
    return factory


def _validate_effect_constructor(effect: Effect) -> None:
    kwargs = _effect_kwargs(effect)
    if "consider_type" in kwargs and kwargs["consider_type"] not in {
        "ignore",
        "separate",
        "interact",
        True,
        False,
    }:
        raise ValueError(
            f"consider_type={kwargs['consider_type']!r} is not supported for {effect.name}"
        )
    if kwargs.get("scaling") == "as.is":
        warnings.warn(
            "scaling='as.is' is deprecated; use 'scaling' is 'none'",
            DeprecationWarning,
            stacklevel=3,
        )
    if effect.name == "tie":
        warnings.warn("tie() is deprecated; use dyad()", DeprecationWarning, stacklevel=3)
    if effect.name == "spUnique":
        warnings.warn(
            "spUnique() is deprecated; use sp(unique=True)",
            DeprecationWarning,
            stacklevel=3,
        )
    if effect.name == "userStat":
        value = effect.args[0] if effect.args else kwargs.get("x")
        if value is not None and bool(np.asarray(pd.isna(value)).any()):
            warnings.warn("userStat contains missing values", UserWarning, stacklevel=3)
    actor_effects = {"send", "receive", "same", "difference", "average", "minimum", "maximum"}
    if effect.name not in actor_effects:
        return
    variable = effect.args[0] if effect.args else kwargs.get("variable")
    if variable is not None and (not isinstance(variable, str) or not variable):
        raise TypeError(f"{effect.name} variable should be a string")
    attributes = effect.args[1] if len(effect.args) > 1 else kwargs.get("attr_actors")
    if attributes is None or variable is None:
        return
    frame = pd.DataFrame(attributes)
    if variable not in frame.columns:
        raise ValueError(f"{variable!r} not in attr_actors")
    if "time" in frame.columns and frame["time"].isna().any():
        raise ValueError("Missing values in the attr_actors time column")
    if frame[variable].isna().any():
        warnings.warn("Missing values in attr_actors", UserWarning, stacklevel=3)


FEtype = _make_effect("FEtype")
inertia = _make_effect("inertia")
reciprocity = _make_effect("reciprocity")
send = _make_effect("send")
receive = _make_effect("receive")
tie = _make_effect("tie")
dyad = _make_effect("dyad")
event = _make_effect("event")
userStat = _make_effect("userStat")
indegreeReceiver = _make_effect("indegreeReceiver")
indegreeSender = _make_effect("indegreeSender")
outdegreeReceiver = _make_effect("outdegreeReceiver")
outdegreeSender = _make_effect("outdegreeSender")
totaldegreeDyad = _make_effect("totaldegreeDyad")
totaldegreeReceiver = _make_effect("totaldegreeReceiver")
totaldegreeSender = _make_effect("totaldegreeSender")
degreeDiff = _make_effect("degreeDiff")
degreeMax = _make_effect("degreeMax")
degreeMin = _make_effect("degreeMin")
same = _make_effect("same")
difference = _make_effect("difference")
average = _make_effect("average")
minimum = _make_effect("minimum")
maximum = _make_effect("maximum")
isp = _make_effect("isp")
itp = _make_effect("itp")
osp = _make_effect("osp")
otp = _make_effect("otp")
sp = _make_effect("sp")
spUnique = _make_effect("spUnique")
psABA = _make_effect("psABA")
psABAB = _make_effect("psABAB")
psABAY = _make_effect("psABAY")
psABB = _make_effect("psABB")
psABBA = _make_effect("psABBA")
psABBY = _make_effect("psABBY")
psABX = _make_effect("psABX")
psABXA = _make_effect("psABXA")
psABXB = _make_effect("psABXB")
psABXY = _make_effect("psABXY")
recencyContinue = _make_effect("recencyContinue")
recencyReceiveReceiver = _make_effect("recencyReceiveReceiver")
recencyReceiveSender = _make_effect("recencyReceiveSender")
recencySendReceiver = _make_effect("recencySendReceiver")
recencySendSender = _make_effect("recencySendSender")
rrankReceive = _make_effect("rrankReceive")
rrankSend = _make_effect("rrankSend")
activeTie = _make_effect("activeTie")
activeReciprocalTie = _make_effect("activeReciprocalTie")
activeDegreeDyad = _make_effect("activeDegreeDyad")
activeDegreeMin = _make_effect("activeDegreeMin")
activeDegreeMax = _make_effect("activeDegreeMax")
activeIndegreeReceiver = _make_effect("activeIndegreeReceiver")
activeOutdegreeSender = _make_effect("activeOutdegreeSender")
activeTotaldegreeDyad = _make_effect("activeTotaldegreeDyad")
activeTotaldegreeReceiver = _make_effect("activeTotaldegreeReceiver")
activeTotaldegreeSender = _make_effect("activeTotaldegreeSender")
activeSharedPartners = _make_effect("activeSharedPartners")
activeSharedPartners_isp = _make_effect("activeSharedPartners_isp")
activeSharedPartners_itp = _make_effect("activeSharedPartners_itp")
activeSharedPartners_osp = _make_effect("activeSharedPartners_osp")
activeSharedPartners_otp = _make_effect("activeSharedPartners_otp")


def _stats_for_event(
    history: EventHistory,
    event_index: int,
    effects: Sequence[Effect],
    memory: str = "full",
    memory_value: float | tuple[float, float] | None = None,
) -> np.ndarray:
    riskset = history.risksets[event_index]
    previous = _previous_events(history.events, event_index, memory, memory_value)
    columns = [
        column
        for effect in effects
        for column in _compute_effect_columns(
            effect,
            history,
            riskset,
            previous,
            event_index,
        )
    ]
    if not columns:
        return np.empty((len(riskset), 0), dtype=float)
    return np.column_stack(columns)


def _compute_effect_columns(
    effect: Effect,
    history: EventHistory,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    event_index: int,
) -> list[np.ndarray]:
    _validate_effect_for_direction(effect, history.directed)
    if effect.name == "FEtype":
        if "event_type" not in riskset.columns or len(history.event_types) < 2:
            raise ValueError(
                "FEtype requires a type-expanded risk set with at least two types"
            )
        return [
            np.asarray(
                (riskset["event_type"] == event_type).to_numpy(),
                dtype=float,
            )
            for event_type in history.event_types[1:]
        ]
    mode = _consider_type_mode(effect)
    base_effect = _without_effect_kwarg(effect, "consider_type")
    common = {
        "event_index": event_index,
        "current_time": history.events.iloc[event_index]["time"],
    }
    if mode == "ignore":
        return [
            _compute_effect(
                base_effect,
                riskset,
                previous,
                history.directed,
                **common,
            )
        ]
    if not history.event_types:
        raise ValueError(f"consider_type={mode!r} requires a typed event history")
    type_history = previous
    if effect.name.startswith("ps") and not previous.empty:
        # Participation shifts are defined by the immediately preceding point
        # in time.  Type separation must not reach back to an older event merely
        # because the latest point contains no event of that type.
        latest_time = previous.iloc[-1]["time"]
        type_history = previous[previous["time"] == latest_time]
    if effect.name.startswith("ps") and mode == "separate" and "event_type" in riskset.columns:
        # Pinned remstats duplicates the immediate point-time pshift pattern
        # across the named type slices when the candidate risk set itself is
        # type-expanded.  The unexpanded path separates by historical type;
        # the expanded historical/candidate cross-product belongs to interact.
        aggregate = _compute_effect(
            base_effect,
            riskset,
            type_history,
            history.directed,
            **common,
        )
        return [aggregate.copy() for _ in history.event_types]
    by_history_type = [
        _compute_effect(
            base_effect,
            riskset,
            type_history[type_history["event_type"] == event_type].copy(),
            history.directed,
            **common,
        )
        for event_type in history.event_types
    ]
    if mode == "separate" or "event_type" not in riskset.columns:
        return by_history_type
    columns: list[np.ndarray] = []
    # Interaction axes use the stable <history type>.<candidate type> order.
    # Keep the history slice outermost and mask it once for every candidate
    # type in the expanded risk set.
    for values in by_history_type:
        for candidate_type in history.event_types:
            candidate_mask = (riskset["event_type"] == candidate_type).to_numpy(dtype=float)
            columns.append(values * candidate_mask)
    return columns


def _validate_effect_for_direction(effect: Effect, directed: bool) -> None:
    for component in effect.components:
        _validate_effect_for_direction(component, directed)
    if directed and effect.name in {"sp", "spUnique"}:
        raise ValueError("sp is only defined for undirected events; use directed shared partners")
    directed_only = {
        "reciprocity",
        "indegreeReceiver",
        "indegreeSender",
        "outdegreeReceiver",
        "outdegreeSender",
        "isp",
        "itp",
        "osp",
        "otp",
    }
    if not directed and effect.name in directed_only:
        raise ValueError(f"{effect.name} is not defined for undirected events")


def _effect_statistic_names(effect: Effect, history: EventHistory) -> list[str]:
    if effect.name == "FEtype":
        if not history.extend_riskset_by_type or len(history.event_types) < 2:
            raise ValueError(
                "FEtype requires a type-expanded risk set with at least two types"
            )
        return [f"FEtype_{event_type}" for event_type in history.event_types[1:]]
    mode = _consider_type_mode(effect)
    base = _without_effect_kwarg(effect, "consider_type").statistic_name
    if mode == "ignore":
        return [base]
    if not history.event_types:
        raise ValueError(f"consider_type={mode!r} requires a typed event history")
    if mode == "separate" or not history.extend_riskset_by_type:
        return [f"{base}.{event_type}" for event_type in history.event_types]
    return [
        f"{base}.{history_type}.{candidate_type}"
        for history_type in history.event_types
        for candidate_type in history.event_types
    ]


def _consider_type_mode(effect: Effect) -> str:
    value = _effect_kwargs(effect).get("consider_type", "ignore")
    if value is True:
        return "separate"
    if value is False:
        return "ignore"
    if value not in {"ignore", "separate", "interact"}:
        raise ValueError(
            "consider_type must be one of 'ignore', 'separate', 'interact', True, or False"
        )
    return str(value)


def _without_effect_kwarg(effect: Effect, key: str) -> Effect:
    return Effect(
        effect.name,
        effect.args,
        tuple((name, value) for name, value in effect.kwargs if name != key),
        tuple(_without_effect_kwarg(component, key) for component in effect.components),
    )


def _previous_events(
    events: pd.DataFrame,
    event_index: int,
    memory: str,
    memory_value: float | tuple[float, float] | None,
) -> pd.DataFrame:
    previous = events.iloc[:event_index].copy()
    if previous.empty:
        previous["__effective_weight"] = pd.Series(dtype=float)
        return previous
    reference_index = max(event_index - 1, 0)
    current_time = events.iloc[reference_index]["time"]
    try:
        time_values = previous["time"].astype(float)
        current_value = float(current_time)
        previous = previous[time_values <= current_value].copy()
    except (TypeError, ValueError):
        current_value = float(event_index + 1)
    if previous.empty:
        previous["__effective_weight"] = pd.Series(dtype=float)
        return previous
    base_weights = previous.get("event_weight", pd.Series(1.0, index=previous.index)).astype(float)
    if memory == "full":
        previous["__effective_weight"] = base_weights
        return previous
    if memory == "window":
        if not isinstance(memory_value, float):
            raise RuntimeError("validated window memory must be a float")
        width = memory_value
        try:
            ages = current_value - previous["time"].astype(float)
            previous = previous[(ages >= 0) & (ages < width)].copy()
        except (TypeError, ValueError):
            previous = previous.tail(max(int(width), 0)).copy()
        previous["__effective_weight"] = previous.get("event_weight", 1.0)
        return previous
    if memory == "interval":
        if not isinstance(memory_value, tuple):
            raise RuntimeError("validated interval memory must be a pair")
        lower, upper = memory_value
        try:
            ages = current_value - previous["time"].astype(float)
            previous = previous[(ages >= lower) & (ages < upper)].copy()
        except (TypeError, ValueError):
            previous = previous.iloc[0:0].copy()
        previous["__effective_weight"] = previous.get("event_weight", 1.0)
        return previous
    if memory == "decay":
        if not isinstance(memory_value, float):
            raise RuntimeError("validated decay memory must be a float")
        half_life = memory_value
        try:
            ages = current_value - previous["time"].astype(float)
        except (TypeError, ValueError):
            ages = pd.Series(np.arange(len(previous), 0, -1, dtype=float), index=previous.index)
        previous["__effective_weight"] = base_weights * np.exp(-ages * (np.log(2.0) / half_life))
        return previous
    raise ValueError(f"memory must be one of full, window, decay, interval; got {memory!r}")


def _validate_memory(
    memory: str, memory_value: float | Sequence[float] | None
) -> float | tuple[float, float] | None:
    if memory == "full":
        return None
    if memory in {"window", "decay"}:
        if not isinstance(memory_value, Real):
            raise ValueError(f"memory_value must be one numeric value for memory='{memory}'")
        value = float(memory_value)
        if memory == "decay" and value <= 0:
            raise ValueError("memory_value must be positive and non-zero for decay memory")
        if memory == "window" and value < 0:
            raise ValueError("memory_value must be non-negative for window memory")
        return value
    if memory == "interval":
        if not isinstance(memory_value, Sequence) or isinstance(memory_value, (str, bytes)):
            raise ValueError("memory_value must contain two values for interval memory")
        values = tuple(float(value) for value in memory_value)
        if len(values) != 2:
            raise ValueError("memory_value must contain two values for interval memory")
        if values[0] >= values[1]:
            raise ValueError("the first interval value must be lower than the second")
        return values
    raise ValueError(f"unsupported memory mode: {memory}")


def _compute_effect(
    effect: Effect | str,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    directed: bool,
    *,
    event_index: int | None = None,
    current_time: Any | None = None,
) -> np.ndarray:
    normalized = Effect(effect) if isinstance(effect, str) else effect
    values = _compute_effect_unscaled(
        normalized,
        riskset,
        previous,
        directed,
        event_index=event_index,
        current_time=current_time,
    )
    if not _is_endogenous_effect(normalized):
        return values
    scaling = _effect_kwargs(normalized).get("scaling", "none")
    if scaling in {"none", "as.is"}:
        return values
    if scaling == "std":
        return _scale_covariate(values, "std", normalized.name)
    if scaling == "prop":
        return _scale_endogenous_proportion(values, normalized.name, riskset, previous, directed)
    raise ValueError(f"invalid scaling for {normalized.name}: {scaling!r}")


def _compute_effect_unscaled(
    effect: Effect,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    directed: bool,
    *,
    event_index: int | None = None,
    current_time: Any | None = None,
) -> np.ndarray:
    name = effect.name
    if name == "FEtype":
        levels = list(pd.unique(riskset["event_type"]))
        if len(levels) < 2:
            raise ValueError("FEtype requires a type-expanded risk set with at least two types")
        return np.asarray((riskset["event_type"] == levels[1]).to_numpy(), dtype=float)
    if name in {"event", "userStat"}:
        if event_index is None:
            raise ValueError(f"{name} requires an event index")
        if name == "event":
            return _event_covariate(effect, riskset, event_index)
        return _user_stat(effect, riskset, event_index)
    if name in {"same", "difference", "average", "minimum", "maximum"}:
        return _actor_covariate(effect, riskset, current_time, transform=name)
    if name in {"send", "receive"} and (effect.args or effect.kwargs):
        return _actor_covariate(effect, riskset, current_time, transform=name)
    if name in {"tie", "dyad"} and (effect.args or effect.kwargs):
        return _dyad_covariate(
            effect,
            riskset,
            current_time,
            directed=directed,
        )
    _validate_endogenous_effect_arguments(effect)
    if name in {"inertia", "tie", "dyad"}:
        return np.array(
            [
                _weighted_count(
                    previous,
                    (previous["sender_id"] == row.sender_id)
                    & (previous["receiver_id"] == row.receiver_id),
                )
                for row in riskset.itertuples()
            ],
            dtype=float,
        )
    if name == "reciprocity":
        if not directed:
            return _compute_effect("inertia", riskset, previous, directed)
        return np.array(
            [
                _weighted_count(
                    previous,
                    (previous["sender_id"] == row.receiver_id)
                    & (previous["receiver_id"] == row.sender_id),
                )
                for row in riskset.itertuples()
            ],
            dtype=float,
        )
    if name in {"send", "outdegreeSender"}:
        return np.array(
            [
                _weighted_count(previous, previous["sender_id"] == row.sender_id)
                for row in riskset.itertuples()
            ],
            dtype=float,
        )
    if name in {"receive", "indegreeReceiver"}:
        return np.array(
            [
                _weighted_count(previous, previous["receiver_id"] == row.receiver_id)
                for row in riskset.itertuples()
            ],
            dtype=float,
        )
    if name == "outdegreeReceiver":
        return np.array(
            [
                _weighted_count(previous, previous["sender_id"] == row.receiver_id)
                for row in riskset.itertuples()
            ],
            dtype=float,
        )
    if name == "indegreeSender":
        return np.array(
            [
                _weighted_count(previous, previous["receiver_id"] == row.sender_id)
                for row in riskset.itertuples()
            ],
            dtype=float,
        )
    if name == "totaldegreeSender":
        return np.asarray(
            _compute_effect("send", riskset, previous, directed)
            + _compute_effect("indegreeSender", riskset, previous, directed),
            dtype=float,
        )
    if name == "totaldegreeReceiver":
        return np.asarray(
            _compute_effect("receive", riskset, previous, directed)
            + _compute_effect("outdegreeReceiver", riskset, previous, directed),
            dtype=float,
        )
    if name == "totaldegreeDyad":
        return np.asarray(
            _compute_effect("totaldegreeSender", riskset, previous, directed)
            + _compute_effect("totaldegreeReceiver", riskset, previous, directed),
            dtype=float,
        )
    if name == "degreeDiff":
        return np.asarray(
            np.abs(
                _compute_effect("totaldegreeSender", riskset, previous, directed)
                - _compute_effect("totaldegreeReceiver", riskset, previous, directed)
            ),
            dtype=float,
        )
    if name == "degreeMax":
        return np.asarray(
            np.maximum(
                _compute_effect("totaldegreeSender", riskset, previous, directed),
                _compute_effect("totaldegreeReceiver", riskset, previous, directed),
            ),
            dtype=float,
        )
    if name == "degreeMin":
        return np.asarray(
            np.minimum(
                _compute_effect("totaldegreeSender", riskset, previous, directed),
                _compute_effect("totaldegreeReceiver", riskset, previous, directed),
            ),
            dtype=float,
        )
    if name in {"osp", "isp", "otp", "itp", "sp", "spUnique"}:
        unique = bool(_effect_kwargs(effect).get("unique", False)) or name == "spUnique"
        return _shared_partner_effect(name, riskset, previous, unique=unique)
    if name.startswith("ps"):
        return _participation_shift(name, riskset, previous, directed=directed)
    if name.startswith("recency"):
        return _recency_effect(name, riskset, previous)
    if name in {"rrankReceive", "rrankSend"}:
        return _rank_effect(name, riskset, previous)
    if name == "activeTie":
        return (_compute_effect("tie", riskset, previous, directed) > 0).astype(float)
    if name == "activeReciprocalTie":
        return (_compute_effect("reciprocity", riskset, previous, directed) > 0).astype(float)
    if name == "activeIndegreeReceiver":
        return (_compute_effect("indegreeReceiver", riskset, previous, directed) > 0).astype(float)
    if name == "activeOutdegreeSender":
        return (_compute_effect("outdegreeSender", riskset, previous, directed) > 0).astype(float)
    if name == "activeTotaldegreeReceiver":
        return (_compute_effect("totaldegreeReceiver", riskset, previous, directed) > 0).astype(
            float
        )
    if name == "activeTotaldegreeSender":
        return (_compute_effect("totaldegreeSender", riskset, previous, directed) > 0).astype(float)
    if name == "activeTotaldegreeDyad":
        return (_compute_effect("totaldegreeDyad", riskset, previous, directed) > 0).astype(float)
    if name == "activeDegreeDyad":
        return (_compute_effect("totaldegreeDyad", riskset, previous, directed) > 0).astype(float)
    if name == "activeDegreeMin":
        return (_compute_effect("degreeMin", riskset, previous, directed) > 0).astype(float)
    if name == "activeDegreeMax":
        return (_compute_effect("degreeMax", riskset, previous, directed) > 0).astype(float)
    if name == "activeSharedPartners":
        return (_compute_effect("sp", riskset, previous, directed) > 0).astype(float)
    if name == "activeSharedPartners_isp":
        return (_compute_effect("isp", riskset, previous, directed) > 0).astype(float)
    if name == "activeSharedPartners_itp":
        return (_compute_effect("itp", riskset, previous, directed) > 0).astype(float)
    if name == "activeSharedPartners_osp":
        return (_compute_effect("osp", riskset, previous, directed) > 0).astype(float)
    if name == "activeSharedPartners_otp":
        return (_compute_effect("otp", riskset, previous, directed) > 0).astype(float)
    if len(effect.components) == 2:
        left, right = effect.components
        return np.asarray(
            _compute_effect(
                left,
                riskset,
                previous,
                directed,
                event_index=event_index,
                current_time=current_time,
            )
            * _compute_effect(
                right,
                riskset,
                previous,
                directed,
                event_index=event_index,
                current_time=current_time,
            ),
            dtype=float,
        )
    raise ValueError(f"effect kernel is not registered: {name}")


def _effect_kwargs(effect: Effect) -> dict[str, Any]:
    return dict(effect.kwargs)


def _attach_deprecated_attributes(
    parsed: Formula,
    attr_actors: pd.DataFrame | None,
    attr_dyads: pd.DataFrame | np.ndarray | None,
) -> Formula:
    actor_effects = {"send", "receive", "same", "difference", "average", "minimum", "maximum"}
    dyad_effects = {"tie", "dyad"}
    terms: list[Effect] = []
    for effect in parsed.terms:
        terms.append(
            _attach_effect_attributes(effect, actor_effects, dyad_effects, attr_actors, attr_dyads)
        )
    return Formula(tuple(terms), parsed.intercept)


def _validate_deprecated_actor_attributes(
    parsed: Formula,
    attributes: pd.DataFrame,
    history: EventHistory,
) -> None:
    actor_effect_names = {
        "send",
        "receive",
        "same",
        "difference",
        "average",
        "minimum",
        "maximum",
    }
    frame = pd.DataFrame(attributes).copy()
    actor_column = next(
        (column for column in ("name", "actor", "id", "actor_id") if column in frame),
        None,
    )
    if actor_column is None:
        raise ValueError("attr_actors must contain a name column")
    if "time" in frame and frame["time"].isna().any():
        raise ValueError("Missing values in the attr_actors time column")

    def actor_effects(effect: Effect) -> list[Effect]:
        selected = [effect] if effect.name in actor_effect_names else []
        for component in effect.components:
            selected.extend(actor_effects(component))
        return selected

    selected = [
        effect
        for term in parsed.terms
        for effect in actor_effects(term)
    ]
    for effect in selected:
        variable = _variable_name(effect)
        if variable not in frame:
            raise ValueError(f"{variable!r} not in attr_actors")
        if frame[variable].isna().any():
            warnings.warn("Missing values in attr_actors", UserWarning, stacklevel=3)

    expected = set(history.actors["actor"].to_list())
    supplied = set(frame[actor_column].to_list())
    missing = expected.difference(supplied)
    if missing:
        raise ValueError(f"Missing actors in attr_actors: {sorted(missing)!r}")
    extra = supplied.difference(expected)
    if extra:
        warnings.warn(
            f"Actors {sorted(extra)!r} in attr_actors are not in the risk set",
            UserWarning,
            stacklevel=3,
        )


def _attach_effect_attributes(
    effect: Effect,
    actor_effects: set[str],
    dyad_effects: set[str],
    attr_actors: pd.DataFrame | None,
    attr_dyads: pd.DataFrame | np.ndarray | None,
) -> Effect:
    kwargs = dict(effect.kwargs)
    if effect.name in actor_effects and attr_actors is not None:
        kwargs.setdefault("attr_actors", attr_actors)
    if effect.name in dyad_effects and attr_dyads is not None:
        kwargs.setdefault("attr_dyads", attr_dyads)
    components = tuple(
        _attach_effect_attributes(component, actor_effects, dyad_effects, attr_actors, attr_dyads)
        for component in effect.components
    )
    return Effect(effect.name, effect.args, tuple(kwargs.items()), components)


def _interaction_effect(left: Effect, right: Effect) -> Effect:
    return Effect(
        f"{left.statistic_name}:{right.statistic_name}",
        components=(left, right),
    )


def _validate_endogenous_effect_arguments(effect: Effect) -> None:
    if effect.args:
        raise TypeError(
            f"{effect.name} does not accept positional arguments in this implementation"
        )
    kwargs = _effect_kwargs(effect)
    scaling = kwargs.pop("scaling", "none")
    consider_type = kwargs.pop("consider_type", "ignore")
    unique = kwargs.pop("unique", False)
    if scaling not in {"none", "as.is", "std", "prop"}:
        raise ValueError(f"invalid scaling for {effect.name}: {scaling!r}")
    if consider_type not in {"ignore", False}:
        raise NotImplementedError(
            f"consider_type={consider_type!r} is not implemented for {effect.name}"
        )
    if not isinstance(unique, (bool, np.bool_)):
        raise TypeError(f"unique must be boolean for {effect.name}")
    if unique and effect.name not in {"sp", "isp", "itp", "osp", "otp"}:
        raise TypeError(f"unique is not defined for {effect.name}")
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected arguments for {effect.name}: {names}")


def _is_endogenous_effect(effect: Effect) -> bool:
    name = effect.name
    exogenous = {
        "FEtype",
        "event",
        "userStat",
        "same",
        "difference",
        "average",
        "minimum",
        "maximum",
    }
    if name in exogenous:
        return False
    if name in {"send", "receive", "tie", "dyad"} and (effect.args or effect.kwargs):
        return not any(key == "variable" for key, _ in effect.kwargs) and not effect.args
    return True


def _scale_endogenous_proportion(
    values: np.ndarray,
    name: str,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    directed: bool,
) -> np.ndarray:
    event_count = len(previous)
    actor_ids = pd.concat(
        [riskset["sender_id"], riskset["receiver_id"]], ignore_index=True
    ).dropna()
    actor_count = int(actor_ids.nunique())
    if actor_count < 2:
        raise ValueError("proportional scaling requires at least two actors")
    fallback = np.full(len(values), 1.0 / actor_count, dtype=float)
    one_event_degree = {
        "indegreeSender",
        "outdegreeSender",
        "indegreeReceiver",
        "outdegreeReceiver",
        "degreeMin",
        "degreeMax",
        "degreeDiff",
        "send",
        "receive",
    }
    two_event_degree = {
        "totaldegreeSender",
        "totaldegreeReceiver",
        "totaldegreeDyad",
    }
    if name in one_event_degree:
        return fallback if event_count == 0 else np.asarray(values, dtype=float) / event_count
    if name in two_event_degree:
        return (
            fallback if event_count == 0 else np.asarray(values, dtype=float) / (2.0 * event_count)
        )
    if name in {"inertia", "tie", "dyad"}:
        if not directed:
            raise ValueError(
                "proportional scaling for inertia is not defined for undirected events"
            )
        denominator = _compute_effect("outdegreeSender", riskset, previous, directed)
        zero_value = 1.0 / (actor_count - 1)
    elif name == "reciprocity":
        if not directed:
            raise ValueError(
                "proportional scaling for reciprocity is not defined for undirected events"
            )
        denominator = _compute_effect("indegreeSender", riskset, previous, directed)
        zero_value = 1.0 / (actor_count - 1)
    else:
        raise ValueError(f"proportional scaling is not defined for {name}")
    return np.asarray(
        np.divide(
            np.asarray(values, dtype=float),
            denominator,
            out=np.full(len(values), zero_value, dtype=float),
            where=denominator != 0,
        ),
        dtype=float,
    )


def _effective_weights(previous: pd.DataFrame) -> pd.Series:
    if "__effective_weight" in previous.columns:
        return previous["__effective_weight"].astype(float)
    if "event_weight" in previous.columns:
        return previous["event_weight"].astype(float)
    return pd.Series(1.0, index=previous.index, dtype=float)


def _weighted_count(previous: pd.DataFrame, mask: pd.Series) -> float:
    if previous.empty:
        return 0.0
    return float(_effective_weights(previous).loc[mask].sum())


def _variable_name(effect: Effect) -> str:
    kwargs = _effect_kwargs(effect)
    value = effect.args[0] if effect.args else kwargs.get("variable")
    if not isinstance(value, str) or not value:
        raise TypeError(f"{effect.name} requires a non-empty string variable name")
    return value


def _actor_covariate(
    effect: Effect,
    riskset: pd.DataFrame,
    current_time: Any,
    *,
    transform: str,
) -> np.ndarray:
    kwargs = _effect_kwargs(effect)
    allowed = {"variable", "attr_actors", "scaling", "absolute"}
    unexpected = set(kwargs).difference(allowed)
    if unexpected:
        raise TypeError(f"unexpected arguments for {effect.name}: {', '.join(sorted(unexpected))}")
    variable = _variable_name(effect)
    if len(effect.args) > 2:
        raise TypeError(f"{effect.name} accepts at most variable and attr_actors positionally")
    attributes = effect.args[1] if len(effect.args) > 1 else kwargs.get("attr_actors")
    if attributes is None:
        raise ValueError(f"{effect.name} requires attr_actors")
    frame = pd.DataFrame(attributes).copy()
    actor_column = next(
        (column for column in ("name", "actor", "id", "actor_id") if column in frame.columns),
        None,
    )
    if actor_column is None or variable not in frame.columns:
        raise ValueError("attr_actors must contain an actor/name column and the requested variable")
    if "time" in frame.columns:
        try:
            eligible = frame[frame["time"].astype(float) <= float(current_time)]
        except (TypeError, ValueError):
            eligible = frame[frame["time"] <= current_time]
        frame = eligible.sort_values("time").groupby(actor_column, sort=False).tail(1)
    if frame[actor_column].duplicated().any():
        raise ValueError("attr_actors contains duplicate actor rows without a usable time column")
    lookup = dict(zip(frame[actor_column], frame[variable], strict=True))

    def values(column: str) -> np.ndarray:
        labels = riskset[column].to_list()
        missing = [label for label in labels if label not in lookup]
        if missing:
            raise ValueError(f"attr_actors is missing actor {missing[0]!r}")
        return np.asarray([lookup[label] for label in labels])

    sender = values("sender")
    receiver = values("receiver")
    if transform == "same":
        # Equality is meaningful for categorical actor attributes. Arithmetic transforms
        # below retain their explicit numeric requirement.
        result = (sender == receiver).astype(float)
        return _scale_covariate(result, kwargs.get("scaling", "none"), effect.name)
    try:
        sender = sender.astype(float)
        receiver = receiver.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{effect.name} requires a numeric actor attribute") from exc
    if transform == "send":
        result = sender
    elif transform == "receive":
        result = receiver
    elif transform == "difference":
        difference = sender - receiver
        result = np.abs(difference) if kwargs.get("absolute", True) else difference
    elif transform == "average":
        result = (sender + receiver) / 2.0
    elif transform == "minimum":
        result = np.minimum(sender, receiver)
    elif transform == "maximum":
        result = np.maximum(sender, receiver)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported actor transform: {transform}")
    return _scale_covariate(result, kwargs.get("scaling", "none"), effect.name)


def _dyad_covariate(
    effect: Effect,
    riskset: pd.DataFrame,
    current_time: Any,
    *,
    directed: bool,
) -> np.ndarray:
    kwargs = _effect_kwargs(effect)
    allowed = {"variable", "attr_dyads", "scaling", "x"}
    unexpected = set(kwargs).difference(allowed)
    if unexpected:
        raise TypeError(f"unexpected arguments for {effect.name}: {', '.join(sorted(unexpected))}")
    if "x" in kwargs:
        if effect.args or "variable" in kwargs or "attr_dyads" in kwargs:
            raise TypeError("x cannot be combined with variable or attr_dyads")
        matrix = np.asarray(kwargs["x"], dtype=float)
        actor_count = len(
            pd.unique(
                pd.concat(
                    [riskset["sender"], riskset["receiver"]],
                    ignore_index=True,
                )
            )
        )
        if matrix.shape != (actor_count, actor_count):
            raise ValueError("tie x dimensions must match the actor count")
        if not np.isfinite(matrix).all():
            raise ValueError("tie x contains missing values")
        if not directed and not np.array_equal(matrix, matrix.T):
            raise ValueError("tie x must be symmetric for undirected events")
        result = matrix[
            riskset["sender_id"].to_numpy(dtype=int) - 1,
            riskset["receiver_id"].to_numpy(dtype=int) - 1,
        ]
        return _scale_covariate(
            np.asarray(result, dtype=float),
            kwargs.get("scaling", "none"),
            effect.name,
        )
    variable = _variable_name(effect)
    attributes = kwargs.get("attr_dyads")
    if attributes is None:
        raise ValueError(f"{effect.name} requires attr_dyads")
    if isinstance(attributes, np.ndarray):
        matrix = np.asarray(attributes, dtype=float)
        max_actor = int(max(riskset["sender_id"].max(), riskset["receiver_id"].max()))
        if matrix.shape != (max_actor, max_actor):
            raise ValueError("attr_dyads matrix dimensions must match the actor count")
        result = matrix[
            riskset["sender_id"].to_numpy(dtype=int) - 1,
            riskset["receiver_id"].to_numpy(dtype=int) - 1,
        ]
    else:
        frame = pd.DataFrame(attributes).copy()
        sender_column = "actor1" if "actor1" in frame.columns else frame.columns[0]
        receiver_column = "actor2" if "actor2" in frame.columns else frame.columns[1]
        if variable not in frame.columns:
            raise ValueError(f"attr_dyads does not contain variable {variable!r}")
        if "time" in frame.columns:
            try:
                frame = frame[frame["time"].astype(float) <= float(current_time)]
            except (TypeError, ValueError):
                frame = frame[frame["time"] <= current_time]
            frame = (
                frame.sort_values("time")
                .groupby([sender_column, receiver_column], sort=False)
                .tail(1)
            )
        lookup = {
            (row[0], row[1]): row[2]
            for row in frame[[sender_column, receiver_column, variable]].itertuples(
                index=False, name=None
            )
        }
        result = np.asarray(
            [lookup.get((row.sender, row.receiver), np.nan) for row in riskset.itertuples()],
            dtype=float,
        )
        if np.isnan(result).any():
            raise ValueError("attr_dyads is missing one or more risk-set dyads")
    return _scale_covariate(result, kwargs.get("scaling", "none"), effect.name)


def _event_covariate(effect: Effect, riskset: pd.DataFrame, event_index: int) -> np.ndarray:
    kwargs = _effect_kwargs(effect)
    variable = _variable_name(effect)
    attributes = kwargs.get("event_attr")
    if attributes is None and len(effect.args) > 1:
        attributes = effect.args[1]
    if attributes is None:
        raise ValueError("event requires event_attr")
    if isinstance(attributes, Mapping):
        values = attributes[variable]
    elif isinstance(attributes, pd.DataFrame):
        if variable not in attributes.columns:
            raise ValueError(f"event_attr does not contain variable {variable!r}")
        values = attributes[variable]
    else:
        values = attributes
    sequence = list(values)
    if event_index >= len(sequence):
        raise ValueError("event_attr has fewer rows than the event history")
    return np.full(len(riskset), float(sequence[event_index]), dtype=float)


def _user_stat(effect: Effect, riskset: pd.DataFrame, event_index: int) -> np.ndarray:
    kwargs = _effect_kwargs(effect)
    values = effect.args[0] if effect.args else kwargs.get("x")
    if values is None:
        raise ValueError("userStat requires x")
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if event_index >= len(array):
            raise ValueError("userStat x has fewer event rows than the history")
        return np.full(len(riskset), array[event_index], dtype=float)
    if array.ndim != 2 or event_index >= array.shape[0]:
        raise ValueError("userStat x must be an event-by-risk-entry matrix")
    if array.shape[1] == len(riskset):
        return np.array(array[event_index], copy=True)
    dyad_indices = riskset["dyad_id"].to_numpy(dtype=int) - 1
    if dyad_indices.max(initial=-1) >= array.shape[1]:
        raise ValueError("userStat x does not contain all risk-set dyads")
    return np.asarray(array[event_index, dyad_indices], dtype=float)


def _scale_covariate(values: np.ndarray, scaling: Any, effect_name: str) -> np.ndarray:
    if scaling in {"none", "as.is"}:
        return np.asarray(values, dtype=float)
    if scaling == "std":
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if sd == 0 or not np.isfinite(sd):
            return np.zeros_like(values, dtype=float)
        return (np.asarray(values, dtype=float) - float(np.mean(values))) / sd
    raise ValueError(f"invalid scaling for {effect_name}: {scaling!r}")


def _shared_partner_effect(
    name: str,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    *,
    unique: bool = False,
) -> np.ndarray:
    weights: dict[tuple[int, int], float] = {}
    for row, weight in zip(previous.itertuples(), _effective_weights(previous), strict=True):
        edge = (int(row.sender_id), int(row.receiver_id))
        weights[edge] = weights.get(edge, 0.0) + float(weight)
    actors = set(previous["sender_id"].astype(int)).union(previous["receiver_id"].astype(int))
    values: list[float] = []
    for row in riskset.itertuples():
        sender = int(row.sender_id)
        receiver = int(row.receiver_id)
        count = 0.0
        for actor in actors.difference({sender, receiver}):
            if name == "osp":
                path = (weights.get((sender, actor), 0.0), weights.get((receiver, actor), 0.0))
            elif name == "isp":
                path = (weights.get((actor, sender), 0.0), weights.get((actor, receiver), 0.0))
            elif name == "otp":
                path = (weights.get((sender, actor), 0.0), weights.get((actor, receiver), 0.0))
            elif name == "itp":
                path = (weights.get((receiver, actor), 0.0), weights.get((actor, sender), 0.0))
            else:
                first = weights.get((sender, actor), 0.0) + weights.get((actor, sender), 0.0)
                second = weights.get((receiver, actor), 0.0) + weights.get((actor, receiver), 0.0)
                path = (first, second)
            contribution = min(path)
            if unique or name == "spUnique":
                contribution = float(contribution > 0)
            count += contribution
        values.append(count)
    return np.asarray(values, dtype=float)


def _participation_shift(
    name: str,
    riskset: pd.DataFrame,
    previous: pd.DataFrame,
    *,
    directed: bool,
) -> np.ndarray:
    if previous.empty:
        return np.zeros(len(riskset), dtype=float)
    latest_time = previous.iloc[-1]["time"]
    latest_events = previous[previous["time"] == latest_time]
    values = np.zeros(len(riskset), dtype=float)
    for last in latest_events.itertuples():
        a = int(last.sender_id)
        b = int(last.receiver_id)
        for position, row in enumerate(riskset.itertuples()):
            x = int(row.sender_id)
            y = int(row.receiver_id)
            if name == "psABAB":
                value = x == a and y == b
            elif name in {"psABA", "psABBA"}:
                value = x == b and y == a
            elif name == "psABAY":
                if directed:
                    value = x == a and y not in {a, b}
                else:
                    value = len({x, y}.intersection({a, b})) == 1
            elif name == "psABB":
                value = x == b and y == b
            elif name == "psABBY":
                value = x == b and y not in {a, b}
            elif name == "psABX":
                value = x not in {a, b}
            elif name == "psABXA":
                value = x not in {a, b} and y == a
            elif name == "psABXB":
                value = x not in {a, b} and y == b
            elif name == "psABXY":
                value = x not in {a, b} and y not in {a, b, x}
            else:
                value = False
            values[position] += float(value)
    return values


def _recency_effect(name: str, riskset: pd.DataFrame, previous: pd.DataFrame) -> np.ndarray:
    values: list[float] = []
    indexed = list(previous.itertuples())
    last_index = len(indexed)
    for row in riskset.itertuples():
        sender = int(row.sender_id)
        receiver = int(row.receiver_id)
        value = 0.0
        for event_position, event in reversed(list(enumerate(indexed, start=1))):
            event_sender = int(event.sender_id)
            event_receiver = int(event.receiver_id)
            match = False
            if name == "recencyContinue":
                match = event_sender == sender and event_receiver == receiver
            elif name == "recencyReceiveReceiver":
                match = event_receiver == receiver
            elif name == "recencyReceiveSender":
                match = event_receiver == sender
            elif name == "recencySendReceiver":
                match = event_sender == receiver
            elif name == "recencySendSender":
                match = event_sender == sender
            if match:
                distance = last_index - event_position + 1
                value = 1.0 / (distance + 1.0)
                break
        values.append(value)
    return np.asarray(values, dtype=float)


def _rank_effect(name: str, riskset: pd.DataFrame, previous: pd.DataFrame) -> np.ndarray:
    if previous.empty:
        return np.zeros(len(riskset), dtype=float)
    values: list[float] = []
    for row in riskset.itertuples():
        sender = int(row.sender_id)
        receiver = int(row.receiver_id)
        if name == "rrankSend":
            contacts = previous.loc[previous["sender_id"] == sender, "receiver_id"]
        else:
            contacts = previous.loc[previous["receiver_id"] == sender, "sender_id"]
        recency_order = list(dict.fromkeys(reversed(contacts.astype(int).to_list())))
        try:
            rank = recency_order.index(receiver) + 1
        except ValueError:
            values.append(0.0)
        else:
            values.append(1.0 / rank)
    return np.asarray(values, dtype=float)


def _parse_formula(source: str) -> Formula:
    text = source.strip()
    if text.startswith("~"):
        text = text[1:].strip()
    if not text:
        return Formula(())
    tokens = _split_top_level(text, "+")
    terms: list[Effect] = []
    intercept: bool | None = None
    for token in tokens:
        token = token.strip()
        if token == "1":
            intercept = True
            continue
        if token in {"0", "-1"}:
            intercept = False
            continue
        if "*" in token:
            left, right = [_parse_call(part.strip()) for part in _split_top_level(token, "*")]
            terms.extend([left, right, _interaction_effect(left, right)])
        elif _has_top_level_sep(token, ":"):
            left, right = [_parse_call(part.strip()) for part in _split_top_level(token, ":")]
            terms.extend([left, right, _interaction_effect(left, right)])
        else:
            terms.append(_parse_call(token))
    return Formula(tuple(terms), intercept)


def _parse_call(token: str) -> Effect:
    token = token.removeprefix("remflow::").removeprefix("remstats::")
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\((.*)\))?", token)
    if not match:
        raise ValueError(f"invalid formula term: {token}")
    name, args_source = match.groups()
    if name not in _EFFECT_NAMES:
        raise ValueError(f"unknown effect: {name}")
    if args_source and args_source.strip():
        args: list[Any] = []
        kwargs: list[tuple[str, Any]] = []
        for item in _split_top_level(args_source, ","):
            item = item.strip()
            if not item:
                continue
            keyword = _split_keyword_argument(item)
            if keyword is None:
                if kwargs:
                    raise ValueError("positional formula arguments cannot follow keyword arguments")
                args.append(_parse_formula_literal(item))
            else:
                key, value = keyword
                if any(existing == key for existing, _ in kwargs):
                    raise ValueError(f"duplicate formula argument: {key}")
                kwargs.append((key, _parse_formula_literal(value)))
        effect = Effect(name, tuple(args), tuple(kwargs))
        _validate_effect_constructor(effect)
        return effect
    effect = Effect(name)
    _validate_effect_constructor(effect)
    return effect


def _split_keyword_argument(value: str) -> tuple[str, str] | None:
    depth = 0
    quote: str | None = None
    for index, char in enumerate(value):
        if quote is not None:
            if char == quote and (index == 0 or value[index - 1] != "\\"):
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            key = value[:index].strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"invalid formula keyword argument: {key!r}")
            return key, value[index + 1 :].strip()
    return None


def _parse_formula_literal(value: str) -> Any:
    aliases = {"TRUE": True, "FALSE": False, "NULL": None, "Inf": float("inf")}
    if value in aliases:
        return aliases[value]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", value):
            return value
        raise ValueError(f"unsupported formula argument value: {value!r}") from None


def _split_top_level(text: str, sep: str) -> list[str]:
    depth = 0
    parts: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == sep and depth == 0 and not _is_namespace_colon(text, index):
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def _has_top_level_sep(text: str, sep: str) -> bool:
    return len(_split_top_level(text, sep)) > 1


def _is_namespace_colon(text: str, index: int) -> bool:
    return text[index] == ":" and (
        (index > 0 and text[index - 1] == ":") or (index + 1 < len(text) and text[index + 1] == ":")
    )


def _match_arg(value: Sequence[str] | str, choices: Sequence[str], name: str) -> str:
    selected = value[0] if isinstance(value, Sequence) and not isinstance(value, str) else value
    if selected not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return str(selected)


__all__ = [
    "Effect",
    "Formula",
    "RemStats",
    "TomStats",
    "AomStats",
    "formula",
    "remstats",
    "tomstats",
    "aomstats",
    "stack_stats",
    "bind_remstats",
    "select_stats",
    "tie_effects",
    "actor_effects",
    "is_remstats_durem",
    "observed_risk_index",
    *_EFFECT_NAMES,
]
