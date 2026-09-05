"""High-level relational-event model facades."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import pandas as pd

from remflow.estimate import RemEstimate, remstimate
from remflow.history import EventHistory, remify
from remflow.stats import (
    Effect,
    Formula,
    RemStats,
    formula,
    indegreeReceiver,
    inertia,
    otp,
    outdegreeSender,
    reciprocity,
    remstats,
)

_EFFECT_ALIASES = {
    "reciprocity": reciprocity,
    "sender_activity": outdegreeSender,
    "receiver_popularity": indegreeReceiver,
    "triadic_closure": otp,
    "inertia": inertia,
}


class RelationalEventModel:
    """Fit and inspect a general tie-oriented relational event model.

    Parameters
    ----------
    effects:
        High-level aliases such as ``sender_activity`` and
        ``receiver_popularity``, or native :class:`~remflow.Effect` objects.
    event_type:
        Optional column containing event types. When omitted, ``event_type``
        or ``type`` is detected automatically. Untyped events are supported.
    event_attributes:
        Optional event columns to preserve in the normalized history.
    extend_riskset_by_type:
        Whether typed risk sets contain every dyad-type combination. The
        default enables expansion when more than one event type is observed.
    backend:
        ``numpy``, ``jax``, ``jax:cpu``, or ``jax:gpu``. A requested GPU never
        silently falls back to CPU.
    ordinal:
        If ``False`` (default), fit an exact-time intensity model. If ``True``,
        fit the ordinal next-event choice model.
    """

    def __init__(
        self,
        effects: Sequence[str | Effect] = (
            "reciprocity",
            "sender_activity",
            "receiver_popularity",
            "triadic_closure",
        ),
        *,
        backend: str = "numpy",
        ordinal: bool = False,
        directed: bool = True,
        riskset: str = "full",
        event_type: str | None = None,
        event_attributes: str | Sequence[str] | None = None,
        extend_riskset_by_type: bool | None = None,
    ) -> None:
        self.effects = tuple(effects)
        self.backend = backend
        self.ordinal = ordinal
        self.directed = directed
        self.riskset = riskset
        self.event_type = event_type
        self.event_attributes = _attribute_names(event_attributes)
        self.extend_riskset_by_type = extend_riskset_by_type
        self.history_: EventHistory | None = None
        self.stats_: RemStats | None = None
        self.fit_result_: RemEstimate | None = None
        self.events_: pd.DataFrame | None = None
        self._formula: Formula | None = None
        self._actor_attributes: pd.DataFrame | None = None
        self._event_type_column: str | None = None
        self._event_attribute_columns: tuple[str, ...] = ()

    def fit(self, events: Any) -> Self:
        """Fit the model and return ``self``."""

        frame = self._coerce_events(events)
        event_type = self._resolve_event_type(frame)
        event_attributes = self._resolve_event_attributes(frame)
        actor_attributes = self._actor_attributes_for(frame)
        model_formula = self._effect_formula(actor_attributes)
        extend_types = self._extend_types(frame, event_type)
        history = remify(
            frame,
            directed=self.directed,
            ordinal=self.ordinal,
            riskset=self.riskset,
            event_type=event_type,
            event_attributes=event_attributes,
            extend_riskset_by_type=extend_types,
        )
        statistics = remstats(history, tie_effects=model_formula, first=2)
        if not isinstance(statistics, RemStats):
            raise RuntimeError("RelationalEventModel requires tie-oriented statistics")
        fitted = remstimate(history, statistics, backend=self.backend)
        if not isinstance(fitted, RemEstimate):
            raise RuntimeError("RelationalEventModel received an actor-oriented fit")
        self.events_ = frame
        self.history_ = history
        self.stats_ = statistics
        self.fit_result_ = fitted
        self._formula = model_formula
        self._actor_attributes = actor_attributes
        self._event_type_column = event_type
        self._event_attribute_columns = event_attributes
        return self

    def _coerce_events(self, events: Any) -> pd.DataFrame:
        return _coerce_relational_events(events)

    def _resolve_event_type(self, frame: pd.DataFrame) -> str | None:
        if self.event_type is not None:
            if self.event_type not in frame.columns:
                raise ValueError(f"event type column not found: {self.event_type}")
            return self.event_type
        return next((name for name in ("event_type", "type") if name in frame.columns), None)

    def _resolve_event_attributes(self, frame: pd.DataFrame) -> tuple[str, ...]:
        missing = [name for name in self.event_attributes if name not in frame.columns]
        if missing:
            raise ValueError(f"event attribute columns not found: {missing}")
        return self.event_attributes

    def _actor_attributes_for(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        return None

    def _effect_formula(self, actor_attributes: pd.DataFrame | None) -> Formula:
        del actor_attributes
        return _effect_formula(self.effects)

    def _extend_types(self, frame: pd.DataFrame, event_type: str | None) -> bool:
        if self.extend_riskset_by_type is not None:
            return self.extend_riskset_by_type
        return bool(event_type and frame[event_type].nunique(dropna=True) > 1)

    @property
    def coef_(self) -> np.ndarray:
        return np.array(self._require_fit().coef, copy=True)

    def summary(self) -> dict[str, Any]:
        """Return coefficient, backend, data, and effect metadata."""

        fitted = self._require_fit()
        result = fitted.summary()
        result["effects"] = [
            effect if isinstance(effect, str) else effect.statistic_name for effect in self.effects
        ]
        result["events"] = int(len(self.events_)) if self.events_ is not None else 0
        result["actors"] = self.history_.N if self.history_ is not None else 0
        result["timing"] = fitted.metadata.get("timing")
        return result

    def predict_next_events(self, top_k: int = 10) -> pd.DataFrame:
        """Rank candidate sender-receiver events after the fitted history."""

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        fitted = self._require_fit()
        history, statistics = self._next_event_design()
        try:
            coefficient_columns = [statistics.names.index(name) for name in fitted.names]
        except ValueError as exc:
            raise RuntimeError(
                "prediction statistics do not match the fitted coefficient names"
            ) from exc
        design = statistics.stats[0][:, coefficient_columns]
        eta = design @ fitted.coef
        eta -= float(np.max(eta))
        probabilities = np.exp(eta)
        probabilities /= probabilities.sum()
        riskset = history.risksets[statistics.event_indices[0]].copy()
        columns = ["sender", "receiver"]
        if "event_type" in riskset.columns:
            columns.append("event_type")
        result = riskset[columns].assign(probability=probabilities)
        return (
            result.sort_values("probability", ascending=False, kind="stable")
            .head(top_k)
            .reset_index(drop=True)
        )

    def _candidate_count(self) -> int:
        history, statistics = self._next_event_design()
        return len(history.risksets[statistics.event_indices[0]])

    def _next_event_design(self) -> tuple[EventHistory, RemStats]:
        self._require_fit()
        assert self.events_ is not None
        assert self.history_ is not None
        assert self._formula is not None
        frame = self.events_.copy()
        actors = self.history_.actors["actor"].to_list()
        if len(actors) < 2:
            raise ValueError("next-event prediction requires at least two actors")
        dummy: dict[str, Any] = {
            "time": _next_time(frame["time"]),
            "sender": actors[0],
            "receiver": actors[1],
        }
        if self._event_type_column is not None:
            dummy[self._event_type_column] = _representative_value(
                frame[self._event_type_column], first=True
            )
        for attribute in self._event_attribute_columns:
            dummy[attribute] = _representative_value(frame[attribute], first=False)
        extended = pd.concat([frame, pd.DataFrame([dummy])], ignore_index=True)
        history = remify(
            extended,
            actors=actors,
            directed=self.directed,
            ordinal=self.ordinal,
            riskset=self.riskset,
            event_type=self._event_type_column,
            event_attributes=self._event_attribute_columns,
            extend_riskset_by_type=self._extend_types(extended, self._event_type_column),
        )
        statistics = remstats(
            history,
            tie_effects=self._formula,
            first=len(extended),
            last=len(extended),
        )
        if not isinstance(statistics, RemStats):
            raise RuntimeError("RelationalEventModel requires tie-oriented statistics")
        return history, statistics

    def _require_fit(self) -> RemEstimate:
        if self.fit_result_ is None:
            raise RuntimeError("fit must be called before this operation")
        return self.fit_result_


def _coerce_relational_events(events: Any) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        frame = events.copy()
    else:
        rows = list(events)
        if rows and not isinstance(rows[0], dict):
            widths = {len(row) for row in rows}
            if len(widths) != 1 or next(iter(widths)) not in {3, 4}:
                raise ValueError("general event tuples must have 3 or 4 fields")
            columns = ["time", "sender", "receiver", "event_type"][: next(iter(widths))]
            frame = pd.DataFrame(rows, columns=columns)
        else:
            frame = pd.DataFrame(rows)
    frame = _normalize_event_columns(frame)
    required = {"time", "sender", "receiver"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"events are missing required columns: {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError("at least two relational events are required")
    return frame.reset_index(drop=True)


def _normalize_event_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "time": ("start_time", "event_time", "timestamp"),
        "sender": ("actor1", "source", "from"),
        "receiver": ("actor2", "target", "to"),
    }
    renamed: dict[str, str] = {}
    for canonical, alternatives in aliases.items():
        if canonical not in frame.columns:
            alias = next((name for name in alternatives if name in frame.columns), None)
            if alias is not None:
                renamed[alias] = canonical
    return frame.rename(columns=renamed)


def _attribute_names(attributes: str | Sequence[str] | None) -> tuple[str, ...]:
    if attributes is None:
        return ()
    if isinstance(attributes, str):
        return (attributes,)
    return tuple(attributes)


def _effect_formula(effects: Sequence[str | Effect]) -> Formula:
    terms: list[Effect] = []
    for requested in effects:
        if isinstance(requested, Effect):
            terms.append(requested)
        elif requested in _EFFECT_ALIASES:
            terms.append(_EFFECT_ALIASES[requested]())
        else:
            raise ValueError(f"unknown high-level effect: {requested}")
    return formula(Formula(tuple(terms)))


def _next_time(values: pd.Series) -> Any:
    last = values.iloc[-1]
    if pd.api.types.is_numeric_dtype(values.dtype):
        differences = pd.to_numeric(values).diff().dropna()
        step = float(differences[differences > 0].median()) if (differences > 0).any() else 1.0
        return float(last) + step
    timestamp = pd.Timestamp(last)
    return timestamp + pd.Timedelta(seconds=1)


def _representative_value(values: pd.Series, *, first: bool) -> Any:
    available = values.dropna()
    if available.empty:
        return np.nan
    return available.iloc[0 if first else -1]


__all__ = ["RelationalEventModel"]
