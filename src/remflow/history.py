"""Relational event history construction."""

from __future__ import annotations

import io
import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

RisksetMode = Literal["full", "active", "active_saturated", "manual"]


@dataclass(frozen=True)
class EventHistory:
    """Normalized relational event history."""

    events: pd.DataFrame
    actors: pd.DataFrame
    risksets: list[pd.DataFrame]
    directed: bool
    ordinal: bool
    model: str
    event_types: list[Any]
    duration: bool = False
    riskset_mode: str = "full"
    extend_riskset_by_type: bool = False
    riskset_decode: str = "labels"
    sender_riskset: np.ndarray = field(default_factory=lambda: np.array([], dtype=int), repr=False)
    receiver_riskset: dict[Any, np.ndarray] = field(default_factory=dict, repr=False)
    durem: dict[str, Any] = field(default_factory=dict)
    weighted: bool = False

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    @property
    def dim(self) -> tuple[int, ...]:
        dimensions: list[int] = []
        if self.E != self.M:
            dimensions.append(self.E)
        dimensions.extend([self.M, self.N])
        if self.event_types:
            dimensions.append(self.C)
        dimensions.append(self.D)
        if self.riskset_mode in {"active", "manual"} and self.model == "tie":
            dimensions.append(self.activeD)
        return tuple(dimensions)

    @property
    def E(self) -> int:
        return len(self.events)

    @property
    def M(self) -> int:
        return int(self.events["time"].nunique())

    @property
    def N(self) -> int:
        return len(self.actors)

    @property
    def C(self) -> int:
        return max(1, len(self.event_types))

    @property
    def D(self) -> int:
        return len(self.risksets[0]) if self.risksets else 0

    @property
    def activeN(self) -> int:
        return int(len(self.sender_riskset))

    @property
    def activeD(self) -> int:
        return self.D if self.riskset_mode in {"active", "manual"} else 0

    @property
    def sender_map(self) -> pd.DataFrame:
        labels = dict(zip(self.actors["actor_id"], self.actors["actor"], strict=True))
        return pd.DataFrame(
            {
                "senderID": self.sender_riskset,
                "actorName": [labels[int(actor_id)] for actor_id in self.sender_riskset],
            }
        )

    @property
    def riskset_info(self) -> dict[str, Any] | None:
        if not self.risksets:
            return None
        included: pd.DataFrame | None
        riskset = self.risksets[0]
        if self.riskset_decode == "none":
            included = None
        elif self.riskset_decode == "ids":
            columns = ["dyad_id", "sender_id", "receiver_id"]
            if "type_id" in riskset.columns:
                columns.append("type_id")
            included = riskset[columns].rename(
                columns={
                    "dyad_id": "dyadID",
                    "sender_id": "actor1ID",
                    "receiver_id": "actor2ID",
                    "type_id": "typeID",
                }
            )
        else:
            columns = ["dyad_id", "sender", "receiver"]
            if "event_type" in riskset.columns:
                columns.append("event_type")
            included = riskset[columns].rename(
                columns={
                    "dyad_id": "dyadID",
                    "sender": "actor1",
                    "receiver": "actor2",
                    "event_type": "type",
                }
            )
        return {
            "mode": self.riskset_mode,
            "decode": self.riskset_decode,
            "with_type": self.extend_riskset_by_type and bool(self.event_types),
            "included": included,
            "riskset_idx": riskset["dyad_id"].to_numpy(dtype=int),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events.copy(),
            "actors": self.actors.copy(),
            "risksets": [riskset.copy() for riskset in self.risksets],
            "directed": self.directed,
            "ordinal": self.ordinal,
            "model": self.model,
            "event_types": list(self.event_types),
            "duration": self.duration,
            "weighted": self.weighted,
            "durem": dict(self.durem),
            "riskset_mode": self.riskset_mode,
            "extend_riskset_by_type": self.extend_riskset_by_type,
            "riskset_decode": self.riskset_decode,
            "riskset_info": self.riskset_info,
            "sender_riskset": np.array(self.sender_riskset, copy=True),
            "receiver_riskset": {
                key: np.array(value, copy=True) for key, value in self.receiver_riskset.items()
            },
            "sender_map": self.sender_map,
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize the complete history using a versioned, stable JSON schema."""

        payload = {
            "schema": "remflow.event_history",
            "schema_version": 1,
            "class": type(self).__name__,
            "events": _frame_to_json_value(self.events),
            "actors": _frame_to_json_value(self.actors),
            "risksets": [_frame_to_json_value(frame) for frame in self.risksets],
            "directed": self.directed,
            "ordinal": self.ordinal,
            "model": self.model,
            "event_types": self.event_types,
            "duration": self.duration,
            "weighted": self.weighted,
            "riskset_mode": self.riskset_mode,
            "extend_riskset_by_type": self.extend_riskset_by_type,
            "riskset_decode": self.riskset_decode,
            "sender_riskset": self.sender_riskset.tolist(),
            "receiver_riskset": [
                {"actor": actor, "receivers": receivers.tolist()}
                for actor, receivers in self.receiver_riskset.items()
            ],
            "durem": self.durem,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_scalar,
        )
        if path is not None:
            Path(path).write_text(serialized, encoding="utf-8")
        return serialized

    @classmethod
    def from_json(cls, value: str | Path) -> EventHistory:
        """Restore a history produced by :meth:`to_json`."""

        source = str(value)
        if isinstance(value, Path) or (
            not source.lstrip().startswith("{") and Path(source).exists()
        ):
            source = Path(source).read_text(encoding="utf-8")
        payload = json.loads(source)
        if payload.get("schema") != "remflow.event_history":
            raise ValueError("JSON does not contain a REMFlow EventHistory")
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported EventHistory JSON schema version")
        history_class = DurationHistory if payload.get("duration") else EventHistory
        return history_class(
            events=_frame_from_json_value(payload["events"]),
            actors=_frame_from_json_value(payload["actors"]),
            risksets=[_frame_from_json_value(frame) for frame in payload["risksets"]],
            directed=bool(payload["directed"]),
            ordinal=bool(payload["ordinal"]),
            model=str(payload["model"]),
            event_types=list(payload["event_types"]),
            duration=bool(payload["duration"]),
            weighted=bool(payload.get("weighted", False)),
            riskset_mode=str(payload["riskset_mode"]),
            extend_riskset_by_type=bool(payload["extend_riskset_by_type"]),
            riskset_decode=str(payload["riskset_decode"]),
            sender_riskset=np.asarray(payload["sender_riskset"], dtype=int),
            receiver_riskset={
                item["actor"]: np.asarray(item["receivers"], dtype=int)
                for item in payload["receiver_riskset"]
            },
            durem=dict(payload["durem"]),
        )

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "events": self.E,
            "time_points": self.M,
            "actors": self.N,
            "event_types": self.C,
            "included_dyads": self.D,
            "directed": self.directed,
            "ordinal": self.ordinal,
            "model": self.model,
            "riskset": self.riskset_mode,
            "weighted": self.weighted,
        }
        if self.event_types:
            result["extend_riskset_by_type"] = self.extend_riskset_by_type
        if (
            self.riskset_mode in {"active", "manual"}
            and self.risksets
            and "event_type" in self.risksets[0]
        ):
            result["dyads_per_type"] = {
                key: int(value)
                for key, value in self.risksets[0]["event_type"].value_counts(sort=False).items()
            }
        if self.duration:
            result["duration"] = dict(self.durem)
        return result

    def __str__(self) -> str:
        summary = self.summary()
        lines = [
            "REMFlow relational event history",
            f"events = {summary['events']}",
            f"time points = {summary['time_points']}",
            f"actors = {summary['actors']}",
            f"event types = {summary['event_types']}",
            f"included dyads = {summary['included_dyads']}",
            f"directed = {summary['directed']}",
            f"riskset = {summary['riskset']}",
        ]
        if "extend_riskset_by_type" in summary:
            lines.append(f"extend_riskset_by_type = {summary['extend_riskset_by_type']}")
        if "dyads_per_type" in summary:
            counts = ", ".join(f"{key}={value}" for key, value in summary["dyads_per_type"].items())
            lines.append(f"per type: {counts}")
        if self.duration:
            lines.append(
                f"duration events: complete={self.durem.get('n_complete', 0)}, "
                f"censored={self.durem.get('n_censored', 0)}"
            )
        return "\n".join(lines)

    def plot(self, actors: Sequence[Any] | None = None) -> dict[str, pd.DataFrame]:
        """Return stable plot-ready event, actor, dyad, and waiting-time data."""

        available = self.actors["actor"].to_list()
        selected = available if actors is None else list(actors)
        unknown = [actor for actor in selected if actor not in available]
        if unknown:
            raise ValueError(f"actors are not present in this event history: {unknown!r}")
        if len(selected) > 50:
            warnings.warn(
                "more than 50 actors requested; selecting the 50 most active actors",
                UserWarning,
                stacklevel=2,
            )
            activity = pd.concat([self.events["sender"], self.events["receiver"]]).value_counts()
            selected = [actor for actor in activity.index if actor in selected][:50]
        mask = self.events["sender"].isin(selected) & self.events["receiver"].isin(selected)
        event_data = self.events.loc[mask].copy()
        if event_data.empty:
            raise ValueError("no events found for the selected actors")
        actor_data = pd.DataFrame({"actor": selected})
        sent = event_data["sender"].value_counts()
        received = event_data["receiver"].value_counts()
        actor_data["sent"] = actor_data["actor"].map(sent).fillna(0).astype(int)
        actor_data["received"] = actor_data["actor"].map(received).fillna(0).astype(int)
        dyad_columns = ["sender", "receiver"]
        dyad_data = event_data.groupby(dyad_columns, sort=False).size().reset_index(name="events")
        if self.ordinal:
            waiting = pd.DataFrame(columns=["event_id", "waiting_time"])
        else:
            numeric_time = pd.to_numeric(event_data["time"], errors="coerce")
            waiting = pd.DataFrame(
                {
                    "event_id": event_data["event_id"],
                    "waiting_time": numeric_time.diff(),
                }
            )
        return {
            "events": event_data,
            "actors": actor_data,
            "dyads": dyad_data,
            "waiting_times": waiting,
        }


class DurationHistory(EventHistory):
    """Duration relational event history marker."""


Remify = EventHistory
RemifyDuration = DurationHistory


def _frame_to_json_value(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(
        frame.to_json(orient="table", date_format="iso", index=False)
    )
    return result


def _frame_from_json_value(value: dict[str, Any]) -> pd.DataFrame:
    return pd.read_json(io.StringIO(json.dumps(value)), orient="table")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def remify(
    edgelist: Any,
    directed: bool = True,
    ordinal: bool = False,
    model: Sequence[str] | str = ("tie", "actor"),
    actors: Sequence[Any] | None = None,
    riskset: Sequence[str] | str = ("full", "active", "active_saturated", "manual"),
    manual_riskset: Any | None = None,
    extend_riskset_by_type: bool = False,
    event_type: str | Sequence[Any] | None = None,
    event_weight: str | Sequence[float] | None = None,
    origin: Any | None = None,
    time_units: Sequence[str] | str = (
        "auto",
        "secs",
        "mins",
        "hours",
        "days",
        "weeks",
        "months",
        "years",
    ),
    aggregate_time: float = 1,
    attach_riskset: bool = True,
    riskset_decode: Sequence[str] | str = ("labels", "ids", "none"),
    riskset_max_decode: int = 200_000,
    event_attributes: str | Sequence[str] | None = None,
    ncores: int = 1,
    duration: bool = False,
    dur_directed_end: bool = False,
    dur_type_exclusive: bool = False,
) -> EventHistory:
    """Create a normalized relational event history.

    Input data may use `sender`/`receiver`, `actor1`/`actor2`, `source`/`target`,
    or `from`/`to` columns. Public actor and event IDs are 1-based. ``ncores``
    controls deterministic concurrent risk-set construction; ``1`` uses the
    reference serial path.
    """

    model_value = _match_arg(model, ("tie", "actor"), "model")
    riskset_value = _match_arg(riskset, ("full", "active", "active_saturated", "manual"), "riskset")
    time_units_value = _match_arg(
        time_units,
        ("auto", "secs", "mins", "hours", "days", "weeks", "months", "years"),
        "time_units",
    )
    riskset_decode_value = _match_arg(riskset_decode, ("labels", "ids", "none"), "riskset_decode")
    if model_value == "actor" and not directed:
        raise ValueError("actor-oriented models require directed=True")
    if model_value == "actor" and extend_riskset_by_type:
        raise ValueError("extend_riskset_by_type=True is not supported for actor models")
    aggregate_value = _validate_aggregate_time(aggregate_time)
    if not isinstance(riskset_max_decode, int) or riskset_max_decode <= 0:
        raise ValueError("riskset_max_decode must be a positive integer")
    ncores_value = _validate_ncores(ncores)
    if duration and isinstance(model, Sequence) and not isinstance(model, str) and len(model) > 1:
        warnings.warn(
            "model set to 'tie' by default for duration history", UserWarning, stacklevel=2
        )
    if duration and dur_type_exclusive and not extend_riskset_by_type:
        warnings.warn(
            "dur_type_exclusive has no effect unless extend_riskset_by_type=True",
            UserWarning,
            stacklevel=2,
        )

    frame = _coerce_edgelist(edgelist)
    sender_col = _first_present(frame, ("sender", "actor1", "source", "from"))
    receiver_col = _first_present(frame, ("receiver", "actor2", "target", "to"))
    time_col = _first_present(
        frame, ("time", "start_time", "event_time", "timestamp"), required=False
    )
    if sender_col is None or receiver_col is None:
        raise ValueError("edgelist must contain sender/receiver or actor1/actor2 columns")

    events = pd.DataFrame()
    events["event_id"] = range(1, len(frame) + 1)
    events["time"] = frame[time_col].to_list() if time_col else list(range(1, len(frame) + 1))
    events["sender"] = frame[sender_col].to_list()
    events["receiver"] = frame[receiver_col].to_list()
    events["event_type"] = _event_column(frame, event_type, len(events))
    events["event_weight"] = _weight_column(frame, event_weight, len(events))
    duration_end: pd.Series | None = None
    if duration:
        duration_end = _duration_end_values(frame, time_col)
        events["end"] = duration_end.to_list()
        who_ended = _duration_who_ended(frame, dur_directed_end)
        if who_ended is not None:
            events["who_ended"] = who_ended
    for attribute in _event_attribute_names(event_attributes):
        if attribute not in frame.columns:
            raise ValueError(f"event attribute column not found: {attribute}")
        if attribute in events.columns:
            raise ValueError(f"event attribute conflicts with a reserved column: {attribute}")
        events[attribute] = frame[attribute].to_list()
    events["time"] = _normalize_event_times(events["time"], origin, time_units_value)
    if duration_end is not None:
        events["end"] = _normalize_optional_event_times(duration_end, origin, time_units_value)
        complete = events["end"].notna()
        if (events.loc[complete, "end"] < events.loc[complete, "time"]).any():
            raise ValueError("End time cannot be before start time")
    events["time"] = _aggregate_event_times(events["time"], aggregate_value)
    if ordinal:
        if duration and "end" in events.columns:
            _ordinal_duration_event_times(events)
        else:
            events["time"] = _ordinal_event_times(events["time"])

    dictionary = _actor_dictionary(events, actors)
    id_by_label = dict(zip(dictionary["actor"], dictionary["actor_id"], strict=True))
    events["sender_id"] = events["sender"].map(id_by_label)
    events["receiver_id"] = events["receiver"].map(id_by_label)
    if not directed:
        reverse = events["sender_id"] > events["receiver_id"]
        if reverse.any():
            old_sender_ids = events.loc[reverse, "sender_id"].copy()
            old_sender = events.loc[reverse, "sender"].copy()
            events.loc[reverse, "sender_id"] = events.loc[reverse, "receiver_id"].to_numpy()
            events.loc[reverse, "receiver_id"] = old_sender_ids.to_numpy()
            events.loc[reverse, "sender"] = events.loc[reverse, "receiver"].to_numpy()
            events.loc[reverse, "receiver"] = old_sender.to_numpy()
    events["dyad_id"] = [
        _dyad_id(sid, rid, len(dictionary), directed)
        for sid, rid in zip(events["sender_id"], events["receiver_id"], strict=True)
    ]

    type_values = [
        value for value in _stable_unique(events["event_type"]) if not pd.isna(value)
    ]
    try:
        types = sorted(type_values)
    except TypeError:
        types = sorted(type_values, key=lambda value: (type(value).__name__, str(value)))
    type_ids = {value: index for index, value in enumerate(types, start=1)}
    events["type_id"] = events["event_type"].map(type_ids).astype("Int64")
    if extend_riskset_by_type and types:
        dyad_count = len(dictionary) * (len(dictionary) - 1)
        if not directed:
            dyad_count //= 2
        events["dyad_id"] = events["dyad_id"] + (events["type_id"] - 1) * dyad_count
    risksets: list[pd.DataFrame] = []
    if attach_riskset:
        observed_events: list[tuple[int, int, Any]] = [
            (int(sender), int(receiver), event_type)
            for sender, receiver, event_type in zip(
                events["sender_id"],
                events["receiver_id"],
                events["event_type"],
                strict=True,
            )
        ]
        riskset_builder = partial(
            _riskset_for_event,
            events=events,
            dictionary=dictionary,
            directed=directed,
            mode=riskset_value,  # type: ignore[arg-type]
            manual_riskset=manual_riskset,
            extend_by_type=extend_riskset_by_type,
            event_types=types,
        )
        if ncores_value == 1 or len(observed_events) < 2:
            risksets = [riskset_builder(observed) for observed in observed_events]
        else:
            with ThreadPoolExecutor(
                max_workers=ncores_value,
                thread_name_prefix="remflow-riskset",
            ) as executor:
                risksets = list(executor.map(riskset_builder, observed_events))

    effective_decode = riskset_decode_value
    if effective_decode == "labels" and risksets and len(risksets[0]) > riskset_max_decode:
        warnings.warn(
            "risk-set label decoding exceeds riskset_max_decode; using ID-only table",
            UserWarning,
            stacklevel=2,
        )
        effective_decode = "ids"
    sender_riskset = np.array([], dtype=int)
    receiver_riskset: dict[Any, np.ndarray] = {}
    if model_value == "actor":
        sender_riskset, receiver_riskset = _actor_risksets(
            events, dictionary, riskset_value, manual_riskset
        )
    duration_info: dict[str, Any] = {}
    if duration:
        censored = int(events["end"].isna().sum())
        duration_info = {
            "n_complete": int(len(events) - censored),
            "n_censored": censored,
            "has_censored": bool(censored),
            "dur_directed_end": bool(dur_directed_end),
            "dur_type_exclusive": bool(dur_type_exclusive and extend_riskset_by_type),
            "has_who_ended": "who_ended" in events.columns,
        }
    cls = DurationHistory if duration else EventHistory
    return cls(
        events,
        dictionary,
        risksets,
        directed,
        ordinal,
        model_value,
        types,
        duration,
        riskset_value,
        extend_riskset_by_type,
        effective_decode,
        sender_riskset,
        receiver_riskset,
        duration_info,
        event_weight is not None or "weight" in frame.columns,
    )


def is_remify_durem(value: object) -> bool:
    return isinstance(value, DurationHistory) or (
        isinstance(value, EventHistory) and bool(value.duration)
    )


def _coerce_edgelist(edgelist: Any) -> pd.DataFrame:
    if isinstance(edgelist, pd.DataFrame):
        return edgelist.reset_index(drop=True).copy()
    if isinstance(edgelist, Mapping):
        return pd.DataFrame(edgelist)
    if isinstance(edgelist, Iterable):
        return pd.DataFrame(list(edgelist))
    raise TypeError("edgelist must be a pandas DataFrame, mapping, or row iterable")


def _event_column(frame: pd.DataFrame, source: str | Sequence[Any] | None, length: int) -> Any:
    if source is None:
        if "type" in frame.columns:
            return frame["type"].to_list()
        return None
    if isinstance(source, str):
        if source not in frame.columns:
            raise ValueError(f"event_type column not found: {source}")
        if source != "type" and "type" in frame.columns:
            warnings.warn(
                f"event_type={source!r} overrides the existing 'type' column",
                UserWarning,
                stacklevel=2,
            )
        return frame[source].to_list()
    values = list(source)
    if len(values) != length:
        raise ValueError("event_type must have one value per event")
    return values


def _weight_column(frame: pd.DataFrame, source: str | Sequence[float] | None, length: int) -> Any:
    if source is None:
        if "weight" in frame.columns:
            values = frame["weight"].astype(float).to_list()
            _validate_weights(values)
            return values
        return 1.0
    if isinstance(source, str):
        if source not in frame.columns:
            raise ValueError(f"event_weight column not found: {source}")
        if source != "weight" and "weight" in frame.columns:
            warnings.warn(
                f"event_weight={source!r} overrides the existing 'weight' column",
                UserWarning,
                stacklevel=2,
            )
        values = frame[source].astype(float).to_list()
        _validate_weights(values)
        return values
    values = list(source)
    if len(values) != length:
        raise ValueError("event_weight must have one value per event")
    numeric = [float(value) for value in values]
    _validate_weights(numeric)
    return numeric


def _validate_weights(values: Sequence[float]) -> None:
    if any(pd.isna(value) for value in values):
        raise ValueError("event_weight cannot contain missing values")


def _event_attribute_names(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    names = list(value)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("event_attributes must contain column names")
    return names


def _validate_aggregate_time(value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or pd.isna(value)
        or not np.isfinite(float(value))
        or float(value) < 1
    ):
        raise ValueError("`aggregate_time` must be a single numeric value >= 1.")
    return int(float(value))


def _normalize_event_times(values: pd.Series, origin: Any | None, units: str) -> pd.Series:
    if values.isna().any():
        raise ValueError("edgelist time cannot contain missing values")
    if pd.api.types.is_numeric_dtype(values.dtype):
        if origin is None:
            return values.copy()
        if isinstance(origin, bool) or not isinstance(origin, Real):
            raise TypeError("origin and numeric event times must have compatible numeric types")
        return values.astype(float) - float(origin)

    try:
        timestamps = pd.to_datetime(values)
    except (TypeError, ValueError) as error:
        raise TypeError("time must contain numeric, date, or timestamp values") from error
    if origin is None:
        if len(timestamps) > 1:
            differences = timestamps.sort_values().diff().dropna()
            average = differences.mean()
        else:
            average = pd.Timedelta(days=1)
        origin_timestamp = timestamps.min() - average
    else:
        try:
            origin_timestamp = pd.Timestamp(origin)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "origin and event times must have compatible date/time types"
            ) from error
    seconds = (timestamps - origin_timestamp).dt.total_seconds()
    selected_units = units
    if selected_units == "auto":
        selected_units = "days" if all(value.hour == 0 for value in timestamps) else "secs"
    divisors = {
        "secs": 1.0,
        "mins": 60.0,
        "hours": 3_600.0,
        "days": 86_400.0,
        "weeks": 604_800.0,
        "months": 2_629_746.0,
        "years": 31_556_952.0,
    }
    return pd.Series(seconds / divisors[selected_units], index=values.index, dtype=float)


def _normalize_optional_event_times(values: pd.Series, origin: Any | None, units: str) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    complete = values.notna()
    if complete.any():
        result.loc[complete] = _normalize_event_times(values.loc[complete], origin, units)
    return result


def _duration_end_values(frame: pd.DataFrame, time_column: str | None) -> pd.Series:
    if time_column is None:
        raise ValueError("duration histories require a start time column")
    if "end" in frame.columns:
        return frame["end"].copy()
    if "end_time" in frame.columns:
        return frame["end_time"].copy()
    if "duration" in frame.columns:
        try:
            return frame[time_column] + frame["duration"]
        except TypeError as error:
            raise TypeError("duration values must be compatible with event start times") from error
    raise ValueError("duration histories require end, end_time, or duration column")


def _duration_who_ended(frame: pd.DataFrame, directed_end: bool) -> list[Any] | None:
    if "who_ended" not in frame.columns:
        if directed_end:
            warnings.warn(
                "dur_directed_end=True without who_ended; actor1 is assumed",
                UserWarning,
                stacklevel=2,
            )
            return ["actor1"] * len(frame)
        return None
    values: list[Any] = list(frame["who_ended"].to_list())
    invalid = [
        value for value in values if not pd.isna(value) and value not in {"actor1", "actor2"}
    ]
    if invalid:
        raise ValueError("who_ended values must be 'actor1', 'actor2', or missing")
    return values


def _aggregate_event_times(values: pd.Series, every: int) -> pd.Series:
    if values.isna().any():
        raise ValueError("edgelist time cannot contain missing values")
    if every == 1 or values.empty:
        return values.copy()
    unique = list(pd.unique(values))
    try:
        unique.sort()
    except TypeError as error:
        raise TypeError("edgelist time values must have one mutually comparable type") from error
    kept = unique[every - 1 :: every]
    if not kept:
        kept = [unique[-1]]
    kept_array = np.asarray(kept)
    positions = np.searchsorted(kept_array, values.to_numpy(), side="left")
    positions = np.minimum(positions, len(kept) - 1)
    return pd.Series(kept_array[positions], index=values.index)


def _ordinal_event_times(values: pd.Series) -> pd.Series:
    unique = list(pd.unique(values))
    try:
        unique.sort()
    except TypeError as error:
        raise TypeError("edgelist time values must have one mutually comparable type") from error
    index = {value: position for position, value in enumerate(unique, start=1)}
    return values.map(index).astype(int)


def _ordinal_duration_event_times(events: pd.DataFrame) -> None:
    """Map duration starts and ends onto one dense public event-time axis."""

    complete_ends = events.loc[events["end"].notna(), "end"]
    all_values = pd.concat([events["time"], complete_ends], ignore_index=True)
    unique = list(pd.unique(all_values))
    try:
        unique.sort()
    except TypeError as error:
        raise TypeError("duration start and end times must be mutually comparable") from error
    index = {value: position for position, value in enumerate(unique, start=1)}
    events["time"] = events["time"].map(index).astype(int)
    mapped_end = events["end"].map(index)
    events["end"] = mapped_end.astype(float) if mapped_end.isna().any() else mapped_end.astype(int)


def _first_present(
    frame: pd.DataFrame, names: Sequence[str], *, required: bool = True
) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(f"missing required column; expected one of {', '.join(names)}")
    return None


def _match_arg(value: Sequence[str] | str, choices: Sequence[str], name: str) -> str:
    selected = value[0] if isinstance(value, Sequence) and not isinstance(value, str) else value
    if selected not in choices:
        raise ValueError(f"{name} must be one of {', '.join(choices)}")
    return str(selected)


def _stable_unique(values: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def _actor_dictionary(events: pd.DataFrame, actors: Sequence[Any] | None) -> pd.DataFrame:
    labels = (
        list(actors)
        if actors is not None
        else _stable_unique([*events["sender"].to_list(), *events["receiver"].to_list()])
    )
    if len(_stable_unique(labels)) != len(labels):
        raise ValueError("actors must not contain duplicate labels")
    missing = set(events["sender"]).union(events["receiver"]).difference(labels)
    if missing:
        raise ValueError(f"actors is missing observed labels: {sorted(missing)!r}")
    return pd.DataFrame({"actor_id": range(1, len(labels) + 1), "actor": labels})


def _dyad_id(sender_id: int, receiver_id: int, actor_count: int, directed: bool) -> int:
    if sender_id == receiver_id:
        raise ValueError("self loops are not part of the default REM risk set")
    if directed:
        offset = 0 if receiver_id < sender_id else -1
        return (sender_id - 1) * (actor_count - 1) + receiver_id + offset
    low, high = sorted((sender_id, receiver_id))
    return sum(actor_count - i for i in range(1, low)) + (high - low)


def _all_dyads(dictionary: pd.DataFrame, directed: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    actor_ids = dictionary["actor_id"].to_list()
    labels = dict(zip(dictionary["actor_id"], dictionary["actor"], strict=True))
    for sender_id in actor_ids:
        for receiver_id in actor_ids:
            if sender_id == receiver_id:
                continue
            if not directed and sender_id > receiver_id:
                continue
            rows.append(
                {
                    "sender_id": sender_id,
                    "receiver_id": receiver_id,
                    "sender": labels[sender_id],
                    "receiver": labels[receiver_id],
                    "dyad_id": _dyad_id(sender_id, receiver_id, len(dictionary), directed),
                }
            )
    return pd.DataFrame(rows)


def _riskset_for_event(
    observed: tuple[int, int, Any],
    *,
    events: pd.DataFrame,
    dictionary: pd.DataFrame,
    directed: bool,
    mode: RisksetMode,
    manual_riskset: Any | None,
    extend_by_type: bool,
    event_types: Sequence[Any],
) -> pd.DataFrame:
    expand_types = False
    if mode == "manual":
        if manual_riskset is None:
            raise ValueError("manual_riskset is required when riskset='manual'")
        base = _normalize_manual_riskset(manual_riskset, dictionary, directed)
        if "event_type" in base.columns:
            unknown_types = set(base["event_type"]).difference(event_types)
            if unknown_types:
                raise ValueError(
                    f"manual_riskset contains unknown event types: {sorted(unknown_types)!r}"
                )
        if extend_by_type and event_types and "event_type" not in base.columns:
            base = base.merge(pd.DataFrame({"event_type": list(event_types)}), how="cross")
        key_columns = ["sender_id", "receiver_id"]
        if extend_by_type and event_types:
            key_columns.append("event_type")
        observed_rows = (
            events[[*key_columns, "sender", "receiver", "dyad_id"]]
            .drop_duplicates(key_columns)
            .copy()
        )
        observed_rows["dyad_id"] = [
            _dyad_id(sender, receiver, len(dictionary), directed)
            for sender, receiver in zip(
                observed_rows["sender_id"], observed_rows["receiver_id"], strict=True
            )
        ]
        base_keys = pd.MultiIndex.from_frame(base[key_columns])
        observed_keys = pd.MultiIndex.from_frame(observed_rows[key_columns])
        missing_rows = observed_rows[~observed_keys.isin(base_keys)]
        if not missing_rows.empty:
            base = pd.concat([base, missing_rows], ignore_index=True)
            warnings.warn(
                "manual_riskset did not contain one or more observed dyads; they were added",
                UserWarning,
                stacklevel=2,
            )
    elif mode in {"full", "active_saturated"}:
        base = _all_dyads(dictionary, directed)
        expand_types = extend_by_type
    elif mode == "active":
        columns = ["sender_id", "receiver_id"]
        if extend_by_type and event_types:
            columns.append("event_type")
        previous = events[columns].drop_duplicates()
        labels = dict(zip(dictionary["actor_id"], dictionary["actor"], strict=True))
        base = previous.assign(
            sender=previous["sender_id"].map(labels),
            receiver=previous["receiver_id"].map(labels),
            dyad_id=[
                _dyad_id(sid, rid, len(dictionary), directed)
                for sid, rid in zip(previous["sender_id"], previous["receiver_id"], strict=True)
            ],
        )
    else:
        raise ValueError(f"unsupported riskset mode: {mode}")

    observed_mask = (base["sender_id"] == observed[0]) & (base["receiver_id"] == observed[1])
    if extend_by_type and event_types and "event_type" in base.columns:
        observed_mask &= base["event_type"] == observed[2]
    if not observed_mask.any():
        labels = dict(zip(dictionary["actor_id"], dictionary["actor"], strict=True))
        added = {
            "sender_id": observed[0],
            "receiver_id": observed[1],
            "sender": labels[observed[0]],
            "receiver": labels[observed[1]],
            "dyad_id": _dyad_id(observed[0], observed[1], len(dictionary), directed),
        }
        if extend_by_type and event_types and not expand_types:
            added["event_type"] = observed[2]
        base = pd.concat(
            [base, pd.DataFrame([added])],
            ignore_index=True,
        )

    if expand_types and event_types:
        base = base.merge(pd.DataFrame({"event_type": list(event_types)}), how="cross")
    if extend_by_type and event_types:
        type_ids = {value: index for index, value in enumerate(event_types, start=1)}
        base["type_id"] = base["event_type"].map(type_ids).astype(int)
        dyad_count = len(dictionary) * (len(dictionary) - 1)
        if not directed:
            dyad_count //= 2
        base["dyad_id"] = base["dyad_id"] + (base["type_id"] - 1) * dyad_count
    # Statistic tensors use the public, 1-based dyad index as their stable
    # column order. In particular, an active risk set must not depend on the
    # first-occurrence order of its observed dyads.
    base = base.sort_values("dyad_id", kind="stable").reset_index(drop=True)
    base.insert(0, "risk_id", range(1, len(base) + 1))
    return base


def _validate_ncores(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("ncores must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError("ncores must be a positive integer")
    return result


def _normalize_manual_riskset(value: Any, dictionary: pd.DataFrame, directed: bool) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
    elif isinstance(value, Mapping):
        frame = pd.DataFrame(value)
    else:
        frame = pd.DataFrame(list(value))
    sender_column = _first_present(frame, ("sender", "actor1", "source", "from"), required=False)
    receiver_column = _first_present(frame, ("receiver", "actor2", "target", "to"), required=False)
    if sender_column is None or receiver_column is None:
        if not {"sender_id", "receiver_id"}.issubset(frame.columns):
            raise ValueError(
                "manual_riskset must contain sender/receiver, actor1/actor2, or ID columns"
            )
    labels = dict(zip(dictionary["actor_id"], dictionary["actor"], strict=True))
    ids = {label: actor_id for actor_id, label in labels.items()}
    if sender_column is not None and receiver_column is not None:
        frame["sender"] = frame[sender_column]
        frame["receiver"] = frame[receiver_column]
        frame["sender_id"] = frame["sender"].map(ids)
        frame["receiver_id"] = frame["receiver"].map(ids)
    else:
        frame["sender"] = frame["sender_id"].map(labels)
        frame["receiver"] = frame["receiver_id"].map(labels)
    if frame[["sender_id", "receiver_id"]].isna().any().any():
        raise ValueError("manual_riskset contains actors outside the event-history dictionary")
    frame["sender_id"] = frame["sender_id"].astype(int)
    frame["receiver_id"] = frame["receiver_id"].astype(int)
    if (frame["sender_id"] == frame["receiver_id"]).any():
        raise ValueError("manual_riskset cannot contain self loops")
    if not directed:
        reverse = frame["sender_id"] > frame["receiver_id"]
        old_sender_ids = frame.loc[reverse, "sender_id"].copy()
        old_sender = frame.loc[reverse, "sender"].copy()
        frame.loc[reverse, "sender_id"] = frame.loc[reverse, "receiver_id"].to_numpy()
        frame.loc[reverse, "receiver_id"] = old_sender_ids.to_numpy()
        frame.loc[reverse, "sender"] = frame.loc[reverse, "receiver"].to_numpy()
        frame.loc[reverse, "receiver"] = old_sender.to_numpy()
    if "type" in frame.columns and "event_type" not in frame.columns:
        frame = frame.rename(columns={"type": "event_type"})
    frame["dyad_id"] = [
        _dyad_id(sender, receiver, len(dictionary), directed)
        for sender, receiver in zip(frame["sender_id"], frame["receiver_id"], strict=True)
    ]
    columns = ["sender_id", "receiver_id", "sender", "receiver", "dyad_id"]
    if "event_type" in frame.columns:
        columns.append("event_type")
    return frame[columns].drop_duplicates().reset_index(drop=True)


def _actor_risksets(
    events: pd.DataFrame,
    dictionary: pd.DataFrame,
    mode: str,
    manual_riskset: Any | None,
) -> tuple[np.ndarray, dict[Any, np.ndarray]]:
    if mode == "full":
        pairs = _all_dyads(dictionary, directed=True)
    elif mode == "active":
        pairs = events[
            ["sender_id", "receiver_id", "sender", "receiver", "dyad_id"]
        ].drop_duplicates(["sender_id", "receiver_id"])
    elif mode == "active_saturated":
        observed = events[
            ["sender_id", "receiver_id", "sender", "receiver", "dyad_id"]
        ].drop_duplicates(["sender_id", "receiver_id"])
        reversed_pairs = observed.rename(
            columns={
                "sender_id": "receiver_id",
                "receiver_id": "sender_id",
                "sender": "receiver",
                "receiver": "sender",
            }
        )
        pairs = pd.concat([observed, reversed_pairs], ignore_index=True)
        pairs["dyad_id"] = [
            _dyad_id(sender, receiver, len(dictionary), directed=True)
            for sender, receiver in zip(pairs["sender_id"], pairs["receiver_id"], strict=True)
        ]
        pairs = pairs.drop_duplicates(["sender_id", "receiver_id"])
    elif mode == "manual":
        if manual_riskset is None:
            raise ValueError("manual_riskset is required when riskset='manual'")
        pairs = _normalize_manual_riskset(manual_riskset, dictionary, directed=True)
        observed_keys = pd.MultiIndex.from_frame(events[["sender_id", "receiver_id"]])
        manual_keys = pd.MultiIndex.from_frame(pairs[["sender_id", "receiver_id"]])
        if not observed_keys.isin(manual_keys).all():
            raise ValueError(
                "manual actor riskset must include every observed sender-receiver pair"
            )
    else:  # pragma: no cover - validated by remify
        raise ValueError(f"unsupported actor riskset mode: {mode}")

    actor_ids = dictionary["actor_id"].to_numpy(dtype=int)
    labels = dict(zip(dictionary["actor_id"], dictionary["actor"], strict=True))
    if mode == "full":
        sender_ids = actor_ids
    else:
        active = set(pairs["sender_id"].astype(int))
        sender_ids = np.asarray(
            [actor_id for actor_id in actor_ids if actor_id in active], dtype=int
        )
    receivers: dict[Any, np.ndarray] = {}
    for sender_id in sender_ids:
        if mode == "full":
            values = actor_ids[actor_ids != sender_id]
        else:
            values = (
                pairs.loc[pairs["sender_id"] == sender_id, "receiver_id"]
                .drop_duplicates()
                .to_numpy(dtype=int)
            )
        receivers[labels[int(sender_id)]] = values
    return sender_ids, receivers
