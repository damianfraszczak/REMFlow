"""High-level misinformation-oriented relational event model facade."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

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
    recencyReceiveReceiver,
    reciprocity,
    remstats,
    same,
)

_EFFECT_ALIASES = {
    "reciprocity": reciprocity,
    "sender_activity": outdegreeSender,
    "receiver_popularity": indegreeReceiver,
    "triadic_closure": otp,
    "recent_exposure": recencyReceiveReceiver,
    "inertia": inertia,
}


class RelationalEventModel:
    """Fit and inspect a relational event model for information propagation.

    Parameters
    ----------
    effects:
        High-level aliases such as ``sender_activity`` and
        ``receiver_popularity``, or native :class:`~remflow.Effect` objects.
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
    ) -> None:
        self.effects = tuple(effects)
        self.backend = backend
        self.ordinal = ordinal
        self.directed = directed
        self.riskset = riskset
        self.history_: EventHistory | None = None
        self.stats_: RemStats | None = None
        self.fit_result_: RemEstimate | None = None
        self.events_: pd.DataFrame | None = None
        self._formula: Formula | None = None
        self._actor_attributes: pd.DataFrame | None = None

    def fit(self, events: Any) -> RelationalEventModel:
        """Fit the model and return ``self``."""

        frame = _coerce_propagation_events(events)
        actor_attributes = _stance_attributes(frame) if "stance" in frame.columns else None
        model_formula = _effect_formula(self.effects, actor_attributes)
        event_type = "action" if "action" in frame.columns else None
        event_attributes = ["stance"] if "stance" in frame.columns else None
        extend_types = bool(event_type and frame[event_type].nunique(dropna=True) > 1)
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
        return self

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
        """Rank candidate sender-receiver-action events after the fitted history."""

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

    def actor_roles(self) -> pd.DataFrame:
        """Return transparent source, amplifier, and intermediary role scores."""

        self._require_fit()
        assert self.events_ is not None
        actors = pd.Index(
            pd.unique(
                pd.concat([self.events_["sender"], self.events_["receiver"]], ignore_index=True)
            )
        )
        sent = self.events_["sender"].value_counts().reindex(actors, fill_value=0).astype(float)
        received = (
            self.events_["receiver"].value_counts().reindex(actors, fill_value=0).astype(float)
        )
        total = sent + received
        source = sent / np.maximum(total, 1.0)
        amplifier = sent / np.maximum(received, 1.0)
        intermediary = 2.0 * sent * received / np.maximum(total, 1.0)
        return pd.DataFrame(
            {
                "actor": actors,
                "source_score": source.to_numpy(),
                "amplifier_score": amplifier.to_numpy(),
                "intermediary_score": intermediary.to_numpy(),
            }
        ).sort_values("source_score", ascending=False, kind="stable", ignore_index=True)

    def detect_sources(self, top_k: int | None = None) -> pd.DataFrame:
        """Rank plausible cascade sources using timing and downstream reach.

        This is a transparent descriptive source score, not a latent-source
        posterior. Earlier first transmissions and larger reachable sets
        increase the score.
        """

        self._require_fit()
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        assert self.events_ is not None
        frame = self.events_.sort_values("time", kind="stable").reset_index(drop=True)
        actors = list(
            pd.unique(pd.concat([frame["sender"], frame["receiver"]], ignore_index=True))
        )
        first_send = frame.groupby("sender", sort=False)["time"].first()
        first_receive = frame.groupby("receiver", sort=False)["time"].first()
        send_order = {actor: position for position, actor in enumerate(first_send.index)}
        denominator = max(1, len(first_send) - 1)
        adjacency = {
            actor: set(frame.loc[frame["sender"] == actor, "receiver"].to_list())
            for actor in actors
        }
        rows: list[dict[str, Any]] = []
        for actor in actors:
            reach = _reachable_actors(actor, adjacency)
            temporal = (
                0.0
                if actor not in send_order
                else 1.0 - float(send_order[actor]) / denominator
            )
            reach_score = len(reach) / max(1, len(actors) - 1)
            sent_at = first_send.get(actor, np.nan)
            received_at = first_receive.get(actor, np.nan)
            rows.append(
                {
                    "actor": actor,
                    "first_send_time": sent_at,
                    "first_receive_time": received_at,
                    "downstream_reach": len(reach),
                    "source_score": 0.5 * temporal + 0.5 * reach_score,
                }
            )
        result = pd.DataFrame(rows).sort_values(
            ["source_score", "first_send_time"],
            ascending=[False, True],
            kind="stable",
            ignore_index=True,
        )
        return result if top_k is None else result.head(top_k).reset_index(drop=True)

    def echo_chamber_metrics(self) -> dict[str, Any]:
        """Measure cumulative same-stance interaction concentration over time."""

        self._require_fit()
        assert self.events_ is not None
        if "stance" not in self.events_.columns:
            raise ValueError("echo_chamber_metrics requires a stance column")
        known = self.events_.loc[
            self.events_["stance"].notna(), ["sender", "stance"]
        ].drop_duplicates("sender", keep="last")
        stance_by_actor = dict(zip(known["sender"], known["stance"], strict=True))
        frame = self.events_.sort_values("time", kind="stable").copy()
        sender_stance = frame["sender"].map(stance_by_actor)
        receiver_stance = frame["receiver"].map(stance_by_actor)
        valid = sender_stance.notna() & receiver_stance.notna()
        same_stance = (sender_stance == receiver_stance) & valid
        cross_stance = (sender_stance != receiver_stance) & valid
        within_cumulative = same_stance.astype(int).cumsum()
        cross_cumulative = cross_stance.astype(int).cumsum()
        comparable = within_cumulative + cross_cumulative
        scores = np.divide(
            (within_cumulative - cross_cumulative).to_numpy(dtype=float),
            comparable.to_numpy(dtype=float),
            out=np.zeros(len(frame), dtype=float),
            where=comparable.to_numpy() != 0,
        )
        trajectory = pd.DataFrame(
            {
                "time": frame["time"].to_numpy(),
                "within_stance_events": within_cumulative.to_numpy(dtype=int),
                "cross_stance_events": cross_cumulative.to_numpy(dtype=int),
                "echo_chamber_score": scores,
            }
        )
        total = int(comparable.iloc[-1]) if len(comparable) else 0
        within = int(within_cumulative.iloc[-1]) if len(within_cumulative) else 0
        return {
            "echo_chamber_score": float(scores[-1]) if len(scores) else 0.0,
            "within_stance_share": float(within / total) if total else float("nan"),
            "comparable_events": total,
            "trajectory": trajectory,
        }

    def simulate_intervention(self, *, blocked_actors: Sequence[Any]) -> dict[str, Any]:
        """Remove blocked actors from the next-event distribution and renormalize."""

        blocked = set(blocked_actors)
        prediction = self.predict_next_events(top_k=max(1, self._candidate_count()))
        removed = prediction["sender"].isin(blocked) | prediction["receiver"].isin(blocked)
        removed_mass = float(prediction.loc[removed, "probability"].sum())
        remaining = prediction.loc[~removed].copy()
        mass = float(remaining["probability"].sum())
        if mass > 0:
            remaining["probability"] /= mass
        return {
            "blocked_actors": list(blocked_actors),
            "probability_mass_removed": removed_mass,
            "next_events": remaining.reset_index(drop=True),
        }

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
        if "action" in frame.columns:
            dummy["action"] = frame["action"].dropna().iloc[0]
        if "stance" in frame.columns:
            dummy["stance"] = frame["stance"].dropna().iloc[-1]
        extended = pd.concat([frame, pd.DataFrame([dummy])], ignore_index=True)
        event_type = "action" if "action" in extended.columns else None
        history = remify(
            extended,
            actors=actors,
            directed=self.directed,
            ordinal=self.ordinal,
            riskset=self.riskset,
            event_type=event_type,
            event_attributes=["stance"] if "stance" in extended.columns else None,
            extend_riskset_by_type=bool(event_type and extended[event_type].nunique() > 1),
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


def _coerce_propagation_events(events: Any) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        frame = events.copy()
    else:
        rows = list(events)
        if rows and not isinstance(rows[0], dict):
            widths = {len(row) for row in rows}
            if len(widths) != 1 or next(iter(widths)) not in {3, 4, 5}:
                raise ValueError("event tuples must have 3-5 fields")
            columns = ["time", "sender", "receiver", "action", "stance"][: next(iter(widths))]
            frame = pd.DataFrame(rows, columns=columns)
        else:
            frame = pd.DataFrame(rows)
    aliases = {"actor1": "sender", "actor2": "receiver", "type": "action"}
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame})
    required = {"time", "sender", "receiver"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"events are missing required columns: {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError("at least two relational events are required")
    return frame.reset_index(drop=True)


def _stance_attributes(events: pd.DataFrame) -> pd.DataFrame:
    stances = list(pd.unique(events["stance"].dropna()))
    codes = {value: float(index) for index, value in enumerate(stances)}
    sender_stances = events.loc[events["stance"].notna(), ["sender", "stance"]].drop_duplicates(
        "sender", keep="last"
    )
    actors = pd.Index(
        pd.unique(pd.concat([events["sender"], events["receiver"]], ignore_index=True))
    )
    lookup = dict(zip(sender_stances["sender"], sender_stances["stance"], strict=True))
    return pd.DataFrame(
        {
            "name": actors,
            "stance": [codes.get(lookup.get(actor), -1.0) for actor in actors],
        }
    )


def _effect_formula(
    effects: Sequence[str | Effect], actor_attributes: pd.DataFrame | None
) -> Formula:
    terms: list[Effect] = []
    for requested in effects:
        if isinstance(requested, Effect):
            terms.append(requested)
        elif requested == "stance_similarity":
            if actor_attributes is None:
                raise ValueError("stance_similarity requires a stance column in events")
            terms.append(same("stance", attr_actors=actor_attributes))
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


def _reachable_actors(source: Any, adjacency: dict[Any, set[Any]]) -> set[Any]:
    reached: set[Any] = set()
    frontier = list(adjacency.get(source, set()))
    while frontier:
        actor = frontier.pop()
        if actor == source or actor in reached:
            continue
        reached.add(actor)
        frontier.extend(adjacency.get(actor, set()).difference(reached))
    return reached


__all__ = ["RelationalEventModel"]
