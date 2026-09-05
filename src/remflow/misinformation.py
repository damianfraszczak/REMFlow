"""High-level model for misinformation propagation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from remflow.model import RelationalEventModel, _effect_formula, _normalize_event_columns
from remflow.stats import Effect, Formula, formula, recencyReceiveReceiver, same


class MisinformationModel(RelationalEventModel):
    """Relational event model specialized for misinformation propagation.

    In addition to the general fitting and next-event API, this subclass maps
    positional ``action`` and ``stance`` fields and provides descriptive
    source, actor-role, echo-chamber, and blocking-intervention analyses.
    """

    def _coerce_events(self, events: Any) -> pd.DataFrame:
        return _coerce_misinformation_events(events)

    def _resolve_event_type(self, frame: pd.DataFrame) -> str | None:
        if self.event_type is not None:
            return super()._resolve_event_type(frame)
        return "action" if "action" in frame.columns else None

    def _resolve_event_attributes(self, frame: pd.DataFrame) -> tuple[str, ...]:
        attributes = list(super()._resolve_event_attributes(frame))
        if "stance" in frame.columns and "stance" not in attributes:
            attributes.append("stance")
        return tuple(attributes)

    def _actor_attributes_for(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        return _stance_attributes(frame) if "stance" in frame.columns else None

    def _effect_formula(self, actor_attributes: pd.DataFrame | None) -> Formula:
        return _misinformation_effect_formula(self.effects, actor_attributes)

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


def _coerce_misinformation_events(events: Any) -> pd.DataFrame:
    if isinstance(events, pd.DataFrame):
        frame = events.copy()
    else:
        rows = list(events)
        if rows and not isinstance(rows[0], dict):
            widths = {len(row) for row in rows}
            if len(widths) != 1 or next(iter(widths)) not in {3, 4, 5}:
                raise ValueError("misinformation event tuples must have 3-5 fields")
            columns = ["time", "sender", "receiver", "action", "stance"][: next(iter(widths))]
            frame = pd.DataFrame(rows, columns=columns)
        else:
            frame = pd.DataFrame(rows)
    frame = _normalize_event_columns(frame)
    if "type" in frame.columns and "action" not in frame.columns:
        frame = frame.rename(columns={"type": "action"})
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


def _misinformation_effect_formula(
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
        elif requested == "recent_exposure":
            terms.append(recencyReceiveReceiver())
        else:
            terms.extend(_effect_formula((requested,)).terms)
    return formula(Formula(tuple(terms)))


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


__all__ = ["MisinformationModel"]
