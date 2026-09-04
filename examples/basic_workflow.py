"""Basic REMFlow workflow."""

from __future__ import annotations

import pandas as pd

from remflow import fit_rem, remify, remstats, remstimate


def main() -> None:
    events = pd.DataFrame(
        {
            "time": [1, 2, 3, 4, 5],
            "sender": ["A", "B", "A", "C", "A"],
            "receiver": ["B", "A", "C", "A", "B"],
            "channel": ["chat", "chat", "ticket", "chat", "ticket"],
        }
    )

    history = remify(
        events,
        actors=["A", "B", "C"],
        event_type="channel",
        riskset="full",
        ordinal=True,
    )
    stats = remstats(history, tie_effects="~ inertia() + reciprocity() + send() + receive()")
    fit = remstimate(history, stats)

    print(history.summary())
    print(stats.summary())
    print(fit.summary())

    pipeline_fit = fit_rem(events, actors=["A", "B", "C"])
    print(pipeline_fit.summary())


if __name__ == "__main__":
    main()
