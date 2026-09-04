"""Exported active-state duration effects."""

import pandas as pd
import pytest

from remflow import activeTie, is_remstats_durem, remify, remstats

START_EFFECTS = [
    "activeTie",
    "activeOutdegreeSender",
    "activeIndegreeReceiver",
    "activeTotaldegreeSender",
    "activeTotaldegreeReceiver",
    "activeSharedPartners_otp",
    "activeSharedPartners_itp",
    "activeSharedPartners_osp",
    "activeSharedPartners_isp",
    "activeReciprocalTie",
    "activeTotaldegreeDyad",
]

END_EFFECTS = [
    "activeDegreeMin",
    "activeDegreeMax",
    "activeDegreeDyad",
    "activeSharedPartners",
]


def _events(*, typed=False):
    frame = pd.DataFrame(
        {
            "time": [1, 2, 3, 6],
            "actor1": ["A", "A", "B", "C"],
            "actor2": ["B", "C", "C", "A"],
            "end": [10, 10, 10, 10],
        }
    )
    if typed:
        frame["type"] = ["x", "y", "x", "y"]
    return frame


@pytest.mark.parametrize("effect", START_EFFECTS)
def test_every_directed_active_start_effect_runs(effect):
    history = remify(_events(), duration=True, model="tie")
    result = remstats(history, start_effects=f"~ {effect}()", first=1)

    assert is_remstats_durem(result)
    assert result.stacked.stat_names_start[-1] == f"{effect}.start"


@pytest.mark.parametrize("effect", END_EFFECTS)
def test_every_undirected_active_end_effect_runs(effect):
    history = remify(_events(), duration=True, model="tie")
    result = remstats(
        history,
        start_effects="~ inertia()",
        end_effects=f"~ {effect}()",
        first=1,
    )

    assert is_remstats_durem(result)
    assert result.stacked.stat_names_end[-1] == f"{effect}.end"


def test_active_effect_scaling_type_modes_and_validation():
    history = remify(
        _events(typed=True),
        duration=True,
        model="tie",
        extend_riskset_by_type=True,
    )
    for mode in ("ignore", "separate", "interact", True, False):
        value = repr(mode) if isinstance(mode, bool) else repr(mode)
        result = remstats(
            history,
            start_effects=f"~ activeOutdegreeSender(consider_type={value})",
            first=1,
        )
        assert is_remstats_durem(result)

    ignored = remstats(
        history,
        start_effects='~ activeOutdegreeSender(consider_type="ignore")',
        first=1,
    )
    separate = remstats(
        history,
        start_effects='~ activeOutdegreeSender(consider_type="separate")',
        first=1,
    )
    standardized = remstats(
        history,
        start_effects='~ activeOutdegreeSender(scaling="std")',
        first=1,
    )
    assert len(separate.stacked.stat_names_start) > len(
        ignored.stacked.stat_names_start
    )
    assert is_remstats_durem(standardized)
    with pytest.raises(ValueError, match="not supported"):
        activeTie(consider_type="nonsense")
