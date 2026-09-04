import numpy as np
import pandas as pd

from remflow import AomStats, EventHistory, RemStats, aomstats, remify, remstats
from remflow.stats import RemStatsDuration


def _events():
    return pd.DataFrame(
        {
            "time": [1.0, 2.0, 4.0, 7.0],
            "sender": ["A", "B", "A", "C"],
            "receiver": ["B", "A", "C", "A"],
            "type": ["post", "reply", "post", "reply"],
            "weight": [1.0, 2.0, 1.5, 0.5],
        }
    )


def test_event_history_json_round_trip_preserves_frames_ids_and_metadata(tmp_path):
    history = remify(
        _events(),
        actors=["A", "B", "C"],
        riskset="active",
        extend_riskset_by_type=True,
    )
    path = tmp_path / "history.json"
    serialized = history.to_json(path)
    restored = EventHistory.from_json(path)

    assert serialized == path.read_text(encoding="utf-8")
    pd.testing.assert_frame_equal(restored.events, history.events, check_dtype=False)
    pd.testing.assert_frame_equal(restored.actors, history.actors, check_dtype=False)
    assert len(restored.risksets) == len(history.risksets)
    for actual, expected in zip(restored.risksets, history.risksets, strict=True):
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    assert restored.event_types == history.event_types
    assert restored.summary() == history.summary()


def test_sampled_remstats_json_round_trip_preserves_design_and_sample_map():
    history = remify(_events(), actors=["A", "B", "C"], ordinal=True)
    statistics = remstats(
        history,
        tie_effects="~ inertia() + reciprocity()",
        first=1,
        sampling=True,
        samp_num=3,
        seed=9,
    )
    restored = RemStats.from_json(statistics.to_json())

    assert restored.names == statistics.names
    assert restored.observed_indices == statistics.observed_indices
    for actual, expected in zip(restored.stats, statistics.stats, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(restored.sample_map, statistics.sample_map, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(
        restored.sampling_weights, statistics.sampling_weights, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_actor_stats_json_round_trip_preserves_masks_and_components():
    history = remify(_events(), actors=["A", "B", "C"], model="actor", ordinal=True)
    statistics = aomstats(
        reh=history,
        sender_effects="~ outdegreeSender()",
        receiver_effects="~ inertia()",
        first=1,
    )
    restored = AomStats.from_json(statistics.to_json())

    assert restored.sender_names == statistics.sender_names
    assert restored.receiver_names == statistics.receiver_names
    for actual, expected in zip(restored.sender_stats, statistics.sender_stats, strict=True):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(restored.receiver_masks, statistics.receiver_masks, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert restored.observed_sender_groups == statistics.observed_sender_groups
    assert (
        restored.receiver_choice_observed_indices
        == statistics.receiver_choice_observed_indices
    )
    for actual, expected in zip(
        restored.receiver_choice_stats, statistics.receiver_choice_stats, strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_duration_stats_json_round_trip_preserves_stacked_processes(tmp_path):
    events = _events().assign(end=[3.0, 5.0, 6.0, 9.0])
    history = remify(events, actors=["A", "B", "C"], duration=True, model="tie")
    statistics = remstats(
        history,
        start_effects="~ inertia() + reciprocity()",
        end_effects="~ inertia()",
        psi_start=2,
        psi_end=0.5,
    )
    assert isinstance(statistics, RemStatsDuration)
    path = tmp_path / "duration-stats.json"
    statistics.to_json(path)
    restored = RemStatsDuration.from_json(path)

    assert restored.psi_start == 2
    assert restored.psi_end == 0.5
    assert restored.stacked.to_dict().keys() == statistics.stacked.to_dict().keys()
    assert restored.stacked.stat_names == statistics.stacked.stat_names
    pd.testing.assert_frame_equal(
        restored.stacked.remstats_stack,
        statistics.stacked.remstats_stack,
        check_dtype=False,
    )
