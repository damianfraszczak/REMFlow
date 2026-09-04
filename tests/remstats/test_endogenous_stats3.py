"""Regression tests for typed endogenous statistic kernels."""

from pathlib import Path

import numpy as np
import pandas as pd

from remflow import remify, remstats

EXPECTED_RESULTS = Path(__file__).parents[1] / "fixtures/endogenous_stats3_expected.npz"


def _history(*, expanded: bool = True, simultaneous: bool = False):
    times = [1, 2, 3, 4, 5, 5, 5, 6, 7, 8] if simultaneous else range(1, 11)
    return remify(
        pd.DataFrame(
            {
                "time": times,
                "actor1": [1, 2, 1, 2, 3, 4, 2, 2, 2, 4],
                "actor2": [3, 1, 3, 3, 2, 3, 1, 3, 4, 1],
                "type": [1, 1, 2, 2, 1, 2, 2, 1, 1, 1],
            }
        ),
        model="tie",
        riskset="active",
        extend_riskset_by_type=expanded,
    )


def _tensor(result, name: str) -> np.ndarray:
    column = result.names.index(name)
    return np.stack([matrix[:, column] for matrix in result.stats])


def _expected_matrix(variable: str) -> np.ndarray:
    """Load a deterministic expected tensor from the Python test fixture."""

    with np.load(EXPECTED_RESULTS) as fixture:
        if variable not in fixture:
            raise AssertionError(f"missing expected matrix {variable}")
        return np.asarray(fixture[variable])


IGNORE_FORMULA = (
    "~ FEtype() + outdegreeSender() + outdegreeReceiver() + "
    "indegreeSender() + indegreeReceiver() + totaldegreeSender() + "
    "totaldegreeReceiver() + totaldegreeDyad() + inertia() + reciprocity() + "
    "isp() + itp() + osp() + otp() + isp(unique=TRUE) + "
    "itp(unique=TRUE) + osp(unique=TRUE) + otp(unique=TRUE) + "
    "psABBA() + psABBY() + psABAB() + psABAY() + psABXA() + psABXB() + "
    "psABXY() + recencyContinue() + recencySendSender() + "
    "recencySendReceiver() + recencyReceiveSender() + "
    "recencyReceiveReceiver() + rrankSend() + rrankReceive()"
)


def _ignored_statistics():
    return remstats(_history(), tie_effects=IGNORE_FORMULA, first=1)


def test_directed_typed_ignore_effects_match_expected_matrices():
    result = _ignored_statistics()
    expected_variables = {
        "outdegreeSender": "outdegreeSender.ig",
        "outdegreeReceiver": "outdegreeReceiver.ig",
        "indegreeSender": "indegreeSender.ig",
        "indegreeReceiver": "indegreeReceiver.ig",
        "inertia": "inertia.ig",
        "reciprocity": "reciprocity.ig",
        "itp": "itp.ig",
        "itp.unique": "itp.unique.ig",
        "otp": "otp.ig",
        "otp.unique": "otp.unique.ig",
        "isp": "isp.ig",
        "isp.unique": "isp.unique.ig",
        "osp": "osp.ig",
        "osp.unique": "osp.unique.ig",
        "psABBA": "psABBA.ig",
        "psABBY": "psABBY.ig",
        "psABAB": "psABAB.ig",
        "psABAY": "psABAY.ig",
        "psABXA": "psABXA.ig",
        "psABXB": "psABXB.ig",
        "psABXY": "psABXY.ig",
        "recencyContinue": "recencyContinue.ig",
        "recencySendSender": "recencySendSender.ig",
        "recencySendReceiver": "recencySendReceiver.ig",
        "recencyReceiveSender": "recencyReceiveSender.ig",
        "recencyReceiveReceiver": "recencyReceiveReceiver.ig",
        "rrankSend": "rrankSend.ig",
        "rrankReceive": "rrankReceive.ig",
    }

    assert not any(
        name.endswith((".1", ".2")) or "TypeAgg" in name for name in result.names
    )
    baseline = _tensor(result, "baseline")
    np.testing.assert_array_equal(baseline, np.ones_like(baseline))
    expected_type = np.zeros_like(baseline)
    expected_type[:, 6:] = 1.0
    np.testing.assert_array_equal(_tensor(result, "FEtype_2"), expected_type)
    for name, variable in expected_variables.items():
        np.testing.assert_allclose(_tensor(result, name), _expected_matrix(variable))

    np.testing.assert_array_equal(
        _tensor(result, "totaldegreeSender"),
        _tensor(result, "indegreeSender") + _tensor(result, "outdegreeSender"),
    )
    np.testing.assert_array_equal(
        _tensor(result, "totaldegreeReceiver"),
        _tensor(result, "indegreeReceiver") + _tensor(result, "outdegreeReceiver"),
    )
    np.testing.assert_array_equal(
        _tensor(result, "totaldegreeDyad"),
        _tensor(result, "totaldegreeSender") + _tensor(result, "totaldegreeReceiver"),
    )


def test_directed_typed_separate_effects_match_expected_slices():
    ignored = _ignored_statistics()
    separate = remstats(
        _history(),
        tie_effects=(
            '~ inertia(consider_type="separate") + '
            'outdegreeSender(consider_type="separate") + '
            'reciprocity(consider_type="separate") + '
            'itp(consider_type="separate")'
        ),
        first=1,
    )
    required = {
        "inertia.1",
        "inertia.2",
        "outdegreeSender.1",
        "outdegreeSender.2",
        "reciprocity.1",
        "reciprocity.2",
        "itp.1",
        "itp.2",
    }
    assert required.issubset(separate.names)
    np.testing.assert_array_equal(
        _tensor(separate, "inertia.1") + _tensor(separate, "inertia.2"),
        _tensor(ignored, "inertia"),
    )
    np.testing.assert_array_equal(
        _tensor(separate, "outdegreeSender.1")
        + _tensor(separate, "outdegreeSender.2"),
        _tensor(ignored, "outdegreeSender"),
    )
    np.testing.assert_array_equal(
        _tensor(separate, "inertia.1"), _expected_matrix("inertia.1")
    )
    np.testing.assert_array_equal(
        _tensor(separate, "reciprocity.1"), _expected_matrix("reciprocity.1")
    )

    unexpanded_history = _history(expanded=False)
    unexpanded = remstats(
        unexpanded_history,
        tie_effects='~ inertia(consider_type="separate")',
        first=1,
    )
    typed_riskset = _history().risksets[0]
    untyped_riskset = unexpanded_history.risksets[0]
    for untyped_column, row in untyped_riskset.iterrows():
        matches = typed_riskset.index[
            (typed_riskset["sender"] == row["sender"])
            & (typed_riskset["receiver"] == row["receiver"])
            & (typed_riskset["event_type"] == 1)
        ]
        if len(matches) == 1:
            np.testing.assert_array_equal(
                _tensor(separate, "inertia.1")[:, matches[0]],
                _tensor(unexpanded, "inertia.1")[:, untyped_column],
            )


def test_directed_typed_interactions_match_partition_rules():
    ignored = _ignored_statistics()
    interacted = remstats(
        _history(), tie_effects='~ inertia(consider_type="interact")', first=1
    )
    assert {
        "inertia.1.1",
        "inertia.1.2",
        "inertia.2.1",
        "inertia.2.2",
    }.issubset(interacted.names)
    riskset = _history().risksets[0]
    type_1 = np.flatnonzero(riskset["event_type"].to_numpy() == 1)
    type_2 = np.flatnonzero(riskset["event_type"].to_numpy() == 2)
    np.testing.assert_array_equal(_tensor(interacted, "inertia.1.1")[:, type_2], 0)
    np.testing.assert_array_equal(_tensor(interacted, "inertia.1.2")[:, type_1], 0)
    assert np.all(_tensor(interacted, "inertia.2.2")[:, type_2] >= 0)
    np.testing.assert_array_equal(
        _tensor(interacted, "inertia.1.1")[:, type_1]
        + _tensor(interacted, "inertia.2.1")[:, type_1],
        _tensor(ignored, "inertia")[:, type_1],
    )
    np.testing.assert_array_equal(
        _tensor(interacted, "inertia.1.2")[:, type_2]
        + _tensor(interacted, "inertia.2.2")[:, type_2],
        _tensor(ignored, "inertia")[:, type_2],
    )


def test_directed_typed_standardization_matches_rowwise_scaling():
    raw = _ignored_statistics()
    effects = [
        "outdegreeSender",
        "outdegreeReceiver",
        "indegreeSender",
        "indegreeReceiver",
        "totaldegreeSender",
        "totaldegreeReceiver",
        "totaldegreeDyad",
        "inertia",
        "reciprocity",
        "isp",
        "itp",
        "osp",
        "otp",
    ]
    formula = "~ " + " + ".join(f'{name}(scaling="std")' for name in effects)
    formula += (
        ' + isp(scaling="std", unique=TRUE)'
        ' + itp(scaling="std", unique=TRUE)'
        ' + osp(scaling="std", unique=TRUE)'
        ' + otp(scaling="std", unique=TRUE)'
    )
    standardized = remstats(_history(), tie_effects=formula, first=1)
    for name in standardized.names[1:]:
        values = _tensor(raw, name)
        deviation = values.std(axis=1, ddof=1, keepdims=True)
        expected = np.divide(
            values - values.mean(axis=1, keepdims=True),
            deviation,
            out=np.zeros_like(values),
            where=deviation != 0,
        )
        np.testing.assert_allclose(_tensor(standardized, name), expected, atol=1e-14)


def test_directed_typed_proportional_scaling_matches_expected_denominators():
    raw = _ignored_statistics()
    effects = [
        "outdegreeSender",
        "outdegreeReceiver",
        "indegreeSender",
        "indegreeReceiver",
        "totaldegreeSender",
        "totaldegreeReceiver",
        "totaldegreeDyad",
        "inertia",
        "reciprocity",
    ]
    proportional = remstats(
        _history(),
        tie_effects="~ " + " + ".join(f'{name}(scaling="prop")' for name in effects),
        first=1,
    )
    event_denominator = np.arange(10, dtype=float)[:, None]
    for name in effects[:4]:
        expected = np.divide(
            _tensor(raw, name),
            event_denominator,
            out=np.full_like(_tensor(raw, name), 0.25),
            where=event_denominator != 0,
        )
        np.testing.assert_allclose(_tensor(proportional, name), expected)
    for name in effects[4:7]:
        expected = np.divide(
            _tensor(raw, name),
            2 * event_denominator,
            out=np.full_like(_tensor(raw, name), 0.25),
            where=event_denominator != 0,
        )
        np.testing.assert_allclose(_tensor(proportional, name), expected)

    sender_outdegree = _tensor(raw, "outdegreeSender")
    expected_inertia = np.divide(
        _tensor(raw, "inertia"),
        sender_outdegree,
        out=np.full_like(sender_outdegree, 1 / 3),
        where=sender_outdegree != 0,
    )
    np.testing.assert_allclose(_tensor(proportional, "inertia"), expected_inertia)
    sender_indegree = _tensor(raw, "indegreeSender")
    expected_reciprocity = np.divide(
        _tensor(raw, "reciprocity"),
        sender_indegree,
        out=np.full_like(sender_indegree, 1 / 3),
        where=sender_indegree != 0,
    )
    np.testing.assert_allclose(
        _tensor(proportional, "reciprocity"), expected_reciprocity
    )


def test_directed_typed_point_time_method_matches_expected_matrices():
    result = remstats(
        _history(simultaneous=True),
        tie_effects="~ FEtype() + inertia() + itp()",
        first=1,
    )
    np.testing.assert_array_equal(_tensor(result, "inertia"), _expected_matrix("inertia.pt"))
    np.testing.assert_array_equal(_tensor(result, "itp"), _expected_matrix("itp.pt"))
