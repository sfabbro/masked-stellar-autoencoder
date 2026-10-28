"""Synthetic split and transitive group-leakage checks."""

import pytest

from masked_stellar_autoencoder.training.multisurvey_splits import build_group_split

FRACTIONS = {"fit": 0.7, "tune": 0.1, "cal": 0.1, "test": 0.1}


def test_transitive_groups_and_exclusions_are_preserved():
    result = build_group_split(
        [44, 11, 33, 22, 55, 66],
        {"cluster:a": [11, 22], "family:b": [22, 33], "cluster:c": [44, 55]},
        role_fractions=FRACTIONS,
        salt="experiment-v1",
        excluded_ids=[33, 999],
    )
    assignments = dict(result["contract"]["assignments"])
    assert assignments[11] == assignments[22] == assignments[33] == "excluded"
    assert assignments[44] == assignments[55]
    assert result["matched_excluded_ids"] == 1 and result["unmatched_excluded_ids"] == 1
    assert sum(result["counts"].values()) == 6


def test_order_and_group_member_order_do_not_change_the_manifest():
    a = build_group_split(
        [1, 2, 3], {"cluster:a": [2, 1]}, role_fractions=FRACTIONS, salt="v1"
    )
    b = build_group_split(
        [3, 1, 2], {"cluster:a": [1, 2]}, role_fractions=FRACTIONS, salt="v1"
    )
    assert a["sha256"] == b["sha256"] and a["contract"] == b["contract"]


@pytest.mark.parametrize(
    "ids,groups,fractions,excluded",
    [
        ([1, 1], {}, FRACTIONS, ()),
        ([1.0], {}, FRACTIONS, ()),
        ([1], {"cluster": [1]}, FRACTIONS, ()),
        ([1], {"cluster:a": [2]}, FRACTIONS, ()),
        ([1], {}, {"fit": 1.0}, ()),
        ([1], {}, FRACTIONS, [-1]),
    ],
)
def test_invalid_or_ambiguous_inputs_fail(ids, groups, fractions, excluded):
    with pytest.raises(ValueError):
        build_group_split(
            ids, groups, role_fractions=fractions, salt="v1", excluded_ids=excluded
        )


def test_existing_roles_propagate_and_conflicts_are_not_silently_mixed():
    result = build_group_split(
        [1, 2, 3, 4],
        {"cluster:x": [1, 2]},
        role_fractions=FRACTIONS,
        salt="v1",
        fixed_roles={1: "devtest", 3: "fit"},
    )
    roles = dict(result["contract"]["assignments"])
    assert roles[1] == roles[2] == "devtest" and roles[3] == "fit"
    with pytest.raises(ValueError, match="conflicting fixed roles"):
        build_group_split(
            [1, 2],
            {"family:x": [1, 2]},
            role_fractions=FRACTIONS,
            salt="v1",
            fixed_roles={1: "fit", 2: "test"},
        )


def test_exclusions_override_and_report_locked_roles():
    result = build_group_split(
        [1, 2, 3],
        {"cluster:x": [1, 2]},
        role_fractions=FRACTIONS,
        salt="v1",
        fixed_roles={1: "fit", 3: "external"},
        excluded_ids=[2],
    )
    roles = dict(result["contract"]["assignments"])
    assert roles == {1: "excluded", 2: "excluded", 3: "external"}
    assert result["overridden_fixed_ids"] == 1
    with pytest.raises(ValueError, match="candidate universe"):
        build_group_split(
            [1], {}, role_fractions=FRACTIONS, salt="v1", fixed_roles={99: "fit"}
        )
