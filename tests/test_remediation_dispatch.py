import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dispatch_remediation_pilots import REQUIRED, ready


def good():
    state = {
        run: dict(run_id=run, status=status, source_commit="a" * 40)
        for run, status in REQUIRED.items()
    }
    state["remediation_inputs_v2"].update(
        readiness=dict(phot_xp_input_pass=True, native_label_association_pass=True)
    )
    state["remediation_source_v2"].update(
        training_source_ready=True, prepared_sha256="b" * 64
    )
    state["remediation_joint6_preflight_v2"].update(
        head_module_sha256="c" * 64,
        checks={k: dict(passed=True) for k in ("gaussian", "mdn", "nsf")},
    )
    return state


def test_actual_gates_and_no_missing_status_success():
    assert ready({}, "a" * 40, "c" * 64) == "pending"
    assert ready(good(), "a" * 40, "c" * 64) == "ready"
    for run in REQUIRED:
        state = good()
        state[run]["status"] = "failed"
        assert ready(state, "a" * 40, "c" * 64) == "blocked"
        state[run]["status"] = "running"
        assert ready(state, "a" * 40, "c" * 64) == "pending"


def test_bindings_and_real_joint_gate():
    for change in ("source", "head", "audit", "nsf"):
        state = good()
        if change == "source":
            state["remediation_source_v2"]["source_commit"] = "wrong"
        elif change == "head":
            state["remediation_joint6_preflight_v2"]["head_module_sha256"] = "wrong"
        elif change == "audit":
            state["remediation_inputs_v2"]["readiness"]["phot_xp_input_pass"] = False
        else:
            state["remediation_joint6_preflight_v2"]["checks"]["nsf"]["passed"] = False
        if change in {"source", "head"}:
            with pytest.raises(ValueError):
                ready(state, "a" * 40, "c" * 64)
        else:
            assert ready(state, "a" * 40, "c" * 64) == "blocked"
