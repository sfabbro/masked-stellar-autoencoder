"""Synthetic metadata fixtures only, no models or real source files."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_expanded_source_contract as contract


def test_headless_guard_before_output(tmp_path, monkeypatch):
    monkeypatch.delenv("MSA_EXPERIMENT_CPU_ONLY", raising=False)
    with pytest.raises(RuntimeError, match="CANFAR"):
        contract.audit({}, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_metadata_only_readable_sources_and_no_repeat(tmp_path, monkeypatch):
    monkeypatch.setenv("MSA_EXPERIMENT_CPU_ONLY", "1")
    monkeypatch.setattr(contract.subprocess, "check_output", lambda *a, **k: "a" * 40)
    source = tmp_path / "source"
    source.write_bytes(b"metadata fixture")
    cfg = dict(
        native=dict(path=str(source), sha256="b" * 64),
        feature_lookup={
            k: str(source) for k in ["legacy_path", "original_path", "cache_path"]
        },
    )
    result = contract.audit(cfg, tmp_path / "result")
    assert result["status"] == "expanded_source_contract_audit_complete"
    assert result["checkpoint_loaded"] is False
    assert result["sources"]["native"]["content_sha256_verified"] is False
    with pytest.raises(FileExistsError):
        contract.audit(cfg, tmp_path / "result")


def test_missing_source_retains_failed_report(tmp_path, monkeypatch):
    monkeypatch.setenv("MSA_EXPERIMENT_CPU_ONLY", "1")
    monkeypatch.setattr(contract.subprocess, "check_output", lambda *a, **k: "a" * 40)
    with pytest.raises(FileNotFoundError):
        contract.audit(
            dict(native=dict(path=str(tmp_path / "missing"))), tmp_path / "out"
        )
    assert '"status": "failed"' in (tmp_path / "out/manifest.json").read_text()
