"""Conservative source-contract invariants; no remote or native model execution."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/survey-label-registry-20261026-v1.json"


def test_nine_survey_contracts_are_versioned_and_conservative():
    data = json.loads(REGISTRY.read_text())
    assert data["schema"] == "msa-survey-label-registry-v1"
    assert data["version"] == "2026-10-26.1"
    assert data["default_targets"] == ["teff", "logg", "fe_h"]
    sources = data["sources"]
    assert {s["survey"] for s in sources} == {
        "APOGEE",
        "GALAH",
        "Gaia RVS",
        "Gaia-ESO",
        "RAVE",
        "DESI",
        "LAMOST",
        "MaStar",
        "SDSS SpecObj",
    }
    assert len({s["id"] for s in sources}) == 9
    assert data["activation_policy"]["registry_does_not_expand_training"]
    assert not data["activation_policy"]["generic_metallicity_alias_to_fe_h"]
    assert len(data["activation_policy"]["current_core_allowlist"]) == 2
    for source in sources:
        assert source["quality_contract"] and source["dependency_contract"]
        assert source["next_action"] and source["official_documentation"]
        assert not source["physical_truth_claim"]
        assert all(
            url.startswith("https://") for url in source["official_documentation"]
        )
        assert all((ROOT / p).is_file() for p in source["local_evidence"])


def test_ambiguous_metallicity_is_not_registered_as_native_iron():
    sources = {s["survey"]: s for s in json.loads(REGISTRY.read_text())["sources"]}
    gaia = sources["Gaia RVS"]["native_fields"]
    assert "fe_h" not in gaia
    assert "M/H" in gaia["m_h_separate"]
    assert "NOT direct Fe/H" in gaia["fe_over_m_separate"]
    assert "native_ingestion" in sources["RAVE"]["ingestion_status"]
    assert "NOT Gaia source_id" in sources["SDSS SpecObj"]["native_fields"]["identity"]


def test_final_public_release_and_carry_forward_are_explicit():
    sources = {s["survey"]: s for s in json.loads(REGISTRY.read_text())["sources"]}
    ges = sources["Gaia-ESO"]
    assert "DR5.1" in ges["release"]
    assert "https://www.eso.org/qi/catalog/show/411" in ges["official_documentation"]
    assert ges["ingestion_status"] == "candidate_not_native_ingestion_verified"
    apogee = sources["APOGEE"]
    assert "DR17" in apogee["release"]
    assert "DR20" in apogee["newer_public_analysis"]
    assert "unchanged" in apogee["newer_public_analysis"]
