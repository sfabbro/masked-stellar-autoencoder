import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "binding",
    Path(__file__).resolve().parents[1] / "scripts/saved_array_source_binding.py",
)
binding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(binding)


def test_report_rejects_wrong_source_and_only_reuses_waiting_outputs(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(dict(prepared_sha256="new", prepared_path="actual.h5"))
    )
    producer = dict(source_manifest_path=str(source), source_sha256="old")
    with pytest.raises(ValueError, match="differ"):
        binding.load_bound_source(producer, lambda _: "new")
    producer["source_sha256"] = "new"
    assert (
        binding.load_bound_source(producer, lambda _: "new")["prepared_path"]
        == "actual.h5"
    )
    output = tmp_path / "report"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(dict(status="waiting")))
    binding.prepare_report_output(output, "waiting", True)
    (output / "plot.png").write_bytes(b"saved artifact")
    with pytest.raises(ValueError, match="processing"):
        binding.prepare_report_output(output, "waiting", True)


def test_population_masks_do_not_expand_when_seismic_rows_are_added(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "population.h5"
    with h5py.File(path, "w") as f:
        f["role"] = np.array([b"fit"] * 20 + [b"devtest"] * 20)
        f["primary_input_eligible"] = np.ones(40, bool)
    masks = binding.population_panel_masks(
        dict(prepared_path=str(path)), np.arange(40), count=2, seed=42
    )
    assert masks["population_fit"].sum() == 2
    assert masks["population_devtest"].sum() == 2
    assert not masks["population_devtest"][:20].any()
    assert np.count_nonzero(~masks["population_devtest"][20:]) == 18
