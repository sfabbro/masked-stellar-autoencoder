"""Compare an actual distance publication against the full independent local design."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from pristine_quality_views import exact_rows

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--cohort", type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
checked = []
maximum_epoch_difference = 0.0
with (
    h5py.File(args.cohort) as actual,
    h5py.File(root / "distance-feature-local-v1/distance-feature-cohort.h5") as design,
):
    assert actual.attrs["complete"]
    assert (
        actual.attrs["posterior_input_sha256"]
        == "0d24f9a748b9664d6ae6888cb38ec0ca50fe91b1d685cd8f377fa09378e2ae36"
    )
    assert (
        actual.attrs["feature_input_sha256"]
        == "b8b502c3c55672557a3e0f32ad488bec8af248062ca74b904ac1bb6c82ee6621"
    )
    assert (
        actual.attrs["benchmark_source_id_sha256"]
        == "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
    )
    mapping = exact_rows(
        actual["hipparcos_reference/source_id"][:],
        design["hipparcos_reference/source_id"][:],
    )
    assert len(np.unique(mapping)) == 99525 and (mapping >= 0).all()

    def compare(name, node):
        global maximum_epoch_difference
        if not isinstance(node, h5py.Dataset):
            return
        other = actual[name]
        assert other.dtype == node.dtype and other.shape == node.shape, name
        for key in node.attrs:
            np.testing.assert_array_equal(
                other.attrs[key], node.attrs[key], err_msg=name + ":" + key
            )
        values = other[:]
        if name.startswith("hipparcos_reference/"):
            values = values[mapping]
        if name == "hipparcos_reference/epoch_corrected_separation_arcsec":
            maximum_epoch_difference = float(np.nanmax(np.abs(values - node[:])))
            np.testing.assert_allclose(values, node[:], atol=1e-7, rtol=1e-6)
        else:
            np.testing.assert_array_equal(values, node[:], err_msg=name)
        checked.append(name)

    design.visititems(compare)
    content = hashlib.sha256()

    def digest(name, node):
        if isinstance(node, h5py.Dataset):
            content.update(name.encode())
            content.update(node[:].tobytes())

    actual.visititems(digest)
    assert content.hexdigest() == actual.attrs["content_id"]
report = dict(
    every_design_dataset_checked=checked,
    all_input_hashes_verified=True,
    content_id=content.hexdigest(),
    maximum_epoch_difference_arcsec=maximum_epoch_difference,
    epoch_tolerance_arcsec=1e-7,
    reference_ids_reordered_by_exact_int64_identity=True,
    not_file_byte_sha_comparison=True,
    training_performed=False,
)
(args.cohort.parent / "publication-array-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(
    json.dumps(
        dict(
            content_id=report["content_id"],
            datasets_checked=len(checked),
            maximum_epoch_difference_arcsec=maximum_epoch_difference,
        )
    )
)
