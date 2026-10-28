import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

root = Path("experiments/xp/results-stellar-improvement/pristine_quality_view_v1")
m = json.loads((root / "manifest.json").read_text())["result"]
dest = root / "pristine-quality-view.h5"
h = hashlib.sha256()
with dest.open("rb") as f:
    while block := f.read(8 * 1024 * 1024):
        h.update(block)
assert h.hexdigest() == m["sha256"]
local = Path(
    "/var/folders/fw/thzgrv6x0lq9h2lxmqmtt5hh0000gp/T/msa-pristine-view-strict-0w8mtyqv/output/pristine-quality-view.h5"
)
names = []
with h5py.File(dest) as got, h5py.File(local) as expected:
    assert got.attrs["complete"] and expected.attrs["complete"]
    assert got.attrs["content_id"] == expected.attrs["content_id"]

    def verify(name, node):
        if isinstance(node, h5py.Dataset):
            other = expected[name]
            assert node.shape == other.shape and node.dtype == other.dtype
            for start in range(0, len(node), 131072):
                np.testing.assert_array_equal(
                    node[start : start + 131072],
                    other[start : start + 131072],
                    err_msg=name,
                )
            names.append(name)

    got.visititems(verify)
    assert not got["validation/strict_supervised_fit_eligible"][:].any()
    assert (
        int(got["train/strict_supervised_fit_eligible"][:].sum())
        == int(got["native/strict_supervised_fit_eligible"][:].sum())
        == 583
    )
result = dict(
    actual_sha256=h.hexdigest(),
    bytes=dest.stat().st_size,
    all_published_arrays_equal_full_local_build=True,
    datasets=names,
    validation_fit_mask_all_false=True,
    matching_native_and_benchmark_fit_count=583,
    ordered_9M_1M_source_ids_equal_local_verified_benchmark=True,
    caveat="Local/remote HDF5 dependency paths differ by design; numeric arrays and content_id match. Requires immutable source files at recorded paths.",
)
(root / "independent-verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
