import hashlib
import json
import pathlib

import h5py

p = pathlib.Path(
    "experiments/xp/results-stellar-improvement/remediation-wave-20261026/desi-unique-overlay-preservation-20261027"
)
m = json.loads((p / "manifest.json").read_text())
meta = json.loads((p / "bundle/overlay-manifest.json").read_text())
linkfile = p / "bundle/overlay.h5"
assert (
    hashlib.sha256(linkfile.read_bytes()).hexdigest() == m["result"]["overlay_sha256"]
)
assert (
    hashlib.sha256((p / "bundle/overlay-manifest.json").read_bytes()).hexdigest()
    == m["result"]["overlay_manifest_sha256"]
)
seen = 0
with h5py.File(linkfile, "r") as h:
    assert set(h) == set(meta["root_links"]) | {"native_observations"}
    for key in meta["root_links"]:
        link = h.get(key, getlink=True)
        assert (
            isinstance(link, h5py.ExternalLink)
            and link.filename == meta["parent"]["path"]
            and link.path == "/" + key
        )
        seen += 1
    for name, source in meta["observation_groups"].items():
        group = h["native_observations"][name]
        assert json.loads(group.attrs["registry_json"]) == source["registry"]
        assert list(group.attrs["source_fields"]) == source["fields"]
        for key in group:
            link = group.get(key, getlink=True)
            assert (
                isinstance(link, h5py.ExternalLink)
                and link.filename == source["file"]
                and link.path == source["group"] + "/" + key
            )
            seen += 1
assert (
    m["result"]["counts"]["roles"]["fit"]["primary_input_desi_logg"]
    == m["result"]["counts"]["roles"]["fit"]["desi_logg_not_previously_observed"]
    + m["result"]["counts"]["roles"]["fit"]["overlap_desi_and_previous_logg"]
)
print(
    json.dumps(
        {
            "status": "pass",
            "verified_links": seen,
            "sources": sorted(meta["observation_groups"]),
            "scope": "small HDF5 link metadata and producer hashes; no local rehash of remote attachments",
            "fit_counts": m["result"]["counts"]["roles"]["fit"],
        }
    )
)
