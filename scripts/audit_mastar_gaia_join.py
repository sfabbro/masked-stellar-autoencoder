"""Audit MaStar parameter-to-publisher Gaia associations using exact native IDs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
from astropy.io import fits
from xmatcher import CrossMatch, MatchRequest, MatchSpec

EXPECTED_SHA1 = "2a3c82483e5807b615228a2e7be469fab13e7fb6"
PARAM_SHA1 = "969aa44afaba9f9500b8c7e1bd6c21200df69fe1"


def digest(path, algorithm):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def join_native_ids(parameters, associations):
    for frame in [parameters, associations]:
        if frame["mangaid"].n_unique() != frame.height:
            raise ValueError("Native MANGAID must be unique within each product")
    result = CrossMatch().crossmatch_request(
        MatchRequest(
            cat1=parameters,
            cat2=associations,
            id_join=True,
            id_column_1="mangaid",
            id_column_2="mangaid",
            spec=MatchSpec(join_type="all1", find="all", fallback_policy="error"),
        )
    )
    if result["gaia_edr3_source_id"].null_count():
        raise ValueError("Parameter MANGAID absent from publisher association product")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parameters", type=Path, required=True)
    p.add_argument("--crossmatch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if digest(args.crossmatch, "sha1") != EXPECTED_SHA1:
        raise ValueError("Gaia crossmatch publisher checksum differs")
    if digest(args.parameters, "sha1") != PARAM_SHA1:
        raise ValueError("Parameter publisher checksum differs")
    args.output.mkdir(parents=True, exist_ok=False)
    with fits.open(args.parameters) as f:
        a = f[1].data
        parameters = pl.DataFrame(
            dict(
                mangaid=np.char.strip(a["MANGAID"].astype("U")),
                parameter_row=np.arange(len(a)),
            )
        )
    with fits.open(args.crossmatch) as f:
        a = f["GOODSTARS"].data
        if a["SOURCE_ID"].dtype.kind not in "iu":
            raise ValueError("Gaia IDs must be exact native integers")
        associations = pl.DataFrame(
            dict(
                mangaid=np.char.strip(a["MANGAID"].astype("U")),
                gaia_edr3_source_id=a["SOURCE_ID"].astype("i8"),
                gaia_cleanmatch=a["GAIA_CLEANMATCH"].astype("i8"),
                contamination=a["CONTAMINATION"].astype("i8"),
                source_epoch=a["EPOCH"].astype("f8"),
                gaia_ref_epoch=a["REF_EPOCH"].astype("f8"),
            )
        )
    joined = join_native_ids(parameters, associations)
    if joined.height != parameters.height:
        raise ValueError("ID join duplicated or dropped parameter rows")
    joined.write_csv(args.output / "mangaid-gaia-candidates.csv")
    ids = joined["gaia_edr3_source_id"].to_numpy()
    valid = ids > 0
    result = dict(
        status="publisher_identity_join_audited",
        parameter_rows=parameters.height,
        crossmatch_rows=associations.height,
        joined_rows=joined.height,
        valid_gaia_id_rows=int(valid.sum()),
        distinct_positive_gaia_ids=int(np.unique(ids[valid]).size),
        publisher_cleanmatch_counts={
            str(k[0]): v.height for k, v in joined.group_by("gaia_cleanmatch")
        },
        parameter_sha256=digest(args.parameters, "sha256"),
        crossmatch_sha256=digest(args.crossmatch, "sha256"),
        crossmatch_publisher_sha1=EXPECTED_SHA1,
        xmatcher_version=__import__("importlib.metadata").metadata.version("xmatcher"),
        training_performed=False,
        training_identity_promoted=False,
        scope="Exact MANGAID join to publisher proper-motion-aware Gaia EDR3 VAC; no fresh positional crossmatch, DR3 archive validation or parameter-quality admission claimed; repeated Gaia identities retained",
    )
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
