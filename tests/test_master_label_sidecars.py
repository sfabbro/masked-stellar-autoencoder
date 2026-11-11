import h5py
import numpy as np
import pandas as pd

from masked_stellar_autoencoder.training import master_label_sidecars as mls
from masked_stellar_autoencoder.training import master_labels as ml
from masked_stellar_autoencoder.training.multisurvey_overlay import (
    write_observation_sidecar,
)


def _obs(source, tier, ids, feh, *, native=None, edge=False, snr=None, teff=5000.0):
    frame = ml.empty_frame(
        len(ids),
        source=source,
        tier=tier,
        release="r",
        pipeline="p",
        dependency_group=source,
    )
    frame["native_row"] = np.arange(len(ids))
    frame["source_id"] = ids
    frame["fe_h"] = feh
    frame["teff"] = teff
    frame["logg"] = 2.0
    frame["quality"] = True
    frame["fe_h_grid_edge"] = edge
    if native is not None:
        frame["native_id"] = native
    if snr is not None:
        frame["snr"] = snr
    return ml.finalize(frame)


def test_blocks_dedupe_censor_merge_and_round_trip(tmp_path):
    obs = pd.concat(
        [
            # DR19: star 1 has a DR17 row -> dropped; star 2 SDSS-V only; star 3 edge
            _obs(
                "sdss_dr19_astra_aspcap",
                "survey",
                [1, 1, 2, 3],
                [-1, -1.1, -0.5, -2.5],
                native=[
                    "a:apo25m:dr17",
                    "a:apo25m:sdss5",
                    "b:apo25m:sdss5",
                    "c:lco25m:sdss5",
                ],
                edge=[False, False, False, True],
            ),
            # SSPP: two rows of star 4 -> best S/N kept
            _obs("sdss_dr12_sspp", "survey", [4, 4], [-2.0, -2.4], snr=[10, 50]),
            # high-res: three papers on star 4, roederer model [M/H] ignored for Fe/H
            _obs("saga_mw", "highres_reference", [4, 4], [-2.6, -2.8]),
            _obs("roederer2014", "highres_reference", [4], [-9.0]),
            _obs("lamost_cat_li2025", "metal_poor_survey", [5], [-2.7]),
        ],
        ignore_index=True,
    )
    obs.loc[obs.source == "roederer2014", "fe_h_kind"] = "m_h_model"
    blocks, registry, counts = mls.build_all(obs)
    dr19 = blocks["sdss_dr19_aspcap_sdss5"]
    assert dr19["source_id"].tolist() == [2, 3]
    assert dr19["observed"][1].tolist() == [True, True, False]  # censored Fe/H
    assert blocks["sdss_dr12_sspp"]["value"][0, 2] == -2.4
    hr = blocks["highres_literature"]
    assert hr["value"][0, 2] == -2.7 and np.isnan(hr["error"][0, 2])
    cat = blocks["lamost_cat_li2025"]
    assert cat["observed"][0].tolist() == [False, False, True]
    assert counts["highres_literature"]["fe_h_below_minus2"] == 1
    receipt = write_observation_sidecar(
        tmp_path / "s.h5", blocks, registry, provenance=dict(test=True)
    )
    with h5py.File(receipt["path"]) as f:
        assert set(f["native_observations"]) == set(blocks)
        g = f["native_observations/sdss_dr12_sspp"]
        assert g["value"].shape == (1, 3) and g["source_id"][0] == 4


def test_highres_extra_leaves_out_stars_already_in_compilations():
    obs = pd.concat(
        [
            _obs("saga_mw", "highres_reference", [1], [-3.0]),
            _obs("li2022_lamost_subaru", "highres_reference", [1, 2], [-3.1, -2.5]),
            _obs("bensby2014", "highres_reference", [3], [-0.2]),
        ],
        ignore_index=True,
    )
    blocks, _, counts = mls.build_all(obs, ["highres_literature_extra"])
    block = blocks["highres_literature_extra"]
    assert block["source_id"].tolist() == [2, 3]
    assert counts["highres_literature_extra"]["stars"] == 2
