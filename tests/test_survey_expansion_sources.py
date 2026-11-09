import numpy as np
import pandas as pd
import pytest

from masked_stellar_autoencoder.training.survey_expansion_sources import (
    assemble_benchmarks,
    exact_optional_ids,
    read_vizier_tsv,
    select_rave,
)


def test_decimal_identity_never_passes_through_float():
    sid = 2**60 + 1
    a = exact_optional_ids([str(sid), ""])
    assert a[0] == sid and pd.isna(a[1])
    for bad in [float(sid), str(sid) + ".0", "-1", "0"]:
        with pytest.raises(ValueError):
            exact_optional_ids([bad])


def test_corrected_benchmark_join_keeps_distinct_iron_and_flags(tmp_path):
    p = tmp_path / "p.tsv"
    a = tmp_path / "a.tsv"
    p.write_text(
        "#Title: corrected\n#Column schema\nrecno\tHIP\tGaiaDR3\tTeff\te_Teff\tlogg\te_logg\tIndirect\n\t\t\tK\tK\tdex\tdex\t\n1\tHIP1\t1152921504606846977\t5000\t50\t3.00\t0.05\t\n2\tHIP2\t\t4500\t60\t2.00\t0.1\t1\n"
    )
    a.write_text(
        "#Title: recommended\n#Column schema\nrecno\tStar\tTeff\tlogg\tFlag\t[FeI/H]\te_[FeI/H]\t[FeII/H]\te_[FeII/H]\n1\tHIP2\t4500\t2.00\tIndirect_theta\t-1\t0.1\t-0.8\t0.2\n2\tHIP1\t5000\t3.00\t\t-0.5\t0.05\t-0.3\t0.1\n"
    )
    b = assemble_benchmarks(p, a, expected_rows=2)
    assert b.iloc[0].source_id == 2**60 + 1
    assert b.iloc[0]["[FeI/H]"] == -0.5 and b.iloc[0]["[FeII/H]"] == -0.3
    assert b.direct_diameter.tolist() == [True, False]
    assert b.spectral_warning.tolist() == [False, True]
    with pytest.raises(ValueError, match="Incomplete"):
        read_vizier_tsv(p, expected_rows=3)
    a.write_text(a.read_text().replace("5000\t3.00", "5100\t3.00"))
    with pytest.raises(ValueError, match="differ between"):
        assemble_benchmarks(p, a, expected_rows=2)


def test_rave_repeat_selection_masks_and_unknown_errors():
    m = pd.DataFrame(
        dict(
            rave_obs_id=["a", "b", "c"],
            teff_cal_madera=[5000.0, 5100.0, 4500.0],
            teff_error_madera=[50.0, 0.0, 60.0],
            logg_cal_madera=[4.0, 4.1, 2.0],
            logg_error_madera=[0.1, 0.1, 0.2],
            snr_madera=[40.0, 80.0, 50.0],
            algo_conv_madera=[0, 0, 0],
        )
    )
    x = pd.DataFrame(
        dict(
            rave_obs_id=["a", "b", "c"],
            source_id=np.array([2**60 + 1, 2**60 + 1, 2**60 + 2], dtype="i8"),
            dist=[0.1, 0.2, 0.1],
        )
    )
    fe = pd.DataFrame(
        dict(
            rave_obs_id=["a", "b", "c"],
            fe_h_gauguin=[-0.2, -0.1, -1.0],
            fe_h_error_gauguin=[0.1, 0.1, 0.2],
            fe_h_nl_gauguin=[2, 2, 0],
        )
    )
    frame, block = select_rave(m, fe, x)
    assert frame.rave_obs_id.tolist() == ["b", "c"]
    assert block["source_id"][0] == 2**60 + 1
    assert block["source_row"].tolist() == [1, 2]
    assert np.isnan(block["error"][0, 0]) and block["native_error"][0, 0] == 0
    assert not block["observed"][1, 2]
    with pytest.raises(ValueError, match="Ambiguous"):
        select_rave(m, fe, pd.concat([x, x.iloc[:1]]))
