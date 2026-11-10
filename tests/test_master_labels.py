import numpy as np
import pandas as pd
import pytest

from masked_stellar_autoencoder.training import master_labels as ml


def test_exact_ids_never_accept_floats():
    assert ml._exact_ids(["123", "", 45, None]).tolist() == [123, 0, 45, 0]
    with pytest.raises(ValueError):
        ml._exact_ids([1.5e17])


def test_sentinels_become_missing_but_hot_teff_survives():
    assert np.isnan(ml._f8([-9999.0])[0])
    assert ml._f8(["12000"])[0] == 12000


def _candidates(rows):
    return pd.DataFrame(
        rows, columns=["key", "source_id", "ra", "dec", "pmra", "pmdec", "ref_epoch"]
    )


def test_choose_matches_propagates_and_requires_epoch_stability():
    targets = pd.DataFrame(dict(key=[0, 1, 2], ra=[10.0, 20.0, 30.0], dec=[0.0] * 3))
    candidates = _candidates(
        [
            # key 0: exact at J2000 after propagating a 100 mas/yr mover back
            (0, 11, 10.0 + 100 * 16 / 3.6e6, 0.0, 100.0, 0.0, 2016.0),
            # key 1: two sources inside 1.5" -> ambiguous count
            (1, 21, 20.0, 0.0, 0.0, 0.0, 2016.0),
            (1, 22, 20.0 + 1.0 / 3600, 0.0, 0.0, 0.0, 2016.0),
            # key 2: fast mover only matches at one of the assumed epochs
            (2, 31, 30.0, 0.0, 500.0, 0.0, 2016.0),
        ]
    )
    out = ml.choose_matches(
        candidates, targets, radius_arcsec=1.5, epochs=(2000.0, 2016.0)
    ).set_index("key")
    assert 0 not in out.index  # 1.6" apart at 2016 although exact at 2000
    single = ml.choose_matches(candidates, targets, radius_arcsec=1.5, epochs=(2000.0,))
    single = single.set_index("key")
    assert single.loc[0, "source_id"] == 11 and single.loc[0, "sep_arcsec"] < 1e-3
    assert out.loc[1, "source_id"] == 21 and out.loc[1, "n_within"] == 2
    assert 2 not in out.index


def _obs(source, ids, feh, tier="survey", quality=True, edge=False):
    n = len(ids)
    frame = ml.empty_frame(
        n,
        source=source,
        tier=tier,
        release="r",
        pipeline="p",
        dependency_group=source,
    )
    frame["native_row"] = np.arange(n)
    frame["source_id"] = ids
    frame["fe_h"] = feh
    frame["quality"] = quality
    frame["fe_h_grid_edge"] = edge
    return ml.finalize(frame)


def test_census_counts_unique_stars_bins_and_existing_overlap():
    obs = pd.concat(
        [
            _obs("a", [1, 2, 2, 3], [-2.6, -1.2, -1.4, -3.1]),
            _obs("b", [3, 4], [-3.3, -2.5], tier="highres_reference"),
            _obs("c", [5, 0], [-2.46, -0.1], edge=True),
        ],
        ignore_index=True,
    )
    result, stars = ml.census(obs, {"old": np.array([1, 9])})
    a = result["sources"]["a"]
    assert a["unique_gaia"] == 3 and a["metal_poor_unique"]["lt_-2.5"] == 2
    assert result["sources"]["c"]["grid_edge_unique_gaia"] == 1
    assert result["sources"]["c"]["metal_poor_unique"]["lt_-2.0"] == 0
    union = result["union"]
    assert union["unique_gaia"] == 5 and union["multi_source"] == 1
    assert union["exclusive_to_source"] == {"a": 2, "b": 1, "c": 1}
    assert union["metal_poor_unique_highres"]["lt_-3.0"] == 1
    assert union["edge_censored_only"] == 1
    assert result["against_existing"]["new_unique"] == 4
    assert result["against_existing"]["combined_unique"] == 6
    star3 = stars.set_index("source_id").loc[3]
    assert star3.n_sources == 2 and star3.fe_h_min_any == pytest.approx(-3.3)


def test_vizier_normalisation_keeps_native_ids_and_quality_rules():
    spec = ml.VizierSource(
        "toy",
        "J/x",
        "metal_poor_survey",
        "rel",
        "pipe",
        "dep",
        "Teff",
        None,
        "[Fe/H]",
        gaia="Gaia",
        gaia_release="edr3",
        native_id="Name",
        quality={"good": ("==", 1)},
    )
    raw = pd.DataFrame(
        {
            "_RAJ2000": ["1.0", "2.0", "3.0"],
            "_DEJ2000": ["0.5", "0.6", ""],
            "Name": ["s1", "s2", "s3"],
            "Gaia": ["1234567890123456789", "", "42"],
            "Teff": ["5000", "", "6000"],
            "[Fe/H]": ["-2.5", "-1.0", ""],
            "good": ["1", "0", "1"],
        }
    )
    out = ml.vizier_observations(spec, raw)
    assert out.source_id.tolist() == [1234567890123456789, 0, 42]
    assert out.id_method.tolist() == ["publisher_edr3", "", "publisher_edr3"]
    assert out.quality.tolist() == [True, False, False]
    assert np.isnan(out.dec[2]) and out.native_id.tolist() == ["s1", "s2", "s3"]


def test_position_from_designation():
    ra, dec = ml.position_from_designation(
        ["J000111.95+032105.0", "x", "J123000.00-450000.0"]
    )
    assert ra[0] == pytest.approx(15 * (1 / 60 + 11.95 / 3600))
    assert dec[0] == pytest.approx(3 + 21 / 60 + 5.0 / 3600)
    assert np.isnan(ra[1]) and dec[2] == pytest.approx(-45.0)


def test_two_mass_designation_and_sexagesimal_parsing():
    ra, dec = ml.position_from_designation(["2MASSJ09544277+5246414"])
    assert ra[0] == pytest.approx(15 * (9 + 54 / 60 + 42.77 / 3600))
    assert dec[0] == pytest.approx(52 + 46 / 60 + 41.4 / 3600)
    ra, dec = ml.parse_sexagesimal(["10 13 34.7", ""], ["-2 7 57.9", ""])
    assert ra[0] == pytest.approx(15 * (10 + 13 / 60 + 34.7 / 3600))
    assert dec[0] == pytest.approx(-(2 + 7 / 60 + 57.9 / 3600))
    assert np.isnan(ra[1])


def test_literature_rows_borrow_positions_from_same_object():
    raw = pd.DataFrame(
        {
            "Name": ["CS 1", "CS 1", "2MASSJ09544277+5246414"],
            "Ref": ["A", "B", "C"],
            "RA": ["10 00 00.0", "", ""],
            "DEC": ["+10 00 00", "", ""],
            "Teff": ["5000", "5100", "4400"],
            "logg": ["2", "2.1", "0.6"],
            "Fe/H": ["-3.0", "-3.1", ""],
            "Priority": ["1", "2", "1"],
        }
    )
    out = ml.jinabase_observations(raw)
    assert out.ra[1] == pytest.approx(150.0) and out.dec[1] == pytest.approx(10.0)
    assert out.ra[2] == pytest.approx(148.678208, abs=1e-5)
    assert out.quality.tolist() == [True, False, False]
    assert out.dependency_group.unique().tolist() == ["literature_highres_compilation"]


def test_name_variants_cover_compact_literature_spellings():
    assert "HD 2665" in ml.name_variants("HD2665")
    assert "CD-30 298" in ml.name_variants("CD-30_298")
    assert "BPS CS 29516-0041" in ml.name_variants("CS29516-041")
    assert "BD+09 2190" in ml.name_variants("BD+9_2190")
    assert "HE 2323-0256" in ml.name_variants("HE2323-0256")
    assert "2MASS J09544277+5246414" in ml.name_variants("2MASSJ09544277+5246414")


def test_name_resolution_keeps_exact_19_digit_gaia_ids():
    obs = _obs("lit", [0, 0, 0], [-3.0, -2.0, -1.0], tier="highres_reference")
    obs["native_id"] = ["A|p1", "B|p2", "C|p3"]
    resolved = pd.DataFrame(
        dict(
            name=["A", "C"],
            simbad_main_id=["a", "c"],
            ra=[1.0, 2.0],
            dec=[3.0, 4.0],
            gaia_dr3=np.array([4671418400651900544, 0], dtype="i8"),
        )
    )
    out = ml.apply_name_resolution(obs, resolved)
    assert out.source_id.dtype == np.int64
    assert out.source_id.tolist() == [4671418400651900544, 0, 0]
    assert out.id_method.tolist() == ["simbad_dr3", "", ""]
    assert out.ra[2] == 2.0 and np.isnan(out.ra[1])
