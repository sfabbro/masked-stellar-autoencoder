from masked_stellar_autoencoder.training.map_policy import family_for, source_map_kinds


def _entry(chosen, n):
    return dict(overlap=n, maps=dict(chosen=chosen))


def test_family_follows_cv_choice_but_is_capped_by_overlap():
    assert family_for(_entry("piecewise_linear", 2370))[0] == "spline"
    assert family_for(_entry("piecewise_linear", 150))[0] == "affine"
    assert family_for(_entry("affine", 12))[0] == "offset"
    assert family_for(_entry("identity", 5000))[0] == "offset"
    assert family_for(None)[0] == "offset"
    assert family_for(dict(overlap=0))[0] == "offset"


def test_source_map_kinds_uses_aliases_and_fixed_rows():
    report = dict(
        staged_surveys=dict(desi=dict(reference=_entry("offset", 166))),
        master_sources=dict(sspp=dict(reference=_entry("piecewise_linear", 266))),
    )
    kinds = source_map_kinds(
        report,
        ["apogee", "desi", "sspp_train", "highres"],
        aliases=dict(sspp_train="sspp", highres=None),
        fixed=dict(apogee="affine"),
    )
    assert {k: v["family"] for k, v in kinds.items()} == dict(
        apogee="affine", desi="offset", sspp_train="spline", highres="offset"
    )
