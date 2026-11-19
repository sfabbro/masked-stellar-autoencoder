"""Write the v7 fine-tune config and the v7 ablation configs from v6.

usage: make_v7_configs.py   (writes configs/finetune-multisurvey-20261118-v7.yaml
                             and configs/ablation-20261118-v7/*.yaml)

v7 = v6 + clean label input + censored survey floors + SLAM domain cut +
proper motions and RA/Dec hidden from the encoder. Evaluation conditions the
head on the measured Gaia parallax (predict_fits_rows --condition-parallax).

The ablations run on a 25 % training subsample (``--max-train-rows``, see the
launcher) for 6 epochs, each changing one thing from v6 or from v7, so the
effect of every idea is measured on the same held-out anchors:
small-sample removal, head context (SFD E(B-V), parallax label), Galactic
positional encoding, a dust branch, no feature noise.
"""

import copy
from pathlib import Path

import yaml

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
HERE = Path(__file__).resolve().parent.parent
V6 = HERE / "configs/finetune-multisurvey-20261118-v6.yaml"
V7 = HERE / "configs/finetune-multisurvey-20261118-v7.yaml"
ABL = HERE / "configs/ablation-20261118-v7"
ABL_RUN = "multisurvey_ablation_20261118_v7"

# [Fe/H] upper limits (dex): records below are "at or below" this value.
# Grid floors from the records (minimum / pile-up): LAMOST LRS and MRS,
# ASPCAP, APOGEE DR17 -2.5; The Payne -1.5; astroNN and ApogeeNet compress
# below about -1.7; RAVE DR6 -2.0. Margin 0.3 dex: on core metal-poor
# references these surveys read +0.3 to +1.0 dex high near their floors.
CENSOR = {
    "lamost_dr12_lrs": {"fe_h": -2.2},
    "lamost_dr12_mrs": {"fe_h": -2.2},
    "sdss_dr19_astra_aspcap": {"fe_h": -2.2},
    "apogee_dr17": {"fe_h": -2.2},
    "sdss_dr19_astra_thepayne": {"fe_h": -1.2},
    "sdss_dr19_astra_astronn": {"fe_h": -1.3},
    "sdss_dr19_astra_apogeenet": {"fe_h": -1.4},
    "rave_dr6": {"fe_h": -1.7},
}
# SLAM (Astra, BOSS) is an M-dwarf model: Teff <= 3994 K, log g 4.65-4.81,
# [Fe/H] >= -0.94; records near its Teff edge are other stars pinned there.
CUTS = {"sdss_dr20_astra_slam": {"teff": [None, 3800.0]}}
NO_SKY = ["pmra", "pmdec", "RA", "DEC"]
V7_MS = dict(
    label_input="clean", censor=CENSOR, record_cuts=CUTS, drop_input_features=NO_SKY
)

POSENC = dict(scales=[1.0, 4.0, 16.0, 64.0], n=16, seed=0)
DUST = dict(scales=[1.0, 4.0, 16.0, 64.0], n=32, hidden=128, seed=1)
SMALL = dict(min_records=1000, min_records_keep_core=True)
# The literature sources repeat the same measurements: SAGA holds 82-92 % of
# Bensby+2014, Placco+2019, Li+2022, Yong+2021 and Aoki+2013 values
# identically, JINAbase and PASTEL overlap SAGA and each other. One value per
# star and parameter: the original paper first, compilations last.
LITERATURE = [
    "bensby2014", "aoki2013_segue", "li2022_lamost_subaru", "smss_yong2021",
    "yong2013_program", "yong2013_literature", "rpa_hansen2018",
    "rpa_holmbeck2020", "sakari2018_rpa", "ezzeddine2020_rpa", "roederer2014",
    "kirby2010_highres", "kirby2010_dsph", "bestbrightest_placco2019",
    "pristine_aguado2019", "smss_dacosta2019_t2", "smss_dacosta2019_t3",
    "jinabase_2021", "saga_mw", "pastel_2020",
]  # fmt: skip
# APOGEE pipelines on the same spectra are already deduplicated (ASPCAP
# first); the remaining fallbacks of The Payne are hot stars at its [Fe/H]
# grid floor (learned noise 330 K), ApogeeNet adds 264 records, astroNN is a
# network trained on ASPCAP labels.
LEAN = ["sdss_dr19_astra_thepayne", "sdss_dr19_astra_apogeenet"]


def _groups_plus_literature(v6):
    g = copy.deepcopy(v6["multisurvey"]["spectra_groups"])
    g["literature"] = list(LITERATURE)
    return g


# name: (base, multisurvey overrides, finetuning overrides)
VARIANTS = {
    "a01_v6": ("v6", {}, {}),
    "a02_clean": ("v6", {"label_input": "clean"}, {}),
    "a03_censor": ("v6", {"censor": CENSOR, "record_cuts": CUTS}, {}),
    "a04_nopm": ("v6", {"drop_input_features": ["pmra", "pmdec"]}, {}),
    "a05_nosky": ("v6", {"drop_input_features": NO_SKY}, {}),
    "a06_small_noncore": ("v6", SMALL, {}),
    "a07_small_all": ("v6", {"min_records": 1000, "min_records_keep_core": False}, {}),
    "b01_v7": ("v7", {}, {}),
    "b02_v7_small": ("v7", SMALL, {}),
    "b03_v7_ctx_ebv": ("v7", {"context": {"features": ["EBV"]}}, {}),
    "b04_v7_ctx_plx": (
        "v7",
        {"context": {"features": ["EBV"], "labels": ["parallax"]}},
        {},
    ),
    "b05_v7_posenc": ("v7", {"context": {"features": ["EBV"], "posenc": POSENC}}, {}),
    "b06_v7_dust": ("v7", {"context": {"features": ["EBV"], "dust": DUST}}, {}),
    "b07_v7_noperturb": ("v7", {}, {"pert_features": False}),
    "c01_v7_apogee_lean": ("v7", {"exclude_surveys": LEAN}, {}),
    "c02_v7_litdedupe": ("v7", {"spectra_groups": "LIT"}, {}),
    "c03_v7_lean_lit_noastronn": (
        "v7",
        {
            "exclude_surveys": [*LEAN, "sdss_dr19_astra_astronn"],
            "spectra_groups": "LIT",
        },
        {},
    ),
}
# Metal-poor / scale experiments on c03 (see scratchpad proposal): a seed
# control, a wider censoring band above the survey floors, [Fe/H]-balanced
# sampling, and wide-binary companions inheriting their primary's [Fe/H].
C03 = {
    "exclude_surveys": [*LEAN, "sdss_dr19_astra_astronn"],
    "spectra_groups": "LIT",
}
CENSOR_BAND = {
    **CENSOR,
    **{s: {"fe_h": -1.8} for s in ("lamost_dr12_lrs", "lamost_dr12_mrs",
                                   "sdss_dr19_astra_aspcap", "apogee_dr17")},
    "rave_dr6": {"fe_h": -1.4},
}  # fmt: skip
BINARY_POOL = ROOT + "binary_pool_20261119/pool.npz"
VARIANTS.update(
    {
        "d00_c03_seed43": ("v7", C03, {"seed": 43, "pert_seed": 43}),
        "d01_c03_censorband": ("v7", {**C03, "censor": CENSOR_BAND}, {}),
        "d02_c03_balance": (
            "v7", {**C03, "balance": {"power": 0.5, "cap": 10.0}}, {},
        ),
        "d03_c03_binary": (
            "v7",
            {**C03, "reference_pool": {"path": BINARY_POOL, "ratio": 0.25}},
            {},
        ),
    }
)  # fmt: skip
# Mixture-of-multivariate-Student-t head (3 components, full covariance each):
# stability and like-for-like check against c03 / d00 before the full run.
VARIANTS["e01_c03_mix3"] = ("v7", {**C03, "n_components": 3}, {})
# Same with Gaussian components (nu = 1000: excess kurtosis 0.006), full and
# diagonal covariance, to decide Student-t vs Gaussian mixtures.
VARIANTS["e02_c03_mix3_gauss"] = ("v7", {**C03, "n_components": 3, "nu": 1000.0}, {})
VARIANTS["e03_c03_mix3_gauss_diag"] = (
    "v7",
    {**C03, "n_components": 3, "nu": 1000.0, "full_covariance": False},
    {},
)
GROUPS = {  # one GPU session each, run in sequence (about 50 min per variant)
    # a01_v6 and b01_v7 finished before the 18:30-18:55 UTC session losses.
    "1": ["a02_clean", "a03_censor", "a04_nopm"],
    "2": ["a05_nosky", "a06_small_noncore", "a07_small_all"],
    "3": ["b02_v7_small", "b03_v7_ctx_ebv", "b04_v7_ctx_plx"],
    "4": ["b05_v7_posenc", "b06_v7_dust", "b07_v7_noperturb"],
    "5": ["c01_v7_apogee_lean", "c02_v7_litdedupe", "c03_v7_lean_lit_noastronn"],
    "6": ["d00_c03_seed43", "d01_c03_censorband"],
    "7": ["d03_c03_binary", "d02_c03_balance"],
    "8": ["e01_c03_mix3"],
    "9": ["e02_c03_mix3_gauss"],
    "10": ["e03_c03_mix3_gauss_diag"],
}
ABL_EPOCHS = 6
ABL_TRAIN_ROWS = 1_200_000  # about 25 % of the v5 table's training split


def v7_config(v6: dict) -> dict:
    c = copy.deepcopy(v6)
    run = ROOT + "multisurvey_finetune_20261118_v7/"
    c["saving"]["model_str"] = run + "finetune.pth"
    c["saving"]["log_file"] = run + "finetune.log"
    c["multisurvey"].update(copy.deepcopy(V7_MS))
    return c


def main():
    v6 = yaml.safe_load(V6.read_text())
    v7 = v7_config(v6)
    V7.write_text(
        "# Generated by scripts/make_v7_configs.py from the v6 config.\n"
        "# v7 = v6 + clean label input (masking/noise only on reconstruction),\n"
        "# censored [Fe/H] floors of LAMOST, APOGEE pipelines and RAVE, the SLAM\n"
        "# domain cut, proper motions and RA/Dec hidden from the encoder.\n"
        "# Evaluate with --condition-parallax 5 (inference-time Gaia parallax).\n"
        "# Everything else as v6 (see finetune-multisurvey-20261118-v6.yaml).\n"
        + yaml.safe_dump(v7, sort_keys=False)
    )
    ABL.mkdir(exist_ok=True)
    for name, (base, ms, ft) in VARIANTS.items():
        c = copy.deepcopy(v6 if base == "v6" else v7)
        ms = copy.deepcopy(ms)
        if ms.get("spectra_groups") == "LIT":
            ms["spectra_groups"] = _groups_plus_literature(v6)
        c["multisurvey"].update(ms)
        c["finetuning"].update(ft)
        c["finetuning"]["num_epochs"] = ABL_EPOCHS
        c["finetuning"]["lr_scheduler_head_step_epochs"] = 4
        run = f"{ROOT}{ABL_RUN}/{name}/"
        c["saving"]["model_str"] = run + "finetune.pth"
        c["saving"]["log_file"] = run + "finetune.log"
        (ABL / f"{name}.yaml").write_text(
            f"# Ablation {name} (base {base}); generated by scripts/make_v7_configs.py\n"
            f"# multisurvey overrides: {ms}\n# finetuning overrides: {ft}\n"
            + yaml.safe_dump(c, sort_keys=False)
        )
    print("wrote", V7.name, "and", len(VARIANTS), "ablation configs")


if __name__ == "__main__":
    main()
