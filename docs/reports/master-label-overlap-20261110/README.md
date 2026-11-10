# [Fe/H] overlap with the high-resolution reference, 2026-11-10

CANFAR session `wq1fimly` (`master_label_overlap_20261110_v1`), produced by
`scripts/run_master_label_overlap.py`
(`training/label_overlap.py`). Inputs: the master-label census v3 observation
table and every staged `native_observations` sidecar under the
stellar-improvement root. Full numbers are in `overlap.json`. Diagnostics only:
no gauge was fixed and no training input changed.

**Reference:** per-star median of the high-resolution literature tier (SAGA,
JINAbase, PASTEL, RPA, Yong, Roederer), 21,465 stars, 2,929 below [Fe/H] = −2
and 687 below −3. For each tested source, reference rows from its own
dependency group, and rows that repeat its own published value (compilations
republish their papers), are excluded. It is pooled literature, not physical
truth, so offsets are relative to it.

**Map complexity:** survey → reference maps are scored by 5-fold CV (median
absolute held-out residual). The simplest map within 0.005 dex (or 2%) of the
best is chosen. Affine needs ≥ 20 overlap stars and piecewise-linear ≥ 200.

## Staged surveys (training-admitted sidecars)

Offset = survey − reference, by reference [Fe/H] bin (number of stars in brackets).

| Survey | Stars | Min [Fe/H] | Overlap | Map | < −3 | −3…−2.5 | −2.5…−2 | New label < −2 / < −2.5 / < −3 |
|---|--:|--:|--:|---|--:|--:|--:|--:|
| LAMOST DR12 LRS (LASP) | 3,265,098 | −2.50 | 2,370 | piecewise | +0.93 (68) | +0.45 (134) | +0.26 (181) | 4,725 / 1,894 / 291 |
| LAMOST DR12 MRS (LASP) | 821,991 | −2.50 | 718 | piecewise | +1.20 (12) | +0.63 (26) | +0.12 (53) | 298 / 160 / 31 |
| BOSSNet DR20 | 928,985 | −3.96 | 567 | piecewise | +1.07 (58) | +0.82 (90) | +0.60 (75) | 1,049 / 387 / 110 |
| DESI DR1 RVSpecFit | 3,681,789 | −3.99 | 166 | offset | +0.26 (48) | +0.17 (41) | +0.24 (27) | 988 / 276 / 65 |
| GALAH DR4 | 631,917 | −3.91 | 749 | affine | +0.45 (9) | +0.20 (40) | +0.14 (74) | 68 / 32 / 18 |
| APOGEE DR17 (admitted) | 200,813 | −2.24 | 955 | identity | — | — | +0.20 (15) | 75 / 12 / 5 |
| RAVE DR6 | 151,160 | −2.36 | 253 | piecewise | — | +0.76 (2) | +0.28 (22) | 19 / 3 / 0 |
| Gaia GSP-Spec | 0 | — | — | — | — | — | — | the staged sidecar has no observed [Fe/H] channel |

The last column counts staged-survey stars whose value is at or above the
threshold while a new metal-poor or high-res source puts them below it. This
is what the ID-only audit could not show. LAMOST DR8 LRS (superseded) behaves
like DR12.

## Master-label sources

| Source | Stars | Overlap | Median offset | Robust σ | Map | Offset < −3 | −3…−2.5 | −2.5…−2 |
|---|--:|--:|--:|--:|---|--:|--:|--:|
| SDSS DR19 ASPCAP | 695,688 | 2,737 | +0.02 | 0.09 | identity | — | +0.37 (21) | +0.17 (118) |
| SDSS DR12 SSPP | 370,437 | 266 | +0.13 | 0.30 | piecewise | +0.32 (118) | +0.08 (74) | +0.10 (14) |
| Gaia-ESO DR5.1 | 94,145 | 434 | +0.02 | 0.13 | piecewise | +0.51 (5) | +0.12 (8) | +0.12 (20) |
| S5 DR1 | 29,581 | 18 | +0.34 | 0.36 | identity | | | |
| PIGS | 13,231 | 4 | +0.12 | 0.33 | identity | | | |
| LAMOST VMP Li+2018 | 9,674 | 413 | −0.08 | 0.28 | piecewise | +0.07 (120) | −0.08 (140) | −0.10 (119) |
| LAMOST CaT Li+2025 | 8,378 | 317 | +0.16 | 0.19 | offset | +0.46 (50) | +0.19 (124) | +0.08 (133) |
| Pristine Aguado+2019 | 990 | 42 | +0.03 | 0.38 | offset | | | |
| Best & Brightest | 796 | 70 | −0.17 | 0.21 | offset | | | |
| SkyMapper Da Costa t2 | 142 | 87 | −0.34 | 0.31 | offset | −0.19 (61) | −0.63 (19) | −0.93 (7) |
| PASTEL | 17,943 | 1,988 | +0.02 | 0.10 | identity | | | |
| SAGA | 6,188 | 2,513 | +0.00 | 0.10 | identity | | | |
| JINAbase | 1,615 | 699 | +0.04 | 0.18 | affine | | | |
| Roederer+2014 | 295 | 271 | −0.09 | 0.28 | piecewise | | | |
| RPA, Yong, Yong+2021 | 22–152 each | | −0.11…+0.10 | 0.04–0.30 | identity/offset/affine | | | |

## Findings

1. **LAMOST LASP and BOSSNet compress the metal-poor tail.** LASP stops at
   −2.50 and reads +0.45 dex high at −3…−2.5 and +0.9–1.2 dex high below −3.
   BOSSNet reads +0.6 to +1.1 dex high below −2 with σ ≈ 0.5. Their maps need
   curvature, and below about −2.5 they carry little metallicity information:
   treat them as censored or heavily down-weighted there, not as precise labels.
2. **The new metal-poor sources fix specific stars:** 4,725 LAMOST LRS,
   1,049 BOSSNet and 988 DESI stars move below −2 when a dedicated metal-poor
   or high-res source is used, and 291 / 110 / 65 of them move below −3.
3. **DESI RVSpecFit is offset, not compressed:** about +0.2 to +0.26 dex at
   every metal-poor bin, so an offset map is enough.
4. **Small catalogues only support simple maps.** With fewer than about 100
   overlap stars, CV never prefers more than an offset; for most high-res
   samples identity is best. SkyMapper FITTER (−0.34), S5 (+0.34, n = 18) and
   the CaT calibration (+0.16, +0.46 below −3) have real offsets to keep.
5. **Staged APOGEE DR17 is cut at −2.24** by the admission quality policy; it
   adds nothing below that. DR19 ASPCAP behaves the same (edge at −2.45).
6. **The GSP-Spec sidecar has no observed [Fe/H] channel.** Its metallicity
   was not mapped to `fe_h` (GSP-Spec reports [M/H]; [Fe/H] is a separate
   flagged quantity). It needs an explicit adapter before it can be compared
   or mapped.
