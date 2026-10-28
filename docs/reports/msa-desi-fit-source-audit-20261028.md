# DESI candidate-label audit — 28 October 2026

The proposed DESI addition is **1,192,466 candidate FIT logg labels**, not a billion stars. The exact post-collation count is still being checked. These are raw survey values and should not be pooled into a common physical gravity scale without further work.

A saved-array audit used 1,113,196 FIT stars with primary input, a quality-selected DESI logg, and exactly one occurrence of their Gaia ID in the complete saved DESI raw-field table. This subset may include old-label overlaps and excludes stars with multiple raw rows, so it is not the full candidate addition. Of these stars, 951,590 (85.5%) have raw DESI logg in [4, 5), 90,841 (8.2%) in [3, 4), and 22,314 (2.0%) in [2, 3). The median quoted random error is 0.044 dex; 31 are missing or nonpositive. Formal errors do not include all systematic or cross-survey errors.

In the same selected FIT cohort, an exact Gaia-ID join to an **earlier** saved native survey sidecar found 4,544 quality-observed logg pairs. All paired native source-row, ID and logg associations were rechecked against their saved source groups. Differences below are native minus raw DESI, in dex; bins are defined by the native survey logg.

| Native logg | GALAH pairs | GALAH median difference | APOGEE pairs | APOGEE median difference |
|---|---:|---:|---:|---:|
| All | 3,151 | -0.282 | 1,393 | -0.618 |
| [2, 3) | 399 | -0.640 | 723 | -0.699 |
| [3, 4) | 652 | -0.324 | 276 | -0.496 |
| [4, 5) | 2,043 | -0.237 | 191 | -0.106 |

The separate [source audit](../../experiments/xp/multisurvey-desi-fit-source-audit-20261028.json) and [independent join check](../../experiments/xp/multisurvey-desi-fit-source-verification-20261028.json) agree on the cohort histogram, counts and medians. The exact original remote FITS and the current selected overlay were not re-read in this saved-array audit; the saved DESI table and census share the original FITS SHA provenance. The older native sidecar need not equal every later remediation-source record. This is a selected training overlap, not a fresh test or an independent measurement of truth. Binning by a noisy native label can itself produce conditional bias; the source audit also reports bins defined by DESI logg. GALAH gravity has photometric/astrometric dependencies. The offsets therefore identify a serious scale/label compatibility problem to investigate; they do not establish which survey is correct or explain the older model's entire logg error.

The [DESI DR1 stellar catalogue paper](https://arxiv.org/html/2505.14787) applies `VSINI < 30 km/s` when validating RVS stellar parameters. Our current DESI sidecar neither saves nor applies that field. The paper notes a deficit of RVS giants below logg 2, a possible gravity bias, and structure among cool dwarfs; its bounded Fe/H correction uses survey overlaps and is not applied by the raw adapter. The [official DR1 VAC description](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws/) describes the scale of the catalogue; none of these numbers imply a billion labelled stars.

Before scaling multi-survey training, measure exact post-repeat-selection coverage and the survey overlap by stellar regime; inspect VSINI and source-quality fields in the already-staged publisher-bound FITS; compare survey-dependent gravity and temperature, including independent physical anchors; and use explicit source-dependent likelihoods and uncertainty assumptions. A constant 0.1-dex illustrative floor does not establish that these systematic differences are resolved. Keep same-star Andrae and held-out source comparisons separate from the FIT overlap.

Reproduce the source audit with `scripts/audit_desi_fit_sources.py --output <new-json>` and its independent arithmetic with `scripts/verify_desi_fit_sources_independent.py`; both read existing saved arrays and do not load a stellar model. The audit JSON records all input hashes and detailed masks.
