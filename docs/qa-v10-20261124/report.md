# QA of the v10 chain (msa-split-v1 TEST)

Evaluator `scripts/evaluate_model.py` (commit `4a5876d1e2`), 2026-11-24T05:11:34 UTC. Split `msa-split-v1` (/arc/projects/astroai/msa/runs/msa_split_v1 (local copy of overrides.parquet, sha256 in README)); partition **TEST** only (QA-reserved objects report-only); rows with table split in [0, 1] dropped (seen by the model).

Conventions (qa.md §0): bias = median(prediction − reference), NMAD = 1.4826 MAD, coverage of the (noisy) reference with Wilson 95 % Bonferroni bands, cells with n < 20 descriptive (*). Teff in K, others in dex or mag.

## Headline

| set | product | n | bias | NMAD | 68 % cov | 95 % cov | strata failing 68 % |
|---|---|--:|--:|--:|--:|--:|--:|
| posterior_v1test | teff | 15473 | +21 | 121 | 0.342 | – | 42 |
| posterior_v1test | logg | 15471 | -0.023 | 0.139 | 0.491 | – | 42 |
| posterior_v1test | fe_h | 15066 | -0.004 | 0.121 | 0.533 | – | 41 |
| posterior_v1test | ebv | 13963 | +0.003 | 0.023 | 0.489 | – | 40 |
| posterior_core | teff | 296 | +21 | 137 | 0.399 | – | 28 |
| posterior_core | logg | 295 | -0.016 | 0.248 | 0.417 | – | 28 |
| posterior_core | fe_h | 296 | -0.012 | 0.169 | 0.453 | – | 28 |
| s1_all_ens | teff | 46148 | -6 | 79 | 0.664 | 0.922 | 16 |
| s1_all_ens | logg | 46134 | +0.004 | 0.194 | 0.665 | 0.925 | 13 |
| s1_all_ens | fe_h | 35202 | +0.001 | 0.117 | 0.661 | 0.921 | 13 |
| s1_all_ens | joint set (own) | 35199 | | | 0.664 |  | 14 |
| s1_all_ens | joint set (own) | 35199 | | |  | 0.923 |  |
| s2_flow | teff | 46148 | -5 | 71 | 0.557 | – | 38 |
| s2_flow | logg | 46134 | -0.007 | 0.151 | 0.564 | – | 38 |
| s2_flow | fe_h | 35202 | -0.001 | 0.101 | 0.654 | – | 19 |
| s2_flow | joint set (own) | 35199 | | | 0.661 |  | 25 |
| s2_flow | joint set (own) | 35199 | | |  | 0.922 |  |
| s3_gmix | parallax_pit | 46491 | | | 0.720 | | 15 |
| s3_gmix | ebv_pit | 33730 | | | 0.898 | | 19 |

- core `posterior_core`: worst |bias| (G ≤ 16, n ≥ 20) +0.041 in [-1.0,-0.5) (296 TEST rows with a core reference)
- core `s1_all_ens`: worst |bias| (G ≤ 16, n ≥ 20) +0.031 in [-1.0,-0.5) (296 TEST rows with a core reference)
- core `s2_flow`: worst |bias| (G ≤ 16, n ≥ 20) +0.057 in [-1.0,-0.5) (296 TEST rows with a core reference)
- wide binaries: Δ[Fe/H] +0.086 / NMAD 0.189 (33321 pairs), |z| < 1 0.57
- open clusters: giant − dwarf +0.328 (224 clusters), median χ²/dof 2.63
- binary pool, primary − companion: 3000-3700 K +0.312; 3700-4200 K +0.256; 4200-4800 K +0.095; 4800-5500 K +0.016

## Held-out labels and calibration

### posterior_v1test (`posterior_v1test`, kind B)

Rows: FIT 97003, DEV 6433, CAL 10316, TEST 15473, QA_RESERVED 702, excluded_seen_in_partition 0, evaluated 15473. Integrity: 0 duplicate ids; non-monotone quantiles teff 0, logg 0, fe_h 0, ebv 0, parallax 0.
References: teff from prediction file (<p>_ref), logg from prediction file (<p>_ref), fe_h from prediction file (<p>_ref), ebv from prediction file (<p>_ref).

**teff** (K) vs reference: bias / NMAD (n) overall +21 / 121 (15473); pull NMAD 2.31; PIT KS 0.321
- 68 % coverage (central quantile interval): 0.342; 42 of 42 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: <0.5, 0.5-0.8, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, rgb, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; ebv_sfd: <0.05, 0.05-0.3, 0.3-1, >1; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: +39 / 101 (6759); 13.5-15: +26 / 117 (3217); 15-16: -26 / 138 (1524); 16-17: -1 / 150 (2165); 17-18: -15 / 151 (1783); 18-19: -493 / 438 (9)*; >19: -414 / 291 (16)*
- by feh_ref: <-3: -4 / 486 (34); -3--2.5: -24 / 180 (118); -2.5--2: +30 / 224 (589); -2--1.5: -8 / 154 (1309); -1.5--1: -4 / 152 (3213); -1--0.5: +18 / 108 (1140); >-0.5: +33 / 103 (8663)
- by stage: dwarf_ms: +60 / 136 (7402); turnoff_sgb: +64 / 135 (1870); rc: -8 / 66 (2154); rgb: -8 / 69 (98); agb_other: -12 / 80 (3949)
- by input_class: X: +21 / 121 (15473)
- by ruwe: <1.4: +18 / 117 (13905); >=1.4: +58 / 137 (1568)
- by abs_b: <5: +8 / 161 (1899); 5-15: +9 / 92 (3065); 15-30: +32 / 117 (3488); 30-60: +28 / 125 (5470); >60: +13 / 137 (1551)

**logg** (dex) vs reference: bias / NMAD (n) overall -0.023 / 0.139 (15471); pull NMAD 1.47; PIT KS 0.211
- 68 % coverage (central quantile interval): 0.491; 42 of 42 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: <0.5, 0.5-0.8, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, rgb, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; ebv_sfd: <0.05, 0.05-0.3, 0.3-1, >1; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: -0.013 / 0.091 (6757); 13.5-15: -0.020 / 0.108 (3217); 15-16: -0.038 / 0.234 (1524); 16-17: -0.076 / 0.261 (2165); 17-18: -0.082 / 0.284 (1783); 18-19: -0.758 / 0.217 (9)*; >19: -0.857 / 0.089 (16)*
- by feh_ref: <-3: -0.202 / 0.972 (34); -3--2.5: -0.036 / 0.445 (118); -2.5--2: -0.042 / 0.424 (589); -2--1.5: -0.085 / 0.289 (1309); -1.5--1: -0.092 / 0.258 (3213); -1--0.5: -0.018 / 0.145 (1140); >-0.5: -0.014 / 0.086 (8662)
- by stage: dwarf_ms: -0.031 / 0.116 (7401); turnoff_sgb: +0.020 / 0.149 (1869); rc: -0.068 / 0.142 (2154); rgb: -0.017 / 0.124 (98); agb_other: +0.014 / 0.195 (3949)
- by input_class: X: -0.023 / 0.139 (15471)
- by ruwe: <1.4: -0.026 / 0.143 (13903); >=1.4: +0.004 / 0.111 (1568)
- by abs_b: <5: -0.017 / 0.197 (1899); 5-15: -0.019 / 0.122 (3064); 15-30: -0.023 / 0.109 (3488); 30-60: -0.029 / 0.135 (5469); >60: -0.014 / 0.210 (1551)

**fe_h** (dex) vs reference: bias / NMAD (n) overall -0.004 / 0.121 (15066); pull NMAD 1.37; PIT KS 0.163
- 68 % coverage (central quantile interval): 0.533; 41 of 42 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: <0.5, 0.5-0.8, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, rgb, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; ebv_sfd: <0.05, 0.05-0.3, 0.3-1, >1; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: +0.005 / 0.076 (6361); 13.5-15: -0.003 / 0.108 (3208); 15-16: -0.024 / 0.195 (1524); 16-17: -0.038 / 0.253 (2165); 17-18: -0.060 / 0.320 (1783); 18-19: -0.177 / 0.431 (9)*; >19: -0.140 / 0.821 (16)*
- by feh_ref: <-3: +0.278 / 0.649 (34); -3--2.5: +0.296 / 0.346 (118); -2.5--2: +0.036 / 0.353 (589); -2--1.5: -0.058 / 0.282 (1309); -1.5--1: -0.064 / 0.271 (3213); -1--0.5: +0.012 / 0.117 (1140); >-0.5: +0.002 / 0.080 (8663)
- by stage: dwarf_ms: -0.018 / 0.140 (7321); turnoff_sgb: -0.012 / 0.119 (1822); rc: -0.000 / 0.081 (2110); rgb: +0.038 / 0.078 (98); agb_other: +0.017 / 0.112 (3715)
- by input_class: X: -0.004 / 0.121 (15066)
- by ruwe: <1.4: -0.005 / 0.123 (13544); >=1.4: +0.006 / 0.105 (1522)
- by abs_b: <5: +0.025 / 0.159 (1794); 5-15: +0.014 / 0.098 (2897); 15-30: -0.004 / 0.104 (3404); 30-60: -0.017 / 0.126 (5429); >60: -0.042 / 0.163 (1542)

**ebv** (mag) vs reference: bias / NMAD (n) overall +0.003 / 0.023 (13963); pull NMAD 1.51; PIT KS 0.218
- 68 % coverage (central quantile interval): 0.489; 40 of 40 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: <0.5, 0.5-0.8, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; feh_ref: -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; feh_pred: -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, rgb, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; ebv_sfd: <0.05, 0.05-0.3, 0.3-1, >1; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: +0.003 / 0.018 (5943); 13.5-15: +0.003 / 0.024 (2980); 15-16: +0.002 / 0.034 (1295); 16-17: +0.006 / 0.030 (2034); 17-18: +0.005 / 0.030 (1709); 18-19: +0.311 / 0.000 (1)*; >19: -0.099 / 0.000 (1)*
- by feh_ref: <-3: -0.008 / 0.022 (6)*; -3--2.5: +0.002 / 0.039 (94); -2.5--2: +0.015 / 0.041 (501); -2--1.5: +0.006 / 0.029 (1196); -1.5--1: +0.004 / 0.030 (2899); -1--0.5: +0.006 / 0.027 (981); >-0.5: +0.002 / 0.020 (8286)
- by stage: dwarf_ms: +0.003 / 0.021 (7079); turnoff_sgb: +0.008 / 0.021 (1612); rc: +0.000 / 0.021 (2109); rgb: +0.004 / 0.029 (83); agb_other: +0.003 / 0.029 (3080)
- by input_class: X: +0.003 / 0.023 (13963)
- by ruwe: <1.4: +0.003 / 0.023 (12529); >=1.4: +0.004 / 0.026 (1434)
- by abs_b: <5: +0.007 / 0.031 (1299); 5-15: +0.002 / 0.022 (2684); 15-30: +0.003 / 0.022 (3266); 30-60: +0.003 / 0.022 (5204); >60: +0.003 / 0.024 (1510)

**IRFM Teff**: -4 / 34 (3150) (training anchor: tests transfer)

**Physical references** (bias / NMAD (n); TEST and reserved stars not seen in training; 0 seen rows dropped):
- anchor_apok2: 22 of 4660 predicted; logg -0.046 / 0.056 (22)
- anchor_apokasc3: 109 of 13871 predicted; logg -0.007 / 0.088 (109)
- anchor_apokasc_dwarfs: 2 of 413 predicted; logg -0.023 / 0.062 (2)*
- anchor_kepler_legacy: 2 of 65 predicted; logg -0.021 / 0.042 (2)*
- gbs_v3: 5 of 192 predicted; teff -48 / 45 (5)*; logg -0.156 / 0.101 (5)*; fe_h +0.021 / 0.060 (5)*

**Competitors** (exact-id overlap; model vs competitor against the same reference):

| catalogue | overlap | product | model bias / NMAD | competitor bias / NMAD | Δ median abs err (95 %) |
|---|--:|---|---|---|---|
| andrae2023 | 14009 | teff | +24 / 122 (11003) | -45 / 136 (11003) | -7 [-10, -4] |
| andrae2023 | 14009 | logg | -0.028 / 0.140 (11002) | -0.029 / 0.167 (11002) | -0.015 [-0.019, -0.012] |
| andrae2023 | 14009 | fe_h | -0.012 / 0.135 (10630) | +0.028 / 0.173 (10630) | -0.022 [-0.026, -0.019] |
| starhorse2023 | 1905 | teff | +96 / 87 (1905) | +4 / 65 (1905) | +60 [+55, +66] |
| starhorse2023 | 1905 | logg | -0.014 / 0.073 (1905) | +0.004 / 0.042 (1905) | +0.024 [+0.021, +0.028] |
| starhorse2023 | 1905 | fe_h | -0.025 / 0.072 (1862) | +0.016 / 0.088 (1862) | -0.007 [-0.011, -0.003] |
| starhorse2023 | 1905 | ebv | +0.006 / 0.016 (1840) | -0.010 / 0.035 (1840) | -0.015 [-0.017, -0.013] |
| shboost | 15450 | teff | +21 / 120 (15450) | +8 / 112 (15450) | +8 [+6, +10] |
| shboost | 15450 | logg | -0.023 / 0.138 (15448) | -0.015 / 0.146 (15448) | -0.004 [-0.007, -0.002] |
| shboost | 15450 | fe_h | -0.004 / 0.121 (15043) | +0.070 / 0.188 (15043) | -0.039 [-0.042, -0.037] |
| shboost | 15450 | ebv | +0.003 / 0.023 (13962) | -0.021 / 0.047 (13962) | -0.018 [-0.019, -0.018] |
| gspphot | 15473 | teff | +30 / 125 (12003) | +23 / 252 (12003) | -80 [-85, -76] |
| gspphot | 15473 | logg | -0.026 / 0.139 (12001) | +0.066 / 0.237 (12001) | -0.062 [-0.067, -0.056] |
| gspphot | 15473 | fe_h | -0.010 / 0.127 (11789) | -0.079 / 0.347 (11789) | -0.176 [-0.181, -0.171] |

- andrae2023: own training stars (in_training_sample) removed from the comparison
- andrae2023: 3006 of 14009 overlap stars are in its training set and are not compared
- starhorse2023: spectroscopic StarHorse runs take the survey Teff, log g and [M/H] as inputs: agreement with survey references is partly by construction
- shboost: trained on spectroscopic (StarHorse) labels of the same surveys; no training flag, so overlap with its training set is not removed
- gspphot: Gaia DR3 GSP-Phot (Aeneas), independent of the survey labels

### posterior_core (`posterior_core`, kind B)

Rows: FIT 27319, DEV 1869, CAL 2982, TEST 4399, QA_RESERVED 2504, excluded_seen_in_partition 4103, evaluated 296. Integrity: 0 duplicate ids; non-monotone quantiles teff 0, logg 0, fe_h 0, ebv 0, parallax 0.
References: teff from prediction file (<p>_ref), logg from prediction file (<p>_ref), fe_h from prediction file (<p>_ref), ebv from prediction file (<p>_ref).

**teff** (K) vs reference: bias / NMAD (n) overall +21 / 137 (296); pull NMAD 2.06; PIT KS 0.287
- 68 % coverage (central quantile interval): 0.399; 28 of 28 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: 0.5-0.8, 0.8-1.2, 1.2-1.8; feh_ref: -1--0.5, >-0.5; feh_pred: -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; ebv_sfd: <0.05, 0.05-0.3, 0.3-1, >1; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: -8 / 124 (124); 13.5-15: +89 / 78 (33); 15-16: +64 / 114 (48); 16-17: -19 / 110 (61); 17-18: +69 / 223 (30)
- by feh_ref: <-3: +72 / 316 (5)*; -3--2.5: +97 / 126 (6)*; -2.5--2: +155 / 297 (9)*; -2--1.5: +311 / 407 (10)*; -1.5--1: +129 / 305 (19)*; -1--0.5: +59 / 143 (45); >-0.5: +1 / 115 (202)
- by stage: dwarf_ms: +6 / 122 (162); turnoff_sgb: +30 / 181 (36); rc: +30 / 138 (31); rgb: -11 / 29 (7)*; agb_other: +52 / 153 (60)
- by input_class: X: +21 / 137 (296)
- by ruwe: <1.4: +29 / 135 (261); >=1.4: -7 / 96 (35)
- by abs_b: <5: -31 / 204 (49); 5-15: +7 / 134 (68); 15-30: +27 / 102 (59); 30-60: +23 / 137 (90); >60: +64 / 97 (30)

**logg** (dex) vs reference: bias / NMAD (n) overall -0.016 / 0.248 (295); pull NMAD 1.98; PIT KS 0.251
- 68 % coverage (central quantile interval): 0.417; 28 of 28 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: 0.5-0.8, 0.8-1.2, 1.2-1.8; feh_ref: -1--0.5, >-0.5; feh_pred: -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; ebv_sfd: <0.05, 0.05-0.3, 0.3-1, >1; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: -0.034 / 0.258 (123); 13.5-15: -0.029 / 0.145 (33); 15-16: -0.021 / 0.179 (48); 16-17: -0.001 / 0.248 (61); 17-18: +0.138 / 0.418 (30)
- by feh_ref: <-3: -0.114 / 0.371 (5)*; -3--2.5: +0.081 / 0.403 (6)*; -2.5--2: -0.032 / 0.343 (9)*; -2--1.5: +0.250 / 0.410 (10)*; -1.5--1: +0.028 / 0.504 (19)*; -1--0.5: -0.012 / 0.269 (45); >-0.5: -0.026 / 0.219 (201)
- by stage: dwarf_ms: -0.017 / 0.171 (161); turnoff_sgb: +0.124 / 0.298 (36); rc: -0.182 / 0.360 (31); rgb: -0.389 / 0.543 (7)*; agb_other: +0.019 / 0.416 (60)
- by input_class: X: -0.016 / 0.248 (295)
- by ruwe: <1.4: -0.015 / 0.244 (260); >=1.4: -0.022 / 0.279 (35)
- by abs_b: <5: -0.111 / 0.443 (49); 5-15: -0.032 / 0.235 (68); 15-30: -0.013 / 0.178 (58); 30-60: +0.000 / 0.202 (90); >60: +0.096 / 0.239 (30)

**fe_h** (dex) vs reference: bias / NMAD (n) overall -0.012 / 0.169 (296); pull NMAD 1.54; PIT KS 0.193
- 68 % coverage (central quantile interval): 0.453; 28 of 28 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: 0.5-0.8, 0.8-1.2, 1.2-1.8; feh_ref: -1--0.5, >-0.5; feh_pred: -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; ebv_sfd: <0.05, 0.05-0.3, 0.3-1, >1; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: +0.002 / 0.125 (124); 13.5-15: -0.017 / 0.164 (33); 15-16: -0.010 / 0.150 (48); 16-17: -0.049 / 0.262 (61); 17-18: -0.071 / 0.375 (30)
- by feh_ref: <-3: +0.349 / 0.125 (5)*; -3--2.5: +0.263 / 0.374 (6)*; -2.5--2: +0.001 / 0.249 (9)*; -2--1.5: +0.014 / 0.137 (10)*; -1.5--1: +0.020 / 0.366 (19)*; -1--0.5: -0.012 / 0.225 (45); >-0.5: -0.020 / 0.140 (202)
- by stage: dwarf_ms: -0.041 / 0.157 (162); turnoff_sgb: -0.039 / 0.140 (36); rc: -0.024 / 0.198 (31); rgb: +0.036 / 0.040 (7)*; agb_other: +0.040 / 0.210 (60)
- by input_class: X: -0.012 / 0.169 (296)
- by ruwe: <1.4: -0.012 / 0.176 (261); >=1.4: -0.009 / 0.136 (35)
- by abs_b: <5: -0.101 / 0.260 (49); 5-15: -0.014 / 0.131 (68); 15-30: +0.023 / 0.194 (59); 30-60: -0.023 / 0.173 (90); >60: +0.011 / 0.157 (30)

**Physical references** (bias / NMAD (n); TEST and reserved stars not seen in training; 2330 seen rows dropped):
- anchor_apok2: 0 of 4660 predicted; no overlap
- anchor_apokasc3: 2 of 13871 predicted; logg -0.210 / 0.219 (2)*
- anchor_apokasc_dwarfs: 2 of 413 predicted; logg +0.133 / 0.238 (2)*
- anchor_kepler_legacy: 0 of 65 predicted; no overlap
- gbs_v3: 4 of 192 predicted; teff +4 / 58 (4)*; logg -0.089 / 0.154 (4)*; fe_h -0.048 / 0.074 (4)*

**Competitors** (exact-id overlap; model vs competitor against the same reference):

| catalogue | overlap | product | model bias / NMAD | competitor bias / NMAD | Δ median abs err (95 %) |
|---|--:|---|---|---|---|
| andrae2023 | 227 | teff | +22 / 131 (209) | -29 / 162 (209) | -29 [-50, -7] |
| andrae2023 | 227 | logg | -0.022 / 0.226 (208) | -0.024 / 0.236 (208) | +0.005 [-0.041, +0.030] |
| andrae2023 | 227 | fe_h | -0.015 / 0.153 (209) | +0.013 / 0.201 (209) | -0.025 [-0.053, +0.002] |
| starhorse2023 | 42 | teff | +43 / 109 (42) | +15 / 96 (42) | +2 [-30, +32] |
| starhorse2023 | 42 | logg | -0.018 / 0.225 (42) | -0.048 / 0.124 (42) | +0.088 [+0.010, +0.167] |
| starhorse2023 | 42 | fe_h | -0.044 / 0.100 (42) | -0.016 / 0.102 (42) | -0.011 [-0.054, +0.072] |
| starhorse2023 | 42 | ebv | – | – | – [–, –] |
| shboost | 296 | teff | +21 / 137 (296) | +17 / 161 (296) | -15 [-34, -1] |
| shboost | 296 | logg | -0.016 / 0.248 (295) | -0.015 / 0.260 (295) | -0.006 [-0.042, +0.021] |
| shboost | 296 | fe_h | -0.012 / 0.169 (296) | +0.031 / 0.241 (296) | -0.051 [-0.077, -0.018] |
| shboost | 296 | ebv | – | – | – [–, –] |
| gspphot | 296 | teff | +26 / 130 (241) | -25 / 276 (241) | -113 [-149, -78] |
| gspphot | 296 | logg | -0.010 / 0.205 (240) | +0.092 / 0.296 (240) | -0.053 [-0.096, -0.007] |
| gspphot | 296 | fe_h | -0.020 / 0.160 (241) | -0.161 / 0.386 (241) | -0.172 [-0.216, -0.142] |

- andrae2023: own training stars (in_training_sample) removed from the comparison
- andrae2023: 18 of 227 overlap stars are in its training set and are not compared
- starhorse2023: spectroscopic StarHorse runs take the survey Teff, log g and [M/H] as inputs: agreement with survey references is partly by construction
- shboost: trained on spectroscopic (StarHorse) labels of the same surveys; no training flag, so overlap with its training set is not removed
- gspphot: Gaia DR3 GSP-Phot (Aeneas), independent of the survey labels

### s1_all_ens (`s1_all_ens`, kind A)

Rows: FIT 900526, DEV 60032, CAL 95639, TEST 144933, QA_RESERVED 4651, excluded_seen_in_partition 98442, evaluated 46491. Integrity: 0 duplicate ids; non-monotone quantiles teff 1491, logg 201, fe_h 332.
References: teff from --labels, logg from --labels, fe_h from --labels.

**teff** (K) vs reference: bias / NMAD (n) overall -6 / 79 (46148); pull NMAD 1.18
- 68 % coverage (model set): 0.664; 16 of 39 strata (n ≥ 20) outside ±0.03 (G: 18-19; bp_rp: <0.5, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -1.5--1; stage: turnoff_sgb; abs_b: <5; ruwe: >=1.4)
- 95 % coverage (model set): 0.922; 19 of 39 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 18-19; bp_rp: <0.5, >2.5; feh_ref: <-3, -2.5--2, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -1.5--1; stage: turnoff_sgb; abs_b: <5, 5-15; ruwe: >=1.4)
- by G: <=13.5: -7 / 78 (21311); 13.5-15: -5 / 69 (9227); 15-16: -6 / 82 (5115); 16-17: -2 / 88 (6095); 17-18: -8 / 94 (4346); 18-19: -4485 / 4834 (37); >19: -234 / 784 (17)*
- by feh_ref: <-3: +17 / 469 (61); -3--2.5: +3 / 101 (132); -2.5--2: -10 / 115 (577); -2--1.5: -5 / 91 (1453); -1.5--1: -10 / 92 (2544); -1--0.5: -1 / 75 (6720); >-0.5: -2 / 62 (23713)
- by stage: dwarf_ms: -4 / 74 (24275); turnoff_sgb: +6 / 108 (5712); rc: -10 / 76 (4708); rgb: -10 / 72 (465); agb_other: -12 / 83 (10988)
- by input_class: X: -6 / 79 (46148)
- by ruwe: <1.4: -6 / 77 (41199); >=1.4: -0 / 95 (4949)
- by abs_b: <5: -25 / 163 (7006); 5-15: -7 / 98 (10926); 15-30: -3 / 77 (11062); 30-60: -4 / 61 (13436); >60: +1 / 44 (3718)

**logg** (dex) vs reference: bias / NMAD (n) overall +0.004 / 0.194 (46134); pull NMAD 1.29
- 68 % coverage (model set): 0.665; 13 of 39 strata (n ≥ 20) outside ±0.03 (G: 18-19; bp_rp: <0.5; feh_ref: <-3, -3--2.5, -2.5--2, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2; stage: turnoff_sgb; abs_b: <5, 5-15; ruwe: >=1.4)
- 95 % coverage (model set): 0.925; 17 of 39 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 15-16, 18-19; bp_rp: <0.5; feh_ref: <-3, -2.5--2, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -1.5--1; stage: turnoff_sgb; abs_b: <5, 5-15; ruwe: >=1.4)
- by G: <=13.5: -0.005 / 0.175 (21302); 13.5-15: -0.002 / 0.157 (9226); 15-16: -0.004 / 0.208 (5115); 16-17: +0.037 / 0.242 (6094); 17-18: +0.065 / 0.321 (4344); 18-19: -1.351 / 1.901 (36); >19: +0.068 / 0.152 (17)*
- by feh_ref: <-3: +0.072 / 0.161 (61); -3--2.5: +0.044 / 0.394 (132); -2.5--2: +0.019 / 0.370 (577); -2--1.5: +0.021 / 0.286 (1453); -1.5--1: +0.014 / 0.269 (2544); -1--0.5: +0.018 / 0.206 (6720); >-0.5: +0.009 / 0.150 (23712)
- by stage: dwarf_ms: -0.001 / 0.167 (24266); turnoff_sgb: +0.054 / 0.217 (5709); rc: -0.031 / 0.216 (4707); rgb: -0.072 / 0.230 (465); agb_other: +0.012 / 0.238 (10987)
- by input_class: X: +0.004 / 0.194 (46134)
- by ruwe: <1.4: +0.005 / 0.194 (41187); >=1.4: +0.002 / 0.188 (4947)
- by abs_b: <5: -0.018 / 0.320 (7005); 5-15: +0.001 / 0.222 (10924); 15-30: +0.009 / 0.173 (11061); 30-60: +0.006 / 0.161 (13427); >60: +0.016 / 0.161 (3717)

**fe_h** (dex) vs reference: bias / NMAD (n) overall +0.001 / 0.117 (35202); pull NMAD 1.11
- 68 % coverage (model set): 0.661; 13 of 39 strata (n ≥ 20) outside ±0.03 (G: 18-19; bp_rp: <0.5; feh_ref: <-3, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -1.5--1; stage: turnoff_sgb, rgb; abs_b: <5, 5-15; ruwe: >=1.4)
- 95 % coverage (model set): 0.921; 19 of 39 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 18-19; bp_rp: <0.5, 0.5-0.8, >2.5; feh_ref: <-3, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -1.5--1; stage: turnoff_sgb; abs_b: <5, 5-15; ruwe: >=1.4)
- by G: <=13.5: +0.008 / 0.068 (10674); 13.5-15: +0.006 / 0.098 (8978); 15-16: -0.003 / 0.141 (5091); 16-17: -0.017 / 0.206 (6074); 17-18: -0.052 / 0.289 (4332); 18-19: -1.253 / 2.394 (36); >19: +0.021 / 0.863 (17)*
- by feh_ref: <-3: -0.070 / 0.701 (61); -3--2.5: -0.039 / 0.555 (133); -2.5--2: -0.028 / 0.467 (578); -2--1.5: -0.008 / 0.305 (1453); -1.5--1: +0.002 / 0.235 (2544); -1--0.5: +0.008 / 0.154 (6720); >-0.5: -0.000 / 0.093 (23713)
- by stage: dwarf_ms: -0.001 / 0.122 (21962); turnoff_sgb: +0.005 / 0.110 (4474); rc: -0.000 / 0.096 (2821); rgb: +0.003 / 0.102 (200); agb_other: +0.005 / 0.118 (5745)
- by input_class: X: +0.001 / 0.117 (35202)
- by ruwe: <1.4: +0.001 / 0.118 (31566); >=1.4: +0.004 / 0.110 (3636)
- by abs_b: <5: +0.004 / 0.184 (4092); 5-15: +0.016 / 0.114 (6686); 15-30: -0.001 / 0.102 (8808); 30-60: -0.005 / 0.113 (12195); >60: -0.002 / 0.121 (3421)

**Joint 68 % set of the model** (Neyman for A, conformal for B0): coverage 0.664 (n 35199); 14 of 39 strata (n ≥ 20) outside ±0.03 (G: 18-19; bp_rp: <0.5, 1.8-2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5; feh_pred: <-3, -3--2.5, -2.5--2; stage: turnoff_sgb; abs_b: <5, 5-15; ruwe: >=1.4)
- by G: <=13.5: 0.659 [0.647, 0.672] (10673); 13.5-15: 0.664 [0.650, 0.677] (8977); 15-16: 0.660 [0.642, 0.677] (5090); 16-17: 0.666 [0.649, 0.682] (6074); 17-18: 0.684 [0.664, 0.702] (4332); 18-19: 0.222 [0.092, 0.445] (36); >19: 0.529 [0.248, 0.794] (17)

**Joint 95 % set of the model** (Neyman for A, conformal for B0): coverage 0.923 (n 35199); 18 of 39 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 18-19; bp_rp: <0.5, >2.5; feh_ref: <-3, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -1.5--1; stage: turnoff_sgb; abs_b: <5, 5-15; ruwe: >=1.4)
- by G: <=13.5: 0.919 [0.912, 0.926] (10673); 13.5-15: 0.919 [0.911, 0.926] (8977); 15-16: 0.915 [0.903, 0.925] (5090); 16-17: 0.932 [0.923, 0.941] (6074); 17-18: 0.942 [0.931, 0.950] (4332); 18-19: 0.250 [0.110, 0.474] (36); >19: 0.588 [0.292, 0.832] (17)

**Physical references** (bias / NMAD (n); TEST and reserved stars not seen in training; 3810 seen rows dropped):
- anchor_apok2: 29 of 4660 predicted; logg +0.020 / 0.104 (29)
- anchor_apokasc3: 118 of 13871 predicted; logg +0.037 / 0.113 (118)
- anchor_apokasc_dwarfs: 2 of 413 predicted; logg +0.191 / 0.384 (2)*
- anchor_kepler_legacy: 0 of 65 predicted; no overlap
- gbs_v3: 4 of 192 predicted; teff +23 / 85 (4)*; logg -0.093 / 0.199 (4)*; fe_h -0.030 / 0.097 (4)*

**Competitors** (exact-id overlap; model vs competitor against the same reference):

| catalogue | overlap | product | model bias / NMAD | competitor bias / NMAD | Δ median abs err (95 %) |
|---|--:|---|---|---|---|
| andrae2023 | 42169 | teff | -6 / 79 (39001) | -76 / 172 (39001) | -73 [-75, -72] |
| andrae2023 | 42169 | logg | +0.002 / 0.193 (38988) | -0.021 / 0.196 (38988) | +0.000 [-0.002, +0.002] |
| andrae2023 | 42169 | fe_h | +0.001 / 0.116 (28878) | +0.025 / 0.162 (28878) | -0.032 [-0.033, -0.030] |
| starhorse2023 | 6265 | teff | +3 / 68 (6264) | -34 / 78 (6264) | -15 [-17, -13] |
| starhorse2023 | 6265 | logg | +0.011 / 0.134 (6264) | +0.024 / 0.082 (6264) | +0.031 [+0.028, +0.034] |
| starhorse2023 | 6265 | fe_h | +0.003 / 0.067 (4900) | +0.059 / 0.098 (4900) | -0.031 [-0.034, -0.027] |
| shboost | 46440 | teff | -6 / 79 (46101) | -30 / 115 (46101) | -30 [-31, -29] |
| shboost | 46440 | logg | +0.004 / 0.194 (46088) | -0.002 / 0.175 (46088) | +0.013 [+0.011, +0.014] |
| shboost | 46440 | fe_h | +0.001 / 0.117 (35159) | +0.066 / 0.172 (35159) | -0.041 [-0.043, -0.039] |
| gspphot | 46491 | teff | -3 / 77 (35644) | -63 / 252 (35644) | -135 [-137, -133] |
| gspphot | 46491 | logg | +0.009 / 0.190 (35633) | +0.059 / 0.254 (35633) | -0.045 [-0.048, -0.042] |
| gspphot | 46491 | fe_h | -0.000 / 0.117 (29369) | -0.108 / 0.297 (29369) | -0.153 [-0.156, -0.149] |

- andrae2023: own training stars (in_training_sample) removed from the comparison
- andrae2023: 2905 of 42169 overlap stars are in its training set and are not compared
- starhorse2023: spectroscopic StarHorse runs take the survey Teff, log g and [M/H] as inputs: agreement with survey references is partly by construction
- shboost: trained on spectroscopic (StarHorse) labels of the same surveys; no training flag, so overlap with its training set is not removed
- gspphot: Gaia DR3 GSP-Phot (Aeneas), independent of the survey labels

### s2_flow (`s2_flow`, kind B0)

Rows: FIT 293815, DEV 19464, CAL 31189, TEST 47122, QA_RESERVED 1171, excluded_seen_in_partition 631, evaluated 46491. Integrity: 0 duplicate ids; non-monotone quantiles teff 0, logg 0, fe_h 0.
References: teff from --labels, logg from --labels, fe_h from --labels.

**teff** (K) vs reference: bias / NMAD (n) overall -5 / 71 (46148); pull NMAD 1.28; PIT KS 0.183
- 68 % coverage (central quantile interval): 0.557; 38 of 39 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18, 18-19; bp_rp: <0.5, 0.5-0.8, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, rgb, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: -7 / 71 (21311); 13.5-15: -4 / 63 (9227); 15-16: -1 / 72 (5115); 16-17: -1 / 74 (6095); 17-18: -1 / 76 (4346); 18-19: -2525 / 3157 (37); >19: -49 / 497 (17)*
- by feh_ref: <-3: +3 / 345 (61); -3--2.5: +26 / 121 (132); -2.5--2: +17 / 99 (577); -2--1.5: +6 / 84 (1453); -1.5--1: -2 / 80 (2544); -1--0.5: +6 / 67 (6720); >-0.5: -4 / 55 (23713)
- by stage: dwarf_ms: -6 / 65 (24274); turnoff_sgb: +14 / 105 (5712); rc: -14 / 67 (4708); rgb: +2 / 56 (465); agb_other: -5 / 72 (10989)
- by input_class: X: -5 / 71 (46148)
- by ruwe: <1.4: -5 / 69 (41199); >=1.4: -2 / 84 (4949)
- by abs_b: <5: -28 / 134 (7006); 5-15: -8 / 85 (10926); 15-30: -3 / 70 (11062); 30-60: -1 / 56 (13436); >60: +1 / 40 (3718)

**logg** (dex) vs reference: bias / NMAD (n) overall -0.007 / 0.151 (46134); pull NMAD 1.25; PIT KS 0.162
- 68 % coverage (central quantile interval): 0.564; 38 of 39 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18, 18-19; bp_rp: <0.5, 0.5-0.8, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5, >-0.5; stage: dwarf_ms, turnoff_sgb, rc, rgb, agb_other; abs_b: <5, 5-15, 15-30, 30-60, >60; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: -0.014 / 0.159 (21302); 13.5-15: -0.006 / 0.128 (9226); 15-16: -0.001 / 0.152 (5115); 16-17: +0.006 / 0.159 (6094); 17-18: +0.001 / 0.169 (4344); 18-19: -0.758 / 0.789 (36); >19: -0.042 / 0.125 (17)*
- by feh_ref: <-3: -0.007 / 0.167 (61); -3--2.5: +0.073 / 0.276 (132); -2.5--2: +0.022 / 0.256 (577); -2--1.5: +0.010 / 0.216 (1453); -1.5--1: -0.016 / 0.209 (2544); -1--0.5: +0.002 / 0.155 (6720); >-0.5: -0.001 / 0.110 (23712)
- by stage: dwarf_ms: -0.019 / 0.118 (24266); turnoff_sgb: +0.089 / 0.188 (5709); rc: -0.065 / 0.200 (4707); rgb: -0.114 / 0.190 (465); agb_other: +0.013 / 0.197 (10987)
- by input_class: X: -0.007 / 0.151 (46134)
- by ruwe: <1.4: -0.008 / 0.151 (41187); >=1.4: -0.002 / 0.147 (4947)
- by abs_b: <5: -0.040 / 0.251 (7005); 5-15: -0.017 / 0.181 (10924); 15-30: -0.002 / 0.137 (11061); 30-60: -0.000 / 0.125 (13427); >60: +0.005 / 0.121 (3717)

**fe_h** (dex) vs reference: bias / NMAD (n) overall -0.001 / 0.101 (35202); pull NMAD 1.04; PIT KS 0.093
- 68 % coverage (central quantile interval): 0.654; 19 of 39 strata (n ≥ 20) outside ±0.03 (G: 13.5-15, 18-19; bp_rp: <0.5, 0.8-1.2; feh_ref: <-3, -3--2.5, -2.5--2, -1.5--1; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1; stage: turnoff_sgb, rc; abs_b: <5, 5-15, 15-30; ruwe: >=1.4)
- by G: <=13.5: -0.001 / 0.060 (10674); 13.5-15: -0.003 / 0.088 (8978); 15-16: -0.004 / 0.126 (5091); 16-17: +0.001 / 0.171 (6074); 17-18: +0.004 / 0.225 (4332); 18-19: -0.570 / 1.807 (36); >19: +0.094 / 0.562 (17)*
- by feh_ref: <-3: +0.579 / 0.573 (61); -3--2.5: +0.443 / 0.315 (133); -2.5--2: +0.164 / 0.272 (578); -2--1.5: +0.062 / 0.224 (1453); -1.5--1: +0.018 / 0.225 (2544); -1--0.5: +0.039 / 0.132 (6720); >-0.5: -0.013 / 0.081 (23713)
- by stage: dwarf_ms: -0.004 / 0.107 (21962); turnoff_sgb: +0.002 / 0.100 (4474); rc: -0.006 / 0.079 (2821); rgb: -0.007 / 0.062 (200); agb_other: +0.010 / 0.097 (5745)
- by input_class: X: -0.001 / 0.101 (35202)
- by ruwe: <1.4: -0.001 / 0.102 (31566); >=1.4: -0.005 / 0.096 (3636)
- by abs_b: <5: -0.011 / 0.137 (4092); 5-15: +0.003 / 0.093 (6686); 15-30: -0.004 / 0.092 (8808); 30-60: -0.001 / 0.101 (12195); >60: +0.003 / 0.112 (3421)

**Joint 68 % set of the model** (Neyman for A, conformal for B0): coverage 0.661 (n 35199); 25 of 39 strata (n ≥ 20) outside ±0.03 (G: 17-18, 18-19; bp_rp: <0.5, 1.2-1.8, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1; stage: turnoff_sgb, rc, agb_other; abs_b: <5, 5-15, 30-60, >60; ruwe: >=1.4)
- by G: <=13.5: 0.656 [0.643, 0.668] (10673); 13.5-15: 0.664 [0.650, 0.677] (8977); 15-16: 0.651 [0.633, 0.669] (5090); 16-17: 0.691 [0.674, 0.706] (6074); 17-18: 0.639 [0.619, 0.659] (4332); 18-19: 0.167 [0.060, 0.385] (36); >19: 0.353 [0.132, 0.662] (17)

**Joint 95 % set of the model** (Neyman for A, conformal for B0): coverage 0.922 (n 35199); 24 of 39 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 15-16, 18-19; bp_rp: <0.5, 1.8-2.5, >2.5; feh_ref: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5; feh_pred: <-3, -3--2.5, -2.5--2, -2--1.5, -1.5--1, -1--0.5; stage: turnoff_sgb, rgb, agb_other; abs_b: <5, 5-15; ruwe: >=1.4)
- by G: <=13.5: 0.914 [0.906, 0.921] (10673); 13.5-15: 0.923 [0.915, 0.930] (8977); 15-16: 0.913 [0.901, 0.923] (5090); 16-17: 0.941 [0.932, 0.949] (6074); 17-18: 0.933 [0.922, 0.942] (4332); 18-19: 0.222 [0.092, 0.445] (36); >19: 0.588 [0.292, 0.832] (17)

**Physical references** (bias / NMAD (n); TEST and reserved stars not seen in training; 330 seen rows dropped):
- anchor_apok2: 29 of 4660 predicted; logg -0.017 / 0.067 (29)
- anchor_apokasc3: 118 of 13871 predicted; logg +0.007 / 0.079 (118)
- anchor_apokasc_dwarfs: 2 of 413 predicted; logg +0.097 / 0.241 (2)*
- anchor_kepler_legacy: 0 of 65 predicted; no overlap
- gbs_v3: 4 of 192 predicted; teff +11 / 96 (4)*; logg -0.078 / 0.149 (4)*; fe_h -0.033 / 0.091 (4)*

**Competitors** (exact-id overlap; model vs competitor against the same reference):

| catalogue | overlap | product | model bias / NMAD | competitor bias / NMAD | Δ median abs err (95 %) |
|---|--:|---|---|---|---|
| andrae2023 | 42169 | teff | -5 / 71 (39001) | -76 / 172 (39001) | -78 [-80, -77] |
| andrae2023 | 42169 | logg | -0.008 / 0.153 (38988) | -0.021 / 0.196 (38988) | -0.027 [-0.029, -0.026] |
| andrae2023 | 42169 | fe_h | -0.002 / 0.102 (28878) | +0.025 / 0.162 (28878) | -0.041 [-0.043, -0.039] |
| starhorse2023 | 6265 | teff | +2 / 65 (6264) | -34 / 78 (6264) | -17 [-19, -15] |
| starhorse2023 | 6265 | logg | +0.016 / 0.117 (6264) | +0.024 / 0.082 (6264) | +0.019 [+0.015, +0.022] |
| starhorse2023 | 6265 | fe_h | -0.003 / 0.063 (4900) | +0.059 / 0.098 (4900) | -0.033 [-0.035, -0.029] |
| shboost | 46440 | teff | -5 / 71 (46101) | -30 / 115 (46101) | -36 [-37, -35] |
| shboost | 46440 | logg | -0.007 / 0.151 (46088) | -0.002 / 0.175 (46088) | -0.016 [-0.018, -0.015] |
| shboost | 46440 | fe_h | -0.001 / 0.101 (35159) | +0.066 / 0.172 (35159) | -0.051 [-0.053, -0.049] |
| gspphot | 46491 | teff | -3 / 70 (35644) | -63 / 252 (35644) | -140 [-142, -138] |
| gspphot | 46491 | logg | -0.004 / 0.145 (35633) | +0.059 / 0.254 (35633) | -0.075 [-0.078, -0.073] |
| gspphot | 46491 | fe_h | -0.001 / 0.104 (29369) | -0.108 / 0.297 (29369) | -0.162 [-0.165, -0.158] |

- andrae2023: own training stars (in_training_sample) removed from the comparison
- andrae2023: 2905 of 42169 overlap stars are in its training set and are not compared
- starhorse2023: spectroscopic StarHorse runs take the survey Teff, log g and [M/H] as inputs: agreement with survey references is partly by construction
- shboost: trained on spectroscopic (StarHorse) labels of the same surveys; no training flag, so overlap with its training set is not removed
- gspphot: Gaia DR3 GSP-Phot (Aeneas), independent of the survey labels

### s3_gmix (`s3_gmix`, kind B)

Rows: FIT 295037, DEV 19531, CAL 31329, TEST 47433, QA_RESERVED 1259, excluded_seen_in_partition 942, evaluated 46491. Integrity: 0 duplicate ids; non-monotone quantiles dist_mod 0, ebv 0.
References: teff from --labels, logg from --labels, fe_h from --labels.

**parallax_pit**: 68 % coverage 0.720 (n 46491); 15 of 20 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 16-17, 18-19; bp_rp: <0.5, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; abs_b: 5-15, 15-30, 30-60, >60; input_class: X; ruwe: <1.4)
- by G: <=13.5: 0.726 [0.717, 0.734] (21383); 13.5-15: 0.720 [0.707, 0.732] (9366); 15-16: 0.702 [0.684, 0.719] (5161); 16-17: 0.727 [0.711, 0.742] (6137); 17-18: 0.704 [0.685, 0.722] (4384); 18-19: 0.829 [0.626, 0.934] (41); >19: 0.789 [0.481, 0.938] (19)

**ebv_pit**: 68 % coverage 0.898 (n 33730); 19 of 19 strata (n ≥ 20) outside ±0.03 (G: <=13.5, 13.5-15, 15-16, 16-17, 17-18; bp_rp: <0.5, 0.5-0.8, 0.8-1.2, 1.2-1.8, 1.8-2.5, >2.5; abs_b: <5, 5-15, 15-30, 30-60, >60; input_class: X; ruwe: <1.4, >=1.4)
- by G: <=13.5: 0.917 [0.909, 0.924] (10174); 13.5-15: 0.894 [0.885, 0.902] (8620); 15-16: 0.885 [0.872, 0.897] (4734); 16-17: 0.885 [0.874, 0.896] (5936); 17-18: 0.890 [0.877, 0.903] (4256); 18-19: 0.750 [0.310, 0.952] (8); >19: 1.000 [0.217, 1.000] (2)

**Physical references** (bias / NMAD (n); TEST and reserved stars not seen in training; 418 seen rows dropped):
- anchor_apok2: 29 of 4660 predicted; no overlap
- anchor_apokasc3: 118 of 13871 predicted; no overlap
- anchor_apokasc_dwarfs: 2 of 413 predicted; no overlap
- anchor_kepler_legacy: 0 of 65 predicted; no overlap
- gbs_v3: 4 of 192 predicted; no overlap

**Competitors** (exact-id overlap; model vs competitor against the same reference):

| catalogue | overlap | product | model bias / NMAD | competitor bias / NMAD | Δ median abs err (95 %) |
|---|--:|---|---|---|---|
| starhorse2023 | 6265 | ebv | model − competitor +0.013 / 0.043 (6265) | | |
| starhorse2023 | 6265 | dist_mod | model − competitor -0.026 / 0.268 (6265) | | |
| shboost | 46440 | ebv | model − competitor +0.030 / 0.054 (46440) | | |
| shboost | 46440 | dist_mod | model − competitor +0.009 / 0.254 (46440) | | |
| gspphot | 46491 | dist_mod | model − competitor +0.119 / 0.451 (35865) | | |

- andrae2023: own training stars (in_training_sample) removed from the comparison
- andrae2023: 2905 of 42169 overlap stars are in its training set and are not compared
- starhorse2023: spectroscopic StarHorse runs take the survey Teff, log g and [M/H] as inputs: agreement with survey references is partly by construction
- shboost: trained on spectroscopic (StarHorse) labels of the same surveys; no training flag, so overlap with its training set is not removed
- gspphot: Gaia DR3 GSP-Phot (Aeneas), independent of the survey labels

## Frozen metal-poor core

### Core: set `posterior_core`

Rows with a core reference: 296; rows: FIT 27319, DEV 1869, CAL 2982, TEST 4399, QA_RESERVED 2504, excluded_seen_in_partition 4103, evaluated 296.

Median offset / NMAD (n) of [Fe/H] − core reference, per [Fe/H]_ref × G (* = n < 20, descriptive):

| [Fe/H]_ref | G ≤ 13.5 | 13.5–15 | 15–16 | 16–17.65 | G ≤ 16 |
|---|---|---|---|---|---|
| [-9.0,-3.0) | +0.264 / 0.000 (1)* | +0.414 / 0.097 (2)* | +0.442 / 0.160 (2)* | – | +0.349 / 0.125 (5)* |
| [-3.0,-2.5) | -0.049 / 0.218 (3)* | +0.217 / 0.000 (1)* | +0.310 / 0.000 (1)* | +0.515 / 0.137 (2)* | +0.098 / 0.218 (5)* |
| [-2.5,-2.0) | +0.039 / 0.194 (5)* | -0.587 / 0.000 (1)* | – | +0.096 / 0.141 (2)* | -0.026 / 0.230 (6)* |
| [-2.0,-1.5) | -0.042 / 0.046 (2)* | – | -0.271 / 0.000 (1)* | +0.061 / 0.139 (5)* | -0.074 / 0.093 (3)* |
| [-1.5,-1.0) | -0.006 / 0.368 (5)* | -0.059 / 0.215 (2)* | -0.217 / 0.216 (4)* | +0.058 / 0.713 (7)* | -0.115 / 0.299 (11)* |
| [-1.0,-0.5) | -0.010 / 0.138 (10)* | +0.107 / 0.177 (6)* | +0.081 / 0.087 (5)* | -0.131 / 0.346 (25) | +0.041 / 0.182 (21) |
| [-0.5,9.0) | -0.003 / 0.125 (98) | -0.046 / 0.083 (21) | -0.060 / 0.129 (35) | -0.128 / 0.255 (50) | -0.020 / 0.125 (154) |

Worst |bias| at G ≤ 16 (n ≥ 20): +0.041 in [-1.0,-0.5) (acceptance ≤ 0.10, qa.md §5.3).

Metal-poor selection P([Fe/H] < −2.5) > 0.9: G<=16: purity 0.60, completeness 0.30 (3/10); G>16: purity –, completeness 0.00 (0/2); all: purity 0.60, completeness 0.25 (3/12)
Tail probability: n 296 (12 true), Brier 0.0139, AUC 0.995.
68 % coverage at [Fe/H]_ref < −2: 0.250 (n 20).

### Core: set `s1_all_ens`

Rows with a core reference: 296; rows: FIT 900526, DEV 60032, CAL 95639, TEST 144933, QA_RESERVED 4651, excluded_seen_in_partition 98442, evaluated 46491.

Median offset / NMAD (n) of [Fe/H] − core reference, per [Fe/H]_ref × G (* = n < 20, descriptive):

| [Fe/H]_ref | G ≤ 13.5 | 13.5–15 | 15–16 | 16–17.65 | G ≤ 16 |
|---|---|---|---|---|---|
| [-9.0,-3.0) | -0.070 / 0.000 (1)* | +0.019 / 0.050 (2)* | +0.044 / 0.286 (2)* | – | -0.015 / 0.101 (5)* |
| [-3.0,-2.5) | -0.030 / 0.508 (3)* | +0.237 / 0.000 (1)* | +0.406 / 0.000 (1)* | +0.271 / 0.480 (2)* | +0.237 / 0.396 (5)* |
| [-2.5,-2.0) | +0.079 / 0.361 (5)* | -0.623 / 0.000 (1)* | – | -0.355 / 0.281 (2)* | -0.043 / 0.334 (6)* |
| [-2.0,-1.5) | -0.112 / 0.111 (2)* | – | -0.244 / 0.000 (1)* | +0.138 / 0.265 (5)* | -0.187 / 0.085 (3)* |
| [-1.5,-1.0) | -0.001 / 0.063 (5)* | -0.070 / 0.111 (2)* | -0.083 / 0.283 (4)* | +0.298 / 0.425 (7)* | -0.061 / 0.149 (11)* |
| [-1.0,-0.5) | -0.017 / 0.207 (10)* | +0.107 / 0.169 (6)* | +0.089 / 0.075 (5)* | -0.135 / 0.293 (25) | +0.031 / 0.139 (21) |
| [-0.5,9.0) | +0.005 / 0.168 (98) | -0.066 / 0.128 (21) | -0.020 / 0.120 (35) | -0.186 / 0.497 (50) | -0.002 / 0.157 (154) |

Worst |bias| at G ≤ 16 (n ≥ 20): +0.031 in [-1.0,-0.5) (acceptance ≤ 0.10, qa.md §5.3).

Metal-poor selection P([Fe/H] < −2.5) > 0.9: G<=16: purity 0.50, completeness 0.20 (2/10); G>16: purity –, completeness 0.00 (0/2); all: purity 0.50, completeness 0.17 (2/12)
Tail probability: n 296 (12 true), Brier 0.0297, AUC 0.976.
68 % coverage at [Fe/H]_ref < −2: 0.700 (n 20).

### Core: set `s2_flow`

Rows with a core reference: 296; rows: FIT 293815, DEV 19464, CAL 31189, TEST 47122, QA_RESERVED 1171, excluded_seen_in_partition 631, evaluated 46491.

Median offset / NMAD (n) of [Fe/H] − core reference, per [Fe/H]_ref × G (* = n < 20, descriptive):

| [Fe/H]_ref | G ≤ 13.5 | 13.5–15 | 15–16 | 16–17.65 | G ≤ 16 |
|---|---|---|---|---|---|
| [-9.0,-3.0) | +0.389 / 0.000 (1)* | +0.672 / 0.200 (2)* | +0.509 / 0.104 (2)* | – | +0.537 / 0.145 (5)* |
| [-3.0,-2.5) | +0.669 / 0.213 (3)* | +0.308 / 0.000 (1)* | +0.429 / 0.000 (1)* | +0.722 / 0.184 (2)* | +0.429 / 0.211 (5)* |
| [-2.5,-2.0) | +0.063 / 0.188 (5)* | -0.512 / 0.000 (1)* | – | +0.212 / 0.113 (2)* | +0.036 / 0.212 (6)* |
| [-2.0,-1.5) | -0.031 / 0.050 (2)* | – | -0.183 / 0.000 (1)* | +0.134 / 0.107 (5)* | -0.064 / 0.099 (3)* |
| [-1.5,-1.0) | -0.182 / 0.383 (5)* | -0.043 / 0.267 (2)* | -0.222 / 0.182 (4)* | +0.137 / 0.675 (7)* | -0.182 / 0.243 (11)* |
| [-1.0,-0.5) | -0.023 / 0.203 (10)* | +0.096 / 0.129 (6)* | +0.083 / 0.038 (5)* | -0.042 / 0.299 (25) | +0.057 / 0.132 (21) |
| [-0.5,9.0) | -0.043 / 0.144 (98) | -0.056 / 0.109 (21) | -0.054 / 0.094 (35) | -0.135 / 0.233 (50) | -0.046 / 0.126 (154) |

Worst |bias| at G ≤ 16 (n ≥ 20): +0.057 in [-1.0,-0.5) (acceptance ≤ 0.10, qa.md §5.3).

Metal-poor selection P([Fe/H] < −2.5) > 0.9: G<=16: purity 1.00, completeness 0.10 (1/10); G>16: purity –, completeness 0.00 (0/2); all: purity 1.00, completeness 0.08 (1/12)
Tail probability: n 296 (12 true), Brier 0.0205, AUC 0.986.
68 % coverage at [Fe/H]_ref < −2: 0.350 (n 20).

## Co-natal checks

**Wide binaries** (El-Badry+2021 MSMS, R_chance < 0.1, pair group in the evaluated partition): 33321 pairs scored (26569 with neither star labelled). Δ = [Fe/H](brighter) − [Fe/H](fainter): median / NMAD (n); |z| < 1 fraction (ideal 0.68).

| subset | all pairs | |z|<1 | neither labelled | |z|<1 |
|---|---|--:|---|--:|
| all | +0.086 / 0.189 (33321) | 0.57 | +0.085 / 0.187 (26569) | 0.59 |
| pair mean [-0.5,9) | +0.089 / 0.187 (31183) | 0.56 | +0.088 / 0.185 (24879) | 0.58 |
| pair mean [-1,-0.5) | +0.029 / 0.242 (2056) | 0.67 | +0.033 / 0.241 (1634) | 0.68 |
| pair mean [-1.5,-1) | -0.022 / 0.365 (63) | 0.70 | -0.003 / 0.339 (45) | 0.73 |
| pair mean [-2,-1.5) | -0.037 / 0.465 (17) | 0.47 | -0.015 / 0.933 (11) | 0.36 |
| pair mean [-2.5,-2) | -0.245 / 0.000 (1) | 1.00 | – / – (0) | – |
| pair mean [-9,-2.5) | +5.619 / 0.000 (1) | 0.00 | – / – (0) | – |
| fainter Teff [3000,3700) | +0.137 / 0.204 (11546) | 0.54 | +0.120 / 0.194 (9179) | 0.57 |
| fainter Teff [3700,4200) | +0.199 / 0.214 (7158) | 0.48 | +0.202 / 0.215 (5868) | 0.49 |
| fainter Teff [4200,4800) | +0.090 / 0.156 (5514) | 0.58 | +0.092 / 0.159 (4575) | 0.59 |
| fainter Teff [4800,5500) | -0.007 / 0.120 (4631) | 0.67 | -0.004 / 0.120 (3681) | 0.69 |
| fainter Teff [5500,7000) | -0.012 / 0.114 (4240) | 0.66 | -0.012 / 0.118 (3090) | 0.67 |
| ΔG [0,1) | +0.019 / 0.110 (10976) | 0.78 | +0.021 / 0.113 (9375) | 0.79 |
| ΔG [1,2) | +0.069 / 0.171 (7589) | 0.60 | +0.076 / 0.172 (6372) | 0.61 |
| ΔG [2,3) | +0.144 / 0.221 (5724) | 0.47 | +0.157 / 0.221 (4579) | 0.46 |
| ΔG [3,9) | +0.235 / 0.226 (8972) | 0.35 | +0.239 / 0.227 (6206) | 0.35 |

**Held-out binary pool** (2141 of 2141 pairs scored): median [Fe/H](primary) − [Fe/H](companion) by companion Teff: 3000-3700 K +0.312 (647); 3700-4200 K +0.256 (681); 4200-4800 K +0.095 (635); 4800-5500 K +0.016 (173); 5500-7000 K – (0); companion G > 16.5: +0.293 (912).

**Open clusters** (HR23, Prob ≥ 0.7, ≥ 20 members, cluster in the evaluated partition): 637 clusters (40392 members). Median within-cluster NMAD 0.296, median χ²/dof 2.63, giant − dwarf +0.328 (224 clusters), faint − bright -0.145, slope -0.023 dex/mag.

**Globular clusters** (QA-reserved, report-only; ≥ 10 members): 113 clusters, 102 with Harris [Fe/H]. Median(pred − Harris): [-1,0): +0.028 (23 GCs, within NMAD 0.340); [-1.5,-1): +0.006 (28 GCs, within NMAD 0.321); [-2,-1.5): -0.088 (41 GCs, within NMAD 0.243); [-3,-2): -0.019 (10 GCs, within NMAD 0.260)

## Not run / missing inputs

- competitor:zhang_green_rix2023: not on ARC (schema verified only)
- competitor:survey_of_surveys_dr3: not on ARC
- competitor:bailer_jones2021: no local copy (remote LSDB HATS only)
- competitor:bayestar19_edenhofer2024: no 3-D dust map on ARC
- competitor:ye2025_ferre_xp: not on ARC
- competitor:gaia_net: Zenodo copy no longer on ARC
