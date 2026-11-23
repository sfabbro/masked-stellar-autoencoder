# C0b: age consensus with the priority labels as fitted sources (2026-11-23)

Status: phase 1 of the priority labels from the age report, run on the full data. The new sources are LDB, EB and GC cluster benchmarks, WD+MS pairs, and El-Badry+21 wide pairs as equality constraints. No label is truth. Every source gets a fitted map and fitted noise, including the benchmarks; cluster ages are fitted latents.
- Code: `src/masked_stellar_autoencoder/training/age_consensus.py`, `scripts/age_consensus_c0.py` (`run`, `plots`), `tests/test_age_consensus_c0.py`.
- Units: t = log10(age/Gyr). Map offset b = map value − t at 4 Gyr (t = 0.6); A = slope − 1; β = [Fe/H] coefficient; all in dex.
- Gauges: **A** = APOKASC-3 identity map (fitted noise); **C** = Xiang 2024 identity map. ARB_A / ARB_C are refits with every benchmark held out (validation only).

## 1. Summary
- **Which scale.** The APOKASC-3 gauge (A) agrees with the independent benchmarks; the Xiang gauge (C) does not.
  - Held-out eclipsing binaries in 15 clusters: EB − latent = **+0.060 ± 0.018** dex in A (χ²/dof 2.7) and **−0.129 ± 0.024** in C (χ²/dof 7.9).
  - LDB clusters (−0.37 vs −0.59) and WD+MS pairs (−0.23 vs −0.45) also sit closer to A.
  - In the main fits, where the benchmarks are fitted sources, EB maps are +0.041 ± 0.028 (A) and −0.086 ± 0.037 (C).
- **The isochrone pipelines read about 0.25 dex older than the seismic giant scale.** In gauge A, Xiang (+0.260), StarHorse (+0.20 to +0.25), GALAH DR4 subgiants (+0.24) and Chaplin seismic dwarfs (+0.26) all sit there, against APOKASC-3.
  - The zero point is stable at +0.20 to +0.26 over 6 model variants and 12 leave-one-catalogue-out refits. The one exception is the unconverged LOO_starhorse23_lamostl7 fit (+0.42).
  - The EBs say the truth is ≈ 0.06 dex above A. On the EB scale the isochrone pipelines therefore read ≈ 0.2 dex (≈ 55 %) too old at the turn-off, and APOKASC-3 ≈ 0.06 dex (15 %) too young.
- **The two gauges are not the same model.** C is not a relabelled A: turn-off and subgiant latents differ by 0.26 dex and giant latents by 0.055, so the giant–subgiant bridge is 0.20 dex apart (§4).
  - The likelihood is not translation invariant: the latent domain (13.8 Gyr bound, lower limits) and the cluster terms see absolute t.
  - So the gauge is a model choice that the benchmarks must arbitrate. Here they choose A.
- **Clusters (gauge A).**
  - Old, giant-dominated clusters are close to the EB ages: NGC 6791 8.07 Gyr (EB 8.45), NGC 6819 2.13 (EB 2.38).
  - Turn-off-dominated clusters sit 0.05–0.10 dex below their EBs: M67 2.86 (EB 3.50), NGC 2243 3.66 (EB 4.35), NGC 6253 3.34 (EB 3.80).
  - NGC 188 is the outlier: 3.75 Gyr against EB 6.2 (5 members).
  - Young clusters are fixed when the LDB records are fitted: Pleiades 142 Myr (LDB 127), IC 2391 64 (58), IC 2602 58 (52), α Per 88 (79). Without those records they land 0.07–0.39 dex older.
  - Hyades and Praesepe still read 1.1 Gyr.
- **Calibration.**
  - Cluster members against their cluster latent: z-NMAD 0.97 (A) / 0.92 (C), so the per-star σ is right where an independent truth exists.
  - Wide-pair members: z-NMAD 0.58, so pair σ is about 1.7× too large.
  - Held-out isochrone catalogues give z-NMAD 0.25–0.55, which is expected. Their σ includes the fitted shared isochrone error η = 0.123 dex, which a comparison inside the family cannot see.
- **Next.**
  1. Model the old-age ceiling of the isochrone pipelines: after the map, residuals fall to −0.2 to −0.3 dex above 8 Gyr. This matters for halo and thick-disc ages.
  2. Find why Hyades/Praesepe and NGC 188 resist.
  3. Phase 2: rotation (monotone gyro maps per colour) and Li with censored upper limits.

## 2. What changed from C0

### 2.1 Fixes (A1–A4)
- **A1 Gauge-covariant co-natal term.** Memberships are fixed external priors (HR23 p; VB21 p ≥ 0.5 for its nine GCs, replacing HR23 there; EB21 1 − R_chance for wide pairs). Each member enters with a Student-t weight (ν = 4) against its group latent. There is no field density, so the likelihood depends on latent differences only. In the synthetic test a fit on anchor X, re-expressed on catalogue Y's scale and refitted, moves by < 2e-3 dex.
  - Two script-level asymmetries also broke covariance on real data, and both are removed:
    - **Component references join along a maximum spanning tree** (Prim, with pairwise link counts). The edge set does not depend on the root, so the seismic–isochrone bridge sets Xiang's reference tier in gauge A and APOKASC-3's in gauge C with the same width (33 independent links in both).
    - **Domain censoring applies to the anchor too**, so the censoring set does not depend on the gauge.
- **A2 Shared-spectra sub-families.** These are nested inside the isochrone shared error:
  - StarHorse-LAMOST (LRS, MRS) with Xiang;
  - StarHorse-GALAH with GALAH DR4.

  Each has its own η_k (y = f(t + u + u_k)); the model handles this with two-level Sherman–Morrison.
- **A3 Identifiability gate.** A map gets slope, [Fe/H] or class terms only when ≥ 30 of its constrained records (the same count the offset tier needs) are in noise units whose variance is identified.
  - A variance is identified by triads (three independent measurements of the star: other shared-input groups or a co-natal partner, with same-spectra pipelines counted as one) or by pairs with an already identified unit, iterated.
  - Otherwise the map is offset-only: with pairwise overlaps only, the errors-in-variables slope follows an unidentified variance ratio.
- **A4 Cluster at log age −2.45 in C0.** This was NGC 1980 (ONC region, a few Myr old). Its latent rested on one GALAH DR4 "subgiant" record of a pre-main-sequence member, mapped far below the range on which the GALAH subgiant map was fitted (the 1st percentile of its data is 0.9 Gyr).
  - Fix: a map is defined on its data range. Mapped records below the 0.1st percentile of the map's constrained records become upper limits at that edge (domain censoring) instead of extrapolated values.
  - NGC 1980 now also has HR23 / CG20 / Dias21 node records.

### 2.2 Young clusters: why C0 (and the first C0b fit) failed
- **The data.** In 477 HR23 clusters younger than 500 Myr, the 982 representative isochrone records of members (StarHorse "MSTO", GALAH and Xiang subgiants) read a median **+0.99 dex older** than the HR23 age.
  - About half are hot (Teff > 7,000 K) upper-main-sequence stars with log g 3.7–4.1, so they pass the log g ≥ 4.1 main-sequence rule.
  - The rest are cooler (non-members, binaries, pre-main-sequence stars).
  - In 174 older clusters the same catalogues read +0.29 dex older than HR23.
- **What the likelihood does.** It counts records. With two or more coherent member records per young cluster against one node, the likelihood prefers to inflate the cluster catalogues' noise and shift their offset, rather than call the members wrong.
- **The first full C0b fit, run without the fixes below, shows exactly this.**
  - η_cluster_cmd = 0.71 dex.
  - The HR23 offset is −1.47 dex; LDB is −0.84.
  - The Pleiades latent is 6.8 Gyr.
  - The isochrone world drifts 0.4 dex older than the seismic anchor: Xiang b = −0.40, against −0.026 in C0. M67 lands at 11.6 Gyr, and its seismic members get member weights of 0.08.
- **Fixes, all inside the model.** No label becomes truth; each is tested on synthetic data.
  1. **A map is flat below its data edge.** Below the latent where a map reaches the lower edge of its data (the same 0.1st-percentile edge as the domain censoring, fixed in measurement units, so gauge-covariant), the map's slope is 0.01, a numerical stand-in for zero. A co-natal member whose latent (set by its group) lies there has that record treated as missing (EM weight 0) in the fit and in the per-star solve. A 1 Gyr reading of a 100 Myr star then carries no information, instead of a pull.
  2. **The cluster node is a fixed member.** Its weight is its prior (1), never a Student-t member weight: the node is the cluster's own record set, not a star that might be a non-member. Its outliers are handled by the record noise models (per-source κ, ω, Student-t record weights, η_cluster_cmd).
  3. **A group with its own records starts from them.** The initial latents of a group's members are the raw mean of the node's records. The fit stays an EM: the members' records become missing only if the fitted latent lies below their map's edge.
- **Synthetic test.** Young clusters with two coherently wrong members each, plus one cluster-level record. Without the fixes the young latents are off by 0.88 dex and the record's offset is −0.22 (truth +0.10). With them the error is < 0.1 dex and the offset is recovered within 0.05.

## 3. Which scale: arbitration by independent benchmarks

ARB_A / ARB_C hold out every benchmark (LDB, EB, GC, WD+MS) and refit. The tables compare each benchmark with the fitted latent of its own stars or clusters: offset = benchmark − latent, with the benchmark's κ and ω fitted against the held-out latent.

| benchmark | n | A: offset (κ, χ²/dof quoted) | C: offset (κ, χ²/dof quoted) | main A fitted map | main C fitted map |
|---|---:|---|---|---|---|
| EB clusters | 15 | **+0.060 ± 0.018** (1.62, 2.7) | −0.129 ± 0.024 (3.81, 7.9) | +0.041 ± 0.028 | −0.086 ± 0.037 |
| LDB clusters | 9 | −0.369 ± 0.089 (0.78, 0.10) | −0.591 ± 0.077 (0.72, 0.04) | −0.044 ± 0.086 | −0.018 ± 0.164 |
| WD+MS (Raddi+22, H, C18) | 4 | −0.228 ± 0.098 (0.54, 2.2) | −0.448 ± 0.095 (0.55, 2.5) | −0.178 ± 0.107 | −0.420 ± 0.110 |
| GC (VB21) | 9 (0 constrained held out) | — | — | +0.086 ± 0.127 | +0.014 ± 0.047 |
| WD+MS (Qiu+21) | 2 | — | — | — | — |

- **EB clusters, A vs C per cluster (Gyr).** The latent is the fitted cluster latent; the EB column is the benchmark record.

  | cluster | members | A latent | C latent | EB | HR23 |
  |---|---:|---:|---:|---:|---:|
  | NGC 6791 | 56 | 8.07 | 9.11 | 8.45 | 3.80 |
  | NGC 6819 | 34 | 2.13 | 2.43 | 2.38 | 1.68 |
  | M67 (NGC 2682) | 114 | 2.86 | 4.94 | 3.50 | 1.69 |
  | Ruprecht 147 | 31 | 2.16 | 3.91 | 2.67 | 0.89 |
  | NGC 2243 | 29 | 3.66 | 6.42 | 4.35 | 2.75 |
  | NGC 752 | 20 | 1.53 | 2.72 | 1.61 | 1.42 |
  | NGC 6253 | 12 | 3.34 | 5.84 | 3.80 | 1.85 |
  | NGC 7789 | 26 | 1.15 | 2.07 | 1.26 | 1.57 |
  | NGC 188 | 5 | 3.75 | 5.81 | 6.19 | 2.63 |
  | NGC 6811 | 24 | 0.70 | 0.77 | 1.05 | 1.08 |

  - Giant-dominated clusters (NGC 6791, NGC 6819) are close to the EBs in both gauges, because seismic giants fix them.
  - Turn-off-dominated clusters differ by ≈ 0.25 dex between gauges, and the EBs side with A in every case except NGC 188.
  - HR23 ages are systematically young for old clusters (shown for comparison; HR23 is a fitted source with offset −0.185 in A).
- **Caveat on ARB_C.** With the benchmarks removed, the C refit slides into the degenerate basin seen in V_no_latent_domain: cluster-catalogue maps go to −0.79 / −0.61 / −0.60 dex (HR23 / CG20 / Dias21), against −0.30 / −0.08 / −0.06 in main C.
  - Its EB offset (−0.129) is therefore partly a symptom of that basin.
  - The main-C fit, with benchmarks in, still misfits the turn-off EB clusters (M67 4.9 vs 3.5 Gyr; NGC 752 2.7 vs 1.6; Ruprecht 147 3.9 vs 2.7). So the verdict does not rest on ARB_C alone.
- **What the benchmarks cannot yet decide.**
  - The 9 GC records have no constrained held-out stars (0 in ARB).
  - LDB latents without benchmarks are prior-dominated (consensus σ ≈ 0.75 dex in ARB).
  - WD+MS has 4 usable pairs.

## 4. Gauge non-invariance and the bridge

- `gauge_check`: the A solution is exactly representable in C (no coefficient outside C's tiers). Yet C refitted from transformed A moves maps by up to 0.20 dex (p99 latent move 0.20). C from scratch lands within 0.0007 dex (median) of C-from-A.
  - So C has its own optimum, and it is not the relabelled A.
  - η_cluster_cmd differs too: 0.35 (A), 0.26 (C from A), 0.58 (C from scratch). The cluster-CMD shared error is weakly identified.
- `c0b_scale_difference.png`: t(C) − t(A) = +0.26 for SGB/MSTO stars at every age and [Fe/H], and +0.055 for giants.
  - The gauges disagree only on the giant ↔ subgiant bridge, which the data link through 33 seismic–isochrone stars, the clusters, and 2 giant+subgiant wide pairs.
- **The flip from C0.** In C0, gauge A put Xiang at −0.026 and gauge C implied +0.25. In C0b, A gives +0.26 and C gives +0.05.
  - The bridge width did not change (33 links). What changed is the co-natal term: gauge-covariant, with no field density, and the young-cluster fixes.
  - These fixes stopped young-cluster members from dragging the isochrone world (§2.2). The clusters now carry the bridge in A.
  - The size of the gauge disagreement (≈ 0.2–0.28 dex) is the same in both versions. It measures how weakly the bridge is tied, not a result of either fit.

| variant / fold | bridge Xiang − APOKASC-3 at 4 Gyr, [Fe/H] = 0 (dex) | shift vs main |
|---|---:|---:|
| A_main | +0.260 ± 0.001 | 0 |
| ARB_A (benchmarks held out) | +0.234 | −0.026 |
| V_no_identifiability | +0.235 | −0.025 |
| V_wd_fit | +0.232 | −0.027 |
| V_conatal_only | +0.224 | −0.036 |
| V_seismic_dwarfs (Chaplin identity) | +0.256 | −0.004 |
| V_no_latent_domain | +0.258 | −0.001 |
| LOO folds (11 converged) | +0.201 to +0.254 | −0.058 to −0.006 |
| LOO_starhorse23_lamostl7 (not converged, 30 outer iterations, max\|Δθ\| 1.1e-2) | +0.424 | +0.164 |

## 5. Young clusters and information shares (gauge A)

The latent T is the fitted cluster latent with all sources; T_noB is the same cluster without its benchmark records, from the per-star conditional solve. Shares are the fraction of the latent's information from members, cluster catalogues and benchmarks.

| cluster | members | latent (Myr) | latent without benchmarks (Myr) | LDB | shares m / cat / bench |
|---|---:|---:|---:|---:|---|
| Pleiades (Melotte 22) | 8 | 142 | 176 | 127 | 0.65 / 0.24 / 0.11 |
| α Per (Melotte 20) | 0 | 88 | 117 | 79 | 0 / 0.53 / 0.47 |
| Blanco 1 | 0 | 151 | 161 | 137 | 0 / 0.78 / 0.23 |
| IC 2391 | 0 | 64 | 73 | 58 | 0 / 0.62 / 0.38 |
| IC 2602 | 1 | 58 | 73 | 52 | 0.11 / 0.60 / 0.30 |
| NGC 2547 | 0 | 49 | 63 | 43 | 0 / 0.71 / 0.30 |
| IC 4665 | 3 | 40 | 78 | 32 | 0.45 / 0.46 / 0.09 |
| NGC 6705 (EB) | 6 | 207 | 361 | 222 (EB) | 0.52 / 0.26 / 0.23 |
| NGC 6871 (EB) | 0 | 9 | 19 | 10 (EB) | 0 / 0.53 / 0.48 |
| Hyades (Melotte 25) | 4 | 1103 | 1804 | 695 (LDB), 899 (EB) | 0.41 / 0.27 / 0.31 |
| Praesepe (NGC 2632) | 2 | 1131 | 1261 | — (lit. 700) | 0.36 / 0.65 / 0 |

- The young-cluster pathology of C0 is gone: Pleiades 2.75 Gyr in C0, 142 Myr now. Without the benchmarks, young latents sit 0.07–0.39 dex above LDB, because the cluster catalogues carry most of their information.
- **Hyades.** Its 4 members disagree violently with each other (member z-NMAD 22). Two or more records read far older and pull the latent to 1.1 Gyr even with LDB and EB records in the fit.
- **Praesepe.** Two members with no benchmark records.
- These two clusters are the next diagnostic target: list the member records and their classes.

## 6. Co-natal consistency

| link | n | z-NMAD A | z-NMAD C | Δt NMAD (dex) A |
|---|---:|---:|---:|---:|
| cluster members (responsibility > 0.5) | 1749 / 1739 | 0.97 | 0.92 | — |
| wide pairs, all | 351 | 0.58 | 0.61 | — |
| wide pairs, subgiant + subgiant | 349 | 0.58 | 0.61 | 0.114 |
| wide pairs, giant + subgiant | 2 | 2.96 | 0.90 | 0.44 |

- Cluster members are calibrated in both gauges. Their z tail beyond +2 (members older than the cluster) is the expected binary and non-member population.
- Wide-pair members agree better than their σ says (0.58). The shared isochrone η = 0.123 dex enters both members' σ independently, but the pairs suggest a smaller per-star scatter for subgiant pairs (Δt NMAD 0.114 dex ≈ √2 × 0.081).
- Before vs after the maps, within-cluster NMAD of member log ages is 0.090 vs 0.093 (median over clusters with ≥ 5 members). The maps fix zero points and slopes, not the per-star scatter.

## 7. Noise and shared errors (gauge A)

- **APOKASC-3 quoted errors are about 2× too large** (κ = 0.40 RGB, 0.47 RC; ω ≤ 0.008). Xiang's are right once the shared isochrone error is modelled (κ = 0.95, ω = 0.005). In C0, without the shared term, Xiang looked 1.8× too small.
- **Shared errors η.**
  - Isochrone family: **0.123**. It floors the consensus σ of isochrone-only stars.
  - Cluster CMD ages: 0.349 (0.26–0.58 across gauges, weakly identified).
  - LAMOST spectra: 0.068.
  - GALAH spectra: 0.005, at the floor. Together with GALAH DR4's own ω at the floor, its subgiant records carry no scatter beyond the isochrone family term.
- **GC benchmarks:** ω = 0.24 dex (12× the quoted σ), with only 4 constrained records. The GC benchmark ages are not consistent with the consensus of their member stars at the quoted precision.

## 8. Held-out validation

- **Leave one catalogue out.** Each held-out catalogue is mapped on half its stars and tested on the other half.
  - Mapped bias ≤ 0.008 dex everywhere except Chaplin (+0.026) and APOKASC-2 (+0.034, n = 45).
  - Mapped NMAD: 0.045 (StarHorse-GALAH) to 0.18 (Willett Kepler L).
  - Raw offsets before the map: +0.20 to +0.41 dex for every isochrone catalogue; +0.28 for Chaplin; +0.15 for LEGACY.
- **Validation-only records.**
  - Seismic giants: z-NMAD 0.82–1.11 (APO-K2-Kepler 0.82, APOKASC-2 0.92, Leung 2023 1.08, Martig 2016 1.11, GALAH DR4 giants 0.97).
  - Isochrone catalogues: 0.18–0.63, because of the shared η (§1).
- `c0b_residuals.png`: after the maps, residuals are flat in [Fe/H] for every catalogue. Against age they are flat from 0.5 to 8 Gyr. Above t ≈ 0.9 (8 Gyr) they fall to −0.2 to −0.3 dex for Xiang, GALAH DR4, StarHorse-RVS, -APOGEE and -LAMOST-MRS: either these pipelines saturate at their old edge or the consensus runs old there; an offset/linear map follows neither. This is the main remaining systematic for old (halo, thick-disc) stars.

## 9. Open issues (ordered)

1. **Old-age ceiling** (§8). Give the isochrone maps a saturating upper branch (a monotone flex term at the old end), or treat records within the pipeline's top-percentile band as lower limits, which is what they are. Test on GC benchmarks and old seismic giants.
2. **Hyades / Praesepe / NGC 188.** Inspect member records. Check whether the Student-t member weight (ν = 4) is too permissive for 2–5-member clusters.
3. **ARB_C basin.** Without benchmarks, gauge C loses its cluster-catalogue zero point (§3). A C-gauge arbitration started from transformed ARB_A would show whether a C optimum that fits clusters exists at all.
4. **Two LOO folds did not converge** (Xiang, StarHorse-LAMOST-LRS; 30 outer iterations).
5. **Wide-pair σ ≈ 1.7× too large;** the GALAH spectra η sits at its floor.
6. **WD+MS can arbitrate only after phase 2** (4 usable pairs; rotation/Li will add the young and mid-age dwarfs).

## 10. Files

- **Plots:** `c0b_arbitration.png`, `c0b_scale_difference.png`, `c0b_clusters.png`, `c0b_maps.png`, `c0b_residuals.png`, `c0b_loo.png`.
- **Tables:**
  - `maps_{A,C,C_fromA,ARB_A,ARB_C,V_*}.csv`, `tiers_*.csv`, `noise_{A,C,V_wd_fit}.csv`;
  - `arbitration_offsets.csv`, `benchmark_maps.csv`, `bridge_summary.csv`, `bridge_variants.csv`;
  - `clusters_{A,C}.csv`, `loo_predictive.csv`, `validation_records.csv`;
  - `measurement_model_{A,C}.json`, `summary.json`.
- **Per-star consensus:** `consensus_stars.parquet`, 1,163,699 stars, in the work directory (not committed).

## Appendix: maps and noise (gauge A)

### maps A
| map | tier | n constr. | ident. | b | A | beta | class terms | flex |
|---|---|---:|---|---|---|---|---|---|
| apokasc3 | identity | 7850 | y |  |  |  |  |  |
| apok2 | offset | 3420 | n | +0.014 ± 0.002 |  |  |  |  |
| apokasc2 | offset | 87 | y | -0.052 ± 0.007 |  |  |  |  |
| bench_eb | offset | 15 | n | +0.041 ± 0.028 |  |  |  |  |
| bench_gc | offset | 4 | n | +0.086 ± 0.127 |  |  |  |  |
| bench_ldb | offset | 9 | n | -0.044 ± 0.086 |  |  |  |  |
| chaplin14 | offset | 301 | y | +0.264 ± 0.009 |  |  |  |  |
| cl_cg20 | linear | 1790 | y | +0.028 ± 0.036 | +0.102 ± 0.011 | +0.086 ± 0.049 |  |  |
| cl_dias21 | linear | 1519 | y | +0.054 ± 0.037 | +0.130 ± 0.011 | +0.036 ± 0.050 |  |  |
| cl_hr23 | offset | 2151 | y | -0.185 ± 0.032 |  |  |  |  |
| galah_dr4_sgb | flex | 65074 | y | +0.242 ± 0.002 | +0.142 ± 0.005 | -0.256 ± 0.002 |  | g_uu +0.06 ± 0.01, g_xu +0.11 ± 0.01, g_xx -0.06 ± 0.00 |
| legacy17_BASTA | offset | 55 | n | +0.156 ± 0.018 |  |  |  |  |
| starhorse23_apogee17 | linear | 9808 | y | +0.250 ± 0.002 | +0.095 ± 0.006 | -0.185 ± 0.004 | MSTO: +0.003 ± 0.002, +0.021 ± 0.009 |  |
| starhorse23_galahdr3 | flex | 48443 | y | +0.224 ± 0.002 | +0.192 ± 0.005 | -0.318 ± 0.002 | SGB: +0.006 ± 0.000, +0.002 ± 0.001 | g_uu +0.10 ± 0.01, g_xu +0.23 ± 0.01, g_xx -0.10 ± 0.00 |
| starhorse23_gesdr5 | offset | 491 | y | +0.230 ± 0.009 |  |  |  |  |
| starhorse23_lamostl7 | flex | 175098 | y | +0.230 ± 0.001 | -0.228 ± 0.002 | -0.146 ± 0.001 | MSTO: +0.029 ± 0.000, +0.113 ± 0.001 | g_uu -0.49 ± 0.00, g_xu +0.18 ± 0.00, g_xx -0.02 ± 0.00 |
| starhorse23_lamostm7 | linear | 9336 | y | +0.249 ± 0.002 | +0.189 ± 0.004 | -0.006 ± 0.004 |  |  |
| starhorse23_ravedr6 | offset | 838 | y | +0.196 ± 0.003 |  |  |  |  |
| starhorse23_rvs | linear | 14812 | y | +0.195 ± 0.002 | +0.064 ± 0.005 | -0.269 ± 0.004 | MSTO: -0.021 ± 0.002, +0.178 ± 0.008 |  |
| starhorse23_sdssdr12 | offset | 540 | y | +0.208 ± 0.004 |  |  |  |  |
| willett26_L | offset | 371 | y | +0.235 ± 0.012 |  |  |  |  |
| willett26_dnu | linear | 10997 | y | +0.010 ± 0.001 | +0.011 ± 0.002 | -0.033 ± 0.003 |  |  |
| xiang24 | offset | 196783 | y | +0.260 ± 0.001 |  |  |  |  |
| key | n_constrained | n_constrained_identified | triad_stars | pairs_with_identified | tier_by_count | tier | component | bridge_width | domain_lo | n_below_domain |
|---|---|---|---|---|---|---|---|---|---|---|
| apokasc3 | 7850 | 7850 | 30 | 3537 | 0 | 0 | 12 |  | -0.505 | 36.000 |
| apok2 | 3420 | 0 | 3 | 3 | 1 | 1 | 12 |  | -0.155 | 7.000 |
| apokasc2 | 87 | 66 | 1 | 20 | 1 | 1 | 0 | 84.000 |  |  |
| bench_eb | 15 | 0 | 15 | 15 | 1 | 1 | 1 | 30.000 |  |  |
| bench_gc | 4 | 0 | 0 | 0 | 1 | 1 | 2 | 3.000 |  |  |
| bench_ldb | 9 | 0 | 9 | 9 | 1 | 1 | 3 | 18.000 |  |  |
| chaplin14 | 301 | 245 | 6 | 26 | 1 | 1 | 4 | 267.000 | -0.169 | 1.000 |
| cl_cg20 | 1790 | 1790 | 1486 | 1790 | 2 | 2 | 5 |  | -2.628 | 2.000 |
| cl_dias21 | 1519 | 1519 | 1453 | 1519 | 2 | 2 | 5 |  | -2.322 | 1.000 |
| cl_hr23 | 2151 | 2151 | 1492 | 1862 | 2 | 1 | 5 | 554.000 | -2.477 | 7.000 |
| galah_dr4_sgb | 65074 | 65074 | 422 | 18381 | 3 | 3 | 13 |  | -0.068 | 257.000 |
| legacy17_BASTA | 55 | 0 | 17 | 25 | 1 | 1 | 6 | 49.000 |  |  |
| mackereth21_tess | 1 | 0 | 0 | 1 | -1 | -1 | 7 |  |  |  |
| starhorse23_apogee17 | 9808 | 9808 | 170 | 4002 | 2 | 2 | 13 |  | 0.025 | 262.000 |
| starhorse23_galahdr3 | 48443 | 48443 | 0 | 582 | 3 | 3 | 13 |  | -0.032 | 236.000 |
| starhorse23_gesdr5 | 491 | 373 | 1 | 10 | 1 | 1 | 8 | 59.000 | 0.123 | 12.000 |
| starhorse23_lamostl7 | 175098 | 175098 | 2 | 863 | 3 | 3 | 13 |  | -0.031 | 12890.000 |
| starhorse23_lamostm7 | 9336 | 9336 | 3 | 602 | 2 | 2 | 13 |  | -0.027 | 1026.000 |
| starhorse23_ravedr6 | 838 | 838 | 1 | 374 | 1 | 1 | 9 | 805.000 | -0.001 | 262.000 |
| starhorse23_rvs | 14812 | 14812 | 20 | 6616 | 2 | 2 | 13 |  | -0.029 | 413.000 |
| starhorse23_sdssdr12 | 540 | 540 | 0 | 249 | 1 | 1 | 10 | 517.000 | 0.090 | 187.000 |
| willett26_L | 371 | 171 | 0 | 0 | 1 | 1 | 11 | 361.000 | -0.596 | 2.000 |
| willett26_dnu | 10997 | 7735 | 0 | 0 | 2 | 2 | 12 |  | -0.344 | 13.000 |
| xiang24 | 196783 | 196783 | 493 | 16265 | 3 | 1 | 13 | 33.000 | -0.009 | 893.000 |
| zinn22_k2gap | 35 | 0 | 1 | 0 | 1 | -1 | 14 |  |  |  |


### noise A
| unit | records | median quoted σ | κ | ω (dex) | scale at median σ | eff. inflation |
|---|---:|---:|---|---|---:|---:|
| apok2:giant | 4660 | 0.137 | 0.49 | 0.076 | 0.101 | 0.74 |
| apokasc2:RGB | 204 | 0.106 | 0.46 | 0.017 | 0.052 | 0.49 |
| apokasc2:all | 35 | 0.095 | 1.99 | 0.031 | 0.192 | 2.02 |
| apokasc3:RC | 5391 | 0.040 | 0.47 | 0.008 | 0.021 | 0.52 |
| apokasc3:RGB | 6665 | 0.051 | 0.40 | 0.005 | 0.021 | 0.42 |
| bench_eb:all | 15 | 0.030 | 1.03 | 0.080 | 0.086 | 2.89 |
| bench_gc:all | 9 | 0.020 | 1.60 | 0.242 | 0.244 | 12.03 |
| bench_ldb:all | 9 | 0.071 | 1.59 | 0.033 | 0.117 | 1.66 |
| chaplin14:SGB | 276 | 0.052 | 0.33 | 0.018 | 0.025 | 0.48 |
| chaplin14:all | 138 | 0.086 | 1.12 | 0.029 | 0.101 | 1.18 |
| cl_cg20:cluster | 1790 |  |  | 0.172 | 0.172 |  |
| cl_dias21:cluster | 1519 | 0.160 | 0.70 | 0.103 | 0.152 | 0.95 |
| cl_hr23:cluster | 6956 | 0.221 | 0.79 | 0.143 | 0.226 | 1.02 |
| galah_dr4:SGB | 158315 |  |  | 0.005 | 0.005 |  |
| legacy17_BASTA:all | 66 | 0.059 | 0.63 | 0.080 | 0.088 | 1.49 |
| starhorse23_apogee17:MSTO | 18141 | 0.028 | 0.67 | 0.058 | 0.061 | 2.16 |
| starhorse23_apogee17:SGB | 14136 | 0.025 | 1.02 | 0.049 | 0.055 | 2.19 |
| starhorse23_galahdr3:MSTO | 40128 | 0.060 | 0.39 | 0.023 | 0.033 | 0.55 |
| starhorse23_galahdr3:SGB | 23993 | 0.040 | 0.33 | 0.019 | 0.023 | 0.58 |
| starhorse23_gesdr5:MSTO | 2709 | 0.068 | 0.35 | 0.050 | 0.055 | 0.81 |
| starhorse23_gesdr5:SGB | 2524 | 0.068 | 0.55 | 0.038 | 0.054 | 0.79 |
| starhorse23_lamostl7:MSTO | 352025 | 0.039 | 0.56 | 0.036 | 0.042 | 1.09 |
| starhorse23_lamostl7:SGB | 115742 | 0.072 | 0.70 | 0.058 | 0.077 | 1.07 |
| starhorse23_lamostm7:SGB | 46784 | 0.030 | 0.76 | 0.072 | 0.076 | 2.52 |
| starhorse23_ravedr6:MSTO | 20166 | 0.093 | 0.24 | 0.080 | 0.083 | 0.89 |
| starhorse23_ravedr6:SGB | 14643 | 0.059 | 1.05 | 0.046 | 0.078 | 1.31 |
| starhorse23_rvs:MSTO | 169356 | 0.085 | 0.32 | 0.070 | 0.075 | 0.88 |
| starhorse23_rvs:SGB | 104149 | 0.048 | 0.38 | 0.051 | 0.054 | 1.13 |
| starhorse23_sdssdr12:MSTO | 8283 | 0.054 | 0.37 | 0.011 | 0.023 | 0.43 |
| starhorse23_sdssdr12:SGB | 7731 | 0.043 | 0.77 | 0.020 | 0.039 | 0.89 |
| willett26_k2_bhm:giant | 4351 | 0.106 | 0.37 | 0.006 | 0.039 | 0.37 |
| willett26_k2_bhm_L:giant | 285 | 0.089 | 2.70 | 0.063 | 0.248 | 2.80 |
| willett26_k2_cor:giant | 531 | 0.112 | 0.20 | 0.012 | 0.025 | 0.23 |
| willett26_k2_cor_L:all | 107 | 0.082 | 1.93 | 0.039 | 0.164 | 1.99 |
| willett26_k2_gal_L:giant | 629 | 0.084 | 1.59 | 0.033 | 0.138 | 1.64 |
| willett26_kepler:RC | 4329 | 0.078 | 0.49 | 0.063 | 0.074 | 0.94 |
| willett26_kepler:RGB | 3450 | 0.058 | 0.63 | 0.005 | 0.037 | 0.63 |
| willett26_kepler:all | 94 | 0.142 | 1.00 | 0.029 | 0.145 | 1.02 |
| willett26_kepler_L:all | 176 | 0.049 | 2.31 | 0.188 | 0.220 | 4.45 |
| willett26_tess:giant | 572 | 0.118 | 0.99 | 0.030 | 0.121 | 1.02 |
| willett26_tess_L:all | 20 | 0.069 | 1.00 | 0.030 | 0.076 | 1.09 |
| xiang24:SGB | 305542 | 0.032 | 0.95 | 0.005 | 0.031 | 0.97 |
| shared effect eta: isochrone |  |  |  | 0.123 | 0.123 |  |
| shared effect eta: cluster_cmd |  |  |  | 0.349 | 0.349 |  |
| shared effect eta: lamost_spectra |  |  |  | 0.068 | 0.068 |  |
| shared effect eta: galah_spectra |  |  |  | 0.005 | 0.005 |  |
