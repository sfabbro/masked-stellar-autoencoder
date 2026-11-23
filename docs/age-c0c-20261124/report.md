# C0c: residual QA, a chemistry check of the gauges, and rotation and lithium as fitted sources (2026-11-23)

Status: phase 2 of the priority labels. Ages are unchanged from C0b: C0b's age maps, noise and shared errors are reused unchanged. Rotation periods and A(Li) are fitted against that consensus as monotone maps of the latent, then every star is re-solved with all its records.
- Code: `src/masked_stellar_autoencoder/training/age_consensus.py` (`MonotoneMaps`, `FrozenAgeMaps`, `prefit_monotone`), `scripts/age_consensus_c0.py run --freeze-from <C0b work>`, `scripts/age_metal_poor_check.py`, `tests/test_age_consensus_c0.py`.
- Units: latent t = log10(age/Gyr); rotation y = ln(P/day); lithium y = A(Li).

## 1. Summary
- **The "old-age ceiling" is not real.**
  - The drop in the C0b residuals above 8 Gyr comes from binning by a noisy consensus (regression to the mean).
  - A null simulation (correct map, consensus perturbed by its own σ) reproduces it: Xiang −0.20 / −0.26 at t = 1.0–1.1 / 1.1–1.2 against −0.15 / −0.27 observed; GALAH −0.19 against −0.25; StarHorse RVS / APOGEE / LAMOST-MRS −0.19 / −0.18 / −0.17 against −0.23 / −0.21 / −0.22.
  - Binned by the record value, the residual at the old end is flat or slightly positive.
  - No model change; the QA figure now shows the null curve and the residual against the record value.
- **The chemistry check contradicts the EB verdict for old stars.** At fixed [Fe/H] and [α/M], giants (seismic ages) and subgiants/turn-off stars (isochrone ages) should share an age distribution.
  - High-α sequence, [Fe/H] −1 to +0.1 (about 4,400 giants, 3,900 subgiants): gauge A puts subgiants 0.21–0.22 dex younger than giants; gauge C agrees to 0.02 dex.
  - Low-α: A −0.10 to −0.15 dex, C +0.06 to +0.10.
  - [Fe/H] < −1 (142 giants, 118 subgiants) favours A (−0.005 against +0.20).
- **One reading reconciles the benchmarks and the chemistry: the bridge depends on age.** By "bridge" we mean the offset between the isochrone (subgiant) and seismic (giant) scales.
  - The EBs, in turn-off clusters of 1–8 Gyr (most at 2–4 Gyr), want a bridge of ≈ 0.20 dex (A's +0.26 minus the EB offset +0.06).
  - Low-α stars (≈ 4–5 Gyr) want ≈ 0.15 dex, roughly midway between A and C.
  - Old high-α stars (≈ 10 Gyr) want ≈ 0.05 dex (gauge C).
  - So the isochrone pipelines read too old for young and intermediate ages and about right for old ages, i.e. they are compressed relative to the seismic scale.
  - A constant bridge, which is what A and C each assume, cannot satisfy both. The 33-star giant–subgiant link cannot constrain a slope; a population constraint per chemistry cell can (§5).
- **Rotation and lithium enter as fitted maps.**
  - The gyrochronology maps, learned from the cluster ladder, follow Mamajek & Hillenbrand 2008 in 4000–6000 K. They pass through the Sun (about 28 d against 25.4 d) and Praesepe's K dwarfs (10.7 d).
  - Lithium maps decline sensibly in the 4800–6300 K bins.
- **Clusters (gauge A):**
  - Praesepe 1.13 → 0.68 Gyr (literature ~0.7).
  - Hyades 1.10 → 1.15 (LDB 0.70, EB 0.90), but member z-NMAD fell from 22 to 0.40.
  - NGC 6811 0.70 → 0.74 (EB 1.05).
  - Worse: Pleiades 0.142 → 0.222 (LDB 0.127), Blanco 1 0.151 → 0.083 (LDB 0.137), NGC 3532 0.46 → 0.15.
  - NGC 188, M67, NGC 752, Ruprecht 147 and NGC 6791 unchanged.
  - Below about 300 Myr rotation converges and lithium plateaus, so the maps carry little age information there, and what they carry is biased.
- **Arbitration** (latents re-solved without benchmark records; the age maps are frozen, so this is not a clean held-out refit):
  - EB +0.077 ± 0.017 (A) against −0.134 ± 0.020 (C).
  - LDB −0.157 against −0.285.
  - WD+MS, now with 6 and 5 usable pairs because main-sequence companions get rotation/Li ages: −0.008 ± 0.111 and −0.072 ± 0.098 (A) against −0.35 and −0.42 (C).
  - Every independent benchmark (mostly ≤ 4 Gyr) favours A; the chemistry check favours C for old stars.
- **New stars.** About 270k dwarfs without an age record gain a consensus from rotation or lithium. Only 25,663 have σ < 0.3 dex (median 0.21), and 1,949 have σ < 0.15.

## 2. What was tried, and why the final setup is staged
1. **Joint fit** (rotation/Li maps, age maps and noise all free). The cluster shared catalogue error ran to 1.03 dex (0.35 in C0b): flexible maps fed by many cluster members trade their slopes against the clusters' shared catalogue error.
   - Fix: non-age records must not redefine the age scale they are learned on. Staged fit with C0b's age maps, noise and shared errors frozen (`--freeze-from`).
2. **Map form.** Cumulative ramps (convex) ran away on synthetic clusters.
   - Now df/dt is a non-negative sum of local bumps (a smoothed degree-1 B-spline basis): monotone by construction, flat outside the knots, anchored at 4 Gyr (c0 = f(4 Gyr)), with bump weights that stay near zero without data.
   - An identifiability gate keeps a map only with ≥ 30 constrained stars tied to age records with identified noise.
3. **GALAH Li, 4800–5700 K.** The scatter ran to its 3-dex cap twice, once with and once without a pre-fit start.
   - Cause: with most records being upper limits below the map, a larger scatter pushes the EM-imputed values further below the map, which raises the scatter again.
   - Fix: non-age scatter held at the pre-fit value (censored Student-t regression on stars with σ_t < 0.25).
   - The 6300–6700 K Li-dip bin was dropped (non-monotone; its map diverged).
4. **Field-tied rotators.** Wide pairs and stars with their own age records flattened the gyro maps between 0.6 and 4 Gyr, and the solve put Praesepe and the Hyades at about 3 Gyr. These records carry detection bias, unresolved binaries and turn-off contamination.
   - Fix: non-age maps are learned from cluster members only; rotation/Li records of stars with an evolved-class age record are dropped.
5. **TARS (TESS) periods.** Completeness falls above about 10–13 d, so in old clusters only the fast tail is detected (Ruprecht 147: TARS median 5.0 d against 26.1 d from K2). Even as solve-only records they pulled the Hyades, NGC 752, Ruprecht 147 and NGC 3532 to 0.1–0.9 Gyr.
   - Fix: TARS is not used.

## 3. Rotation and lithium maps (gauge A; values at the bin centre)

| map | 0.1 Gyr | 0.25 | 0.63 | 1.6 | 4.0 | 10 | scatter (fixed) |
|---|---|---|---|---|---|---|---|
| rot 4500–5000 K (ln P) | 1.09 | 2.06 | 2.36 | 2.55 | 3.00 | 3.44 | 0.08 |
| rot 5000–5500 K | 1.49 | 1.80 | 2.15 | 2.48 | 3.14 | 3.79 | 0.18 |
| rot 5500–6000 K | 1.21 | 1.55 | 2.00 | 2.57 | 3.29 | 3.74 | 0.31 |
| Li GALAH 4800–5300 K | 1.85 | 1.35 | 0.77 | 0.51 | 0.45 | 0.43 | 0.61 |
| Li GALAH 5300–5700 K | 2.34 | 2.19 | 1.92 | 1.17 | 0.42 | 0.18 | 0.44 |
| Li GALAH 5700–6000 K | 2.58 | 2.54 | 2.39 | 2.19 | 2.06 | 2.01 | 0.36 |
| Li GES 4800–5300 K | 2.14 | 1.46 | 1.23 | 1.03 | 0.52 | 0.02 | 0.18 |

Full tables: `mono_maps_{A,C}.csv`, `mono_info_{A,C}.csv`, `measurement_model_{A,C}.json`.

## 4. Clusters (gauge A, Gyr; C0b → C0c)

| cluster | members C0b → C0c | C0b | C0c | EB / LDB |
|---|---|---|---|---|
| Praesepe | 2 → 341 | 1.13 | 0.68 | (lit. 0.7) |
| Hyades | 4 → 50 | 1.10 | 1.15 | 0.90 / 0.70 |
| NGC 6811 | 24 → 69 | 0.70 | 0.74 | 1.05 |
| Pleiades | 8 → 150 | 0.142 | 0.222 | — / 0.127 |
| Blanco 1 | 0 → 47 | 0.151 | 0.083 | — / 0.137 |
| NGC 2516 | 0 → 56 | 0.31 | 0.084 | — |
| NGC 3532 | 1 → 21 | 0.46 | 0.15 | — |
| M67 | 114 → 257 | 2.86 | 2.90 | 3.50 |
| Ruprecht 147 | 31 → 56 | 2.16 | 2.16 | 2.67 |
| NGC 188 | 5 → 5 | 3.75 | 3.75 | 6.2 |

Co-natal consistency:
- Cluster members (3,301): z-NMAD 0.67.
- Dwarf–dwarf wide pairs (99): z-NMAD 0.15.
- The σ of rotation/Li-only stars is conservative.

## 5. Still wrong
1. **Gauge.** EBs and WD+MS favour A; old high-α stars favour C.
   - Next step: an age-dependent (linear) bridge. Constrain it with a population term per chemistry cell (equal giant and subgiant age distributions at fixed [Fe/H], [α/M]) with ~10^4 stars, alongside the EBs. Arbitration then tests the slope, not a constant.
2. **Young clusters (≤ 300 Myr).** Rotation and lithium pull Pleiades, Blanco 1 and NGC 3532 the wrong way. Options: restrict non-age records to latents above about 300 Myr through a flat lower map edge (as for the age maps), or model the young rotation bimodality.
3. **NGC 188 and Hyades.** NGC 188 has only 5 members. The Hyades' member records are now consistent with each other, but the latent sits above its LDB and EB ages.
4. **Truncated period catalogues** need a truncation-aware likelihood before TARS can return.
5. **Convergence.** The staged fits stop at 15 outer iterations with raw coefficient moves of 0.7–8 (probably slack directions of bumps without data). A function-space convergence check is needed.

## 6. Files
- Plots: `c0c_rotation_lithium.png`, `c0c_clusters.png`, `c0c_arbitration.png`, `c0c_metal_poor_check.png`, `c0c_residuals.png`, `c0c_maps.png`, `c0c_scale_difference.png`.
- Tables: `clusters_{A,C}.csv`, `arbitration_offsets.csv`, `benchmark_maps.csv`, `metal_poor_check.csv`, `noise_{A,C}.csv`, `maps_{A,C}.csv` (C0b's age maps), `summary.json`.
- Commits on `feat/age-c0`: ecb0b56, 8abdf2b, e7c2d18, 982be4b, 469440e, b52df80, 63e8421, 5a0e3d2, 7d11992, e63aa11.
