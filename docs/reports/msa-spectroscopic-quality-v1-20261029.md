# Provisional spectroscopic quality masks

These masks are stored separately from the unmodified RAVE DR6 and Gaia-ESO DR5.1 measurements. They are **not** training selection, match certification, physical truth, or the full RAVE published recommended sample. The same observations may appear multiple times; counts are rows, not independent stars. The existing HDF5 sidecar and receipt are in the private project directory `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/rave_dr6_cnn_recovery_sources_20261028_v1/`, and all files were SHA-checked by remote readback. Inputs are pinned by SHA in the sidecar. Code: `scripts/spectroscopic_survey_quality.py`; exact results: `experiments/xp/spectroscopic-quality-v1-completion-20261029.json`.

## Policy

- All tiers require finite values in broad deliberately permissive limits: Teff 1,000–50,000 K, logg −2 to 7, metallicity −6 to +2 dex, RAVE alpha/Fe −2 to +2 dex, and Gaia-ESO native neutral-element values −5 to +15 dex. These are sanity bounds, not survey calibration ranges or claims of accuracy. Gaia-ESO neutral-element values are retained on their native scale and are not treated as [alpha/Fe].
- RAVE MADERA loose accepts documented algorithm codes 0, 2, 3 or 4 and rejects non-convergence code 1 or unknown codes. The strict tier adds S/N ≥ 20 and a finite positive uncertainty bounded at 1,000 K for Teff or 1 dex for logg and M/H. Code 4 is specifically a low-S/N metal-poor giant treatment, so the strict S/N cut removes that cohort; loose remains available. The release paper recommends additional RV and fit-quality checks that are **not** represented here. [RAVE DR6 paper](https://research-management.mq.edu.au/ws/portalfiles/portal/164923962/164628021.pdf).
- RAVE GAUGUIN loose requires a unique observation-key match to MADERA, its accepted algorithm code, and a positive contributing line/element count. Strict adds MADERA S/N ≥ 20, a finite positive abundance uncertainty ≤ 1 dex and finite nonnegative fit statistic. This is a provisional screen, **not** the published RAVE combined-fit Q selection. Its Fe/H and alpha/Fe share the MADERA inputs and spectra; they cannot be counted as independent observations of separate objects.
- Gaia-ESO loose excludes survey-defined SNR and SRP flags, and excludes NIA only for non-iron elemental abundances. Unknown flag tokens fail closed. Strict additionally requires positive finite quoted error (≤ 1,000 K or ≤ 1 dex), excludes BIN and excludes SSP/PSC for atmospheric parameters or SSA for elemental abundances. The strict tier is intentionally conservative: the release warns that SSP/SSA may concern only some measurements. The [survey flag guidance](https://www.eso.org/sci/publications/announcements/sciann17584.html) supports using SNR, SRP and NIA to clean a sample. Positive uncertainty availability is separately recorded; missing uncertainties are never interpreted as zero.
- Gaia-ESO identity is **unknown**, not falsely labeled unmatched. RAVE's EDR3 source IDs are published best-neighbour candidates, not all-neighbour ambiguity certification. Training roles, ancestor exposure and feature eligibility remain unset.

## Coverage examples

| Catalogue/label | Loose rows | Strict rows | Loose rows below −2 dex | Strict rows below −2 dex |
| --- | ---: | ---: | ---: | ---: |
| Gaia-ESO Fe/H | 93,423 | 50,945 | 1,176 | 601 |
| RAVE GAUGUIN Fe/H | 317,006 | 308,945 | 52 | 50 |
| RAVE MADERA M/H | 500,182 | 459,439 | 5,166 | 3,559 |

The two RAVE quantities have different definitions and the rows overlap. Gaia-ESO counts include no Gaia identity requirement; no value above is a FIT or test count. No cut is silently applied to other source catalogues.

## Gaia-ESO DR3 crossmatch

A three-row candidate-only xmatcher probe to the configured Gaia DR3 CDS/TAP source failed with a connection reset. A discovered local Gaia cache has only 3.5 million entries and explicitly says `complete: false`; it cannot be treated as an all-sky catalogue. Gaia-ESO's input astrometry comes from different catalogues, and its per-row epochs are not in the staged release. A genuine proper-motion-aware match needs an accessible candidate source and per-row epoch/provenance, or an explicitly bounded epoch-uncertainty model with all candidate alternatives. Neither a successful match nor a rejected Gaia identity is inferred from this probe.

A subsequent bounded ESA Gaia archive probe succeeded using xmatcher `gaia_esa` and a 0.1-degree cone. For the first three Gaia-ESO rows, the 10-arcsec candidate search returned eight Gaia pairs, all with finite proper motions. Propagating the Gaia positions to each **assumed** source epoch (2000, 2010, 2016) and matching at 1 arcsec returned the same one candidate for each source at all three epochs. An independent ESA query reproduced the first candidate's exact 64-bit ID. This is a three-row example and cannot establish positional epochs, association reliability, or complete sky coverage. It provides a tested route for a properly bounded catalogue-wide candidate search, not the catalogue-wide result. Receipt `experiments/xp/gaia-eso-xmatcher-esa-probe-20261029.json`.
