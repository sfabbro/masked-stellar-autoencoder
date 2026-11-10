# Native population correction and completed comparison QA

The native population audit found **3,767,056 DESI/LAMOST Gaia IDs absent from
the previous candidate parent**. The builder had incorporated quality-selected
sidecars and an older feature registry, rather than explicitly including the
complete staged native observation identity sets. It now includes the raw IDs
before quality, feature or XP selection.

## Counts and scope

| Staged native product | Observation rows | Unique positive Gaia IDs | Absent from old candidate parent |
|---|---:|---:|---:|
| LAMOST DR12 v2 LRS AFGK parameters | 8,376,135 | 5,829,702 | 2,255,802 |
| LAMOST DR12 v2 MRS parameters | 3,224,901 | 1,499,584 | 305,791 |
| DESI DR1 coadded stellar VAC | 6,372,607 | 4,920,596 | 1,254,465 |
| Sum of observation rows / deduplicated star union | 17,973,643 | 11,165,172 | 3,767,056 |

Per-source missing counts overlap; their sum is not the missing-star union.
LAMOST LRS/MRS share 748,508 IDs. DESI overlaps LRS by 314,072 IDs and MRS by
55,960 IDs. All joins and union operations used integer IDs, with no float
conversion, XP predicate or quality selection. LAMOST sidecars were freshly
verified against their recorded hashes; DESI's ID column was reread from the
full native GAIA extension. Its earlier publisher-verified whole-file SHA was
recorded, not recomputed in this ID audit.

The candidate parent across these and the other included surveys expands from
**13,592,882 to 17,359,938 unique stars**. This is an identity parent, not a count
of quality-admitted labels or FIT stars. New identities remain unassigned until
protected benchmarks, clusters, calibrator families and inherited roles are
handled. Native rows without a usable Gaia association also require a separate
matching audit. These three files do not exhaust every possible release/VAC.

The approximately 10M earlier prediction or feature footprint must not be used
as a native catalogue size or as a training-population cap. The previous 13.59M
Gaia-stage matching result is valid for its recorded parent but does not prove
completeness of the newly expanded parent.

The [LAMOST release page](https://www.lamost.org/dr12/v2.0/) separately lists
12,602,862 LRS spectra, 8,376,135 AFGK parameter records and 3,224,901 MRS
parameter records. Its M-star catalogue contains additional LASP Teff/logg and
native M/H channels, which require their own source admission rather than an
automatic Fe/H rename. The [DESI VAC documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws/)
also lists 10,012,925 single-epoch parameter/RV records. Those repeats and their
coadds share inputs; they cannot be counted as independent stars or multiplied
as independent likelihoods. The [LAMOST release portal](https://www.lamost.org/lmusers/)
lists DR12 v2 public and DR13 v1 internal, while the
[NADC DR13 record](https://nadc.china-vo.org/res/r103329/?lang=en) labels sharing
scope public. DR13 access/release eligibility remains to be established; this
audit does not assume either availability or exclusion from an incomplete
metadata listing.

## Execution

- `native_desi_lamost_population_20261110_v1`, session `dolylw69`: complete;
  exact native union and expanded ID arrays remain on ARC.
- `full_gaia_native_parent_refresh_20261110_v1`, session `m9pwdvrf`: running;
  matches the audited 17.36M parent to full Gaia, refreshes legacy photometry,
  then counts unique-star coverage with inherited roles and native XP flags.
  It does not admit labels, assign new roles, normalize data or fit a model.
- `tess_selected_checkpoint_qa_20261110_v1`, session `vmuapd4t`: complete;
  final-selected-checkpoint TESS slices use existing bound draws, 12,017 DEV stars.

The refresh checks the population-audit manifest hash, validates the explicit
parent ID array and records its hash. All outputs have new directories and
source bundles. Prior sessions and outputs were reconciled before submission.

## Completed selected-checkpoint comparison

The older comparison architecture finished QA on 21,561 stars, bound to update
71,906 / epoch 2 and selected-checkpoint SHA
`ebee9539bb6a53683899fba9d4979a57e96c8a9e5b409a085bfd51c80bf534dd`.

| Metric, identical comparison stars | Selected checkpoint | Earlier supervised-adapted NSF |
|---|---:|---:|
| Benchmark Teff MAE, 120 stars | 198.5 K | 324.6 K |
| Benchmark logg MAE, 120 stars | 0.3337 dex | 0.3903 dex |
| Benchmark Fe I/H MAE, 120 stars | 0.1966 dex | 0.2324 dex |
| APOKASC development logg MAE, 737 stars | 0.0571 dex | 0.0692 dex |
| APOKASC reference-convolved 90% coverage | 93.49% | 82.50% |
| APOKASC median 90% interval width | 0.3169 dex | 0.2432 dex |
| TESS DEV logg MAE, 12,017 stars | 0.0724 dex | 0.0869 dex |
| TESS DEV logg NMAD | 0.0769 dex | 0.0735 dex |
| TESS reference-convolved 90% coverage | 84.56% | 86.87% |
| 47 Tuc common Fe/H NMAD, 632 members | 0.2283 dex | 0.1667 dex |

The larger intervals accompany better seismic coverage, but are above nominal
90% for APOKASC. TESS point MAE improves while NMAD and coverage worsen, so the
seismic result does not generalize uniformly. Benchmark iron NMAD worsens despite better MAE, and cluster iron dispersion
remains much wider than same-star native references. These results do not
establish a universally superior or calibrated six-dimensional physical PDF.
Ancestor/teacher dependencies and shared Gaia inputs remain, and DEV has been
used for development. The corrected full-Gaia pretrained trainer is still to be
integrated and validated; these results describe the older comparison model.

Full saved-array report and manifests are alongside this file; large draws,
checkpoints, native catalogues and ID arrays stay on ARC.
