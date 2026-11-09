# Expanded multi-survey model: integration and QA proposal

## Interpretation of the existing checks

The common prediction and a new survey's catalogue are not automatically on the
same scale. The RAVE PIT/coverage results in the previous audit are external
reference-compatibility diagnostics, not an isolated measurement of calibration
of the latent common-reference distribution. A new survey needs its own
observation mapping and measurement model, estimated on development stars and
evaluated on disjoint stars. Cluster scatter remains a useful diagnostic, but
comparing common-coordinate scatter with native-coordinate scatter also needs
the transformation's local scale and uncertainties to be considered.

## Model and reference definition

Keep one feature-only distribution q(theta | XP, photometry, parallax, masks),
where theta currently contains Teff, log g and Fe/H. For each source s, predict
its observations through p_s(y_s | theta, observing metadata). Its likelihood
integrates over the full common distribution; the source mapping is not another
independent stellar predictor. Missing labels simply supply no corresponding
likelihood term. Retain the separate masked reconstruction objective.

Start new sources by freezing the completed encoder/common head and learning
only their regularized observation mappings/noise terms on development stars.
Then compare joint adaptation with that frozen baseline. Use overlap stars and
parameter-space support to identify the mappings; report unsupported regions.
Source error models and intrinsic uncertainty can trade off, so repeat spectra,
independent references and constrained noise models are needed.

The current zero-sum survey gauge defines a relative coordinate, not an absolute
physical scale. Preserve a fixed original reference convention for old/new
comparisons, or define a versioned physical anchor convention. Do not silently
recenter over the expanded source list. Any anchors used in fitting must be
disjoint from the benchmark evaluation set, including calibration lineage.

## Proposed source units

A source is survey + release + parameter pipeline + relevant instrument/setup
or observing mode + calibration version, not merely a survey name.

| Addition | Initial integration contract |
|---|---|
| SEGUE / SDSS legacy | Identify SSPP product, valid stellar domain, flags, repeats and Gaia association; do not count all SDSS spectra as usable stellar labels. |
| BOSS / eBOSS | Select an explicitly documented stellar-parameter product; keep instrument/pipeline distinctions from legacy SEGUE, with hierarchical sharing only if validated. |
| MaStar | Separate parameter methods and their dependencies; distinguish parameters inferred from MaStar spectra from imported target-selection catalogue labels. Multiple reductions of one spectrum are correlated. |
| LAMOST | Separate LRS/MRS and native/data-driven pipelines. Audit teacher labels, particularly APOGEE/GALAH-derived scales, rather than treating each catalogue as independent evidence. |
| Gaia RVS | Use GSP-Spec, its target-specific flags and calibration provenance. Teff/log g can enter directly. M/H is not Fe/H; a physical abundance conversion needs the relevant chemical information and covariance. |
| Gaia-ESO | Preserve UVES/GIRAFFE/setup and homogenization provenance, recommended parameters, quality flags and audited Gaia IDs. |
| RAVE | Start with spectroscopic MADERA plus compatible GAUGUIN Fe information. Keep BDASP and seismic-calibrated alternatives as separate dependent products. |

If we want to use generic M/H measurements substantially, investigate extending
the common chemical state, for example with alpha information. Do not force
M/H to be a universal function of Fe/H alone. Retain both native values and
publisher corrections; avoid applying a correction twice through a learned map.

## QA programme

| QA | Comparison / experiment | What it establishes or catches |
|---|---|---|
| 1. Source admission | Immutable source versions; unique stars, visits, valid scalar labels, flags, uncertainty units, missingness and crossmatch ambiguity. | Actual usable scale; duplicates and quantity mismatches. |
| 2. Leakage and lineage | Group all observations of each star; hold entire clusters/binary groups for dedicated tests; audit calibration standards and teacher-training sets. | Prevents repeated spectra, label transfer and standards from becoming false independent tests. Already inspected devtest is development evidence for subsequent choices. |
| 3. Overlap connectivity | Pairwise shared-star counts and coverage in Teff/log g/FeH; connected components per target; map uncertainty in weakly connected regions. | Whether a common scale is identifiable rather than extrapolated from nonoverlapping populations. |
| 4. Native observation prediction | On held-out stars, compare the full pushed-forward common distribution with each source's native labels, convolving its measurement model. | Survey-space bias, residual scatter, tails and coverage without confusing coordinate offsets with prediction failures. |
| 5. Mapping validity | Inspect corrections/Jacobians against parameters and observing metadata; check folds, extreme stretches, boundaries, complexity and forward/inverse consistency where invertible. | Prevents a flexible map from hiding astrophysical or uncertainty failures. |
| 6. Physical benchmarks | Independent interferometric/bolometric Teff; seismic or dynamical gravity; carefully documented high-resolution iron references. | Physical-scale accuracy, with reference uncertainties and method dependence preserved. Fe/H is not a purely geometric fundamental standard. |
| 7. Benchmark triangle | For the same stars: common prediction vs benchmark; native survey vs benchmark; mapped common prediction vs native survey; optionally inverse-mapped survey vs benchmark. | Separates common-model error, survey bias and observation-map error. Propagate the full distributions through every transformation. |
| 8. Distribution calibration | Marginal PIT/rank checks, exact-mixture or sampled NF quantiles, 50/68/90/95% coverage, NLL and CRPS; joint energy score and validated joint credible-region coverage. | Accuracy, sharpness, tails and Teff–gravity–iron dependence. Include finite-sample uncertainty and subgroup results. |
| 9. Cluster consistency | Independently selected astrometric members; same-star comparisons; cluster means, Fe/H scatter and conditional RGB widths in common coordinates; both predictive means and posterior draws. | Real improvement versus shrinkage or coordinate compression. Never require zero Fe/H scatter for every cluster. |
| 10. Cluster and spatial holdouts | Entire unseen clusters plus matched represented-cluster tests; trends with radius, G, crowding, extinction and distance. | Memorization, environmental systematics and selection effects. Infer more eligible members to improve statistics without adding their labels to fitting. |
| 11. Wide binaries / coeval systems | Component Fe/H consistency with full errors, temperature-dependent trends, vetted physical associations. | Common-composition checks across different stellar temperatures; do not require identical Teff or log g. |
| 12. Population/domain slices | Giants/dwarfs, RGB/RC where independently classified, hot/cool and metal-poor tails, magnitude/SNR, extinction, sky, distance, missing-input patterns and survey overlap. | Failures hidden by an overall average or by the largest survey. |
| 13. Repeats and perturbations | Native repeat spectra test source-noise models; perturb XP/photometry/parallax consistently with measurement covariance; remove modalities and vary masks. | Repeat noise, sensitivity to inputs, robustness and overconfidence. Reusing one Gaia input with multiple spectroscopic visits is not repeated independent model inference. |
| 14. Survey holdout / ablation | Leave one survey out of common training; fit only its map on its development subset; evaluate on other stars. Compare each addition with the frozen three-source baseline. | Transfer, new information, negative transfer and reference drift. |
| 15. Survey-size balance | Star-weighted and survey-balanced objectives, overlap reweighting, fixed evaluation panels; report metrics by source and rare population. | Whether LAMOST or another large source dominates at the expense of other populations. |
| 16. Architecture and numerical QA | Same data/splits/compute for Gaussian, current mixture and conditional NF; nonlinear push-forward sampling versus current affine linearization; independent likelihood/gradient checks. | Whether richer distributions improve proper scores/coverage, and whether the observation approximation is adequate. |
| 17. Large Gaia-XP comparisons | Exact Gaia-ID matches to Andrae, Zhang/Green/Rix, SHBoost, Gaia Net and Ye/FERRE; catalogue-wide agreement plus paired performance against common external references. | Large-scale systematics, model disagreements, domain failures, relative accuracy and sharper predictions without sacrificing calibration. |

### Benchmark presentation

For every reference family show count, domain, provenance, training/calibration
exposure, median bias, robust scatter, tail/outlier rate, coverage and reference
uncertainty. Use identical matched stars for native/common comparisons. Retain
benchmark-specific method/model dependencies; do not label a correlated
compilation as independent truth. Display fitted development anchors separately
from untouched evaluation anchors.

### Evaluation governance

Maintain distinct FIT, selection, calibration and final evaluation roles.
Freeze a fresh group-aware final panel before further architecture/QA-driven
choices; never relabel previously used test results as a fresh confirmation.
Choose acceptance thresholds before scoring that panel. Bootstrap by the
relevant independent unit (star, cluster, binary system or calibration family),
not scalar-label count. Show small-N limitations rather than universal numerical
pass thresholds.

## Recommended order

1. Freeze existing predictions and reference convention; complete source/lineage
   and overlap censuses for every proposed addition.
2. Establish audited benchmark and cluster-group splits before adaptation.
3. Fit new observation maps with the existing stellar distribution frozen;
   run native-space and benchmark-triangle QA.
4. Jointly adapt the shared model with reconstruction retained, comparing
   source-balanced arms and checking old-source regressions.
5. Compare mixture/NF alternatives only after the source/scale checks pass;
   evaluate final calibration and cluster generalization on the fresh panel.

## Primary documentation informing this proposal

- MaStar's multiple parameter methods and imported/heterogeneous source scales:
  https://www.sdss4.org/dr17/mastar/mastar-stellar-parameters/
- SDSS product access, including separate optical products and legacy SSPP:
  https://www.sdss.org/dr20/data_access/
- LAMOST LRS native parameter catalogue and Gaia DR3 identifier fields:
  https://www.lamost.org/dr12/v2.0/table/stellar
- Updated GBSv3 fundamental parameters (Soubiran2024), using the corrected catalogue:
  https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/682/A145
- Updated recommended abundances (Casamiquela2026), keeping FeI/H and FeII/H distinct:
  https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/705/A167
- Existing per-source registry/provenance contracts:
  configs/survey-label-registry-20261026-v1.json

This is an integration/QA proposal, not a claim that these added sources have
already been ingested or fitted.

Release audit correction (6 November): start from DR20, with Astra 0.8.1 for
SDSS-V BOSS. DR20 directs completed SEGUE/BOSS/eBOSS/MaStar to their final DR17
products; retain actual release/pipeline versions. MaStar publisher Gaia EDR3
VAC is checksum-verified and joined by xmatcher; Gaia-ESO full matching remains
pending, with its unknown source epochs and candidate ambiguities explicit.

## Large Gaia-XP catalogue comparison — added 6 November

Use the [evaluation-only comparator registry](../../configs/gaia-xp-comparator-qa-20261106.json).
Start with Andrae, Rix & Chandra's **2023 XGBoost catalogue** (175M), keeping it
separate from Gaia GSP-Phot. Add Zhang, Green & Rix (2023, 220M), **SHBoost
(2024, 217M)**, **Gaia Net (2025, about 219M XP rows)** and **Ye et al.
(2025, 68.4M)**. These are published catalogue sizes, not measured overlaps with
our current predictions. Pin actual files, schema, version and quality cuts
before scoring; use the latest validated author release rather than assuming
an archive metadata update represents new predictions.

Two complementary panels are required:

1. **Large common-sample agreement:** join all available model predictions by
   exact Gaia ID, preserve FIT/TUNE/CAL/devtest/external roles, and plot common
   Teff/logg/metallicity differences, Kiel diagrams and sky/parameter trends.
   FIT agreement remains descriptive. For future feature-only inference,
   extend to millions of stars after role/ancestry assignment; the recovered
   4.76M inputs are not yet predictions or a held-out score.
2. **Paired reference accuracy:** on identical stars and finite targets,
   compare each catalogue and our correctly survey-mapped predictions against
   the same spectroscopy. Show common predictions separately against physical
   benchmarks and cluster members under the fixed reference convention.
   Report raw scale compatibility and any development-fitted corrections
   separately. A tighter catalogue locus or agreement between two methods
   alone does not establish higher physical accuracy.

For each comparator report per-target match counts, selection losses, median
bias/MAE/NMAD/RMSE, outliers and paired error differences with spatial/cluster
block uncertainty intervals. Show pairwise overlaps first; the intersection
of every catalogue is a secondary, potentially selective panel. Slice by G,
colour, extinction, parallax SNR, sky/crowding, giant/dwarf/RC, hot/cool and
metal-poor regimes. Repeat cluster widths and Fe/H dispersion on the same
members for every method, retaining posterior draws to expose shrinkage.

Compare known comparator-training members, known nonmembers and unknowns
separately, together with our own historical exposure. In particular, retain
Andrae's `in_training_sample`. Shared XP inputs and overlapping APOGEE/GALAH/
LAMOST/StarHorse teachers prevent an unqualified independent-validation claim.
Parallax, infrared bands and population priors differ between methods; record
those input differences. Our current masked feature contract does not expose
all the parallax information used by some released competitors.

Our full mixture/NF predictions get PIT, interval coverage, CRPS/NLL and joint
scores against compatible external references, with reference noise treated
explicitly. Released point estimates or scalar uncertainties are evaluated
as supplied; never manufacture a joint PDF or treat their means as truth for
an accuracy/calibration ranking. Preserve **[M/H] versus [Fe/H]**, native solar
scales, log-temperature transforms and Zhang's K-versus-kK release distinction.

Primary references: [Andrae XGBoost](https://arxiv.org/abs/2302.02611),
[Zhang catalogue/schema](https://dc.zah.uni-heidelberg.de/tableinfo/xpparams.main),
[SHBoost author archive](https://data.aip.de/projects/shboost2024.html),
[Gaia Net CDS release](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/ApJ/984/58?format=html&tex=true),
[Ye/FERRE 2025](https://arxiv.org/abs/2411.19105).

Execution status: this adds the comparison contract. Previous Andrae results
refer to older checkpoints; recompute all overlaps and scores for the selected
current checkpoint. No new large-catalogue performance result is claimed here.

Gaia Net version pin: the author API resolves the latest public dataset to
[Zenodo 15048995](https://zenodo.org/records/15048995), DOI
`10.5281/zenodo.15048995`. Its XP Parquet is 9,049,244,346 bytes with publisher
MD5 `4cc09524fc1b4fe774dbeeb9dc0d50c2`; this metadata pin is not a completed
file download or readback verification. The registry preserves it separately
from the older author record and the CDS republication.
