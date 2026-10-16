# Metallicity, age, distance and optimization experiments

The authoritative accepted-job record is `experiments/xp/launch-stellar-improvement-20261015.json`; `configs/stellar-improvement-plan.yaml` also contains unsubmitted dependency/time-gated plans. Never resubmit an accepted ID. All changes are isolated from main pretraining and the immutable 9M/1M benchmark.

The first supervised pilots use at most100K training labels, retaining rare stars before caps, with source-ID holdouts and inverse cap-inclusion weights. They are experiments on the existing labelled feature table, not10M independently labelled stars. The reconstruction LR/batch runs use the full10M benchmark and20M presentations, not20M unique stars. Native-label repetitions wait for the new GALAH/APOGEE registry; legacy GALAH quality flags were discarded and cannot be recovered from the feature table alone.

The native registry preserves raw measurements, errors, rejected quality rows, survey and input-row provenance, distinguishing APOGEE `[M/H]` from native `[Fe/H]`. Survey abundance scales are not assumed identical. XP-derived and photometric metallicities remain candidate selectors or explicitly weak labels. DESI ingestion requires published temperature/gravity corrections, validity limits and the exact Gaia extension join before scientific use. The official Pristine1000-row sample is not the full release; Huang–Beers/LAMOST access metadata is not a downloaded catalogue.

`dataset_view_v1` indexes all10M existing benchmark stars against registryv1. `dataset_view_v2` waits for the native APOGEE/GALAH registryv2. Both preserve the original train/validation assignment, verify source-ID SHA256 in original train-then-validation row order and uniqueness/no overlap, and expose every original field, packed correlation, scaler and row/shard provenance through read-only external links. These are indexed datasets, requiring the benchmark and registry at their recorded paths; they are not standalone copies. No new unique20M-star sample is claimed.

Read the indexed data with an explicit split:

```python
import h5py
import numpy as np

with h5py.File("dataset-view.h5", "r") as f:
    g = f["train"]  # or f["validation"] for evaluation
    # Every original feature/error field remains available here.
    records = g["benchmark/records"]
    block = records[:4096]
    label_row = g["selected_label_row"][:4096]
    # Fit SFT only on benchmark train stars with label split0.
    fit = (label_row >= 0) & (g["label_holdout_split"][:4096] == 0)
    labels = f["label_registry/selected/labels"]
    # h5py fancy indexes require sorted unique positions; restore original order.
    unique, inverse = np.unique(label_row[fit], return_inverse=True)
    matched_labels = labels[unique][inverse]
```

`posterior_labels_v1` preserves all5/16/50/84/95 percentiles and flags from the existing StarHorse APOGEE17/GALAHDR3 tables. Age isGyr and distance iskpc, verified against FITS units. Finite positive monotonic percentile validity is separate from survey-quality eligibility. These are derived model-dependent posteriors, not independent asteroseismic ages or geometric distances. Flag interpretation and independent evaluation remain required. The initial parallax-prediction pilots hide parallax and its aliases; agreement with Gaia labels alone cannot establish distance improvement beyond Gaia.

All results must compare paired IDs/truth/masks/units, parent/scaler hashes, fixed adaptation budgets and checkpoint reloads. Report metal-poor bins and uncertainty calibration alongside population-weighted performance; oversampling can change the objective and the corrected variant explicitly compensates sampling probability. Do not infer representation or SOTA performance from reconstruction MAE alone.

The scheduled photometry/age pilots wait until2026-10-16T00:47:49Z and are submitted by the monitor rather than consuming sleeping GPU sessions. Photometry ablations quantify existing survey groups on identical test IDs and common availability footprints. They cannot quantify an unimported survey yet. Future import priorities require measured coverage, photometric uncertainties, extinction/zero-point conventions, source matching and common-footprint downstream tests.

Native-label comparisons include ordinary-Huber SFT and frozen-encoder controls, plus a sampling-probability-corrected balanced variant. These three controls isolate encoder adaptation and objective changes on exactly the new registry; provisional legacy-label pilot scores are not matched controls for native-label experiments. Native reconstruction and balanced Huber runs compare with native ordinary-Huber SFT, not the noisy-NLL variant.

`desi_labels_v1` verifies the full combined FITS against the publisher checksum, stages it in scratch, checks RVTAB/FIBERMAP/SPTAB TARGETID equality, and takes DR3 identifiers from the separate GAIA extension. It independently audits sky agreement and retains a conservative1arcsec subset. RVS iron-abundance correction follows the published bounded piecewise polynomial at immutable reference commit10b6ab6f409397dde646184c42bc9dddfb452971. All raw measurement rows and errors remain available, with rejected values and correction-invalid NaNs preserved. Selected unique stars require SUCCESS, PRIMARY, RVS_WARN=0, stellar class, SN_R>=10, native positive parameter errors and0<FEH_ERR<0.5dex. The recorded native random error is not asserted to be a total calibrated uncertainty; a separate temperature propagation assumes independence, and logg branch switching/covariances/correction systematics remain unquantified. This first ingestion handles RVS only; SP outputs are preserved in the immutable original FITS.

`pigs_labels_v1` parses the official CDS ReadMe together with tablesD2/D3, preserves every raw field and missingness, and requires native goodF/goodU plus double=false and measured errors. Published FeH uncertainties already contain the0.13dex floor; it is not added twice. FERRE is preferred to ULySS without averaging correlated pipeline values. Distance posteriors join through unique Star-name/GaiaDR3 associations. The released D3-minus-D2 repeated metallicity differs consistently by~0.0008903dex (measured full-table range0.0008902–0.0008905dex); an explicit0.002dex audit bound is applied and both values/differences retained without shifting them. All10379names match; duplicate name/ID mappings are audited rather than assumed one-to-one. StarHorse distances remain derived posteriors, not independent geometric truth.

`enriched_label_view_v1` creates separate native, PIGS and Li2025 CaT silver layers with unchanged source values, measured errors and original source rows. It independently rechecks all10M ordered benchmark IDs, uniqueness and the9M/1M assignment. Each layer records source-ID label split, benchmark membership, broader fit eligibility (label split0 excluding benchmark validation) and strict fit eligibility (also requiring benchmark train). Sources remain separate: repeated stars across layers must not be concatenated as independent observations. No abundance correction or inverse-error averaging is applied; DESI is excluded pending calibrated uncertainty modelling. Li2025 flag1 is a valid two-line CaT estimator, and duplicate observations use highest SNR rather than smallest metallicity error. Its published0.2dex Ca/Fe allowance remains separate from reported errors because inclusion in the electronic-table errors is unresolved. Silver labels are not promoted to high-resolution gold NLL targets. These indexes associate labels to the existing10M stars and do not materialize20M new unique stars or perform training.

`pristine_value_frozen_v1` is one bounded inference/head-fitting job: a frozen epoch57 encoder feeds three fixed-alpha10 ridge heads (encoder alone, encoder plus two processed Gaia colours, and those plus CaHK−G). Each is fitted on two explicitly named cohorts, strict benchmark-train/label-split0 and broader label-split0 excluding benchmark validation. All six fits use the same common input-quality cohort and source-ID test split2; added features never include photometric metallicity. Head scaling uses each fit cohort only, with identical zero initial standardized predictions across variants and no encoder/SGD updates. Coefficients, paired predictions/latent/photometry/native truth, original rows, split/membership, input hashes, FITS unit audit and exact head/encoder reloads are preserved. This first pilot uses processed giant-solution dereddened Pristine/Gaia magnitudes, so it measures processed catalogue value, not an isolated raw-CaHK information gain. No new sky-holdout claim or independent high-resolution/unsupervised-pretraining independence claim is made.
