# Current production training contract — 8 November 2026

This describes `expanded_six_nsf_fullfit_20261107_v1`, not the separate seismic-control suites. It follows the deployed script/configuration, effective runtime masks and a read-only input census. The model configuration lists 140 possible input slots; that list alone does not describe active features in this run.

## Populations and sampling

The current supervised FIT cohort contains **7,693,002 unique stars** and **9,203,880 source observation records per epoch**, for two full record epochs. Survey overlaps remain multiple observations of the same star. Separate TUNE, CAL and DEV cohorts contain 1,041,809, 804,075 and 883,819 stars. There are 4,792 usable reserved external stars, giving 10,427,497 usable stars across admitted roles.

The reconstruction stream draws random contiguous 256-star blocks from a **150,000,000-star cache**. This is an auxiliary stream in the current fine-tune, not 150M labelled stars and not a full 150M reconstruction pass each epoch. The planned 71,906 joint updates give 18,407,936 reconstruction presentations, with repeats possible. The inherited encoder's earlier SSL training is a separate phase.

A normal step contains 256 uniform source records plus 32 repeated FIT seismic records and 32 StarHorse auxiliary records (16/8/8 across its three channels), plus the independent 256-star reconstruction batch. Encoder, decoder, NSF and free survey-map parameters all update jointly. A preceding 1,000-step warm-up adjusted newly admitted survey maps with encoder and NSF fixed.

The full reconstruction-pool census finds **150,000,000/150,000,000 rows with all 110 XP coefficients**, and zero rows without XP input. This is separate from the labelled cohort.

The actual FIT input census finds **7,669,152 stars with all 110 XP coefficients and 23,850 with none**. Counts refer to coefficients supplied to this model, not an independent audit of official Gaia publication status. No partial-XP FIT vectors were found. Thus the model includes photometry/parallax-only inputs, but they are only about 0.31% of FIT; this does not establish broad no-XP/all-Gaia predictive validity.

## Effective architecture

Supervised prediction:

`110 XP coefficients + W1/W2/G/BP/RP + measured parallax → encoder → 256-D latent → MLP → 16-D flow context → six-dimensional NSF → common-reference draws`

Reconstruction, using separate masked batches and shared encoder weights:

`masked XP + W1/W2/G/BP/RP → same encoder → decoder → masked MAE on observed XP/magnitudes`

The encoder is the pretrained residual network with feature embeddings. The active supervised inputs are the five listed magnitude bands, 55 BP and 55 RP XP coefficients, and signed measured parallax where available. Missing inputs remain missing. **EBV, RA, DEC, pmra and pmdec are inactive**, as are the extra optical/2MASS bands listed in the broad model schema. None is concatenated separately at the prediction head. Formal parallax errors are retained for QA but are not a separate encoder/head context feature.

The decoder formally has additional output slots, but the current reconstruction target mask scores only the five magnitude bands and XP. Parallax and EBV are not reconstruction targets in this phase. Masking ratios are 50% of available XP coefficients and 20% of available magnitudes; reconstruction weight is 0.03. There is no systematic full-XP modality dropout added by this recipe.

There is one common six-dimensional conditional NSF, not independent NSF heads per survey or target. Its MLP uses width 128 and emits 16 context features; the NSF uses three transforms, 16 spline bins and internal hidden widths 64/64. The unused inherited Gaussian importance proposal is frozen and has zero objective weight.

## Targets, observation maps and references

Every eligible input can produce Teff, logg, Fe/H, alpha/Fe, age and distance jointly. Age and distance are represented internally in log coordinates and reported in Gyr and kpc. They are learned on **StarHorse posterior-median teacher scales**, not independent true-age/distance measurements. Parallax is a predictive input; a physical parallax–distance likelihood is not imposed.

All supported source-target edges use deterministic continuous monotone piecewise-linear maps with positive slopes and invertible tails. The production maps act separately on each target; their gravity corrections do not independently depend on temperature or metallicity. Contextual seismic maps belong to the separate experiment suites, not this production run.

| Target | Fixed identity reference | Other source maps |
|---|---|---|
| Teff | APOGEE | Learned for other supported spectroscopic channels |
| logg | APOKASC seismic | Learned for spectroscopic channels; seismic map itself fixed identity |
| Fe/H | APOGEE | Learned for other supported spectroscopic channels |
| alpha/Fe | DESI | Only DESI alpha labels are currently admitted; no cross-survey alpha calibration is learned |
| age and distance | StarHorse APOGEE teacher | Learned for StarHorse GALAH and catalogue teacher channels |

Age/distance mappings are therefore **not all identity**. Unsupported survey-target edges are not scored or invented. APOKASC supplies 5,321 FIT gravity anchors with quoted uncertainties and repeated auxiliary supervision; TESS is QA only. A common reference is an explicit convention plus relative calibration, not an automatically unbiased union of survey labels.

Admitted spectroscopy: APOGEE, DESI, GALAH, LAMOST DR12 LRS/MRS, RAVE DR6, Gaia DR3 GSP-Spec and SDSS DR20 BOSSNet. Gaia GSP-Spec supplies Teff/logg here; its M/H is not substituted for Fe/H. Gaia-ESO and MaSTAR remain deferred, and legacy SEGUE/eBOSS coverage has not been separately certified.

## Objective, diagnostics and limits

Training scores samples after survey mapping and measurement-noise convolution, using a masked energy score over each source's actually observed target block, plus masked reconstruction and map regularization. The selected operational label noise is independent Gaussian with quoted errors where known and a fixed positive floor where unknown. This noise assumption does not make the NSF's predicted stellar distribution Gaussian. Correlated label-noise and common-mode calibration uncertainty are not learned by this production model.

Native residuals and common residuals are both exported:

- Native: `map(prediction) - native_observation`.
- Common: `prediction - inverse_map(native_observation)`.

The plotting sign is prediction minus reference, opposite to the sign in the user's proposed expressions. Observations remain noisy after inverse mapping. Multiple surveys' observations are retained separately instead of averaged into one true six-vector per star. Proper source-space predictive scores, interval widths/compatibility, independent-reference checks and per-star coupled corners supplement point residuals.

Full-cohort plots for all six targets and FIT/TUNE/CAL/DEV are currently bound to the update-2,000 snapshot. The completed-epoch update-35,953 has newer fixed-panel benchmark, cluster and seismic QA; it is not yet the checkpoint used for the full 10.43M prediction table.

Only 1,281 FIT stars have all six labels across sources. Source-wise scoring does not directly identify every atmospheric–age–distance dependence, even when a star has separate records. The flow provides genuine coupled draws, but their physical covariance and all six uncertainty distributions are not thereby certified. Current large-run predictions have not received a final CAL-based interval correction; fitted survey-map parameter uncertainty is not marginalized into the exported predictions.

[Input census status](https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/current_run_contract_audit_20261108_v1/manifest.json) · [Training configuration](/Users/seb/src/masked-stellar-autoencoder/configs/expanded-six-nsf-fullfit-20261107.yaml)

The completed input census is saved in [current-run-input-census.json](/Users/seb/src/masked-stellar-autoencoder/docs/reports/msa-insights-20261108/current-run-input-census.json). Its full-cache counts are authoritative; the periodic progress field records the last intermediate checkpoint before completion.
