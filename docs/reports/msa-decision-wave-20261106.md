# Six-target training decision wave — 6 November 2026

Decision target: **7 November 2026, approximately 19:30 Perth time** (36 hours from the request).

**The submitted CANFAR jobs continue with the laptop closed.** Sources, execution commands, hashes and outputs are persisted on ARC. This is an experiment/QA wave; no final large training job is scheduled or launched automatically.

## Controlled training wave

Eight arms share the verified c544 large-pretrained parent, the historical 1,736,618 FIT identity pool, a 10,000-update budget, source/label-balanced sampling and reconstruction weight 0.03. Heads are Gaussian, full-covariance mixture and unrestricted coupled 6D NSF; each has photometry+XP and measured-parallax profiles. Two extra NSF arms use head LR 3e-4 rather than 1e-3. These pilots are distinct from the completed three-target multi-survey model and are **not trained on the newly expanded survey cohort**. The current three-target model is separately retained for frozen source-map and catalogue QAs.

Selection is by fixed TUNE macro normalized median MAE; PDF quality is measured separately. Unrestricted NSF training uses a masked sample energy score, not the previously failed sparse/noisy NLL estimator. Previous numerical failures are not reclassified. Age and distance are posterior-reference targets; alpha conventions and shared teacher ancestry remain explicit. Historical FIT/benchmark exposure remains in these controlled pools; the final protected bundle must exclude the updated benchmark IDs and linked holdout groups.

## Jobs

| Run | Session | Resources | Submission |
|---|---|---|---|
| decision6_gaussian_parallax_20261106_v1 | wpkuk4w1 | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_gaussian_photxp_20261106_v1 | qau40ujl | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_mdn_parallax_20261106_v1 | xd67cic9 | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_mdn_photxp_20261106_v1 | fy21b76l | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_nsf_energy_lr3e4_parallax_20261106_v1 | ow1dv01n | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_nsf_energy_lr3e4_photxp_20261106_v1 | vm57z70n | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_nsf_energy_parallax_20261106_v1 | tf5ol2y2 | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_nsf_energy_photxp_20261106_v1 | pti5lsfz | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision6_pdf_qa_worker_20261106_v1 | uh9gy7on | {'gpu': 1, 'cores': 4, 'ram': 64} | accepted |
| decision_dr20_legacy_source_audit_20261106_v1 | kdpbzkfs | {'cpu': 4, 'ram': 64, 'gpu': 0} | accepted |
| decision_dr20_mrs_label_admission_20261106_v1 | gf3f4b5v | {'cpu': 4, 'ram': 32, 'gpu': 0} | accepted |
| decision_expanded_parallax_audit_20261106_v1 | es4y8r67 | {'cpu': 4, 'ram': 32, 'gpu': 0} | accepted |
| decision_frozen_new_survey_maps_20261106_v1 | utd9dmii | {'cpu': 4, 'ram': 32, 'gpu': 0} | accepted |
| decision_source_census_20261106_v1 | qkltlmzo | {'cpu': 4, 'ram': 32, 'gpu': 0} | accepted |
| decision_wave_array_qa_report_20261106_v1 | txf2kymg | {'cpu': 2, 'ram': 16, 'gpu': 0} | accepted |
| decision_xp_catalogue_qa_20261106_v1 | mkoy9zq3 | {'cpu': 4, 'ram': 32, 'gpu': 0} | accepted |

The receipt says accepted, not completed science. Per-run snapshots and current ARC manifests supply actual progress. Eight training arms were observed making finite-gradient updates; Gaussian/mixture arms had selected first TUNE checkpoints.

## Unattended QA and data work

- PDF worker consumes final selected checkpoints and computes marginal sample CRPS, PIT/coverage, interval widths and masked joint energy from true six-vector samples; separate quoted-noise scores retain unknown-error masks. Bounded draw chunks address the prior evaluation OOM. A single GPU QA worker waits for ready outputs and processes models sequentially.
- CPU report worker verifies saved NPZ CRCs, recomputes corner covariances, hashes selected checkpoints, scores per-target point/reference-compatibility results and writes `decision_wave_array_qa_report_20261106_v1/decision-report.md`. It does not launch final training.
- Andrae 175M and latest Gaia Net XP catalogue acquisition is running, with publisher size/checksum gates, exact Gaia-ID matching and paired comparisons against the latest frozen survey-mapped predictions. Catalogue agreement is separated from native-reference compatibility.
- Completed new-survey offset/affine map comparisons for DR12 LRS, RAVE and Gaia GSP-Spec keep the old common gauge fixed. Gaia unknown-error results are compatibility proxies with explicit illustrative noise floors, not physical calibration certificates.
- Completed DR20 BOSSNet source acquisition (1,803,883 rows, publisher checksum verified), final legacy SSPP acquisition (1,843,200 rows) and DR12 MRS schema audit (3,224,901 rows). Separate candidate admission completed: **929,032 DR20 BOSSNet** and **821,991 DR12 MRS LASP** unique candidate stars. Two DR20 benchmark IDs are flagged for reservation; no raw acquisition or candidate count is called trained stars.
- Completed source census confirms 4,756,867 expanded XP/photometry input candidates. Their current feature table exposes **zero measured-parallax inputs**; a separate exact-ID cache-parallax recovery/audit completed for **4,756,093** cache-associated coordinates, but supplies **zero known formal errors** and does not certify native measurement provenance. No profile was activated and no errors were invented.

## Decision-critical gaps

The historical six-target FIT counts are Teff 506,331, gravity 493,548, Fe/H 506,315, alpha 1,197,641, age 115,834 and distance 117,573. Only **277 FIT stars have all six labels**; six-dimensional physical correlations are not established by merely emitting joint samples. Final training must incorporate admitted survey mappings and the expanded sources, rather than treating this historical pooled source as the complete new training set.

Before final launch, lock the chosen head and input profile; check per-domain calibration/accuracy and inherited exposure; admit chemical definitions, missing errors and posterior teachers; construct benchmark/cluster/dependency-protected roles; bind parallax recovery if selected; and define the all-Gaia inference partition manifest, missing-input/OOD flags, PDF representation, throughput and storage budget. The 150M cache and recovered labelled subset do not establish completeness over all Gaia sources.

Gaia-ESO full external crossmatching is **not submitted**. Automatic approval review rejected uploading 114,916 privately staged positions/row IDs to ESA because explicit permission for that payload/destination was absent. An approval question is pending. No workaround or indirect upload was attempted; all unaffected jobs continued.

Verification: six targeted sampling/PDF/map/pipeline checks passed; Ruff passed for the new decision-wave code. Native checkpoint/model execution is on CANFAR CUDA only. No upstream push.

Final pre-close snapshot at approximately 07:54 Perth: 16 submissions; eight
training arms making finite-gradient updates, one PDF QA worker, seven CPU
analysis/source workers. Five CPU data/map/coverage jobs completed. Andrae
transfer was nearly complete and Gaia Net had passed 3.2GB of 9.05GB at the last
transfer check. Latest scientific manifests, rather than acceptance receipts,
remain authoritative. The pending Gaia-ESO upload approval is the sole rejected
external action; no ESA query carrying those source positions was submitted.

## Gaia-ESO approval and local-input correction

The user explicitly approved sending all 114,916 Gaia-ESO positions and row
identifiers to ESA and pointed out that our full raw Gaia input set already
contains RA/Dec. Two routes are therefore prepared and submitted: an approved
ESA TAP_UPLOAD/xmatcher candidate query and a wholly local scan of the original
raw input shards, retaining source IDs, RA/Dec, proper motions and raw row/shard
provenance. The local route does not transmit our input set externally. Its
raw source is 474,657,279,648 bytes; it is our actual staged input footprint,
not an asserted complete 1.8-billion-source Gaia catalogue.

Original crossmatch workers stopped before entering their Python scripts. The
local failure explicitly reported an unavailable Pixi Conda/PyPI mapping fetch.
For the ESA attempt, authenticated private ARC metadata established that the
source bundle exists but the script output directory does not; since that
directory is created before any ESA submission, no query/upload ran. Original
attempts are preserved. Distinct v2 workers include the missing mapping/DNS
hosts. Their results and source bundles inherit the existing private source
permissions. Matching uses multiple assumed source epochs because the native
Gaia-ESO table does not supply per-row epochs; neither route promotes a
candidate directly into training identity.

This approval supersedes the earlier pending/rejected status above. The
local motion/retrieval test passed, including an RA-wrap/high-motion case;
Ruff passed for both crossmatch scripts.

### Confirmed coordinate inventory and crossmatch execution

Our actual raw input contains **218,131,884 rows** with RA, DEC, proper motions,
parallax and `e_parallax`, as well as proper-motion errors/correlation. A
512-row, four-shard audit confirms finite, distinct, degree-scale coordinates
and positive formal astrometric errors. The earlier cache-derived feature
exports masked these fields and omitted their errors; that did **not** establish
that the raw data lacked astrometry. No inverse-parallax distances are inferred.
The first local scan passed **56.6 million raw rows** and collected **31,856
candidate raw rows** near Gaia-ESO positions; these are candidate records, not
certified matches or unique training stars. The scan continues on CANFAR.

ESA received the approved upload. The full spatial query was aborted by the
archive. A first bounded batch then encountered HTTP 500 while changing its
execution limit; its existing upload was resumed without re-upload. That query
also terminated ABORTED. The distinct indexed v5 attempt uses the geometrically
equivalent reversed predicate so the ESA Q3C rewrite can use the Gaia-side
index; results remain pending. Q3C requires its indexed coordinates in the
second coordinate pair ([primary documentation](https://github.com/segasai/q3c));
this optimization is an inference from the archive's SQL error and that rule.
All original failures, archive identifiers and private source permissions are
retained; no duplicate active query was started. Local-only matching continues
independently of ESA.

Latest verified crossmatch progress: indexed ESA batch 0–5,000 COMPLETED with
62,496 candidate pairs; batch 5,000–10,000 submitted. Local full-input scan
reached 68,531,193 rows and 33,851 candidate raw rows. Neither pair counts nor
raw candidate rows are unique or promoted training identities.

Latest [decision readout](msa-decision-readout-20261106.md) records completed
Gaussian/mixture PDFs, large-XP comparisons and both completed Gaia-ESO
candidate crossmatches. It supersedes earlier progress snapshots.
