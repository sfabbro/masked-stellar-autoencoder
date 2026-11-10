# MSA handoff — 2026-11-08, Vancouver

Start here when resuming on another computer. This is a WIP research handoff, not a release or a claim that the corrected full-Gaia model has been trained.

## Cloud resume — 2026-11-09, Vancouver

### Continuation — 2026-11-10, Vancouver

All Nov 9 recovery, census, selected-checkpoint CUDA QA and report jobs completed
successfully. Latest results and population correction are in
`docs/reports/msa-cloud-resume-20261110/progress.md` and its bound manifests.
The older comparison checkpoint improves benchmark MAEs and APOKASC gravity,
but cluster Fe/H scatter remains worse than the earlier model; it is not the
corrected full-Gaia model.

**The 13.59M candidate parent was incomplete.** User reiterated that DESI+LAMOST
must not be limited to the approximately 10M prepared/prediction footprint.
A fresh native ID audit measured 17,973,643 observation records and 11,165,172
unique Gaia stars in the staged LRS/MRS/DESI files. 3,767,056 of those stars were
absent from the candidate parent. Restoring them expands the all-included-survey
identity parent to 17,359,938 stars. The raw-ID builder is fixed; it now includes
native observation IDs before quality or XP/feature selection. No 10M cap is
admitted. Record counts, unique stars, quality labels and FIT stars are separate.

Native audit `native_desi_lamost_population_20261110_v1` (`dolylw69`) completed;
expanded and missing-ID arrays remain on ARC. The audited parent is being used
by `full_gaia_native_parent_refresh_20261110_v1` (`m9pwdvrf`) to refresh full-Gaia
matching, legacy photometry and census in fresh directories. An expected audit
manifest hash and parent ID hash bind this execution. No role assignment,
normalization, label admission or production training occurs. Recheck receipt,
session and each step's manifest before any retry.

`tess_selected_checkpoint_qa_20261110_v1` (`vmuapd4t`) completed on 12,017 DEV
stars: logg MAE improves .0869 to .0724 dex, but NMAD worsens .0735 to .0769 and
reference-convolved C90 falls 86.87% to 84.56%. Its reporter now accepts the new
selected-checkpoint status while preserving earlier snapshots. The historical
job table below is superseded by the Nov 10 manifests.

The native-parent refresh is running its Gaia step on exactly 17,359,938 IDs.
28 relevant tests pass together, including raw native union/overlap arithmetic,
recovery of IDs outside selected sidecars, and explicit audited-parent binding.

Remaining priorities: validate native quality/domain/dependency cuts; investigate
LAMOST M-star and other VAC/release channels separately; restore native band
errors/flags for real no-XP stars; assemble protected roles and FIT-only
normalization on the corrected parent; integrate the actual pretrained model
with full inputs/context and the justified common-coordinate/noise model.

The following setup notes describe the Nov 9 executor state and remain relevant:

Resumed branch `codex/cloud-resume-20261109` from `26f59be` in the managed
cloud executor under `/workspace`. Current network status is unrestricted,
enforced, with current observations and matching spec revision 1. Fresh requests
reach CANFAR/CADC; the previous executor's proxy CONNECT 403 is resolved.

The pinned CANFAR 1.4.1 client was installed at
`/workspace/.venvs/canfar`, revision `0d96a72761cc6dc7063d6cd97049ae3743133efa`.
Pixi's executable is absent locally, but `.pixi/envs/default/bin/python` works.
The official `canfar` skill was recovered from `opencadc/canfar` and installed
at `.agents/skills/canfar/SKILL.md`; read it and its references. Its content hash
is recorded in `skills-lock.json`. AGENTS.md now points to that available skill.

User certificate login succeeded, but cloud TLS inspection made X.509 requests
arrive anonymously. The user then completed CADC's advertised token login with
`/workspace/canfar-token-login.py`. The private token is stored outside Git at
`/workspace/.canfar-runtime/cadc-token` (0600). Do not print or commit it. Runtime
session clients require both `token=` and
`url="https://ws-uv.canfar.net/skaha/v1"`; storage accepts
`filesystem("arc", token=...)`. Credentials and these local runtime files are
not portable to another executor. No cloud TLS/proxy bypass was configured.

Read-only status, including the newly submitted jobs:

```bash
/workspace/.venvs/canfar/bin/python scripts/inspect_msa_handoff.py --token-file /workspace/.canfar-runtime/cadc-token
```

Before any submission, reconciled all 158 existing decision-wave receipts with
authenticated sessions: none of their session IDs remained retained. Refreshed
all handoff manifests and inspected ARC outputs. Evidence is under
`docs/reports/msa-cloud-resume-20261109/`. The older comparison run completed
71,906 joint updates and two full FIT epochs, selecting epoch 2. A CANFAR CPU
audit independently verified `selected-checkpoint.pth`, 137,372,427 bytes, SHA256
`ebee9539bb6a53683899fba9d4979a57e96c8a9e5b409a085bfd51c80bf534dd`.

Remote continuation (snapshots; recheck before acting):

| Work | Session | Last observed |
|---|---|---|
| full_gaia_resume_audit_20261109_v1 | vexwzwnb | Completed; checkpoint digest verified |
| full_gaia_corrected_sources_20261109_v1 | noz8klgb | Failed on native LAMOST undeclared ASCII nulls, after completing full partition metadata audit |
| full_gaia_corrected_sources_20261109_v2 | eo0ywcd8 | Running; repaired age and alpha extraction complete, input census follows |
| expanded_selected_checkpoint_qa_20261109_v1 attempt 1 | m1n37991 | Failed before model execution: historical source branch was deleted |
| expanded_selected_checkpoint_qa_20261109_v1 attempt 2 | gqc68dfi | Running; exact-commit acquisition preserves the original source pin |
| expanded_selected_checkpoint_report_20261109_v1 | fwib7zhu | Running watcher, waiting for CUDA arrays |

Every new job has an immutable source bundle and receipt in the existing
locations. Attempt 1's branch failure was reconciled before retry. The GPU
launcher now fetches the exact pinned commit rather than requiring its old
branch name. New jobs use the live-listed `astroai/base:latest`; a resolved image
digest was not established. CPU jobs save Pixi lockfiles to their bundles after
successful execution. No unrelated sessions were changed or deleted.

The metadata audit verified 13,592,882 labelled Gaia records plus 513,698
reservoir rows across 11,394 partitions; photometry has 1,754 partitions and
10,931,306 observation rows, alpha 301 partitions, and old age staging 24.
The photometry manifest reports zero no-XP observation rows, so its legacy XP
parent cannot supply all missing-band coverage. This does not replace unique
star/quality auditing. The new census validates integer candidate-row joins,
counts unique-star band/error availability by inherited role and native Gaia XP
flag, and leaves new stars unassigned. It does not perform quality admission or
FIT normalization. The v2 bundle includes protected `external`/`excluded` roles;
the v1 census implementation would reject them, but was never reached because
age extraction halted first.

Age-source recovery now supports native ASCII FITS tables and explicit nullable
integer parsing. LAMOST's observed undeclared `---` numeric null marker is an
explicit per-product rule recorded in the manifest. AMBRE, APOKASC-2, both
APO-K2 files and LAMOST were previously silently skipped by binary-only parsing.
Native source-row indices and HDU identity are retained, and `TAU_GRAN_SSI`
(granulation timescale) is excluded from age candidates. These are extraction
repairs, not age publication/quality/dependency admission. The source driver
halts before subsequent steps when a source reports an error.

Native photometric header evidence is in `native-photometry-headers.json`:
SkyMapper DR4 has `gaia_dr3_id1/id2`, match distances/ambiguity fields, native PSF
photometry, errors and flags; 726 compressed shards were listed. Existing ADQL
products contain 469,001,402 2MASS rows and 119,663,765 SDSS rows, with errors and
quality flags. CatWISE has 678,990,741 rows: its `source_id` is a string catalogue
identifier and `id2` is integer; verify the latter's Gaia release/association
before using it as a Gaia ID. Do not infer identity solely from a column name.
The GALAH WISE/2MASS/Gaia VAC also retains native errors/quality and Gaia IDs.
These products were inspected, not fully joined or quality-admitted.

Selected-final comparison QA binds the completed producer and expected SHA,
uses a distinct `large_selected_checkpoint` artifact key, and preserves earlier
snapshot defaults. The report accepts both snapshot types and describes the
older architecture explicitly. It is configured to include common/source-native comparisons,
Kiel diagrams, seismic/benchmark/cluster checks and whole-vector corners.
No corrected full-Gaia production training was submitted: native admission,
role-safe source assembly/FIT normalization and actual pretrained-trainer
integration remain necessary. Preserve the scientific requirements below.

Validation: 23 relevant pipeline, full-Gaia, selected-checkpoint binding,
saved-array source-binding and survey-map tests pass together. Tests include
ASCII null preservation, exact 18-digit IDs, original rows, no-XP eligibility,
duplicate-star counting and protected roles. New scripts compile. This is a
bounded continuation, not a claim that the accumulated tree is fully tested.

## Get the working state

```bash
git clone --branch codex/cloud-resume-20261109 https://github.com/sfabbro/masked-stellar-autoencoder.git
cd masked-stellar-autoencoder
pixi install
pixi run pytest tests/test_pipeline.py -q
pixi run -e monitor python scripts/inspect_msa_handoff.py
```

For an existing checkout, fetch origin and switch to this branch, then fast-forward if the checkout is clean. Do not reset local changes. Push only to sfabbro's `origin`, never `upstream`. Read AGENTS.md. CANFAR credentials must be configured separately on the new computer; credentials and environments are not committed. Read the official `.agents/skills/canfar/SKILL.md` before CANFAR operations. Real model execution belongs on CANFAR CUDA, not the laptop.

The status script is read-only. No job submission is needed simply because you changed computers: accepted jobs continue remotely. Reconcile receipt/session/output before any retry. Completed CANFAR sessions may disappear while their ARC outputs remain.

## Durable sources and execution provenance

ARC root: `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/`.

Authenticated laptop access: `from canfar.storage import filesystem; fs=filesystem('arc')`; paths start `projects/k-pop/...`. In a CANFAR container use `/arc/projects/k-pop/...`. Large catalogues, reconstructed data, full prediction tables and checkpoints remain on ARC. This commit includes code, configs, submission receipts, reports, plots and bounded diagnostic arrays; it does not copy the training catalogues or checkpoints into Git.

Every decision-wave job has a receipt under `experiments/xp/decision-wave-20261106/`. Durable immutable source bundles are under `decision_wave_source_bundles_20261106_v1/<run-or-attempt>/` on ARC. They include source hashes and environment provenance. Generic launchers are `scripts/launch_decision_cpu.py` and `scripts/launch_decision_wave.py`. GPU launcher base is origin's branch `codex/multisurvey-abundance-union-20261028`, pinned commit `60c47fcf609f5ee80a1f985cc95ab8ca648793e9`, with explicit source overlays; it is not an implicit execution of this checkout's HEAD. Torchregress pin: `82ca1996cc4f3ca0be3e730d38a99db482a79378`. Preserve those pins and bundles to reproduce as-run code.

## Current jobs and last verified state

| Run | Session | Last verified state |
|---|---|---|
| expanded_six_nsf_fullfit_20261107_v1, attempt 7 | lohnjtr7 | Training at 71,000 / 71,906 joint updates; epoch 2, one full FIT record epoch complete |
| msa_morning_report_refresh_20261107_v3 | l44pw7oe | CPU-only live report watcher; output is **msa_morning_report_20261107_v3**, not the refresh run name |
| full_gaia_gaia_stage_20261108_v2 | v1kyrpx1 | Source staging complete: 3,386 shards, 1,811,709,771 Gaia rows scanned, all 13,592,882 candidate IDs matched |
| full_gaia_photometry_stage_20261108_v1 | fxc6kwlb | Source staging complete |
| full_gaia_alpha_stage_20261108_v1 | lbhf0ucu | Source staging complete; semantic/quality admission pending |
| full_gaia_ages_stage_20261108_v1 | fvll9eut | Source staging complete; semantic/quality admission pending |
| full_gaia_contract_pilot_20261108_v1 | hshxs209 | Engineering pilot complete: 3 arms × 600 updates, 48,625 matched stars and 164 real no-XP SSL rows |

These are snapshots, not live assertions. Updated manifests are committed under `docs/reports/msa-insights-20261108/<run>.json`. Recheck remote manifests with the status script before acting. The initial Gaia stage v1 assumed `.h5` shards; v2 handles the actual `.hdf5` files. The pilot depends on v2.

The candidate union is from existing registry plus configured native sidecars and native APOGEE/GALAH releases before XP filtering. It is **not** proof that every possible survey label product has been discovered. Gaia reservoir sampling is explicitly stratified by shard/chunk and XP availability, not population-uniform. All-Gaia matching is now demonstrated; complete full-Gaia feature coverage and a production-ready rebuilt dataset are not.

## User's requested final model and accepted decisions

- Six targets: Teff, logg, Fe/H, alpha, age and distance, with a dependent NSF density and whole-vector single-star corner plots. Marginal diagnostics are acceptable but must not replace correlation/nonlinearity inspection.
- Parent and label matching must include all Gaia, independent of XP availability. Missing XP is an optional modality, not exclusion from SSL or label supervision.
- Restore reliable WISE/Gaia/SkyMapper/SDSS/Pan-STARRS/2MASS photometry, native uncertainties and flags. Encode/reconstruct all available bands; include actual no-XP stars and whole-XP modality dropout.
- Add explicit auxiliary head context: EBV, signed measured parallax with error, sky position and proper motion, plus masks. Do not use reciprocal parallax as a distance label. Handle dependency/double-counting when measurements also inform teacher labels.
- Keep masked reconstruction **MAE**, weight **0.03**. Correlated XP reconstruction Gaussian NLL was tried previously and did not improve it.
- Common coordinates must not privilege APOGEE, DESI, StarHorse or seismic data as fixed physical truth. The symmetric overlap-weighted map prototype fixes a gauge; agreement alone does not identify an absolute physical scale. Preserve source systematics and noise/dependencies.
- Alpha is source/pipeline-specific: alpha/M, alpha/Fe and Mg/Fe are not interchangeable. Admit native alpha/individual-element response channels explicitly, not silent renaming.
- Ages are by publication/estimator/release, not just survey. Include separately published VACs. No privileged StarHorse identity reference. Track isochrone/IMF/SFH priors, seismic calibration lineage and shared inputs. Published posteriors are not independent likelihoods; don't multiply or duplicate them as such.
- Retain star-level FIT/TUNE/CAL/DEV separation and protected benchmark/cluster/seismic dependency groups. FIT optimizes parameters, TUNE selects choices/checkpoints, CAL adjusts calibration thresholds, DEV measures held-out behavior; frequent DEV inspection makes it a development set, not a pristine final test.
- QA: native-source and common-reference residuals/coverage for all six targets; metallicity-colored Kiel diagrams; benchmark stars; globular clusters; TESS/K2/Kepler/seismic logg; Andrae and other Gaia-XP catalogues on explicit overlap footprints. Andrae has ~175M entries; a ~10M comparison footprint is our prediction footprint, not the size of Andrae.

## What exists versus what is still missing

`training/full_gaia_inputs.py` implements full-band/no-XP eligibility and explicit auxiliary context. `training/consensus_survey_map.py` implements a symmetric positive invertible own-coordinate spline gauge with no fixed reference survey. These are opt-in; they do not change the older training job.

`run_full_gaia_correction_sources.py` produces native source stages. `run_full_gaia_contract_pilot.py` demonstrates a small fresh MLP encoder/decoder plus auxiliary conditioning, six-dimensional NSF and map integration. It has three arms: restricted inputs, restored inputs, restored inputs + consensus maps. It is **not pretrained SSL fine-tuning**, not a production architecture ranking and not evidence that new alpha/age products are already supervising the model. It still uses the existing admitted label channels. Its restored-input comparison changes several inputs together; finer ablations remain necessary.

The older fullfit job remains a comparison model: 7,693,002 unique FIT stars, 9,203,880 observation records/epoch, existing eight spectroscopic channels plus StarHorse teacher channels and APOKASC gravity repeats. Its encoder effectively sees 110 XP coefficients, W1/W2/G/BP/RP and signed parallax; other photometry and requested head auxiliaries are inactive. Its reconstruction pool is 150M XP-present rows. It fixes APOGEE Teff/FeH, seismic logg, DESI alpha and StarHorse APOGEE age/distance conventions. Do not promote it as fulfillment of the corrected architecture. See `docs/reports/msa-current-production-contract-20261108.md`.

Next work:

1. Inspect completed source manifests and output partitions; quantify unique stars and usable labels after native quality/domain cuts, including XP/no-XP and every band/auxiliary. Preserve exact integer Gaia IDs and original table row provenance.
2. Complete missing native photometric crossmatches, alpha quantity/solar-convention adapters and published-age provenance/dependency audits. Candidate age registry: `configs/published-age-source-registry-20261108.json`. Source extraction alone does not certify quality.
3. Assemble a role-safe corrected full-Gaia labelled/reconstruction source, inherit existing roles, protect benchmarks/clusters/calibrator families and compute FIT-only normalization. Do not use an XP-selected catalogue as the parent.
4. Integrate full inputs/context and the justified survey/noise model into the actual pretrained encoder/decoder + NSF trainer. Run controlled tests for input additions, no-XP dropout, reference gauge, seismic inclusion/mapping, and label error correlation while holding other choices fixed.
5. Read out the older completed fullfit checkpoint and repeat QA on the selected final checkpoint; earlier full-population QA is bound to update 2,000. Epoch-1 QA is bound to update 35,953, not the final model.
6. Launch corrected large training only when the actual data/trainer integrations and QA support it. User authorized autonomous decisions/experiments/launches; no additional permission is needed for routine continuation. Never silently claim an engineering pilot is the final run.

## Scientific findings that must not be lost

Seismic supervision helped protect logg against drift under the changed training recipe. A blanket claim that seismic mapping does not help was corrected: context-dependent mapping showed a modest native-source gain, with seed sensitivity. Compare mapped predictions against source-native labels separately from common predictions against raw labels. APOGEE ingest used calibrated LOGG, so older models can have indirect seismic calibration even without explicit APOKASC records. See `docs/reports/msa-insights-20261108/seismic-mapping-clarification.md`.

Epoch-1 expanded checkpoint versus older c59 NSF, on the same QA stars: APOKASC g MAE improved (.0631 vs .0692), TESS MAE improved slightly (.0847 vs .0869) but NMAD/coverage worsened, and benchmark g MAE worsened (.4305 vs .3903). No universal winner or causal mapping conclusion. See `docs/reports/msa-insights-20261108/epoch1-update.md` and `insights.md`.

Older c59 was already a supervised-adapted encoder, not pristine frozen SSL. Do not describe frozen ablations starting from it as pristine SSL. Multiple earlier reports are historical snapshots and contain superseded statements; prioritize explicit corrections and the most recent bound manifests.

## Validation at handoff

Passed together: **21 tests** across pipeline, full-Gaia contract/source jobs, six-survey maps, NSF confirmation and seismic ablation. New source/pilot scripts compile. This is a WIP snapshot; the whole accumulated research tree has not been claimed fully tested. Real CUDA integration completed remotely. Credential-pattern scan found no likely secrets among new text files. Large scientific catalogues/checkpoints and ignored local environments/caches stay outside Git.
