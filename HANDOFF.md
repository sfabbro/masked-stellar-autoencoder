# MSA handoff — 2026-11-08, Vancouver

Start here when resuming on another computer. This is a WIP research handoff, not a release or a claim that the corrected full-Gaia model has been trained.

## Cloud resume — 2026-11-09, Vancouver

The cloud checkout initially contained `0c01ce7`; it was fetched and
fast-forwarded to origin/main `6358d94` before continuing. Git fetch and a
push dry run to a new `codex/` branch succeeded. `gh auth status` separately
reports an invalid `GH_TOKEN`; Git transport success does not prove API access.

CANFAR 1.4.1 is installed in `/workspace/.venvs/canfar`, at the monitor
environment's pinned source revision
`0d96a72761cc6dc7063d6cd97049ae3743133efa`. Use
`/workspace/.venvs/canfar/bin/python scripts/inspect_msa_handoff.py` after
authentication/network setup. Installation used a separate uv environment and
`UV_CACHE_DIR=/workspace/.cache/uv`; the existing scientific environment was
left intact. Pixi's executable is absent in this cloud image, but its existing
environment's Python is available at `.pixi/envs/default/bin/python`.

Live CANFAR reconciliation remains blocked: proxy CONNECT requests to
`ws-uv.canfar.net` and `ws-cadc.canfar.net` returned 403, and no CADC certificate
is configured. Add CANFAR/CADC service domains in environment settings and bind
credentials through secrets, never Git or chat. The installed client's default
certificate selector is `~/.ssl/cadcproxy.pem`; configure its path to a supplied
secret file when available. No job was launched or retried. The referenced
`canfar-lab-workflow` skill is not present in this cloud skill catalog or local
skill search; recover it before CANFAR execution. The job states below remain
historical snapshots.

The full-Gaia input policy now supports explicit `encoder_parallax=True` while
keeping reconstruction restricted to mags+XP and retaining parallax in head
context. Eligibility still requires photometry or XP; parallax alone does not
admit a source. The default and existing pilot remain unchanged. Any trainer
using this option must use `policy.reconstruction` for decoder columns, rather
than assuming its columns equal `policy.encoder`. This is a policy extension,
not an integrated pretrained six-target training run.

## Get the working state

```bash
git clone --branch codex/multisurvey-flexible-20261105 https://github.com/sfabbro/masked-stellar-autoencoder.git
cd masked-stellar-autoencoder
pixi install
pixi run pytest tests/test_pipeline.py -q
pixi run -e monitor python scripts/inspect_msa_handoff.py
```

For an existing checkout, fetch origin and switch to this branch, then fast-forward if the checkout is clean. Do not reset local changes. Push only to sfabbro's `origin`, never `upstream`. Read AGENTS.md. CANFAR credentials must be configured separately on the new computer; credentials and environments are not committed. Read the canfar-lab-workflow skill before CANFAR operations. Real model execution belongs on CANFAR CUDA, not the laptop.

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
