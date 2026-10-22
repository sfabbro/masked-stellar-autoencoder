# 02 — Failure / repair dossier: prediction & scoring routines (torchregress + MSA)

Author: `failures-audit` (read-only investigation, 2026-10-22). Only file written: this one.
Reading key: **verified** = re-read in code/artifact/git this session; **producer** = number
quoted by a producer report/ledger and corroborated by an independent verifier entry;
**claimed** = single-source, needs re-check. Tolerances quoted as published (`rtol=atol=5e-5`
unless stated). No tolerance was relaxed or "fixed" anywhere in this audit.

## 1. Failure register (sorted by label impact)

| # | Routine / arm (run, session, source·library) | Symptom (exact numbers) | Root-cause status | Preserved artifacts | Blocks a human-goal label? |
|---|---|---|---|---|---|
| F1 | Float32 CPU batch-vs-batch density replay, pure-contrastive NSF `x0m40mgn` (7aeb4ac·1d735c6) | CPU float32 batch128 vs batch1 **fails 1 row, max 7.6294e-5**; end-to-end float64 → **8.17e-14**; manual vs producer float64 EXACT; patched 128-trained-batch clone **1.15e-13** | **Proven mechanism** (float32 context/log-Jacobian amplification; reproduced qualitatively §4) | `scripts/run_multilabel_screen.py:275-346`; 5 regression tests; CI 570p/4s | No — density/NLL ranking only |
| F2 | Strict CPU replay of saved **GPU** densities (held-out verifier) — NSF/BPF/noisy-NSF/anchored/noisy-Gaussian/Gaussian43/long | Retained FAILs: noisyGaussian recon **19/97695** (1.2469e-4); noisy/no-recon **51/97695** (3.109e-4); Gaussian43 **1/97695** (5.4836e-5); long selected **767/97695** (9.35364e-3), cal **14 std / 5 native**; anchored NSF **131 test + 3 cal** (1.67847e-4 / 1.49727e-4); BPF **1 test** (1.02043e-4); noisy-NSF **578+15 intrinsic**, **8+1 observed**; density-wave NLL **1 cal** (8.39e-5), contrastive **1 test** (7.9155e-5); NSF v2 pilot **1/128 cal** (8.39e-5) | **Plausible, not proven**: failures isolate to the *standardized* field; native-unit fields pass; recomputation from saved GPU mean/var passes; unit shift Σlog(scale)=**5.4979676** makes rtol+atol gate non-invariant (variogram toy in log) | per-run `independent-*-report/failure.json`, cached components NPZ, `density-independent-preparation-2026102*` | No — but blocks every flow density/NLL/level-set claim |
| F3 | **Float64 CUDA certificate** of existing checkpoints | Never run; patch is CPU-only ("actual GPU float64 certificate still pending", 1004) | Not attempted | — | No (blocker for F1/F2 closure) |
| F4 | BPF identity tails + ignored requested capacity, `yy7oyuw9` (34e3e29·d94a969) | Pinned BPF extrapolation shifts tails by inverse-solver epsilon → **±1e-5 jump** and non-identity tails; factory silently ignored `n_transforms`/`hidden_features` | **Proven**; fixed in library `8973bcd` (not pinned); fixed map is state-compatible but **changes density values** | MSA 791 MB checkpoint local; `docs/reports/msa-density-review-20261022.html` row "BPF … strict density FAILED" | No point/CP; **yes** for BPF density |
| F5 | Forward inversion `multi_forward_xp_s42_v1` / **npaoc1lo** | Report status `complete_strict_replay_failed`: **29,162/97,695 test + 623/2,046 cal** label diffs >5e-5; grad-norm fails **3,722/62**; Fisher condition **1**; CPU−GPU label median .00418 K/.0000123 dex/.00000617 dex, 95th .09863 K/.000189/.000146, **max 24.0788 K/.02455/.02391**; 4 test objects beyond objective gate; marginal CP passes, **Bonferroni fails 2 dims**; independent relative-stationarity 97.9149%/96.7253% | **Mixed**: local 3-D Gauss–Newton stationarity ≠ global MLE (proven by reported/indep. stationarity fractions); CPU/GPU sensitivity plausible, unproven | `multi_forward_xp_s42_v1/independent-forward-{report,failure}.json`, trajectory cache 85901, CP diagnostic | Yes — but candidate for **retire/demote** (§5) |
| F6 | Forward-EIV screen `forward_eiv_xp_screen_v1` / **p84fpd54** | Manifest **FAILED** at `integrated_flat` GH64/128 gate; mean NLL/XP GH16 .5940319065 → 32 .4822936770 → 64 .4085627239 → 128 .3397031032 → 256 .2554719057 (**not converged**); FlowEB never completed | **Proven**: under-resolved fixed-order GH quadrature | failed manifest/log/checkpoints/pairs collected (73597); quad diagnostic 77574 | Yes (if forward inversion is kept at all) |
| F7 | Scalar contrastive flow + old GH, `prediction_e72_contrastive_flow_xp_10m_v1` / **asuoy0u4** | Obsolete scalar grid-MLE formulation; original grid-MLE GH non-convergence retained | **Proven obsolete** — superseded by native 3-label NSF contrastive (`x0m40mgn`, `pu0bbxpb`) | ledger + density-review "obsolete formulation" note | No → **retire** |
| F8 | Classifier NRE strict replay (4 mechanism runs, exec10452) | index27 CPU standardized MLE **−7.47869253** vs saved GPU **−7.41596079**, one adjacent grid step **0.06273174**, score gap **2.9802322e-8**; both CPU layouts agree | **Proven**: 201-point grid argmax near-tie, not a numerical bug | `independent-verification-failure.json` | No → retire or gate with margin reporting |
| F9 | Flow TTT CP mismatch (Gaussian42 TTT 79314) | Failed final matched-clean CP byte-equality; diagnosis: **3 coordinates of 1 object**, max **4.7683716e-7**, inside 5e-5; producer recomputed clean baseline in a separate forward pass | **Proven verifier-contract error**; fixed via `ttt-matched-clean-predictions.npz`; long TTT 70195 **PASS** (max mean err 1.5378e-5/1.3113e-5) | TTT object caches, `independent-ttt-cp-diagnostic.json` | No |
| F10 | NSF v1 CUDA OOM / **k0vcrws4** | OOM in `output_test→predictions→head.moments` inverse spline (256 fixed nodes) **after** training; recovery v2 `rdu7gowy` chunk16, peak **353,460,224 B**, identical estimator | **Proven**; repaired | v1 manifest/log, v2 collection 30103 | No |
| F11 | MSA pipeline production path | `pipeline/train.py` only counts stars; HATS reader, survey-selection shift, binned leave-one-survey-out tables, torchregress residual plots **unwired** (`docs/pipeline.md:15,55`) | **Proven gap** (not a numerical failure) | `pipeline/steps.py` functions exist unwired | **Yes** — no all-Gaia production train/infer path |
| F12 | Legacy `eval_ensemble`/`infer_msa` numerics + label hardcode | float32 cast + `nan_to_num(-9999)` (`eval_ensemble.py:229-231`); **index 5 = parallax, index 0 = teff hardcoded** (`:51-62`, `:74-95`); `infer_msa.py:111` hardcodes 6 names; no float64 option; batch size/dtype not recorded in metrics | **Proven** | code only | **Yes** for distance/age/alpha and for any float64/GPU certificate |
| F13 | Legacy CQR CP arithmetic (`conformal.py`) | Plug-in `np.quantile(s, 1−α)` instead of finite-sample rank ⌈(n+1)(1−α)⌉: at n=2046, α=0.1 offset differs by **+6.02e-4 scaled units (1 order stat)**; α=1/3 → +6.25e-4; per-tail 1−α ⇒ two-sided nominal ≥1−2α by union bound while docstring claims 1−α; **silent zero offsets when <5 finite rows** (`:60-63`) | **Proven** (arithmetic), impact on current runs small | `tests/test_conformal.py` (does not test ranks) | **Yes** — parallax CQR "90%" is not 90% |
| F14 | Age transform / units | Age = label 4, unit Gyr; only `scripts/run_native_age.py` uses standardized-log Huber + positive exp; **no `age_target_space` in any config**; legacy eval has no positivity/log guard | **Proven gap** | native-age runners verified | **Yes** (age) |
| F15 | Alpha label definition | No native alpha labels trained; APOGEE alpha/M ≠ alpha/Fe; cross-survey scale unaudited | **Proven definitional gap** | BRIEF-verified | **Yes** (alpha) |
| F16 | Distance | Only `joint_prediction.signed_log_distance_nll` (log kpc; `exp(−log d)` mas — **unit math correct**, `:290-297`) + frozen-encoder ridge pilots (4,452 fit / 4,166 HIP refs / 22,932 identity); **no production distance head, no PDF/CP, no parallax-space scoring** | **Proven gap** | distance cohort HDF5s local | **Yes** (distance) |
| F17 | Density level sets & weighted CP | `density_calibration.py` arithmetic is **correct and independently asserted** (exact rank `:31-37`, library==numpy `:220`) but consumes F2 scores; library **weighted** CP radius unverified (per-row weights/classifier state not saved) | Score dependency (F2); weighted CP **producer-only** | `density-calibration-*.json/npz` | No ranking until F2 closed |
| F18 | Other retained failures (training/launch, repaired) | `distance_value_frozen_v1/v2` FAILED before inference at benchmark-ID read (v3 `z38fdw5u` PASS); `joint-native-preparation-v1` source-layout fail; 4 mechanism v1 runs "**Pinned parent digest differs**" (missing `MSA_PRETRAIN_RESUME`); frozen VIR/CNF v1 fail at `np.savez_compressed` (test_z overwritten); CNF NLL control **1/100620** row (0.000235915 std) | All **proven** and repaired in versioned successors; original IDs retained | per-run failure logs/manifests | No |
| F19 | Partial-mask (missing XP) semantics | Legacy path imputes NaN→**−9999 sentinel after float32 cast**, no per-row mask persisted (`eval_ensemble.py:229-231`); pipeline has explicit masks (`pipeline/batch.py:103-157`) but legacy inference does not use them | **Proven gap** | — | **Yes** — "all Gaia source_id incl. no-XP" needs mask-aware inference |

Also **library-version drift in reporting**: `density-review-20261022/metrics.json` records
`evaluated_torchregress_commit = 4e81a72`, while the producers were scored at `d94a969`
(17 native arms) and `1d735c6` (3 density arms) → three library versions in one table.

## 2. Code findings (read-only; `file:line`)

1. `scripts/run_multilabel_screen.py:275-346` — the float64 patch is correct and total (encoder+head
   deep copies, f8 labels/scales/errors, native Jacobian `:337`, saved nodes not resampled) **but**
   `batch=128` is hardcoded and dtype/device/batch are not written to a per-run receipt.
   For `kind=="huber"` the head returns NaN density (`multilabel_experiments.py:232-233`) so
   `predictions.npz` silently stores NaN logs for Huber arms; guards at `:306`/`:329` skip them.
2. `multilabel_experiments.py:114` — `min_variance=torch.finfo(torch.float32).tiny` baked in; under a
   float64 judging path the clamp is still 1.18e-38 (dtype-conditional constant, benign but should be
   derived from the evaluation dtype). `:211-212` accumulates flow moments in float64 then **downcasts
   to `latent.dtype`**, so float32 moments are re-quantized (verifiers must not expect f64 equality).
3. `multilabel_experiments.py:247-266` — CP uses library `MultiTargetConformal` (finite-sample correct),
   Bonferroni via `alpha/3`; fine. `:187-198` contrastive negatives are a cyclic `roll` of the same
   batch (training only).
4. `eval_ensemble.py:46-63,74-95` — label inverse is positional: `i==5` parallax, `i==0` teff; adding
   distance/age/alpha or reordering labels silently mis-inverts. `:229-231` float32 + −9999 sentinel.
   `:566-616` CQR offsets applied in scaled space then inverted (correct order); coverage quoted in
   native units via `interval_coverage`.
5. `infer_msa.py:47-73` — inference never promotes to float64 and never checks batch stability; `:111`
   hardcodes six label names; `:131-137` `present = [c for c in cols if c in names]` silently drops
   missing features instead of failing on schema mismatch (a wrong catalogue dies later in the scaler).
6. `conformal.py:54,64-65` — plug-in quantile (see F13); `:60-63` silent zero-offset fallback;
   `:18-19` two-sided score implemented as two one-sided gaps but the level is not split.
7. `density_calibration.py` — reference implementation of the exact contract (`numpy_threshold` `:31-37`,
   membership `:238`, library-vs-numpy assertion `:216-220`); **this is the arithmetic the strict gate
   should be re-declared against**. `:171-176` uses zero placeholder targets for CTI (documented).
8. `forward_eiv.py:56-96` — fixed 64-node GH with detached local-linear proposal; `:96` normalizes by
   width *after* summing (per-coefficient NLL); `:104-109` rejects non-uniform grids; `:138` boundary
   mass counts 4 nodes only. Rank-deficient/global-mode failure is architectural (F5/F6).
9. `joint_prediction.py:240-302` — signed-parallax Gaussian NLL, `exp(−log_distance)` mas (kpc→mas is
   exact `1/d`), no silent clipping (`:292-294` raises); **no 1000/×d bug found**. `:173-181` and
   `:286-289` promote to ≥float32 and disable autocast (good), but nothing promotes to float64.
   No distance quantile/CP head exists.
10. `prediction_mechanisms.py:47-56` — NRE grid MLE = argmax over 201 points, so a 3e-8 score gap flips
    the label (F8); `:56` grid from fit-label min/max ±0.5. `:60-70` corrected score with `sigma_x=0,
    eps=0` (exact declared sigmas, no invented floor — good).
11. `pipeline/steps.py:26-27` `QUANTILES=(0.16,0.5,0.84)`, `:584-607` per-label `CQR` with finite mask,
    `:623-639` coverage after monotone inverse (guarantee stays scaled space — correct); `:116-183`
    explicit missingness/masking primitives that the legacy path does not reuse (F19).
12. Missing gates: no test asserts (a) library commit, (b) BPF tail identity, (c) CQR finite-sample rank,
    (d) batch/device invariance of densities, (e) label-name/target-space mapping. `tests/` has 24 relevant
    tests (all pass, §4) but none covers these.

## 3. torchregress pin decision

Facts (verified this session): pin `1d735c6a` in `pyproject.toml:28`; installed dist `0.3.0rc1`
`direct_url.commit_id = 1d735c6a…`; local checkout HEAD `0e77eee`, pin **is an ancestor**;
`1d735c6 → 4e81a72 → 2db6fa1 → 8973bcd → e7c446b → 0e77eee` (last two **docs-only**).

| post-pin commit | content | MSA exposure | required for correctness? |
|---|---|---|---|
| `4e81a72` | NFM endpoint direction/time batching, CFM odd width, IRLS diagnostics, **`BinnedDensityLoss` float32→target-dtype edge precision + nonfinite guard** | MSA uses **none** of NFM/CFM/IRLS/`BinnedDensityLoss` (verified grep) | **No today**; **yes before** any future MSA NFM/binned/IRLS arm. (This is exactly the float32-edge class of bug behind F1/F2.) |
| `2db6fa1` | adds `JointBinnedDensityLoss` (multi-target binned CDF) + WALDO critical-value regression | purely additive; MSA uses neither | No |
| `8973bcd` | **`_IdentityTailBPF`** (de Casteljau, exact identity beyond bound, no epsilon jump) + `create_flow_model` **honours `n_transforms`/`hidden_features`** for BPF | `multi_bpf_recon_s42_v1` (BPF, degree 8) is an accepted, re-scorable MSA arm | **Yes** for BPF density correctness (F4) |

Source compatibility: pin→HEAD diff touches only `nflows.py` (factory branch + new BPF classes),
`noise_aware_density.py` (**additive** class + `BinnedDensityLoss` internals), `waldo.py`, `irls_fit.py`,
`flow_matching.py`, `normalizing_flow_map.py`, `losses/__init__.py`, `method_catalog.py`. No signature
change to `create_flow_model`, `NormalizingFlowLoss`, `ContrastiveFlowLoss`, `NoiseAwareDensityLoss`,
`ConformalizedDensity`, `ContrastiveFlowConformal`, `MultiTargetConformal` → **source-compatible with all
current MSA code** (grep-verified call sites: `multilabel_experiments.py:118-134`,
`pipeline/model.py:95-99`, `density_calibration.py:162-165`).
Capacity nuance: MSA requests exactly the zuko defaults (`n_transforms=3`, `hidden_features=[64,64]`,
`degree=8`), so pre-fix "ignored capacity" did **not** change MSA's built architecture — old BPF
checkpoints load under the new pin; only the **tail map changes values** (`_IdentityTailBPF` is
"state-compatible" by construction).
Tests: pin has 75 test files under `tests/{losses,models,algorithms}`, HEAD 77 (+`test_bpf_identity_tails.py`,
+`test_binned_joint.py`); both are **library** tests, not MSA gates. MSA has no test asserting the
installed library commit or BPF tail continuity.
**Decision: bump to `8973bcd` (numerically identical to HEAD; prefer the numerics commit over the docs
commits). Do not "clear" any old output.** Old verdicts are re-adjudicated, not erased: BPF densities
change systematically, so `rescored-v2-lib8973bcd/` outputs are new artifacts with
`supersedes{old_report_sha, old_verdict, reason}` provenance; the old strict FAIL stays visible.

## 4. What was actually reproduced locally (CPU, no checkpoints)

- `pixi run pytest tests/test_multilabel_scoring_precision.py tests/test_multilabel_verification.py
  tests/test_conformal.py tests/test_density_calibration.py tests/test_forward_eiv.py -q` → **24 passed
  in 3.32 s** (no test covers F13 ranks, F4 tails, or batch invariance).
- Installed library commit confirmed `1d735c6a` (`direct_url.json`), `_IdentityTailBPF` **absent**.
- Throwaway `/tmp` synthetic tensors (no checkpoints): float32 vs float64 BPF batch256-vs-batch1
  log-density max |Δ| = **4.768e-06** vs **1.243e-14** (random-init weights; trained flows amplify this
  to the observed 7.6e-5…1.7e-4). Single-row float32−float64 max |Δ| = **1.146e-05**. → **Mechanism
  reproduced qualitatively; production magnitudes inferred from saved artifacts, not reproduced.**
- Same script: plug-in CP quantile under-shoots the exact rank by **+6.02e-4** (α=.1, n=2046) and
  **+6.25e-4** (α=.1/3) scaled units; and `rtol+atol` allowance is **2.60e-4 standardized vs 5.35e-4
  native** at the documented unit shift → gate is not unit-invariant.
- Local checkpoint availability for re-scoring (no retraining): all 12 short native arms **1 `.pth` each**
  (762–795 MB), long arm **31**, Gaussian42 2, xpnoise/mask + MSE imbalance 1 each, forward 1 (486 MB).
  Verified by directory listing only; not loaded (would be a model replay).

## 5. Repair wave (ordered, bounded; each phase has its own gate)

**P0 — re-declare the scoring contract (0.5 d, local, no launches).**
Write the gate before touching models: (a) compare **float64 CPU vs float64 GPU** first (isolates device
effects from precision); (b) apply tolerances to **shift-invariant** quantities (per-object joint *native*
log-density, and the CP/level-set *scores*), never to an offset log-density with a flat rtol; (c) declare
`atol` from dtype (f64: 1e-10 abs on log-density, 1e-12 on means/variances; f32 diagnostics only as
non-gating references); (d) one batch size per artifact, recorded. Deliverable: `SCORING_CONTRACT.md`
+ `tests/test_scoring_contract.py` (asserting the contract itself, synthetic distributions).

**P1 — production float64 scoring + certificate (1 d).**
Refactor `predictions()` into a reusable `scoring.py` used by production (device selectable, batch and
dtype recorded, `kind=="huber"` raises instead of NaN). Emit `prediction-certificate.json` per run:
```json
{"run_id": "...", "canfar_session": "...", "source_sha": "...", "library_commit": "8973bcd...",
 "checkpoint_sha256": "...", "scaler_sha256": "...", "cohort_sha256": "...",
 "labels": ["teff","logg","fe_h"], "units": ["K","dex","dex"], "target_transforms": ["linear","linear","linear"],
 "eval": {"dtype": "float64", "device": "cuda", "batch_size": 128, "torch": "..."},
 "n": {"fit": 19299, "tune": 2118, "cal": 2046, "test": 97695},
 "gates": {"mean_max_abs": 0.0, "var_max_rel": 0.0, "joint_native_logdensity_max_abs": 0.0,
           "cp_order_stat_match": true, "cdf_levelset_match": true},
 "verdicts": {"point": "pass", "density": "pass|fail", "cp": "pass"},
 "supersedes": {"old_report_sha256": "...", "old_verdict": "complete_strict_failed",
                "reason": "float64 device certification under re-declared gate"}}
```
**Coordination (seen in-tree at audit time, not authored here):** `configs/density-precision-certificate-20261022.yaml`
+ `scripts/run_density_precision_certificate.py` + `tests/test_density_precision_certificate.py` implement a
*bounded* CUDA-independent float64 certificate (128 actual rows per checkpoint, 3 density-wave NSF arms) at the
**old pin `1d735c6`**, explicitly "no full-role recertification". That covers part of F3 only: it must be
extended to all families, full roles, and the `8973bcd` pin (P3) before it can close F2/§5-P2; do not treat it
as a substitute for the certificate schema below.
**Gate P1:** for each of ≥1 checkpoint per family (Gaussian, noisy-Gaussian, NSF, MAF, cINN, BPF,
noisy-NSF, long-selected, anchored-NSF): float64-CPU vs float64-CUDA max |Δ| ≤ 1e-10 (log-density) and
≤1e-12 (mean/var); float64-CUDA vs float64-CPU batch1 identical IDs/roles/labels. **No retraining.**

**P2 — versioned re-scoring of the whole inventory (2 d, inference only).**
Re-score all completed arms from their local checkpoints into **new** dirs
`results-stellar-improvement/<run>/rescored-v2-f64-<libcommit>/` (never overwrite; old dirs read-only):
12 short native arms + long selected (31 checkpoints, select the 99M state as already declared) + forward
+ the three density-wave arms. Recompute point MAE, marginal/Bonferroni CP, global+Mondrian 90/95 level
sets, and the FeH<−2 tail (n=141) with `density_calibration.py` as the arithmetic reference.
**Gate P2:** `reconciliation.json` per arm: identical source IDs/roles/labels, point means agree with
saved producer arrays within the *old* declared tolerance, and the old-vs-new verdict delta explained
field by field; every previously-failed row is re-classified `pass|fail` under P0 (expected: F2 residuals
vanish under f64; **no** retroactive "clearance" of the f32 report).

**P3 — pin bump to `8973bcd` + BPF re-adjudication (1 d).**
Bump `pyproject.toml:28`, refresh env lock, rerun full CI (570p/4s baseline). Add
`tests/test_library_provenance.py`: asserts installed commit == pin and BPF tail identity
(`f(x)==x` for `|x|>bound`, continuity at `±bound` to 1e-12, monotone inverse round-trip). Re-score
`yy7oyuw9` (BPF) under the new map in a new dir; keep the old FAIL. **Gate:** CI green + BPF certificate
present + a one-line note in the density review that BPF numbers are lib-version-specific.

**P4 — code-path defects (2–3 d).**
(a) `conformal.py`: finite-sample order statistic `⌈(n+1)(1−α)⌉`, two-sided score `max(q_lo−y, y−q_hi)`
at one level **or** per-tail α/2, small-n raises instead of silent 0; + rank tests. (b) `eval_ensemble.py`
/`infer_msa.py`: label list + per-label target-space/units from config (kill `i==5`/`i==0`), optional
float64 scoring, persist per-row missing-feature mask (replace −9999 sentinel), record batch/dtype/device
in `metrics.json`. (c) `infer_msa.py:131-137`: fail loudly on missing feature columns. **Gate:** new tests
+ a legacy-config regression that reproduces current numbers to <1e-6 in float64.

**P5 — forward inversion: retire/demote, do not patch (decision, 0.5 d).**
Recommend **demote** `multi_forward_xp_s42_v1` (npaoc1lo) to *diagnostic, non-label-bearing*: it is XP-only,
plugin-label, no label-error integration, local-stationarity only, and its 5e-5 label-equality gate is the
wrong contract for an inverse problem. Keep F5/F6 artifacts as failures. Rebuild **only** if a label-bearing
forward/EIV goal is funded: pre-register (i) quadrature convergence gate GH16→256 with a fixed relative
ceiling, (ii) label-error integration, (iii) objective-based (not label-equality) replay, (iv) shift-invariant
gate from P0.

**P6 — new-label readiness (pre-registration, 1 d).**
Distance: score in parallax space against signed raw mas, never naive `1/ϖ`; units kpc vs mas stated;
hold out the 99,574 HIP/seismic reference IDs (already prepared) and report `d`-space MAE **and** ϖ-space
bias; add a distance quantile/CP head — the existing `exp(−log d)` math is correct, do not "fix" a 1000× factor.
Age: declare `age_target_space: log` + positivity guard in config, and report Gyr + log-Gyr coverage.
Alpha: block until the native alpha/Fe source and cross-survey scale are defined (APOGEE α/M is not α/Fe).

**Retire / demote list:** F7 scalar contrastive (obsolete), F8 NRE grid MLE (near-tie argmax; keep with
margin reporting or drop), F5/F6 forward inversion (demote; rebuild only on decision), F4 old BPF
verdict (superseded, not cleared). **Keep and re-score:** every native 3-label arm, CP, level sets,
long-selected point/CP subset.

**Invalidation policy (mandatory):** immutable outputs are never edited or deleted; a repair writes
`rescored-*`/`certificate-*` siblings with `supersedes` provenance and the library/source/dtype triple.
Any claim made after repair must cite the certificate SHA, not the old manifest. The human-goal labels
(Teff/logg/FeH point+CP) may proceed on certificate P1/P2 output; **density/NLL/level-set rankings stay
blocked** until P2+P3 certificates exist, and distance/age/alpha labels stay blocked until P4/P6.

## 6. What this implies for the plan

1. The failures are **two families**, not one: (i) *precision/contract* failures that are fixed by
   float64 scoring + a re-declared shift-invariant gate with **no retraining** (F1–F4, F9, F17), and
   (ii) *method* failures that a tolerance change cannot fix (F5–F8: under-resolved integration, local
   stationarity, grid near-ties) — schedule them as retire/demote, not as "repair".
2. The current point/CP numbers for Teff/logg/FeH are not blocked by any failure: they rest on
   means + `MultiTargetConformal` that passed independent replay. Only the **density/NLL ranking** is blocked.
3. Pin bump is cheap and safe (`8973bcd`, source-compatible, state-compatible), but it **changes BPF
   values** — so pin it *with* a versioned re-score, never as a silent upgrade behind old claims.
4. The largest code debt for the human goal is not numerical: it is the **legacy label hardcoding and
   the missing production inference path** (F11/F12/F19) — without it, distance/age/alpha and the
   all-Gaia (no-XP) product cannot be produced at all.
5. Estimate: P0–P3 ≈ 4.5 working days local CPU (plus one short GPU float64 job once the 150M session
   frees), P4 ≈ 2–3 d, P5/P6 decisions ≈ 1.5 d. Everything up to P2 requires **no retraining**.
