# 06 — CANFAR experiment campaign: named, sized, ordered, gated

Author `campaign-design`, 2026-10-22, read-only at HEAD `0476628`; only file written is this one. `[V]` read in
code/config/manifest this session · `[P]` producer number (manifest/receipt) · `[E]` explicit estimate, never quotable
as measured. Companion to `PLAN.md` §WS3/M3 — this is the run-level execution plan for the ledger. Every run: 1 GPU
(MIG 1g.12GB = 12 GiB) + 8 CPU + 64–137 GB RAM, `kind=headless`, image `images.canfar.net/astroai/base:latest`,
own output dir, never overwritten.

## 0. Contract for every run

1. **Matched unit = 10M presentations = 306 updates** (10e6/32768, `configs/multilabel-screen-20261020.yaml:8-10`);
   270M = 8,250 updates; supervised = 8×256 = 2,048 fit-row draws/update ⇒ **306 updates ≈ 32 passes over the 19,299 fit
   labels, 8,250 ≈ 875 passes** `[V]` (`run_multilabel_screen.py:943-961`). Presentation budget alone cannot move the
   metal-poor tail; W3–W5 must change inputs/targets, not just spend compute.
2. Frozen bindings unless a row says otherwise: parent86 `d1ed550e…fb3`; cohort `321f2723…120`; sidecar `3998fe21…85b`;
   benchmark ordered-id `41c459bd…abc`; native3 19,299/2,118/2,046/97,695 (tail FeH<−2 **141**, <−3 **14**). Latest-big
   runs use `checkpoint-002250000000.pth` `a0191929…c24` now, re-pointed to the ~4.5B final after 24 Oct.
3. **No run is scored before W0 closes for its family.** A manifest without `prediction-certificate.json` is
   `producer-only` and cannot enter a ranking; no tolerance relaxed, old FAILs visible, test read once, selection on tune.
   Cohorts never mix: native3 · scalarFeH (100,620/185) · distance (4,452 fit / 22,932 identity / 4,166 Hipparcos, 159
   non-positive ϖ) · age (4,440 fit / 22,830 identity / 50 seismic available). Seed 42; 43/44 only for finalists.

## 1. Runnable infrastructure (read, not executed)

| runner | stage binding | knobs it really has | binds to | budget semantics | cannot express |
|---|---|---|---|---|---|
| `run_multilabel_screen.py` | `multilabel-screen`, `-density-screen`, `-long` (`canfar_entrypoint.sh:131-149`) | `--run-id --output --config --checkpoint{,-cohort,-sidecar,-native-qa,-benchmark} --smoke`; arm keys `kind ∈ {huber,gaussian,flow}`, `flow_type ∈ {nsf,maf,cinn,bpf}`, `flow_objective ∈ {nll,contrastive,nll_contrastive}`, `contrastive_weight`, `noise_aware`, `reconstruction_weight`, `ttt`, `library_tta`, `seed` | native3 complete case; parent86 family | `reconstruction_presentations` ÷ `effective_batch`; `checkpoint_interval_presentations` + `selection_policy: reserved_tune_mean_observed_nll` | full-cov, low-rank, MDN, quantile, expectile, SLS, SLSConformal, NFM/CFM, 6-target, survey LOO, mask indicators, peak-GPU receipt |
| `run_prediction_comparison.py` | `prediction-comparison`, `targeted-prediction`, `large-huber`, `prediction-continuation` (`:189-208`) | `--run-id --output --config --checkpoint --benchmark`; `head ∈ {residual,mlp,batchensemble,neural_ratio,contrastive_flow}`, `loss ∈ {huber,mse,gaussian_nll,gaussian_crps,lds_huber,group_dro_huber,gaussian_nll_label_error,bmc,mse_ranksim,corrected_score,contrastive_flow}` | scalar 3-label + 150M cache (`residual-huber-150m.yaml`) | `reconstruction_presentations` 10M…4.5B; `early_checkpoint_presentations` | joint densities; distance/age/α; astrometry tiers |
| `run_joint_continuation.py` | **no CANFAR stage — pixi only** (`docs/joint-continuation-30e.md:193`) | `--config --resume --smoke --stop-after-steps`; `epochs`, `reconstruction_rows_per_epoch`, `distance_enabled`+`parallax_weight`, `include_reddening`; `include_positions`/`include_proper_motions` **raise** (`:81`), `age_enabled`/`alpha_enabled` **raise** (`:85`), `min_gpu_memory_gib` | parent73 `fdbee7ba…` per config; sidecar v1 (unsafe) | `epochs × rows_per_epoch`; fail-closed on staging/promotion gates | PM/position inputs, age/α heads, 6-target |
| `run_native_age.py` · `run_distance_value.py` · `run_pristine_value.py` | `native-age` `:224` · `distance-value` `:219` · `pristine-value` `:229` | **no config**: run-id whitelists; age `frozen|sft|recon` on standardized-log Huber; distance `parallax_free|parallax_assisted` ridge α=10, zero SGD | parent57 `9b033ae6…` (ARC-only) + their cohorts | age = hardcoded steps/warmup; distance frozen encoder, fit 4,452 | age NLL/quantile/anchors; distance head/quantiles/CP/PM/EBV/zero-point; new parents |
| `run_pretrain_experiments.py` | `pretrain-experiments` `:284` | `--experiments --train-rows --presentations --micro-batch-size --runs --evaluation-only` | legacy 140-slot in-RAM loader | `--presentations`, `--train-rows` | 1.8B scale; joint heads |
| `run_density_precision_certificate.py` | `density-precision-certificate` `:126` | `--run-id --output` + `configs/density-precision-certificate-20261022.yaml` | 3 NSF ckpts × **128 rows**, pin `1d735c6` | zero updates, bounded | full roles, other families/pins (W0 extends) |
| `launch_stellar_improvement.py` | all | per-job `resources {cores,ram,gpu}`, `env` overrides, idempotent accepted-ID ledger | `configs/stellar-improvement-plan.yaml` | one session per run | no wall field — enforce `max_wall_hours` in config |

## 2. Measured cost basis (use these; do not invent)

Measured GPU wall time, MIG 1g.12gb, 10M/306 updates `[P]` (`experiments/xp/results-stellar-improvement/*/manifest.json`):
Huber 931 s · diag-Gauss recon 934/951/897 s (s42/43/44) · noise-aware recon 944 s · diag no-recon 403 s · noise-aware
no-recon 411 s · MAF 1,004 · cINN 998 · BPF 2,530 · NSF NLL 1,808 · NSF contrastive 1,849 · NSF NLL+con 1,854 · NSF recon
v1 1,338 (v2 chunked 3,660 after the F10 OOM repair) · noise-aware NSF 1,749 · forward-XP 391 (0 presentations).
270M: noise-aware diag 16,572 s · residual-Huber long 15,931 s. 512-presentation smokes 11–14 s.

**Cost model** `[E]` from the two noise-aware points: `t[s] ≈ 342 + 60.1 × (P/1e6)`; steady state 16.6 k presentations/s
= **59.8 M/h**, matching `dd27e718`'s 58–62 M/h. Family multiplier on (t−342): Huber 0.98 · MAF 1.10 · cINN 1.09 · NSF-NLL
2.44 · NSF+con 2.51 · noisy-NSF 2.34 · BPF 3.64. **No manifest of the 22 arms carries a peak-GPU field** `[V]` (only the
flow-moment preflight writes `peak_allocated_bytes`, `run_multilabel_screen.py:931`) → W0 makes it mandatory. Memory class
`[E]`: diag/full-cov/quantile/MDN ≤6 GiB at `micro_batch 16384`; NSF/MAF/cINN/BPF/MDN-8 8–12 GiB (BPF and the 256-node
inverse spline bind, cf. F10). **Policy: lower `micro_batch` to 8192 before ever requesting a bigger GPU.**

## 3. Code gaps and pre-launch tests (fail-closed)

| gap | file(s) | minimum before launch | run |
|---|---|---|---|
| head kinds full-cov/low-rank/MDN/quantile/expectile/SLS | `multilabel_experiments.py:84,104-106` (16-output `flow` branch is the template); `run_multilabel_screen.py:876-884` | extend kind union + output dims; one synthetic test per family (finite loss, analytic NLL on known Σ/π/τ) + `--smoke` | W1 |
| SLSConformal / Mondrian / weighted CP | `torchregress.losses.conformal` exists at installed pin `1d735c6` `[V]` | post-fit only; arithmetic reuses `density_calibration.py:31-37` exact rank | W1/W0 |
| NFM/CFM | `flow_objective` branch; endpoint + odd-width fixes are **post-pin** | pin `8973bcd` (R0.4) then a 2-epoch endpoint-direction smoke | W1 |
| joint continuation has **no CANFAR stage** | `canfar_entrypoint.sh`, `launch_stellar_improvement.py` | add `joint-continuation` stage + plan entry; refuse if `MSA_PRETRAIN_RESUME` ≠ config parent SHA | W3 |
| 10M/20M cache staging | `stage_joint_reconstruction.py`, `training/reconstruction_sample.py` | CPU staging job bound to parent run signature; fill `sample_request_sha256`/`pilot_policy_sha256`/`promotion_report` | W3 |
| `include_positions`/`include_proper_motions` raise (`:81`) | `run_joint_continuation.py` | implement, or drop the context tier and say so | W3/W4 |
| age NLL/quantile head; distance quantile/CP head | `run_native_age.py` whitelist; `joint_prediction.py` (only full-cov + signed-ϖ point NLL) | new heads + `age_target_space: log` guard/positivity; distance CP **in parallax space** | W3 |
| 6-target partial label | `joint_prediction.py:60` (`target_dim ≤ 8` OK); `TARGET_NAMES` in `multilabel_experiments.py` | observed-mask path exists; block until α/age labels exist | W3 |
| survey LOO / mask indicators | `pipeline/splits.py:53` (unwired), `pipeline/steps.py:116-183` | wire into role construction + 100-row synthetic test | W4 |
| peak-GPU/dtype/batch receipt · BPF identity tail + library provenance | screen manifest; `run_joint_continuation.py:973` has it · `pyproject.toml:28` | add `peak_allocated_bytes`/dtype/device/batch to every manifest; `tests/test_library_provenance.py` with `f(x)=x` beyond bound to 1e-12 | W0 |

**In-flight — adopt, do not duplicate:** `density_fp64_precision_certificate_v1` / session `wetmolvn` (3 NSF ckpts × 128
actual rows, pin `1d735c6`, zero updates, source `65e2243`, Completed 2026-10-22 ~15:37 UTC, output not yet fetched)
`[P]`. W0.1 verifies it; W0.2 extends it to all families + full roles at `8973bcd`.

## 4. Campaign waves
### W0 — Repairs and certification (inference only; no training, no new labels)
| run | closes (report line) | runner / path | GPU-h | gate |
|---|---|---|---|---|
| `w0_cert_adopt_wetmolvn_v1` | bounded certificate (29) | fetch + independent recompute from `density_fp64_precision_certificate_v1/`; **no submission** | 0.1 | output SHA + 3-case reconciliation vs producer log |
| `w0_cert_allfamilies_v1` | full-role FP64 certificate (30/31) | `run_density_precision_certificate.py` + extended config: 9 families × full roles at `8973bcd` | 2.5 | f64-CPU vs f64-CUDA ≤1e-10 log-density, ≤1e-12 mean/var; batch-1 identity |
| `w0_rescore_inventory_v1` | versioned rescoring (30) | new `scripts/scoring.py` → `rescored-v2-f64-8973bcd/`: 18 10M arms + long 99M + forward + 3 density arms | 4.6 | `reconciliation.json`; ids/roles/labels identical; old FAIL retained |
| `w0_rescore_verify_v1` | independent verify + global/tail metrics (31) | `density_calibration.py` on arrays only (no ckpt) | 0.0 | two independent paths agree; 141-vs-145 mask stated |
| `w0_bpf_map_v1` | corrected BPF tail map + provenance (32) | rescore `multi_bpf_recon_s42_v1` at `8973bcd`, new dir | 0.3 | `f(x)=x` beyond bound to 1e-12, C⁰ at ±bound, old verdict kept |
| `w0_recon_source_v1` · `w0_mask_provenance_v1` | raw recon-source + CUDA mask provenance (34) | `row_sha256`/`reconstruction_source` audit vs benchmark train 9M + staged chain; replay `mask_rng` state CPU-vs-CUDA, record generator device | 0.2 | row-ID set equality + SHA chain; mask stream byte-equal or a device-independent seed policy declared |
| `w0_ttt_fullrole_v1` | flow TTT checks (35) | matched-clean protocol over NSF/cINN/MAF/BPF, full roles | 0.3 | CP byte-equality vs matched-clean baseline |
| `w0_weighted_cp_v1` | weighted-CP provenance (36) | save per-row weights + domain-classifier state/seed, recompute radius | 0.2 | weight/state SHAs + recomputed radius (subset geometry is not proof) |
| `w0_mondrian_cp_v1` | Mondrian CP metric layer | `density_calibration.py` + reporting on FeH<−2, logg branch, G, tier | 0.0 | strata n≥200, coverage ±0.05 |
| **W0 total** | 10 run IDs, 0 training updates · repairs #5 forward inversion = **RETIRE** (demote to diagnostic: `npaoc1lo` local-stationarity only, 29,162 label deltas) | | **8.2 h (0.34 GPU-day)** | exit: certificate per family + reconciliation + incumbent tag |

### W1 — Head/loss screen at 10M (matched: parent86, native3, 9M benchmark rows, 306 updates, recon w=0.03, `ttt=false`, seed 42)
| run | head / loss (library class) | GPU-h | reads as |
|---|---|---|---|
| `w1_quantile_s42` · `w1_expectile_s42` | `MultiQuantileLoss` 0.16/0.5/0.84 + crossover · `MultiExpectileLoss` (3 τ) | 0.50 | pinball/point · tail shape |
| `w1_fullcov_s42` · `w1_lowrank_s42` | `MultivariateGaussianLoss` (Σ lower-tri) · `LowRankGaussianLoss` rank 2 | 0.50 | joint NLL + level set |
| `w1_mdn4_s42` · `w1_mdn8_s42` | `MDNLoss` K=4 · K=8 | 0.65 | mixture NLL/CRPS, capacity |
| `w1_sls_s42` · `w1_slsconformal_v1` | `SLSLoss`; `SLSConformal` level-set CP on SLS + W1 top-2 (post-fit) | 0.35 | score + coverage/region size |
| `w1_nfm_s42` · `w1_cfm_s42` | NFM (post-`4e81a72` endpoint fix) · CFM (odd width) | 1.00 | joint NLL |
| `w1_fullcov_norecon_s42` · `w1_mdn4_norecon_s42` | no-reconstruction controls (w=0) | 0.35 | controls |
| `w1_diagG_s42` | **reuse** `multi_gaussian_recon_s42_v1` (incumbent) | 0 | control |
| **W1 total** | 12 new IDs (11 train + 1 post-fit); 12 arms reused (diag s42/s43/s44, Huber, noise-aware diag, diag no-recon, NSF NLL/contrastive/NLL+con, cINN, MAF, BPF-rescored) | **3.5 h (0.15 GPU-day)** | gate: a family advances only with its W0 certificate **and** no point-MAE regression vs `w1_diagG_s42` |

### W2 — Matched comparisons and distribution metrics (mostly inference on W1 artifacts)
| run | purpose (report line) | GPU-h | gate |
|---|---|---|---|
| `w2_nsf_noise_norecon_s42` | **new train**: noise-aware NSF, w=0, `ttt=false` | 0.50 | matched clean/noisy pair on identical rows |
| `w2_clean_vs_noise_nsf_v1` | clean vs noise-aware NSF after corrected scoring (54) | 0.20 | ΔNLL CI excludes 0 **and** no point-MAE regression |
| `w2_objective_threeway_v1` | NLL vs pure contrastive vs NLL+0.1 + Mondrian density-CP (55) | 0.30 | three-way same rows; strata reported |
| `w2_ensemble_mixture_v1` | 3-seed Gaussian scored as a **predictive mixture** (56) | 0.30 | mixture NLL/CRPS/PIT; no calibrated-spread claim |
| `w2_sample_metrics_v1` · `w2_region_size_v1` | energy/variogram + marginal CRPS/PIT/quantile from samples (57/58); 90/95 % level-set size beside coverage (59) | 0.50 | no Gaussian proxy for a flow; size+coverage together |
| `w2_fullcov_s43` · `w2_fullcov_s44` · `w2_mdn4_s43` · `w2_mdn4_s44` | replicated seeds of the two passing families (60) | 1.00 | tune-only selection; seed spread reported |
| **W2 total** | 10 run IDs (5 train, 5 analytic) | **2.8 h (0.12 GPU-day)** | exit: one certificate-backed density ranking |

### W3 — Target-specific heads (20M presentations, latest big state, joint-continuation + staged 20M cache; distance cohort for `w3_dist_*`, age cohort for `w3_age_*`)
Prerequisite **W3.0 plumbing** (CANFAR stage, staging job, heads, `age_target_space: log`) = 3–4 working days, on the critical path.

| run | head / loss / inputs | GPU-h | gate |
|---|---|---|---|
| `w3_dist_featonly_20m` · `w3_dist_assist_20m` | `distance_enabled`+`parallax_weight>0`, astrometry masked · measured ϖ as declared **measurement-assisted** tier | 1.20 | no regression vs raw Gaia **0.960 mas** (4,166 HIP) in high-S/N; gain in low-S/N and d>3 kpc; tiers never mixed |
| `w3_dist_context_20m` · `w3_dist_zp_20m` | PM/position/EBV context (needs the `:81` guard) · audited per-star zero-point tensor in `signed_log_distance_nll` | 1.20 | 03 §5 falsifiers (map-shuffle, d>3 kpc only); ϖ-space bias CI |
| `w3_dist_cp_v1` | distance quantile/CP head **in parallax space** (post-fit) | 0.30 | CP90 in mas; never `1/ϖ` as truth |
| `w3_age_lognll_20m` · `w3_age_quantile_20m` | log-age NLL with positive output/asymmetric percentiles · 0.16/0.5/0.84 age quantiles | 1.20 | signed bias vs 37 quality seismic (50 available) **and** MAE improve vs `native_age_strict_recon_v1`; no "true age" |
| `w3_joint6_partial_20m` | 6-target partial-label joint head (3 labels + ϖ + age + α) | 0.80 | **blocked** until α/age labels pass L1 gates; mask path already exists |
| `w3_dust_av_20m` | joint d–A_V with 3D-map prior | — | **deferred**: no 3D map ingested (§7) |
| **W3 total** | 8 run IDs (7 train, 1 post-fit), 1 deferred | **4.7 h (0.20 GPU-day)** + one CPU staging job (11.2 GB) | exit: distance beats raw Gaia only where declared, else ships as prior |

### W4 — Input / architecture ablations (20M presentations, latest big state, native3 cohort unless noted)
| run | axis | GPU-h | gate |
|---|---|---|---|
| `w4_xp_off_20m` · `w4_mask_ind_20m` | all 110 XP masked · explicit missingness indicators (replace −9999) | 1.20 | per-tier ΔMAE; identical-mask regression vs incumbent |
| `w4_loo_galah_20m` · `w4_loo_apogee_20m` | leave-one-survey-out training (wire `pipeline/splits.py:53`) | 1.20 | per-survey ΔMAE; no merged-scale claim |
| `w4_recon001_20m` · `w4_ebv_20m` | reconstruction weight 0.01 (default 0.03) · `include_reddening: true` (2D SFD) | 1.20 | recon is an aux task; EBV gain only if extinction quartiles improve without calibration loss |
| `w4_parent_latest_20m` · `w4_long_150m` | parent 2.25B/4.5B state vs parent86 on the same stream · 150M presentations (1 pass over the 87 GB cache) | 3.60 | isolates parent from budget; does budget buy anything at fixed labels? |
| `w4_ens_20m_s43` · `w4_ens_20m_s44` · `w4_astro_ctx_age_20m` | two extra seeds · astrometry context **only** for the age head | 1.80 | ensemble beats single seed on tune; context declared as extra information |
| **W4 total** | 11 run IDs | **9.0 h (0.38 GPU-day)** | exit: dust/PM/position claim resolved one way or the other |

### W5 — Finalists (latest 4.5B state, native3 → `native3_v3` if L1.3 lands; 3 seeds; calibration; locked test read once)
| run | setting | GPU-h | gate |
|---|---|---|---|
| `w5_final_20m_s42/s43/s44` | winning head + inputs, 20M, 3 seeds | 1.80 | tune-selected; seed spread reported |
| `w5_final_150m_s42` (+`s43` if tune improves) | 150M presentations | 6.00 | only if 150M > 20M on tune |
| `w5_calibrate_v1` · `w5_lockedtest_v1` | Mondrian CQR/level-set calibration on cal split · single locked-test read + scorecard + F01–F14 | 1.00 | coverage ±0.02 marginal / ±0.05 Mondrian n≥200; pre-registered, no re-read |
| **W5 total** | 7 run IDs (5 train, 2 read/report) | **8.8 h (0.37 GPU-day)** | exit: finalist + certificate + scorecard + honest α/distance/age status |

**Campaign totals:** 58 run IDs (39 train-bearing, 19 inference/analytic) · ≈37 GPU-h ≈ **1.5 GPU-day raw**, budget
**3 GPU-days** with scoring/re-runs · 39 new states × 410 MB ≈ 16 GB. **Critical path:** W0 certificates → W1/W2 ranking →
**W3.0 code** → W3 → W4 → W5. W4-parent and W5 wait on `dd27e718` (24 Oct); W0–W2 can use a second session from 23 Oct.

## 5. Feasibility, staging, storage, GPU memory, scheduling

- **Sessions:** headless deadline ≈14 d (`dd27e718` expiry ~3 Nov); one GPU per session, several sessions allowed within
  quota; the launcher has **no wall field** → set `max_wall_hours` per config and stop via `--stop-after-steps`.
- **No collision:** `dd27e718` is Running at 2.25B/4.5B (22 Oct 14:20), ETA 24 Oct 03–06 UTC. No new GPU session before
  23 Oct 18 UTC; W0 CPU work 22–23 Oct; short GPU W0 jobs 23–24 Oct; nothing resubmits `dd27e718` or `wetmolvn`.
- **Staging & storage:** 10M benchmark 121 GB (ARC) · 150M cache ~87 GB · a 10M/20M f32 cache is ~5.6/11.2 GB, built
  CPU-only by `stage_joint_reconstruction.py` bound to the parent run signature, never while holding a GPU. Each state is
  410,367,373 B ⇒ ~16 GB new states + ~18 GB saved arrays (0.3 GB/arm), all on `/arc/projects/k-pop`; `/scratch` = caches.
- **GPU memory risk:** full-cov/MDN/quantile only widen outputs ⇒ ≤6 GiB; the binding risks are NSF/BPF test-time spline
  inversion (F10: 256 nodes OOM, chunk-16 repair peak 353 MB) and the 150M-row joint continuation (`micro_batch 8192`,
  config demands 12 GiB). A 24/40 GB request needs a measured `peak_allocated_bytes`; none exists, so W1 decides it.
- **Labels land in parallel:** campaign is label-light by design; swap-in points are W3 α/6-target on L1.2+L1.11, and
  W5 cohort → `native3_v3` if the DESI-FeH pool (L1.3) lands before 20 Nov, else W5 ships on native3 with the 141-star
  tail caveat stated.

| window (2026) | work | exit |
|---|---|---|
| 22–23 Oct | W0 CPU runs (recon-source, mask, tail reconciliation, contract); fetch `wetmolvn` | W0.1 verified |
| 23–24 Oct | `w0_cert_allfamilies_v1` (fresh 1-GPU session); 24 Oct `dd27e718` completes → preserve 32 states | certificate per family |
| 24–26 Oct | `w0_rescore_inventory_v1` + BPF + TTT + weighted/Mondrian CP | reconciliation; freeze incumbent tag |
| 27–29 Oct | W1 screen (12 arms, 2 sessions × 6) | family gate |
| 30 Oct–1 Nov | W1 analysis; `w2_nsf_noise_norecon_s42` + W2 seeds + analytic runs | one certificate-backed ranking |
| 2–5 Nov | **W3.0 plumbing** (stage, heads, staging job, target spaces) | staging hash + smoke |
| 6–8 Nov | W3 distance (4) + age (2) arms | distance/age gates |
| 10–13 Nov | W4 ablations (11 runs, 3 sessions) | input verdicts |
| 16–20 Nov | W5 20M × 3 seeds; promote 150M if tune says so | finalist + calibration |
| 23–26 Nov | locked-test read once, scorecard, F01–F14 | campaign close |
| 27–30 Nov | buffer: for-cause re-runs only | ledger + report |

## 6. Coverage matrix — every checklist item in `docs/reports/msa-density-failures-and-missing-experiments-20261022.md`
| item (line) | disposition | run ID(s) |
|---|---|---|
| bounded FP64 CUDA certificate `wetmolvn` (29) | adopt + verify, do not duplicate | `w0_cert_adopt_wetmolvn_v1` |
| versioned full-test/cal rescoring Gaussian+NSF (30) | do | `w0_rescore_inventory_v1` |
| independent verify + global/metal-poor metrics (31) | do | `w0_rescore_verify_v1` |
| corrected BPF tail map + library provenance (32) | do | `w0_bpf_map_v1` |
| forward-inversion replay/stationarity (33) | **RETIRE** (demote to non-label-bearing diagnostic; method failure F5/F6, not a tolerance fix; rebuild only if a label-bearing forward goal is funded) | — |
| raw benchmark recon-source + CUDA mask provenance (34) | do | `w0_recon_source_v1`, `w0_mask_provenance_v1` |
| flow TTT independent checks (35) | do | `w0_ttt_fullrole_v1` |
| weighted conformal provenance (36) | do | `w0_weighted_cp_v1` |
| SLS (42) · SLSConformal (43) · MDN (44) · full-cov (45) · low-rank (46) | do; SLSConformal post-fit only | `w1_sls_s42`, `w1_slsconformal_v1`, `w1_mdn4_s42`, `w1_mdn8_s42`, `w1_fullcov_s42`, `w1_lowrank_s42` (+`_norecon` controls) |
| quantile (47) · expectile (48) | do | `w1_quantile_s42`, `w1_expectile_s42` |
| NFM (49) · CFM (50) with corrected implementations | do after pin `8973bcd` | `w1_nfm_s42`, `w1_cfm_s42` |
| clean vs noise-aware NSF after corrected scoring (54) | do | `w2_clean_vs_noise_nsf_v1`, `w2_nsf_noise_norecon_s42` |
| NLL vs contrastive vs NLL+contrastive + Mondrian density-CP (55) | do | `w2_objective_threeway_v1`, `w0_mondrian_cp_v1` |
| distributional ensembles as predictive mixtures (56) | do | `w2_ensemble_mixture_v1` |
| predictive-sample energy/variogram (57) · marginal CRPS/PIT/quantile (58) | do | `w2_sample_metrics_v1` |
| density-region size alongside coverage (59) | do | `w2_region_size_v1` |
| replicated seeds for promising systems (60) | do | `w2_fullcov_s43/s44`, `w2_mdn4_s43/s44`, W5 seeds |
| population-wide **and** FeH<−2 separately (64) | do — reporting gate on every run | all; enforced via `w0_rescore_verify_v1` + `w0_mondrian_cp_v1` |
| cohorts kept separate, native 97,695/141 vs scalar 100,620/185 (64) | do — contract §0.4 | all; never merged |

24 items done, 1 retired with reason, 0 silently dropped.

## 7. Impossible with current data, and the smallest unblock

| blocked | why now | smallest dependency |
|---|---|---|
| joint distance–A_V (`w3_dust_av_20m`) | no 3D dust map ingested, no package pinned; 2D SFD EBV flagged `safe_for_unqualified_physical_extinction_input: false` | ingest one named 3D map (Edenhofer/Green/Baystar) + per-star A_V uncertainty → 1 CPU job + a validity flag |
| α head (`w3_alpha_tagged_20m`, inside `w3_joint6_partial_20m`) | **0 α rows anywhere**; APOGEE α/M ≠ α/Fe; DESI `ALPHAFE` dropped at ingest | L1.2 re-read of the checksummed DESI FITS (hours) + L1.11 scale audit; no new code after that |
| metal-poor tail claims (all waves) | native3 tail 141 (14 below −3); one star moves coverage 0.7 pp | L1.3 DESI-FeH pool (+78 k at <−2): policy + calibration, not acquisition |
| age truth · all-Gaia no-XP product | 4,440 fit rows on StarHorse posteriors, 37 quality seismic refs · legacy loader ~117 GB per 10M rows | L1.8 cluster/APOKASC anchors or DR4 FLAME (2 Dec 2026) · WS2 data plane (out of scope; `w4_xp_off_20m` only measures the cost) |
| flow NLL / level-set ranking | every flow arm has retained strict-replay failures | `w0_cert_allfamilies_v1` at `8973bcd` (this campaign) |

## 8. What the Lead must put in the plan

1. **W0 is the launch gate**: no new training run before `w0_cert_allfamilies_v1` + `w0_rescore_inventory_v1`; this
   58-ID campaign and its gates replace PLAN.md §WS3's generic bounded matrix.
2. **State the matched unit** (10M presentations = 306 updates = 32 fit-label passes; 8,250 updates = 875 passes) and the
   consequence: presentation budget cannot move the tail — W3–W5 must change inputs/targets.
3. **Budget:** 3 GPU-days (37 GPU-h raw), 16 GB new states, one staging CPU job per cache size; sessions are 12 GiB MIG
   and a bigger GPU requires a measured `peak_allocated_bytes`.
4. **Adopt the dated sequence**, with W3.0 plumbing (2–5 Nov) as the real gate and W4-parent/W5 after `dd27e718` (24 Oct).
5. **Ratify** one retire (forward inversion) and three deferrals (joint d–A_V, α head, 6-target) with §7's unblock
   conditions, so the silence is not read as an omission.
6. **Name owners** for the five launch-blocking gaps (head-family union, joint-continuation CANFAR stage, staging keys,
   age/distance target spaces, `pipeline/splits.py` wiring) and **keep the evidence rules** (certificate per run,
   producer-only rows never ranked, old FAILs visible, cohorts never mixed, one locked-test read).
