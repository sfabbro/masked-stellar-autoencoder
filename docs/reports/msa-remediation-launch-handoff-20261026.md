# Remediation launch handoff — 2026-10-26

All **14 pilots accepted once**. This document and `experiments/xp/remediation-submitted-20261026.json` are the durable starting point for collecting results. Submission acceptance is not completed SGD, numerical admission, or scientific success.

## Immutable code and inputs

- Code tag: `msa-remediation-pilots-20261026`, pointing to actual launched commit `7b29799cee65ebf9e6a0a348ed0e6844b9139a41`.
- torchregress: `6bd8b93b35fe4fe216319870e72e94eb78429106`; large parent checkpoint SHA `c544548ba203f249e11509d0e0f9832df0e8366e6f7b0c7a639460752061a2f9`.
- Source producer: `remediation_source_v3` at `d87d7c60d44ae7ea374886959a7bd082ac3e909a`; prepared H5 SHA `112fb4b5708c6073ffd2388012a6a9dc7e5bd3eca540db15acbac0fe235f0a1c`.
- Input proof: `remediation_inputs_v4` (`whakp3zf`), extending only four missing Gaia-VAC photometry proofs and retaining the passed v3 native/legacy checks. Head proof: `remediation_joint6_preflight_v3` (`tgcwpgsi`), retaining old Gaussian/MDN proofs and testing the actual coupled six-target NSF.
- Order: Teff, logg, FeH, alpha, age, distance. FIT counts: 506331, 493548, 506315, 1197641, 115834, 117573; 1736618 unique FIT identities. Full-six coobservations: 277; pairwise support remains in the source manifest. No physical six-dimensional truth claim.
- Each pilot has 1000 updates, fresh head/optimizer, common large parent, fixed source-balanced/label-balanced sampler and common TUNE selection. This is not 1000 epochs or 450M presentations. Source roles are development roles consuming previously inspected test objects.

## Runtime check — 2026-10-26

Original manifests show seven pilots completed 1,000 updates and seven stopped at zero updates. Three no-reconstruction fine-tuned Gaussian/MDN runs failed the initial encoder-gradient check; four coupled NSF runs failed the real sparse/noisy integration admission gate. This is an execution check, not a scientific ranking. Original manifests remain untouched. Exact statuses are recorded in `experiments/xp/remediation-pilot-status-20261026.json`; successful artifacts still require independent collection and verification. The observed NSF estimator gaps do not justify weakening the gate or treating the flow as independent scalar models.

## Gradient-fix recovery runs

The three fine-tuned Gaussian/mixture jobs that stopped before their first update have distinct recovery runs at source commit `d95608991673454a9ac9a60fdd599dd0d7082505`, with the same input/parent/library, seed and 1,000-update budget. The corrected guard permits the known zero-initialization first step only when the output layer has a finite nonzero gradient and then requires a supervised encoder gradient on step two. Full CI: 1,098 passed, four skipped. The three jobs were accepted once:

- `remediation_gaussian6_ft_gradientfix_v2`: `u5ir9g7k`.
- `remediation_mdn6_ft_gradientfix_v2`: `k24zucq6`.
- `remediation_gaussian6_ft_parallax_gradientfix_v2`: `bxrtggg5`.

They use the same output-root and collection recipe below. The first actual manifest was admitting; two manifests were not yet available at the initial check. Acceptance is not training completion. The four NSF integration failures are separate and remain unresolved.

## Coupled NSF integration probe

The distinct inference-only `remediation_nsf6_qmc_probe_v1` (`e62pubx4`) completed at `07fb40a39a0e2c891bb790c629b3137a1468dd3b` with the original prepared input, parent, seed and head. It tested 128, 256, 512 and 1,024 QMC particles on the same model-blind panel in 8-row chunks, using the original ESS/replica/refinement thresholds. The 128-particle replica gap p95 was 0.10985 and failed the 0.10 gate; 256 particles passed (replica p95 0.07426, refinement p95 0.03434), as did 512 and 1,024. Small arrays were preserved and their SHA256, NPZ CRC and independently recomputed gate statistics passed. The first passing count is not a scientific accuracy or later-checkpoint certificate.

The output directory is `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/remediation_nsf6_qmc_probe_v1/`; local receipts and small arrays are under `experiments/xp/results-stellar-improvement/remediation-wave-20261026/qmc-probe-preservation/`, with a compact tracked `experiments/xp/remediation-nsf-qmc-probe-result-20261026.json`. A separate four-arm recovery is being prepared with 256 particles for both training and admission, 16-row gradient accumulation, 8-row diagnostics, and unchanged numerical thresholds. Its training-time GPU memory and selected-checkpoint gates still require actual verification. Preserve the four original failed NSF runs.

## 256-particle coupled NSF recovery runs

All four distinct recovery jobs were accepted once at `4b8d0b5b780366517baecf10e00165a9c9414b21`. They use 256 particles for the actual joint training objective and initial/selected integration checks, 16-row supervised gradient accumulation, 8-row diagnostic chunks and unchanged gates. Finetuned encoders with batch normalization are rejected; dropout realizations need not match an unchunked run. The initial GPU check showed 50 finite-gradient updates for the frozen arm; the other three manifests were not yet available at that check. No selected checkpoint or scientific result is certified yet.

- `remediation_nsf6_frozen_qmc256_v2`: `bpw6joq8`.
- `remediation_nsf6_ft_qmc256_v2`: `yv16nrs1`.
- `remediation_nsf6_ft_recon003_qmc256_v2`: `hx177dr5`.
- `remediation_nsf6_ft_parallax_qmc256_v2`: `q6lyfrhv`.

Retrieve summaries using the same output-root recipe below and `experiments/xp/remediation-nsf-qmc256-recovery-20261026.json`. Preserve original failed attempts and never resubmit accepted IDs. The actual selected-head QMC gate must also pass before treating a run as complete.

## Actual accepted run IDs and sessions

- `remediation_huber_frozen_v1` — `qrysdbbm`.
- `remediation_huber_ft_v1` — `ipbuzbqo`.
- `remediation_huber_ft_recon003_v1` — `tnyfsly4`.
- `remediation_gaussian6_frozen_v1` — `puc5wm6y`.
- `remediation_gaussian6_ft_v1` — `tx9fe185`.
- `remediation_gaussian6_ft_recon003_v1` — `hs6sjt07`.
- `remediation_mdn6_frozen_v1` — `qv0f7cod`.
- `remediation_mdn6_ft_v1` — `c1vozo13`.
- `remediation_mdn6_ft_recon003_v1` — `ujligcsn`.
- `remediation_nsf6_frozen_v1` — `uracqg4f`.
- `remediation_nsf6_ft_v1` — `gz5istkd`.
- `remediation_nsf6_ft_recon003_v1` — `n3n20n5a`.
- `remediation_gaussian6_ft_parallax_v1` — `me6fjv9e`.
- `remediation_nsf6_ft_parallax_v1` — `juby59r8`.

## Where the summaries live

Each run's output is `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/<run_id>/`. Prefer direct artifact reads over polling completed sessions.

- `manifest.json`: actual runtime state, input/code/library bindings, completed updates, finite gradients, unique label coverage, selected update and TUNE/development metrics. Terminal success is exactly `remediation_pilot_complete`; preserve any `failed` manifest and error. `Running` alone proves no SGD.
- `metrics.jsonl`: per 100-update TUNE macro normalized median MAE, per-target normalized MAEs, supervised/reconstruction objective histories.
- `normalization.npz`: FIT normalization plus native FIT scale; multiply normalized per-target MAE by `native_fit_scale` to obtain native-unit MAE.
- `tune-predictions.npz`, `development-predictions.npz`: source IDs, references/masks/errors/source roles, native medians and empirical intrinsic/observed PITs. The evaluation panel is deliberately bounded; it is not the entire 4.44M source union or a fresh test set.
- Density heads: `whole-vector-corners.npz`, actual 8192 draws per common star, standardized/native covariance and observed reference masks. Huber has no PDF/PIT/sampler and intentionally no corner file.
- NSF: `integration-initial.{json,npz}`, `integration-selected.{json,npz}` record finite-QMC ESS/replica/refinement checks. A failed gate is a failed admission, not an excuse to change tolerances or report an exact partial-flow likelihood.
- `selected-checkpoint.pth`: keep remote; no local native checkpoint/model execution. On completion, manifest `artifacts` binds bytes and SHA256 of saved files. Never fetch heavy arrays/checkpoints while the run is unfinished or redownload files already preserved and verified.

## Reproducible collection, without running a model

Use the current checkout containing the submitted-job manifest (the tag itself intentionally identifies the earlier numerical code). The following is a metadata-only collection recipe; run it with the configured CANFAR client runtime. It makes no jobs, installs, or model calls:

```python
import json
from pathlib import Path
from canfar.storage import filesystem

launch = json.loads(Path("experiments/xp/remediation-submitted-20261026.json").read_text())
fs = filesystem("arc")
out = Path("/tmp/msa-remediation-summaries-20261026")
out.mkdir(exist_ok=True)
for job in launch["jobs"]:
    remote = job["output"].removeprefix("/arc")
    payload = fs.cat_file(remote + "/manifest.json")
    actual = json.loads(payload)
    assert actual["run_id"] == job["run_id"]
    assert actual["source_commit"] == job["source_sha"]
    (out / (job["run_id"] + ".json")).write_bytes(payload)
    print(job["run_id"], job["session_id"], actual["status"],
          actual.get("completed_updates"), actual.get("development_per_label"))
```

Treat absent/404/503 metadata as unavailable, not model failure or permission to retry. If complete, copy only missing report/prediction/normalization/corner files first, verify their manifest byte counts and SHA256, and test NPZ CRC. Inventory file sizes and local free space before large collection. Never resubmit accepted or ambiguous IDs. Never infer physical performance for unobserved labels; keep genuine no-XP separate from masked-XP and inspect the actual covariance of coupled samples before interpreting corner plots.

Generate six-panel prediction/reference and metal-poor residual plots, available empirical PITs and common-axis whole-vector corners from saved arrays with NumPy/matplotlib only. Preserve old diagrams and failed outputs. This launched wave is frozen while the new multi-survey atmospheric-calibration plan is implemented; do not silently replace its code, source, estimands or summaries with the new architecture.

No recurring monitor was created. The finite first-attempt dispatcher is done; its state is `experiments/xp/results-stellar-improvement/remediation-wave-20261026/pilot-dispatch-v1/state.json` (`all_pilots_accepted`).

## Recovery status and preserved artifacts, 27 October

The three gradient-fix Gaussian/MDN recoveries completed all 1,000 updates with finite gradients. Their small prediction, normalization, metrics and whole-vector corner files were preserved and checked against the producer's SHA256 and NPZ CRC; the receipt is `experiments/xp/remediation-gradient-recovery-artifact-receipts-20261027.json`. Selected checkpoints remain on ARC. The full bounded manifest snapshot, including the original terminal failures, is `experiments/xp/remediation-recovery-runtime-20261027.json`.

For the four QMC256 NSF recoveries, frozen `bpw6joq8` and reconstruction `hx177dr5` were running at the recorded observation. Fine-tune `yv16nrs1` and parallax fine-tune `q6lyfrhv` stopped before any update because their zero-initialized projections prevent a first-step encoder gradient. A separately versioned startup correction requires finite nonzero gradients in both projections at step one and a supervised encoder gradient at step two; it does not change the 256-particle objective or integration thresholds. Original failed outputs remain intact. No model-ranking conclusion follows from these runtime receipts.

The two distinct startup-fix recoveries were accepted once at source `7ee3b57425daba406fa0b9a69c518e90a90b2412`, still using the original experiment's torchregress `6bd8b93b35fe4fe216319870e72e94eb78429106` and the same QMC256 settings: `remediation_nsf6_ft_qmc256_startupfix_v3` → `zzt96g5j`; `remediation_nsf6_ft_parallax_qmc256_startupfix_v3` → `k557za1j`. The accepted submission receipts and exact output paths are in `experiments/xp/remediation-nsf-startupfix-submitted-20261027.json`. Acceptance does not prove the initial or selected integration gates, completed updates, or scientific performance. The later multi-survey architecture uses a separate newer torchregress revision and must not change these immutable runs.

Subsequently, both earlier QMC256 NSF jobs `bpw6joq8` and `hx177dr5` reached 1,000 finite-gradient updates but **failed the selected-checkpoint integration gate**. The 128-row selected panels had median relative effective sample sizes 0.00622 and 0.00622; replica log-density gap p95 was 11.21 and 13.10 respectively (required at most 0.10). The 65 alpha-only rows had median ESS of one particle out of 256. These are failed numerical integrations, not successful six-target likelihood results. Saved-array statistics and SHA receipts: `experiments/xp/remediation-nsf-selected-gate-failure-20261027.json`; original terminal manifests are in `experiments/xp/remediation-nsf-selected-gate-evidence-20261027/`. The diagnostics indicate severe importance-weight concentration; they do not by themselves identify why the fitted proposal and flow diverged. No tolerance was relaxed.

At newer source `06d5438f8a2ac045940c0d5182b14bb2b7627032` and torchregress `906a1c17639dcbf308fdb835270701bbc04f91e0`, the first new source/flow jobs were accepted once: `multisurvey_desi_dr1_raw_source_v1` → `ywde0t2t` and `remediation_copula_nsf6_frozen_v1` → `rstbv2c0`. Their full submission receipts and output paths are in `experiments/xp/multisurvey-copula-first-submitted-20261027.json`. The DESI job prepares quality-selected raw rows only; it has not yet established feature joins or FIT counts. The pilot is a coupled Gaussian-copula spline flow, with a restricted copula and exact intrinsic missing-label marginals; its actual device and initial/selected numerical gates must pass before interpretation. Neither acceptance is completion.

After `rstbv2c0` passed its actual copy-only device preflight and initial 128-row integration gate (median relative ESS 0.9968, minimum 0.4938, replica gap p95 0.00115), the remaining two spline-flow variants were accepted once at the same source and library: `remediation_copula_nsf6_ft_v1` → `u17b6yfr`; `remediation_copula_nsf6_ft_recon003_v1` → `pg0qvelr`. See `experiments/xp/multisurvey-copula-remaining-submitted-20261027.json` and `experiments/xp/remediation-copula6-initial-gate-20261027.json`. Each run still must pass its own actual preflight and selected-checkpoint gate; the initial gate is not a trained-head or physical-correlation certificate.

### 27 October: newer-library coupled spline results

DESI raw-source preparation `ywde0t2t` and coupled spline pilots `rstbv2c0`, `u17b6yfr`, `pg0qvelr` used MSA tag `msa-multisurvey-copula6-20261027` and torchregress `906a1c17639dcbf308fdb835270701bbc04f91e0`. The DESI source completed with 3,683,136 quality-selected measurement rows, before feature matching, role exclusions and deduplication. Frozen and fine-tuned spline pilots completed 1,000 updates and their 128-row selected integration gates; fine-tuned plus reconstruction failed that gate. Original unrestricted NSF startup recoveries `zzt96g5j` and `k557za1j` also failed their selected integration gates after 1,000 updates. Do not resubmit, replay or promote these failures.

The [saved-array report and representative corners](msa-remediation-copula6-20261027/index.md) explains the model's restricted Gaussian copula and the scientific limits. Checksummed small predictions, integration panels and original 8,192-draw bundles are under `experiments/xp/results-stellar-improvement/remediation-wave-20261026/final-copula-and-nsf-20261027/`; the tracked [preservation receipt](../../experiments/xp/remediation-copula6-preservation-20261027.json) records the producer output paths, job IDs and hashes. Selected checkpoints and the large DESI sidecar remain on ARC. Generate the shown figures and repeat saved-array checks using the frozen Python environment and `scripts/verify_remediation_copula6.py` with that directory and a new output directory as arguments; it never loads a model or checkpoint.
