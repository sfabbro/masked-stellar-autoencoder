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
