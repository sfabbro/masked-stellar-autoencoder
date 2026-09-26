# Experiment log (multitask rationale, Phase 2)

Record pilot and full-scale results here after each run. **Do not** edit paper tables by hand; copy `results/.../metrics_table.tex` from `masked_stellar_autoencoder.training.eval_ensemble`.

## Pred-only vs recon+pred (pilot)

**Config:** duplicate `configs/finetune.yaml` → `configs/pilot_local.yaml`; set `finetuning.num_epochs: 5`, `finetuning.ensemble: false`, local paths for `ft_datafile` and `saved_weights`.

| Run ID | multitask | lambda_pred / lambda_rec | max_train_rows | Final val loss (from log) | Notes |
|--------|-----------|--------------------------|----------------|---------------------------|--------|
| A | false | N/A | 8192 | | pred-only |
| B | true | 0.8 / 0.2 | 8192 | | recon+pred |

Commands:

```bash
pixi run python -m masked_stellar_autoencoder.training.finetune_msa --config configs/pilot_local.yaml --max-train-rows 8192 --max-valid-rows 2048
```

Flip `finetuning.multitask` between runs; keep seed and data identical.

## CANFAR execution smoke (2026-09-26)

**Revision:** `8f8c610`; image `images.canfar.net/astroai/base:latest`; GPU
`NVIDIA H100 NVL MIG 1g.12gb`.

| Stage | Work bound | Train loss | Validation loss | Output |
|-------|------------|------------|-----------------|--------|
| Pretraining | 1 epoch; 2 train shards and 1 validation shard; 512 rows per shard | 0.905803 | 0.900108 | `/arc/projects/k-pop/msa_runs/pretrain/checkpoints/msa_pretrain_pilot.pth` |
| Fine-tuning | 1 epoch; at most 8192 train and 2048 validation rows | 0.408444 | 0.396878 | `/arc/projects/k-pop/msa_runs/finetune/checkpoints/msa_finetune_pilot.pth` |

Both checkpoints were confirmed on persistent project storage. The CANFAR
catalogue preflight found compatible model settings and all configured fields.
The mounted legacy catalogues lack magnitude uncertainties for `W1`, `W2`,
`G`, `BP`, and `RP`, so those uncertainty mappings remain explicitly unset.
These runs verify execution and checkpoint output; their losses are not a model
comparison. The source-ID index is being generated from the mounted GaiaSource
files under `/arc/projects/k-pop/msa_runs/preprocess/`; its original catalogue
directory destination proved read-only and was moved in revision `68d92a2`.
Full preprocessing remains blocked by the absent
`/gaia/dr3/xp_continuous_mean_spectrum` mount. Dust maps also still need to be
fetched.

## Full-scale (Phase 3)

| Git tag | Checkpoint paths | eval_ensemble `--out` | Decision (A vs B) |
|---------|------------------|----------------------|-------------------|
| | | | |

## Linear probe baseline

**C0 (linear probe):** set `linearprobe: true`, `multitask: false`, `finetuning.lf: mae` (or `mse`). The wrapper trains a frozen encoder + `nn.Linear` head; checkpoints include `linear_probe: true` and work with `eval_ensemble.py`.
