# AGENTS.md — masked-stellar-autoencoder

Personal fork of Aydan's MSA. Ready for local / `sfabbro` work; do **not**
push to `upstream` unless explicitly asked.

## Remotes

| Remote | Points at | Use |
|--------|-----------|-----|
| `origin` | `sfabbro/masked-stellar-autoencoder` | Push WIP / keep fork current |
| `upstream` | `aydanmckay/masked-stellar-autoencoder` | Fetch only (no push by default) |

```bash
git fetch upstream
# integrate upstream when ready: rebase/merge locally, push to origin only
git push origin HEAD
```

## Environment

Pixi (`[tool.pixi]`). Prefer `pixi install` / `pixi run` over ad-hoc
`requirements.txt` when the Pixi env covers the task.

## Verification

Use the repo's documented pixi/pytest entrypoints when present. Leave one
small runnable check for non-trivial logic changes.

The stellar-parameter pipeline is `masked_stellar_autoencoder.pipeline`
([docs/pipeline.md](docs/pipeline.md)). Check it with
`pixi run pytest tests/test_pipeline.py -q`. The original `training/` scripts
are a separate path. Do not push this fork to `upstream`.

## Env hygiene

- Prefer `pixi run` / `pixi run python` over bare `python3` when Pixi exists.
- Never `pip install --user` or install into `~/.local` / `$HOME/.local` (esp. CANFAR `/arc/home`).
- Headless/batch: `export PYTHONNOUSERSITE=1` and `unset PYTHONPATH`.
- On CANFAR: read skill `canfar-lab-workflow` (mounts, quotas, resources, headless, ports).
