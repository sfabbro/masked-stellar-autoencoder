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
