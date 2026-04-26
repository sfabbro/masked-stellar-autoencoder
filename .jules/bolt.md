## 2026-04-24 - Pre-loading Checkpoints for Evaluation

**Optimization:** Optimized `eval_ensemble.py` by pre-loading PyTorch state dictionaries into a list in RAM (`map_location='cpu'`) instead of repeatedly reading them from disk during evaluation across different datasets (`X_test`, `X_off`, and conformal calibration steps).

**Learning:** Loading PyTorch model files from disk multiple times in a script (especially large checkpoints in an ensemble loop) is a significant and unnecessary I/O bottleneck. Pre-loading them into memory once, mapping to CPU to avoid filling VRAM, drastically improves performance.
