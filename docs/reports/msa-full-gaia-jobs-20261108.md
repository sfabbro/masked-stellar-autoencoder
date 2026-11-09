# Correction jobs launched — 2026-11-08 (Vancouver)

These jobs address the full-Gaia scope corrections. They do not retroactively change the restricted production run, and source staging is not scientific admission into training.

| Work | Run | Session | Output |
|---|---|---|---|
| All-Gaia matching, Gaia photometry/astrometry and actual no-XP reconstruction reservoir | full_gaia_gaia_stage_20261108_v2 | v1kyrpx1 | Exact-ID native candidate join against GaiaSource shards; unmatched IDs; stratified XP/no-XP SSL reservoir |
| Restored photometric bands and auxiliary head measurements | full_gaia_photometry_stage_20261108_v1 | fxc6kwlb | Raw native-ID matched observations and errors for all declared available columns; per-shard availability counts |
| Multi-survey alpha | full_gaia_alpha_stage_20261108_v1 | lbhf0ucu | Native alpha/element fields with identifiers, flags and units; no alpha/M-to-alpha/Fe renaming |
| Multiple published age estimators | full_gaia_ages_stage_20261108_v1 | fvll9eut | Native age/error fields and identity/quality columns from existing age tables, including APOGEE latent-age CSV |
| CUDA input/head/map integration | full_gaia_contract_pilot_20261108_v1 | hshxs209 | Three small fresh-network arms: restricted inputs, restored inputs, restored inputs plus symmetric six-target maps; masked MAE .03 and joint six-dimensional NSF |

Source scripts and configuration hashes are preserved in immutable ARC bundles. Submission receipts live in experiments/xp/decision-wave-20261106. The initial Gaia worker assumed a .h5 extension; the replacement handles the verified .hdf5 files. The pilot depends on the replacement.

The input pilot waits for actual staged source parts and actual no-XP Gaia rows, fails explicitly on unavailable dependencies, preserves existing FIT/TUNE/CAL/DEV roles where known, and uses FIT-only normalization. It is an engineering pilot with a small fresh MLP, not a continuation of the pretrained encoder or a scientifically controlled ranking of production architectures. New age/alpha source products remain outside its training until semantic, quality and dependency audits are complete. Population-wide feature completeness, full native photometric crossmatches, covariance models and protected-group admission remain production tasks.

The candidate ID union before any XP requirement has 13,592,882 stars from the existing registry, configured native sidecars and native APOGEE/GALAH releases. This is a candidate count, not a certified all-survey usable-label count or an all-Gaia prediction count.

Local verification: tests/test_full_gaia_source_jobs.py, tests/test_full_gaia_contract.py and tests/test_pipeline.py completed together: 10 passed. New worker scripts also compile. Actual progress snapshots are in msa-insights-20261108/full-gaia-correction-jobs.json.
