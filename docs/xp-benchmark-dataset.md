# Fixed 10M-star XP benchmark

The CPU-only builder `scripts/benchmarks/build_xp_benchmark.py` samples 9M training and 1M validation stars without replacement, proportionally across the original HDF5 shards. Validation membership follows the epoch-27 checkpoint. Seed 42 fixes selection. Every original record field is retained, including G-normalized XP coefficients, their errors and unmodified magnitudes. Scaling is applied in the training reader; the parent checkpoint path identifies the existing RobustScaler for matched continuations.

Each split has `records` (original compound dtype), `source_shard`, `source_row`, packed `xp_correlations` (stars × BP/RP × 1485), `xp_band_metadata` and `xp_covariance_available`. Band metadata retain basis IDs, parameter counts, relevant-basis counts and standard deviations. Correlations have implicit unit diagonal, with strict lower-triangle ordering. The loader reconstructs covariance from coefficient errors, and divides by both scaler scales. G-normalization uncertainty and BP/RP cross covariance are not supplied by this sidecar. No spectra are MC sampled.

Source IDs are indexed once in memory for the selected 10M stars. Original CSV shards are streamed concurrently, returning only selected rows to the single HDF5 writer. The complete source table is scanned once for sampling. Duplicate IDs, invalid correlations or missing covariance matches fail the build. An existing output is never overwritten. The `.building` file and manifest are partial products; only a successful round-trip/coverage check publishes the final `.h5`.

The grouped layout requires a reader selecting `train/records` or `validation/records`; it is not a drop-in replacement for the existing root-shard loader. Values in records remain in stored-table units, before RobustScaling. Interpret unavailable measurements using the existing field-cleaning conventions. New scalers must be fit only on the training split.

Expected correlation storage is about 104 GB, plus records and metadata. Runtime depends on scanning/decompressing the entire raw CSV collection, not merely parsing the selected 10M stars. The one-CSV timing is therefore not an end-to-end production estimate. Outputs belong on ARC; no full scratch copy is required.

Behavioral verification: `pixi run pytest tests/test_xp_benchmark.py -q`.
