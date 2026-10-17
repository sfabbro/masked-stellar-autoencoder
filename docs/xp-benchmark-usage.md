# Verified 10M-star XP benchmark

File: `/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5`

Final size: 121,485,775,663 bytes (121.49 GB). Replacement build wall time: 7,946.21 seconds (2 h 12 min 26 s), including sample reuse and compressed publication. Independent verification is saved in `experiments/xp/final-verification.json`; the original builder manifest is saved in `experiments/xp/benchmark-final-manifest.json`.

Full checks passed: exactly 9M train and 1M validation records; all record and row/shard provenance bytes identical to the retained original sample; all correlations finite and within range; covariance availability complete; unique IDs and disjoint splits; original sample hash; parent epoch-27 scaler arrays and checkpoint hash. The publisher also compared staging and final schema and round-trip probes.

## Explicit grouped reader

```python
import h5py

path = "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5"
with h5py.File(path, "r") as f:
    group = f["train"]  # or f['validation']
    for start in range(0, len(group["records"]), 4096):
        stop = min(start + 4096, len(group["records"]))
        records = group["records"][start:stop]
        correlations = group["xp_correlations"][start:stop]
        metadata = group["xp_band_metadata"][start:stop]
        # Apply the reference scaler and original field-cleaning/masking policy here.
```

The existing root-shard HDF5 loader cannot read this layout automatically. Packed correlation bands are BP then RP, each with 1,485 strict lower-triangle entries in `numpy.tril_indices(55, -1)` order and an implicit unit diagonal. Every original record field is retained.

## Confirmed legacy error-unit mismatch

The 128-source raw-XP audit tested all 55 BP and RP coefficients and errors. Stored coefficients are G-normalized, but stored `bpe_1..55` and `rpe_1..55` remain in raw archive units. Median stored/raw error ratios were 1.000000000 for both bands; coefficients matched raw coefficients divided by `flux = 10**((8.5-G)/2.5)`. Earlier descriptions and the original builder manifest's claim that errors were already G-normalized are incorrect. The file faithfully preserves this legacy table; it does not silently change errors.

Before using an uncertainty or covariance loss, divide the stored XP errors by this flux factor. For normalized coefficient covariance, use `C_ij = correlation_ij * (error_i/flux) * (error_j/flux)`. In network units, further divide each error by its corresponding reference RobustScaler scale, or divide covariance by both feature scales. Keep magnitudes in their stored units until RobustScaling; do not apply the XP flux factor to magnitudes. This does not propagate G uncertainty or supply BP/RP cross covariance.

The normalization audit is a 128-star check from one raw shard, not an all-shard raw-error audit. Existing uncertainty experiments need re-evaluation with corrected error units before their results can establish whether covariance weighting helps.
