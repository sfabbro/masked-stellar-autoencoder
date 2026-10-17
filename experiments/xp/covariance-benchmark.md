# One-CSV XP covariance sidecar benchmark

CPU-only CANFAR session qqz2231x, 4 CPUs requested, 8 GiB RAM. Streaming parser is single-process. Output: /arc/projects/k-pop/msa_runs/xp-covariance-benchmark, *-correlations.h5.

Completed original CSV shard: 87,090 sources; 903,418,804 bytes; 10,373.393 bytes/source. Float32 correlations have shape [87090,2,1485], with implicit unit diagonal and source IDs. Basis identifiers, parameter counts, relevant-basis counts and band standard deviations are retained. Compression: gzip level 1 + shuffle. No Monte Carlo sampling, covariance factorization or dense covariance storage.

Measured elapsed times: 10,240 rows at 14.94 seconds; 81,920 at 76.64 seconds. Thus throughput including initial setup is approximately 1,069 sources/s; complete extraction is approximately 80 seconds, but exact final extraction time was not emitted. The later full-HDF5 alignment scan was stopped while still running, so no completed alignment or end-to-end training-shard timing is claimed. Production should index HDF5 source IDs once, rather than scan all 135 compound-table shards for every CSV.

Linear storage projections: 10M sources = 103.73 GB; 20M = 207.47 GB; 218M = 2.261 TB. Decimal units. Extraction-only projections from the measured 81,920-row checkpoint: 10M ~2.6 h; 20M ~5.2 h; 218M ~56.7 h on one worker. These assume the shard is representative and exclude alignment, covariance reconstruction, loss evaluation and parallel shared-storage contention.

Two dense 55x55 float32 matrices require 24,200 bytes/source before compression, compared with 11,880 bytes/source for packed off-diagonal correlations before metadata/compression. Covariance can be reconstructed from the existing normalized errors, with saved scaler scales applied at training time. This preserves within-band covariance only; no measured BP/RP cross covariance or uncertainty in G normalization is added.
