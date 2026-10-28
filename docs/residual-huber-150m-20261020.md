# Parent86 residual-Huber continuation on 150M training rows

Human-authorized separate CANFAR run, `residual_huber_e86_150m_30e_v1`.
Starts from the latest preserved pretraining checkpoint86 (full encoder/decoder
and its Adam state), with a fresh residual FeH head and 500 head-only warmup
updates. It does not resume the stopped session or substitute the older
parent72 long fine-tune. Native FeH fit/test and photometry/XP prediction policy
remain the tested residual-Huber policy. Reconstruction retains the original
pretraining feature/masking objective; this is not the joint Gaussian template.

Before SGD, stage exactly150M distinct original TRAIN rows without replacement,
proportionally by original shard, preserving integer IDs and original row
indices. Exclude every original validation ID, verify source split/counts,
parent/scaler/source identity and cache checksums. Refuse incomplete caches,
underfill, duplicate IDs and inadequate disk. Existing original catalogues only;
no acquisition. The staging phase occupies the allocated session and may take
substantial time before the first training update. No150M staging claim is made
until its actual manifest passes. GPU availability and original128XP audit gate
training; the numerical smoke remains a fixture rather than a production audit.

Budget30passes =4.5B reconstruction presentations. Checkpoints after the first
9M and50M presentations, then every150M pass (32 full periodic checkpoints,
roughly13GB), including both Adam states and RNG. Each checkpoint has SHA/size
receipt; no overwrite of previous runs. The scaled sample takes about87GB.
Learning rates encoder1.25e-6/head1e-3, effective batch32768/micro4096,
masked reconstruction weight.01, eight256-label batches per update. Fixed
validation and native predictions are archived at checkpoint boundaries; no
test-based checkpoint selection or early stopping. A fresh segment can start
from a preserved full checkpoint, but this runner does not automatically restart
a session or claim exact interrupted-row resume.

This is reconstruction pretraining with residual-Huber measured-FeH supervision,
not a jointly calibrated stellar posterior. Original pretraining overlap with
spectroscopic test objects is not excluded. Future multi-label heads remain a
separate comparison. Actual GPU/resource/source/runtime findings must be read
from the accepted run manifest, not inferred from local checks.
