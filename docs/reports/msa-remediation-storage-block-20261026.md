# Remediation storage blocker — 2026-10-26

The fourteen supervised pilot arms are implemented and tested, including four genuinely coupled six-target NSF arms. None has been submitted. The distinct recovery audit/source/NSF-preflight jobs were accepted once and all failed with `Errno122: Disk quota exceeded` before they could supply the required admission receipts. They are preserved; there is no automatic retry or recurring monitor.

ARC project metadata reports a 20 TB quota and 20,000,005,095,420 bytes used. The launcher now checks quota before future submissions and requires 20 GB of output reserve. The complete updated test suite passed 1,090 tests with 4 skips; 13 focused checks validate the quota and launch guards.

A concrete archive proposal is available in `experiments/xp/results-stellar-improvement/remediation-wave-20261026/quota-block/archive-candidates.json`. The four `expanded_eval_*_s51_v2/batch-cache` directories total 28.34 GB. These are archive candidates, not certified redundant files. Do not discard them. Keep 83.11 GB of consolidated prediction arrays and all unique checkpoints, source labels, reports, bounded pilots and corner samples in place.

To resume: either obtain project quota/additional storage, or approve an archive destination with capacity for those exact cache trees. Before removing any original, copy and verify full content hashes, preserve manifests/source bindings and record the archive locations. No deletion or transfer has occurred. Once storage is available, preserve these failed v2 attempts and use distinct versioned recovery outputs; only actual successful prerequisite receipts may admit the fourteen training pilots.
