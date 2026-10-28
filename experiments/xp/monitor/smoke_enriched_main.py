"""Full actual-input CPU integration; completed source gate substitutes mounted XP audit.

No new original-benchmark unit certification: production still reruns its128
matched-source XP audit. Local scores are workflow checks, not science results.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_enriched_value as r

base = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
r.ROOT = base
r.PARENT = "/tmp/msa-standard-checkpoint-57.pth"
unit = json.loads((base / "enriched_feature_cohort_v1/manifest.json").read_text())[
    "result"
]["unit_audit"]
r.probe.audit_fits_xp_units = lambda *a, **k: dict(
    unit, LOCAL_SMOKE_ONLY_reuses_completed_source_audit=True
)
out = base / "enriched-main-local-smoke-v1"
sys.argv = ["smoke", "--run-id", "LOCAL_SMOKE_ONLY", "--output", str(out)]
r.main()
(out / "LOCAL_SMOKE_LIMITATIONS.json").write_text(
    json.dumps(
        dict(
            actual_full_cohort_parent_inputs=True,
            CPU_integration_only=True,
            original_benchmark_not_local=True,
            no_new_original_benchmark_unit_certification=True,
            production_must_rerun_actual128unit_audit=True,
            not_scientific_results=True,
        ),
        indent=2,
    )
    + "\n"
)
print((out / "checkpoint_verification.json").read_text())
