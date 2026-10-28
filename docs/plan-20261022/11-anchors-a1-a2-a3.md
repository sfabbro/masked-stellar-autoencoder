# A1/A2/A3 — independent age anchors and multi-teacher training

**Prepared:** 2026-10-22. Status after the multi-teacher module landed
(`training/multi_teacher_age.py`, 36 tests passing) and the first anchor label layer
was built (`age_teacher_anchors_v2`).

The point of A1/A2/A3 is to stop validating age only against the StarHorse labels
the model trained on. Seismic and cluster anchors give a second, independent scale;
the multi-teacher model lets us combine them without averaging.

---

## 1. Anchor label layer (built today)

`scripts/build_age_anchor_labels.py` → `experiments/.../age_teacher_anchors_v1/`

| Teacher | Rows | Recommended age | Gaia ID status | Scale |
|---|---:|---:|---|---|
| APOKASC-3 | 15,808 | 12,376 | **direct Gaia DR3 IDs** | `apokasc3` |
| Willett+2026 Kepler | 10,169 | — | Gaia column constant (bug) | `willett_kepler` |
| Willett K2 APG BHM | 10,015 | — | same | `willett_k2_bhm` |
| Willett K2 APG COR | 11,140 | — | same | `willett_k2_cor` |
| Willett K2 GAL BHM | 9,808 | — | same | `willett_k2_gal_bhm` |
| Willett TESS | 1,920 | — | same | `willett_tess` |
| **Willett total** | **43,052** | **35,948** | crossmatch via KIC/TESS/APOGEE IDs required | per-survey |

Caveats:

- APOKASC3 and Willett are seismic/model-dependent **scales, not truth**.
- Willett’s released `GAIADR3_ID` is constant across rows; the anchor build keeps
  KIC/TESS/APOGEE/GALAH identifiers and sets `gaia_id_usable=False`. A dedicated
  crossmatch is required before a Gaia join.
- Cluster/globular-cluster anchors are **not yet included** (next acquisition).

---

## 2. A1 — multi-teacher age training

**Objective:** train an amortized age head with several teachers, a reference scale,
and cluster coevality; no per-star free latents.

- **Head:** `MultiTeacherAgeHead(teacher_names, reference_teacher=...)` on frozen
  MSA encoder features; `TeacherLabelBatch` masks missing teachers; use explicit `objective="joint"` for the shared-latent joint Gaussian likelihood.
  The historical default remains a weighted composite of marginal likelihoods;
  its teacher-specific denominators are not a joint likelihood.
- **Reference scale:** run two arms:
  1. `ref=apokasc3` — target the giant seismic scale; StarHorse MSTO/SGB is the
     heterogeneous teacher; clusters bridge.
  2. `ref=starhorse` — continuity with the current sample; APOKASC3/Willett are
     cross-checks; easier for the 10M pilot.
- **Data:**
  - 10M benchmark pilot: StarHorse strict 4,440 + broader 105,953; APOKASC3 only
    **672** rows in the 10M benchmark; Willett crossmatch pending. Pilot is for code
    path/smoke, not a final age model.
  - 150M cache: expected ~10k APOKASC3 and ~? Willett after crossmatch; use the
    150M cache for the real A1 run.
  - [C/N] weak labels later (A4); not in A1.
- **Regularization:** `cluster_coeval_loss` on open/globular clusters once cluster
  memberships are acquired; `selection_weights` for non-uniform anchor selection.
- **Cost:** head-only on frozen embeddings; 20M presentations ≈ current 10M screen
  cost (1g slice, ~15–20 min); 150M presentations scaled accordingly.
- **Evaluation:** leave-one-teacher-out; cluster holdout; FeH/stage slices;
  tail n; no truth claim.

---

## 3. A2 — few-shot meta-correction

**Objective:** correct a frozen base age prediction onto a chosen anchor scale
without retraining the encoder.

- Base: `native_age_strict_recon_v1` predictions (152k age cohort) or the A1 head.
- `MetaCorrection(target_scales=("apokasc3","willett_tess","cluster"))`; fit only on
  anchor rows, monotone correction `τ_c = c(x) + s(x)[...]`.
- Evaluate on held-out anchors/clusters; report bias/NMAD before/after and the
  systematic envelope.
- Requires the Willett crossmatch and cluster anchors; today it is a code path and a
  synthetic test, not a production result.

---

## 4. A3 — cluster and seismic anchors

- **Seismic:** APOKASC3 + Willett (above); use as held-out scale and teacher.
- **Open clusters:** acquire the Hunt & Reffert catalogue (ages, memberships,
  selection function). Coeval loss:
  `sum_i (τ_i − τ_cluster)² / (σ_i² + σ_cluster²)` with cluster random effects.
- **Globular clusters:** literature ages + Gaia memberships; small n but the only
  old/metal-poor anchor.
- **Selection:** cluster membership and age–metallicity–reddening degeneracies are
  systematics; report selection weights and do not treat cluster ages as truth.
- **Deliverables:** `cluster_anchors_v1.parquet`, membership/selection manifest,
  cluster-holdout metrics.

---

## 5. Launch sequence

| Step | Work | Resource | Gate |
|---|---|---|---|
| A3.1 | Willett KIC/TESS/APOGEE → Gaia crossmatch; finalize anchor table | CPU | matched fraction and provenance |
| A3.2 | Acquire open/globular cluster memberships + ages | CPU | selection/membership audit |
| A1.0 | A1 10M pilot (ref=starhorse, APOKASC3+StarHorse; smoke) | 1g, ~20 min | loss finite, diagnostics, leave-one-teacher-out |
| A1.1 | A1 150M real run (ref=starhorse or apokasc3) | 1g, ~1–2 h | anchor validation before long run |
| A2.0 | Meta-correction on anchors (frozen base) | CPU/head-only | held-out anchor improvement |
| A1.2 | Cluster-coeval loss added | 1g | cluster holdout consistency |

No production age claim until A1/A2 pass held-out anchors and the tail is reported
with its n.

---

## 6. Open blockers

- Willett Gaia-ID crossmatch (constant column bug).
- Cluster catalogue acquisition and membership/selection audit.
- A1 runner integration (head-only over frozen embeddings + label masking).
- APOKASC3 in the 10M benchmark is only 672 rows → real giant-age training must use
  the 150M cache.
- Gaia DR4 FLAME (Dec 2026) will add a large teacher layer; ingest as an additional
  scale, not truth.

## Repair status after independent checkup

APOKASC3 v1 bound columns were misinterpreted signed uncertainties; v2 uses
`age16=age50+e_age` and `age84=age50+E_age` from the catalogue ReadMe,
with invalid/nonpositive bounds masked. Existing v1 artifacts are retained.
Only 12,301 of 15,808 rows have valid positive ordered bounds.
Native audit found three APOKASC IDs in the old age tuning cohort; new
anchor holdouts must be assigned by identity/family before A1/A2 evaluation.
The new bounded prepared-embedding pilot provides FIT-only gradients/TUNE
selection and explicit source/units/SHA contracts for age and distance.
It is a code-path validation, not a replacement for stage/cluster holdouts.
Alpha evidence and raw Gaia parallax/zero-point inputs remain required gates.
