"""Per-source survey-map families chosen from overlap diagnostics.

Map complexity should follow how much a source overlaps the common reference:
cross-validation on the high-resolution overlap
(``label_overlap.model_selection``) almost never supports more than an offset
for a catalogue with fewer than ~100 shared stars, while large low-resolution
surveys that compress the metal-poor tail need a spline. This module turns an
``overlap.json`` report into ``{training_source: family}`` for
``SixSurveyMap(source_kinds=...)``.

Families are per source (all targets), chosen from the [Fe/H] overlap, which
is the only target with a broad independent reference. Identity is never
assigned to a trained source: a free offset is cheap and absorbs the zero-point
difference between the training gauge and the literature scale.
"""

from __future__ import annotations

CV_TO_FAMILY = {
    "identity": "offset",
    "offset": "offset",
    "affine": "affine",
    "piecewise_linear": "spline",
}
MIN_OVERLAP = {"affine": 20, "spline": 200}


def overlap_entry(report: dict, name: str) -> dict | None:
    for section in ("staged_surveys", "master_sources"):
        entry = report.get(section, {}).get(name)
        if entry is not None:
            return entry.get("reference", {})
    return None


def family_for(entry: dict | None) -> tuple[str, str]:
    """(family, reason) for one overlap entry."""
    if not entry or not entry.get("overlap"):
        return "offset", "no reference overlap"
    chosen = (entry.get("maps") or {}).get("chosen")
    family = CV_TO_FAMILY.get(chosen, "offset")
    n = int(entry["overlap"])
    while family != "offset" and n < MIN_OVERLAP[family]:
        family = "affine" if family == "spline" else "offset"
    return family, f"cv={chosen} overlap={n}"


def source_map_kinds(
    report: dict, names, *, aliases: dict | None = None, fixed: dict | None = None
) -> dict:
    """Families for training sources; ``aliases`` map names to report entries.

    ``fixed`` overrides (e.g. the six legacy sources whose family comes from
    the selected decision run) are returned unchanged.
    """
    aliases = aliases or {}
    fixed = fixed or {}
    out = {}
    for name in names:
        if name in fixed:
            out[name] = dict(family=fixed[name], reason="fixed by selected decision")
            continue
        key = aliases.get(name, name)
        family, reason = family_for(None if key is None else overlap_entry(report, key))
        out[name] = dict(family=family, reason=reason, overlap_entry=key)
    return out
