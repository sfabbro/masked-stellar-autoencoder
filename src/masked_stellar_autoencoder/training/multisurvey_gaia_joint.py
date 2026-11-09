"""Gaia native abundance observations for a six-target joint sample model.

The caller supplies existing eligible Gaia IDs and FIT/TUNE/etc. roles; this
module assigns no roles. Its five coordinates are ONE GSP-Spec observation,
not five independent surveys. The covariance of the native quantities is not
provided. Percentiles remain descriptive bounds, not Gaussian uncertainties.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

FIELDS = (
    "teff_gspspec",
    "logg_gspspec",
    "mh_gspspec",
    "alphafe_gspspec",
    "fem_gspspec",
)
UNITS = ("K", "dex(cm/s2)", "dex", "dex", "dex")
DEPENDENCY = "gaia-dr3-gspspec-rvs"
GROUP = "native_observations/gaia_dr3_gspspec"


def _ids(value, *, unique=True):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind not in "iu" or value.dtype.itemsize != 8:
        raise ValueError("Gaia IDs must be exact 64-bit integers")
    if np.any(value <= 0) or (
        value.dtype.kind == "u" and np.any(value > np.iinfo("i8").max)
    ):
        raise ValueError("Gaia IDs must be positive signed-int64 range")
    value = value.astype("i8", copy=False)
    if unique and len(np.unique(value)) != len(value):
        raise ValueError("Gaia IDs must be unique")
    return value


def _state(path):
    x = os.stat(path)
    return x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns


def _verify(path, expected):
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(c not in "0123456789abcdef" for c in expected)
    ):
        raise ValueError("Explicit lowercase SHA256 is required")
    before = _state(path)
    with open(path, "rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected or before != _state(path):
        raise ValueError("Pinned Gaia source changed or has wrong SHA256")
    return before


def _take(dataset, rows):
    """h5py only accepts sorted unique indices; restore request order."""
    rows = np.asarray(rows, dtype="i8")
    if not len(rows):
        return np.empty((0, *dataset.shape[1:]), dtype=dataset.dtype)
    order = np.argsort(rows, kind="stable")
    reverse = np.argsort(order, kind="stable")
    return np.asarray(dataset[rows[order]])[reverse]


def build_gaia_records(
    sidecar, joined, requested_ids, *, sidecar_sha256, joined_sha256
):
    """Return one native Gaia record per matched requested star; no role assignment.

    Input hashes bind exactly the completed sidecar and Fe/M join. Missing Fe/M
    stays missing and all unmatched requested IDs remain represented. Only the
    published selected masks are used; unknown Fe upper-limit flags never become
    detections. Neither the 16/84 percentiles nor Fe-line scatter is reinterpreted
    as a known Gaussian measurement error.
    """
    sidecar, joined = map(Path, (sidecar, joined))
    before = (_verify(sidecar, sidecar_sha256), _verify(joined, joined_sha256))
    requested = _ids(requested_ids)
    n = len(requested)
    values = np.full((n, 5), np.nan, dtype="f8")
    bounds = np.full((n, 5, 2), np.nan, dtype="f8")
    selected = np.zeros((n, 5), dtype=bool)
    source_row = np.full(n, -1, dtype="i8")
    sidecar_index = np.full(n, -1, dtype="i8")
    fem_row = np.full(n, -1, dtype="i8")
    fe_flag = np.full(n, 255, dtype="u1")
    fe_uncertainty = np.full(n, 255, dtype="u1")
    with h5py.File(sidecar, "r") as sd, h5py.File(joined, "r") as fd:
        if (
            fd.attrs.get("schema") != "gaia-gspspec-fem-exact-join-v1"
            or fd.attrs.get("sidecar_sha256") != sidecar_sha256
        ):
            raise ValueError("Fe/M join schema or exact sidecar binding differs")
        gaia = sd[GROUP]
        sid = _ids(gaia["source_id"][:])
        order = np.argsort(sid, kind="stable")
        ordered = sid[order]
        idx = np.searchsorted(ordered, requested)
        matched = idx < len(ordered)
        ix = np.flatnonzero(matched)
        matched[ix] &= ordered[idx[ix]] == requested[ix]
        ix = np.flatnonzero(matched)
        sidecar_index[ix] = order[idx[ix]]
        rows = sidecar_index[ix]
        source_row[ix] = _take(gaia["source_row"], rows)
        for key in ("value", "observed", "percentile16", "percentile84"):
            if gaia[key].shape != (len(sid), 3):
                raise ValueError("Gaia atmospheric arrays have incorrect shape")
        atmospheric = _take(gaia["value"], rows)
        values[ix, :2] = atmospheric[:, :2]
        selected[ix, :2] = _take(gaia["observed"], rows)[:, :2]
        bounds[ix, :2, 0] = _take(gaia["percentile16"], rows)[:, :2]
        bounds[ix, :2, 1] = _take(gaia["percentile84"], rows)[:, :2]
        for k, name in ((2, "mh"), (3, "alphafe")):
            values[ix, k] = _take(gaia[name + "_gspspec"], rows)
            bounds[ix, k, 0] = _take(gaia[name + "_gspspec_lower"], rows)
            bounds[ix, k, 1] = _take(gaia[name + "_gspspec_upper"], rows)
            mask_name = "selected_observed_mh" if k == 2 else "selected_observed_alpha"
            selected[ix, k] = _take(gaia[mask_name], rows)
        fid = _ids(fd["source_id"][:])
        if np.any(fid[1:] <= fid[:-1]):
            raise ValueError("Fe/M source IDs must be strictly sorted")
        pos = np.searchsorted(fid, requested)
        fm = pos < len(fid)
        j = np.flatnonzero(fm)
        fm[j] &= fid[pos[j]] == requested[j]
        j = np.flatnonzero(fm)
        fem_row[j] = pos[j]
        if (
            np.any(~matched[j])
            or not _take(fd["sidecar_matched"], pos[j]).astype(bool).all()
        ):
            raise ValueError("Fe/M source match missing in sidecar")
        if not np.array_equal(_take(fd["sidecar_row"], pos[j]), sidecar_index[j]):
            raise ValueError("Fe/M exact-ID sidecar row association differs")
        if not np.isclose(
            _take(fd["mh_gspspec"], pos[j]),
            values[j, 2],
            atol=2e-5,
            rtol=0,
            equal_nan=True,
        ).all():
            raise ValueError("Fe/M native M/H differs from Gaia sidecar")
        values[j, 4] = _take(fd["fem_gspspec"], pos[j])
        bounds[j, 4, 0] = _take(fd["fem_gspspec_lower"], pos[j])
        bounds[j, 4, 1] = _take(fd["fem_gspspec_upper"], pos[j])
        fe_flag[j] = _take(fd["fe_upper_limit_flag"], pos[j])
        fe_uncertainty[j] = _take(fd["fe_uncertainty_flag"], pos[j])
        candidate = _take(fd["provisional_derived_feh_candidate"], pos[j]).astype(bool)
        detected = _take(fd["fe_detection_flag_zero"], pos[j]).astype(bool)
        if np.any(candidate & (~detected | (fe_flag[j] != ord("0")) | ~selected[j, 2])):
            raise ValueError("Fe/M quality/upper-limit evidence conflicts")
        selected[j, 4] = candidate
    if before != (_state(sidecar), _state(joined)):
        raise ValueError("Gaia source changed while reading")
    if np.any(selected & (~np.isfinite(values) | (values == -9999))):
        raise ValueError("Selected Gaia value is absent or nonfinite")
    raw = np.isfinite(values) & (values != -9999)
    return {
        "source_id": requested.copy(),
        "source_row": source_row,
        "sidecar_index": sidecar_index,
        "fem_join_row": fem_row,
        "source_matched": matched,
        "value": values,
        "raw_observed": raw,
        "observed": selected,
        "error": np.full((n, 5), np.nan),
        "error_known": np.zeros((n, 5), dtype=bool),
        "percentile_bounds": bounds,
        "fe_upper_limit_flag_ascii": fe_flag,
        "fe_uncertainty_flag_ascii": fe_uncertainty,
        "fields": FIELDS,
        "units": UNITS,
        "dependency_id": DEPENDENCY,
        "sidecar_sha256": sidecar_sha256,
        "join_sha256": joined_sha256,
        "scope": "One source observation per star; native Fe/M and M/H are dependent. No derived Fe/H observation; no assigned roles or known covariance.",
    }


class GaiaNativeObservationMap(nn.Module):
    """Map one shared six-vector latent sample into native Gaia five-coordinates.

    Native six-vector order: Teff [K], logg [dex], Fe/H [dex], alpha [dex],
    age, distance. Corrections use fixed FIT normalization and learn separate
    source-scale adjustments. At initialization M/H and Fe I/H both equal Fe/H
    and alpha is unchanged; this is a numerical starting point, not a physical
    equality or calibrated transformation. Fe I/M is defined by their
    difference, preserving the native reference identity in every sample.
    """

    def __init__(self, center, scale):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float64)
        scale = torch.as_tensor(scale, dtype=torch.float64)
        if (
            center.shape != (6,)
            or scale.shape != (6,)
            or not torch.isfinite(center).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Fixed six-coordinate FIT center and scale required")
        self.register_buffer("center", center.clone())
        self.register_buffer("scale", scale.clone())
        self.correction = nn.Linear(6, 3, dtype=torch.float64)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, latent):
        if (
            latent.ndim != 3
            or latent.shape[-1] != 6
            or not torch.isfinite(latent).all()
        ):
            raise ValueError("Shared finite latent samples must be [sample,star,6]")
        z = (latent.to(dtype=self.center.dtype) - self.center) / self.scale
        delta = self.correction(z)
        mh = latent[..., 2] + delta[..., 0]
        iron = latent[..., 2] + delta[..., 1]
        alpha = latent[..., 3] + delta[..., 2]
        return torch.stack(
            (latent[..., 0], latent[..., 1], mh, alpha, iron - mh), dim=-1
        )


def gaia_native_energy_score(
    latent_samples,
    mapper,
    records,
    *,
    scale,
    noise_generator,
    unknown_error_floor,
    assume_independent_gaussian_errors,
    reduction="mean",
):
    """One joint sample score in native Gaia units with explicit noise assumptions.

    Known diagonal errors are not asserted here: all Gaia errors are marked
    unknown until a covariance model or a separate documented policy is supplied.
    This optional approximation explicitly assumes independent Gaussian
    measurement noise, which is unverified for GSP-Spec; supplied positive
    floors are numerical choices, not inferred errors. Scores are per native
    Gaia record, not a product of independent scalar or derived-Fe/H terms.
    """
    from masked_stellar_autoencoder.training.multisurvey_masked_energy import (
        masked_noise_aware_energy_score,
    )

    if (
        tuple(records.get("fields", ())) != FIELDS
        or records.get("dependency_id") != DEPENDENCY
    ):
        raise ValueError(
            "Native Gaia field order and one-record dependency are required"
        )
    if assume_independent_gaussian_errors is not True:
        raise ValueError(
            "Gaia source covariance is unknown; explicit diagonal Gaussian assumption required"
        )
    mapped = mapper(latent_samples)
    target = torch.as_tensor(records["value"], device=mapped.device, dtype=mapped.dtype)
    observed = torch.as_tensor(
        records["observed"], device=mapped.device, dtype=torch.bool
    )
    error = torch.as_tensor(records["error"], device=mapped.device, dtype=mapped.dtype)
    known = torch.as_tensor(
        records["error_known"], device=mapped.device, dtype=torch.bool
    )
    if known.any():
        raise ValueError("Gaia percentile bounds are not validated measurement errors")
    return masked_noise_aware_energy_score(
        mapped,
        target,
        observed,
        error,
        known,
        scale=scale,
        generator=noise_generator,
        assume_independent_errors=True,
        unknown_error_floor=unknown_error_floor,
        reduction=reduction,
    )


class GaiaFitView:
    """Pinned Gaia record view for exactly an existing verified FIT population."""

    def __init__(self, gate, sidecar, joined, *, sidecar_sha256, joined_sha256):
        from masked_stellar_autoencoder.training.multisurvey_training import (
            FitIdentityGate,
        )

        if not isinstance(gate, FitIdentityGate):
            raise ValueError("A verified existing FIT identity gate is required")
        self.gate = gate
        self.sidecar, self.joined = str(sidecar), str(joined)
        self.records = build_gaia_records(
            sidecar,
            joined,
            gate.ids,
            sidecar_sha256=sidecar_sha256,
            joined_sha256=joined_sha256,
        )
        self.states = (_state(sidecar), _state(joined))
        for value in self.records.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def get(self, ids):
        if self.states != (_state(self.sidecar), _state(self.joined)):
            raise ValueError("Pinned Gaia source changed after FIT view construction")
        ids = _ids(ids)
        pos = np.searchsorted(self.gate.ids, ids)
        if np.any(pos >= len(self.gate.ids)) or not np.array_equal(
            self.gate.ids[pos], ids
        ):
            raise ValueError("Gaia view contains no outside-FIT identities")
        return {
            key: value[pos].copy() if isinstance(value, np.ndarray) else value
            for key, value in self.records.items()
        }


def prepare_joint_fit_batch(gate, gaia_view, ids):
    """Reconstruct verified FIT records and replace any old Gaia-only record.

    Original three-target batch is revalidated by FitIdentityGate before its
    Gaia subrecord is removed. The Gaia replacement is built from the pinned
    sidecar, one record per star; non-Gaia sources and natural inputs persist.
    """
    import copy

    from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
    from masked_stellar_autoencoder.training.multisurvey_training import FitIdentityGate

    if not isinstance(gate, FitIdentityGate) or gaia_view.gate is not gate:
        raise ValueError("Shared FIT identity gate and Gaia view required")
    ids = _ids(ids)
    if not len(ids):
        raise ValueError("A nonempty unique FIT identity batch is required")
    pos = np.searchsorted(gate.ids, ids)
    if np.any(pos >= len(gate.ids)) or not np.array_equal(gate.ids[pos], ids):
        raise ValueError("Requested star is outside the pinned FIT split")
    original = [gate.dataset[int(i)] for i in pos]
    full = collate_multisurvey(original, repeat_policy=gate.repeat_policy)
    gate.verify_batch(full)
    gaia = gaia_view.get(ids)
    remaining = []
    for i, star in enumerate(original):
        adjusted = copy.deepcopy(star)
        found = []
        for record in adjusted["observations"]:
            if record["dependency_id"] == DEPENDENCY:
                if record["source"] != "gaia_dr3_gspspec":
                    raise ValueError(
                        "Other survey shares Gaia dependency; explicit joint model required"
                    )
                found.append(record)
        if len(found) > 1:
            raise ValueError("Repeated Gaia source record requires explicit resolution")
        if found:
            old = found[0]
            if (
                not gaia["source_matched"][i]
                or old["source_row"] != gaia["source_row"][i]
            ):
                raise ValueError("Gaia source-record provenance differs")
            if np.any(old["observed"][2:]) or not np.array_equal(
                old["observed"][:2], gaia["observed"][i, :2]
            ):
                raise ValueError("Gaia legacy and native masks differ")
            if not np.allclose(
                old["value"][:2], gaia["value"][i, :2], rtol=0, atol=0, equal_nan=True
            ):
                raise ValueError("Gaia legacy and native values differ")
            adjusted["observations"] = [
                r for r in adjusted["observations"] if r is not old
            ]
        remaining.append(adjusted)
    base = collate_multisurvey(remaining, repeat_policy=gate.repeat_policy)
    return {
        "base": base,
        "gaia": gaia,
        "original": full,
        "role": "fit",
        "scope": "Composite per-record scores on shared six-target samples; no cross-survey error covariance claim.",
    }


class ThreeTargetSurveyMap(nn.Module):
    """Source-specific affine corrections on common six-vector native draws.

    At initialization each observed atmosphere equals the first three latent
    coordinates. Relative offsets obey the declared per-coordinate source
    weight gauge, which does not identify physical truth. Sources may contribute
    only where their gauge weight is positive.
    """

    def __init__(self, center, scale, weights):
        super().__init__()
        center = torch.as_tensor(center, dtype=torch.float64)
        scale = torch.as_tensor(scale, dtype=torch.float64)
        weights = torch.as_tensor(weights, dtype=torch.float64)
        if (
            center.shape != (6,)
            or scale.shape != (6,)
            or not torch.isfinite(center).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Fixed six-coordinate FIT center and scale required")
        if (
            weights.ndim != 2
            or weights.shape[1] != 3
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
            or (weights.sum(0) <= 0).any()
        ):
            raise ValueError("Positive per-target source gauge weights required")
        self.register_buffer("center", center.clone())
        self.register_buffer("scale", scale.clone())
        self.register_buffer("weights", weights.clone())
        self.correction = nn.Linear(6, 3 * len(weights), dtype=torch.float64)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, latent):
        if (
            latent.ndim != 3
            or latent.shape[-1] != 6
            or not torch.isfinite(latent).all()
        ):
            raise ValueError("Shared finite latent samples must be [sample,star,6]")
        z = (latent.to(dtype=self.center.dtype) - self.center) / self.scale
        offset = self.correction(z).reshape(*latent.shape[:2], len(self.weights), 3)
        gauge = (offset * self.weights).sum(-2, keepdim=True) / self.weights.sum(0)
        return latent[..., :3].unsqueeze(-2) + offset - gauge


def joint_multisurvey_energy_score(
    latent,
    batch,
    *,
    gaia_map,
    survey_map,
    source_floor,
    anchor_floor,
    gaia_floor,
    survey_scale,
    gaia_scale,
    noise_generator,
    assume_independent_gaussian_errors,
):
    """Composite per-record energy score; all records use the same latent draws.

    This is a mean of available source-record scores per star, then a mean over
    stars. No source-record multiplication, population-weight, common-scale
    identification, or cross-survey covariance claim is made. A single
    declared diagonal Gaussian error approximation is required.
    """
    from masked_stellar_autoencoder.training.multisurvey_masked_energy import (
        masked_noise_aware_energy_score,
    )

    if assume_independent_gaussian_errors is not True:
        raise ValueError(
            "Explicit diagonal Gaussian measurement-noise assumption required"
        )
    base = batch["base"]
    gaia = batch["gaia"]
    if (
        not isinstance(latent, torch.Tensor)
        or latent.ndim != 3
        or latent.shape[-1] != 6
    ):
        raise ValueError("Latent samples must have shape [sample,star,6]")
    n = latent.shape[1]
    if len(base["source_id"]) != n or not np.array_equal(
        base["source_id"].cpu().numpy(), gaia["source_id"]
    ):
        raise ValueError("Latent samples, survey and Gaia identities must align")
    device = latent.device
    mapped = survey_map(latent)
    mask = base["observed"].to(device=device, dtype=torch.bool)
    if (
        mask.ndim != 3
        or mask.shape[0] != n
        or mask.shape[-1] != 3
        or mapped.shape[2] != len(base["source_registry"])
    ):
        raise ValueError("Three-target survey registry and masks differ")
    source = base["source_index"].to(device=device)
    anchors = base["is_anchor"].to(device=device)
    stars, records = mask.any(-1).nonzero(as_tuple=True)
    total = latent.sum(dim=(0, 2)) * 0
    count = torch.zeros(n, dtype=latent.dtype, device=device)
    if len(stars):
        chosen = source[stars, records]
        anchor = anchors[stars, records]
        if (anchor & (chosen != -1)).any() or (
            (~anchor) & ((chosen < 0) | (chosen >= mapped.shape[2]))
        ).any():
            raise ValueError("Source/anchor index differs")
        safe = chosen.clamp(min=0)
        if (
            mask[stars, records] & ~anchor[:, None] & (survey_map.weights[safe] <= 0)
        ).any():
            raise ValueError("A survey lacks gauge weight for an observed target")
        projected = mapped[:, stars, safe, :]
        projected = torch.where(anchor[None, :, None], latent[:, stars, :3], projected)
        values = base["value"].to(device=device, dtype=projected.dtype)[stars, records]
        errors = base["error"].to(device=device, dtype=projected.dtype)[stars, records]
        known = base["error_known"].to(device=device, dtype=torch.bool)[stars, records]
        observed = mask[stars, records]
        floors = torch.as_tensor(source_floor, dtype=projected.dtype, device=device)
        anchors_floor = torch.as_tensor(
            anchor_floor, dtype=projected.dtype, device=device
        )
        if floors.shape != (mapped.shape[2], 3) or anchors_floor.shape != (3,):
            raise ValueError(
                "Survey and anchor error floors must align with the source registry"
            )
        floor = torch.where(anchor[:, None], anchors_floor[None, :], floors[safe])
        scores = masked_noise_aware_energy_score(
            projected,
            values,
            observed,
            errors,
            known,
            scale=survey_scale,
            generator=noise_generator,
            assume_independent_errors=True,
            unknown_error_floor=floor,
            reduction="none",
        )
        total = total.index_add(0, stars, scores.to(dtype=total.dtype))
        count = count.index_add(0, stars, torch.ones_like(scores, dtype=count.dtype))
    active = np.flatnonzero(gaia["observed"].any(axis=1))
    if len(active):
        record = {
            key: (value[active] if isinstance(value, np.ndarray) else value)
            for key, value in gaia.items()
        }
        scores = gaia_native_energy_score(
            latent[:, active, :],
            gaia_map,
            record,
            scale=gaia_scale,
            noise_generator=noise_generator,
            unknown_error_floor=gaia_floor,
            assume_independent_gaussian_errors=True,
            reduction="none",
        )
        rows = torch.as_tensor(active, device=device)
        total = total.index_add(0, rows, scores.to(dtype=total.dtype))
        count = count.index_add(0, rows, torch.ones_like(scores, dtype=count.dtype))
    if not (count > 0).any():
        raise ValueError("At least one selected source observation is required")
    return (total[count > 0] / count[count > 0]).mean()
