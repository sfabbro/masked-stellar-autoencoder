"""Common-scale stellar labels from survey records and the anchored survey maps.

Survey s reports y_s = x + delta_s(x) + noise (``anchored_survey_map``), with x
the star's common-scale (Teff, log g, [Fe/H]). A fine-tuned anchored
checkpoint's maps turn every uncensored record into a common-scale measurement
by solving x + delta_s(x) = y_s for the record's observed parameters (Newton,
started at the posterior head's prediction; unobserved parameters stay at the
head's prediction). Its variance is J^-1 diag(e^2 + sigma_s^2) J^-T: the quoted
error e and the survey's learned map noise sigma_s, with J the map Jacobian
at the solution. A star's records are combined per parameter by inverse
variance. With ``iterations=0`` the map is evaluated at the head's prediction
(x = y_s - delta_s(x_head)); Newton removes the head's influence wherever a
map has a slope. Offset-only and identity maps (the [Fe/H] core, the anchors)
give the same value either way.

Records go through the same selection as in training (``AnchoredRecords``:
excluded surveys, record cuts, spectra-group deduplication, ``min_records``)
and censored records (upper limits) are left out.

``frozen_core_reference`` is the model-independent high-resolution [Fe/H]
reference of a table row: one record per star, original papers first, then
Gaia-ESO, then compilations (``CORE_PRIORITY``), taken from the raw records
whatever a variant later excludes or deduplicates. It defines the frozen
evaluation set (``scripts/core_eval_rows.py``); ``fe_h_core`` is the same
record mapped to the common scale (core maps are offsets).
"""

from __future__ import annotations

import numpy as np
import torch

from .anchored_multisurvey import MAPPED, AnchoredRecords
from .anchored_survey_map import AnchoredSurveyMaps

CORE_PRIORITY = (
    "bensby2014", "aoki2013_segue", "li2022_lamost_subaru", "smss_yong2021",
    "yong2013_program", "yong2013_literature", "rpa_hansen2018", "rpa_holmbeck2020",
    "sakari2018_rpa", "ezzeddine2020_rpa", "roederer2014", "kirby2010_highres",
    "gaia_eso_dr5_1", "jinabase_2021", "saga_mw", "pastel_2020",
)  # fmt: skip

# Field names of labels.npz (a contract read by other pipelines).
LABEL_FIELDS = (
    "source_id", "table_row", "teff", "e_teff", "logg", "e_logg", "fe_h", "e_fe_h",
    "n_teff", "n_logg", "n_feh", "has_core_feh", "split",
)  # fmt: skip


def frozen_core_reference(rec, core_prefixes, n_rows=None):
    """(ref, error, survey index) of the frozen [Fe/H] core reference per row.

    ``rec``: survey records (``records.npz`` arrays). Only surveys whose name
    starts with one of ``core_prefixes`` count; among them ``CORE_PRIORITY``
    decides, then the remaining core surveys in record order. Rows without a
    core record get NaN and survey -1.
    """
    names = [str(n) for n in rec["survey_names"]]
    fe = [str(p) for p in rec["parameters"]].index("fe_h")
    value = np.asarray(rec["value"], dtype="f8")[:, fe]
    error = np.asarray(rec["error"], dtype="f8")[:, fe]
    row = np.asarray(rec["row"], dtype="i8")
    survey = np.asarray(rec["survey"], dtype="i8")
    n = int(row.max()) + 1 if n_rows is None else int(n_rows)
    core = [i for i, s in enumerate(names) if s.startswith(tuple(core_prefixes))]
    order = [names.index(s) for s in CORE_PRIORITY if s in names]
    order = [i for i in order if i in core]
    order += [i for i in core if i not in order]
    ref, err = np.full(n, np.nan), np.full(n, np.nan)
    src = np.full(n, -1, dtype="i8")
    for s in reversed(order):  # earlier in the priority overwrites later
        m = (survey == s) & np.isfinite(value) & (row < n)
        ref[row[m]], err[row[m]], src[row[m]] = value[m], error[m], s
    return ref, err, src


def table_split(path, config, n_rows):
    """Split per table row: 0 train, 1 valid, 2 test, -1 not kept."""
    from .finetune_data import kept_rows, split_row_positions

    kept = kept_rows(str(path), config)
    mp = config["finetuning"].get("metal_poor") or {}
    split = np.full(int(n_rows), -1, dtype="i1")
    for k, pos in enumerate(split_row_positions(len(kept), mp)):
        split[kept[pos]] = k
    return split


def load_anchored_maps(state, device="cpu", dtype=torch.float64):
    """The checkpoint's ``AnchoredSurveyMaps`` (eval mode), with its tiers."""
    if state.get("map_kind") != "anchored":
        raise ValueError("Common-scale labels need an anchored checkpoint")
    sd = state["survey_maps_state_dict"]
    maps = AnchoredSurveyMaps(
        np.asarray(state["map_tier"]), np.asarray(state["map_core"]),
        np.asarray(state["map_counts"]), basis=int(sd["mix"].shape[-1]),
        hidden=int(sd["phi.0.weight"].shape[0]),
    )  # fmt: skip
    keys = list(maps.state_dict())
    missing = [k for k in keys if k not in sd]
    if missing:
        raise ValueError(f"Survey-map state is missing {missing}")
    maps.load_state_dict({k: sd[k] for k in keys})
    return maps.to(device=device, dtype=dtype).eval()


def map_to_common_scale(
    maps, survey, value, error, observed, x0, *, iterations=5, chunk=1 << 18
):  # fmt: skip
    """Common-scale values and variances of survey records (scaled units).

    ``survey`` (R,) checkpoint survey index, ``value``/``error`` (R, 3) the
    record (NaN where unobserved), ``observed`` (R, 3) bool (finite and not
    censored), ``x0`` (R, 3) the star's starting point (posterior head). Returns
    (x, var, converged): x and var (R, 3) are NaN where not observed;
    ``converged`` (R,) is False where Newton did not converge, in which case the
    head-point evaluation (``iterations=0``) is returned for that record.
    """
    dtype = next(maps.parameters()).dtype
    dev = next(maps.parameters()).device
    R = len(survey)
    xs, vs, cs = [], [], []
    eye = torch.eye(len(MAPPED), dtype=dtype, device=dev)
    with torch.no_grad():
        for i in range(0, R, chunk):

            def t(a, dt=dtype):
                return torch.as_tensor(np.asarray(a[i : i + chunk]), device=dev).to(dt)

            s = t(survey, torch.long)
            o = t(observed, torch.bool)
            v = torch.nan_to_num(t(value))
            e = torch.nan_to_num(t(error))
            x = torch.nan_to_num(t(x0))
            oo = o[:, :, None] & o[:, None, :]
            sig2 = maps.sigma(s).to(dtype) ** 2
            noise = torch.where(o, e * e + sig2, torch.zeros_like(e))

            def solve(x):
                y, jac = maps(x, s)
                jm = torch.where(oo, jac, eye)
                return torch.where(o, y - v, torch.zeros_like(v)), jm

            y0, _ = maps(x, s)
            x = torch.where(o, v - (y0 - x), x)  # delta at the head's prediction
            first = x.clone()
            for _ in range(int(iterations)):
                f, jm = solve(x)
                x = x - torch.linalg.solve(jm, f.unsqueeze(-1)).squeeze(-1)
            f, jm = solve(x)
            ok = torch.isfinite(x).all(-1) & (f.abs().amax(-1) < 1e-6)
            if int(iterations) == 0:
                ok = torch.ones_like(ok)
            x = torch.where(ok[:, None], x, first)
            if not bool(ok.all()):
                _, jm0 = solve(first)
                jm = torch.where(ok[:, None, None], jm, jm0)
            inv = torch.linalg.inv(jm)
            var = torch.einsum("rij,rj,rij->ri", inv, noise, inv)
            nan = torch.full_like(x, float("nan"))
            xs.append(torch.where(o, x, nan).cpu())
            vs.append(torch.where(o, var, nan).cpu())
            cs.append(ok.cpu())
    if not xs:
        empty = np.empty((0, len(MAPPED)))
        return empty, empty.copy(), np.empty(0, bool)
    return (
        torch.cat(xs).numpy().astype("f8"),
        torch.cat(vs).numpy().astype("f8"),
        torch.cat(cs).numpy(),
    )


def combine_inverse_variance(row, x, var, n_rows):
    """Per-row inverse-variance mean, error and record count, (n_rows, P) each."""
    row = np.asarray(row, dtype="i8")
    P = x.shape[1]
    mean = np.full((n_rows, P), np.nan)
    err = np.full((n_rows, P), np.nan)
    count = np.zeros((n_rows, P), dtype="i4")
    for k in range(P):
        ok = np.isfinite(x[:, k]) & np.isfinite(var[:, k]) & (var[:, k] > 0)
        w = np.where(ok, 1.0 / np.where(ok, var[:, k], 1.0), 0.0)
        sw = np.bincount(row, w, minlength=n_rows)[:n_rows]
        swx = np.bincount(row, w * np.where(ok, x[:, k], 0.0), minlength=n_rows)
        count[:, k] = np.bincount(row[ok], minlength=n_rows)[:n_rows]
        has = sw > 0
        mean[has, k] = swx[:n_rows][has] / sw[has]
        err[has, k] = 1.0 / np.sqrt(sw[has])
    return mean, err, count


def _survey_index(record_names, checkpoint_names):
    """Checkpoint survey index of each records survey (-1: not in the maps)."""
    ck = [str(n) for n in checkpoint_names]
    return np.array([ck.index(n) if n in ck else -1 for n in record_names], "i8")


def build_common_scale_labels(config, state, rec, head_scaled, *, iterations=5,
                              device="cpu", log=print):  # fmt: skip
    """Common-scale labels of every table row.

    ``head_scaled`` (n_rows, 3): the posterior head's (Teff, log g, [Fe/H]) in
    the checkpoint's scaled label units (the Newton start). Returns a dict of
    per-row arrays in physical units: ``teff``, ``e_teff``, ``logg``,
    ``e_logg``, ``fe_h``, ``e_fe_h``, ``n_teff``, ``n_logg``, ``n_feh``,
    ``fe_h_core``/``e_fe_h_core`` (the frozen core reference on the common
    scale), ``has_core_feh``, ``fe_h_ref``/``e_fe_h_ref``/``fe_h_ref_survey``
    (the raw reference), and ``diagnostics`` (a dict).
    """
    ms = config["multisurvey"]
    pproc = config.get("preprocessing") or {}
    if pproc.get("teff_target_space", "linear") != "linear":
        raise ValueError("Survey records are mapped in linear Teff only")
    label_names = [str(n) for n in state["label_names"]]
    scalers = state["label_scalers"]
    head_scaled = np.asarray(head_scaled, dtype="f8")
    n_rows = len(head_scaled)
    records = AnchoredRecords(
        rec, scalers, label_names, n_rows, torch.device("cpu"),
        core=ms.get("fe_h_core", ()), spectra_groups=ms.get("spectra_groups"),
        cuts=ms.get("record_cuts"), censor=ms.get("censor"),
        min_records=int(ms.get("min_records", 0)),
        min_records_keep_core=bool(ms.get("min_records_keep_core", True)),
        exclude=ms.get("exclude_surveys") or (),
    )  # fmt: skip
    index = _survey_index(records.names, state["survey_names"])
    ptr = records.ptr.numpy()
    row = np.repeat(np.arange(n_rows), np.diff(ptr))
    value = records.value.numpy().astype("f8")
    error = records.error.numpy().astype("f8")
    censored = records.censored.numpy()
    observed = np.isfinite(value) & ~censored
    survey = index[records.survey.numpy()]
    if (survey[observed.any(1)] < 0).any():
        bad = sorted({records.names[s] for s in records.survey.numpy()[survey < 0]})
        raise ValueError(f"Surveys without a map in the checkpoint: {bad}")
    maps = load_anchored_maps(state, device=device)
    x, var, conv = map_to_common_scale(
        maps, survey, value, error, observed, head_scaled[row], iterations=iterations
    )
    shift = np.nan
    if iterations:
        x0, _, _ = map_to_common_scale(
            maps, survey, value, error, observed, head_scaled[row], iterations=0
        )
        d = np.abs(x - x0)
        shift = [float(np.nanmedian(d[:, k])) if np.isfinite(d[:, k]).any()
                 else float("nan") for k in range(3)]  # fmt: skip
    mean, err, count = combine_inverse_variance(row, x, var, n_rows)
    li = [label_names.index(p) for p in MAPPED]
    mu = np.array([float(scalers[j].mean_[0]) for j in li])
    sd = np.array([float(scalers[j].scale_[0]) for j in li])
    phys, ephys = mean * sd + mu, err * sd
    out = {}
    for k, (name, cnt) in enumerate(zip(MAPPED, ("n_teff", "n_logg", "n_feh"),
                                        strict=True)):  # fmt: skip
        out[name], out["e_" + name], out[cnt] = phys[:, k], ephys[:, k], count[:, k]
    # Frozen core reference, mapped through its survey's (offset) map.
    ref, eref, src = frozen_core_reference(rec, ms.get("fe_h_core", ()), n_rows)
    has = np.isfinite(ref) & (src >= 0)
    rows = np.flatnonzero(has)
    rnames = [str(n) for n in rec["survey_names"]]
    ck = _survey_index(rnames, state["survey_names"])[src[rows]]
    if (ck < 0).any():
        bad = sorted({rnames[s] for s in src[rows][ck < 0]})
        raise ValueError(f"Core surveys without a map in the checkpoint: {bad}")
    k = MAPPED.index("fe_h")
    v = np.full((len(rows), 3), np.nan)
    e = np.zeros((len(rows), 3))
    v[:, k] = (ref[rows] - mu[k]) / sd[k]
    e[:, k] = np.nan_to_num(eref[rows]) / sd[k]
    start = np.where(np.isfinite(mean[rows]), mean[rows], head_scaled[rows])
    obs = np.zeros((len(rows), 3), bool)
    obs[:, k] = True
    xc, vc, _ = map_to_common_scale(maps, ck, v, e, obs, start, iterations=iterations)
    out["fe_h_core"] = np.full(n_rows, np.nan)
    out["e_fe_h_core"] = np.full(n_rows, np.nan)
    out["fe_h_core"][rows] = xc[:, k] * sd[k] + mu[k]
    out["e_fe_h_core"][rows] = np.sqrt(vc[:, k]) * sd[k]
    out["has_core_feh"] = np.isfinite(out["fe_h_core"])
    out["fe_h_ref"], out["e_fe_h_ref"], out["fe_h_ref_survey"] = ref, eref, src
    out["diagnostics"] = dict(
        records=int(observed.any(1).sum()),
        censored_excluded=int(censored.sum()),
        not_converged=int((~conv & observed.any(1)).sum()),
        newton_median_shift_scaled=shift,
        iterations=int(iterations),
        rows_with_core=int(out["has_core_feh"].sum()),
        complete_rows=int(np.isfinite(phys).all(1).sum()),
        dropped_surveys=list(records.dropped),
    )
    log(f"common-scale labels: {out['diagnostics']}")
    return out


def write_labels(config, state, rec, predictions, fits_path, out_path, *, iterations=5,
                 device="cpu", core_split=None, log=print):  # fmt: skip
    """Build and write ``labels.npz`` (``LABEL_FIELDS`` + frozen core reference).

    ``predictions``: the posterior head's predictions per table row
    (``frozen_latents.encode_table``: ``source_id``, ``post_scaled``).
    ``core_split``: optional ``rows_split.npz`` (``scripts/core_eval_rows.py``)
    whose split must agree with the table split. Returns the diagnostics dict.
    """
    n = len(predictions["source_id"])
    lab = build_common_scale_labels(
        config, state, rec, predictions["post_scaled"], iterations=iterations,
        device=device, log=log,
    )  # fmt: skip
    diag = lab.pop("diagnostics")
    split = table_split(fits_path, config, n)
    if core_split is not None:
        cs = np.load(core_split)
        ok = cs["table_row"] < n
        bad = int((split[cs["table_row"][ok]] != cs["split"][ok]).sum())
        diag["core_split_mismatch"] = bad
        if bad:
            raise ValueError(
                f"{bad} core rows have a different split than {core_split}"
            )
    diag["split_counts"] = [int((split == k).sum()) for k in (0, 1, 2, -1)]
    np.savez(
        out_path, source_id=np.asarray(predictions["source_id"]),
        table_row=np.arange(n), split=split,
        survey_names=np.array([str(s) for s in rec["survey_names"]]), **lab,
    )  # fmt: skip
    return diag
