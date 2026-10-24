"""CANFAR CUDA predictive draws and corner figures for explicitly selected models.

Requires model descriptors chosen from TUNE only and independently verified
numerical reports. This tool does not select a winner or interpret density
contours as conformal regions, physical posteriors, or empirical six-D truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run_xp_representation_probes as architecture
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.final_six_recipes import SparseSixHead

LABELS = (
    r"$T_{eff}$ [K]",
    r"$\log g$ [dex]",
    r"$[Fe/H]$ [dex]",
    r"$[\alpha/Fe]$ [dex]",
    "ln(age / Gyr)",
    "ln(distance / kpc)",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(path, expected):
    if sha(path) != expected:
        raise ValueError(f"Changed required artifact: {path}")


def draw_standardized(head, latent, count, *, generator):
    """Joint model samples: shared MDN component, independent scalar-NSF bases."""
    if latent.ndim != 2 or len(latent) != 1 or count < 2:
        raise ValueError("One fixed-object latent and at least two draws required")
    epsilon = torch.randn(
        (count, 6), device=latent.device, dtype=latent.dtype, generator=generator
    )
    if head.kind == "gaussian":
        mean, log_variance = head.head(latent).chunk(2, -1)
        return mean + (0.5 * log_variance).exp() * epsilon
    if head.kind == "mdn":
        weight, means, log_std = head.head._mixture_parameters(head.head(latent))
        # ONE mixture component per six-dimensional draw, not six components.
        component = torch.multinomial(
            weight[0].exp(), count, replacement=True, generator=generator
        )
        return means[0, component] + log_std[0, component].exp() * epsilon
    if head.kind != "factorized_nsf":
        raise ValueError("Unsupported predictive model family")
    result = []
    for column, scalar_head in enumerate(head.heads):
        distribution = scalar_head.distribution(latent)
        # DO NOT reuse shared one-dimensional point_nodes across coordinates:
        # that would manufacture a perfectly coupled six-dimensional copula.
        values = [
            distribution.transform.inv(block[:, None, None])[:, 0, 0]
            for block in epsilon[:, column].split(256)
        ]
        result.append(torch.cat(values))
    return torch.stack(result, -1)


def native_and_plot_samples(standardized, center, scale):
    coordinate = standardized * scale + center
    native = coordinate.clone()
    native[:, 4:] = native[:, 4:].exp()
    if not torch.isfinite(native).all():
        raise FloatingPointError("Nonfinite native samples; no clipping allowed")
    return native, coordinate  # First four native; last two natural-log axes.


def histogram_mass_levels(histogram, probabilities=(0.68, 0.95)):
    values = np.asarray(histogram, dtype="f8")
    if (
        values.ndim != 2
        or np.any(values < 0)
        or not np.isfinite(values).all()
        or not values.sum()
    ):
        raise ValueError("Finite nonempty nonnegative 2D histogram required")
    ordered = np.sort(values.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    thresholds = [
        float(ordered[min(np.searchsorted(cumulative, p), len(ordered) - 1)])
        for p in probabilities
    ]
    return thresholds, [
        float(values[values >= level].sum() / values.sum()) for level in thresholds
    ]


def plot_corner(draws, selection, row, model_name, kind, limits, png, pdf):
    """Histogram density contours enclose approximate predictive sample mass."""
    reference = np.asarray(selection["value"][row], dtype="f8").copy()
    good = np.asarray(selection["quality"][row], bool) & np.asarray(
        selection["observed"][row], bool
    )
    good &= np.isfinite(reference)
    if np.any(good[4:] & (reference[4:] <= 0)):
        raise ValueError("Nonpositive plotted age/distance reference")
    reference[4:][good[4:]] = np.log(reference[4:][good[4:]])
    pair_count = np.asarray(selection["fit_pair_observation_counts"])
    figure, axes = plt.subplots(6, 6, figsize=(17, 17), constrained_layout=False)
    for y in range(6):
        for x in range(6):
            axis = axes[y, x]
            if y < x:
                axis.set_visible(False)
                continue
            axis.set_xlim(*limits[x])
            if y == x:
                axis.hist(
                    draws[:, x],
                    bins=45,
                    range=limits[x],
                    density=True,
                    color="#366b9b",
                    alpha=0.8,
                )
                if good[x]:
                    axis.axvline(reference[x], color="#a52424", linewidth=1.4)
                axis.text(
                    0.96,
                    0.9,
                    f"FIT n={int(pair_count[x, x])}",
                    transform=axis.transAxes,
                    ha="right",
                    fontsize=8,
                )
            else:
                axis.set_ylim(*limits[y])
                histogram, xedges, yedges = np.histogram2d(
                    draws[:, x], draws[:, y], bins=45, range=[limits[x], limits[y]]
                )
                thresholds, _ = histogram_mass_levels(histogram)
                levels = sorted(set(thresholds))
                xx, yy = (xedges[1:] + xedges[:-1]) / 2, (yedges[1:] + yedges[:-1]) / 2
                axis.imshow(
                    histogram.T,
                    origin="lower",
                    extent=(*limits[x], *limits[y]),
                    aspect="auto",
                    cmap="Blues",
                    interpolation="nearest",
                    alpha=0.65,
                )
                if levels and levels[0] < histogram.max():
                    axis.contour(
                        xx,
                        yy,
                        histogram.T,
                        levels=[
                            level for level in levels if 0 < level < histogram.max()
                        ],
                        colors="#133d69",
                        linewidths=0.9,
                    )
                for coordinate, orient in ((x, "x"), (y, "y")):
                    if good[coordinate]:
                        if orient == "x":
                            axis.axvline(
                                reference[coordinate],
                                color="#a52424",
                                linewidth=0.7,
                                alpha=0.75,
                            )
                        else:
                            axis.axhline(
                                reference[coordinate],
                                color="#a52424",
                                linewidth=0.7,
                                alpha=0.75,
                            )
                if good[x] and good[y]:
                    quoted = np.asarray(selection["error_observed"][row], bool)
                    error = np.asarray(selection["error"][row], dtype="f8")
                    xerr = (
                        error[x]
                        if x < 4
                        and quoted[x]
                        and np.isfinite(error[x])
                        and error[x] > 0
                        else None
                    )
                    yerr = (
                        error[y]
                        if y < 4
                        and quoted[y]
                        and np.isfinite(error[y])
                        and error[y] > 0
                        else None
                    )
                    axis.errorbar(
                        reference[x],
                        reference[y],
                        xerr=xerr,
                        yerr=yerr,
                        fmt="+",
                        color="#a52424",
                        capsize=2,
                        markersize=5,
                    )
                    quantiles = np.asarray(selection["posterior_quantiles"][row])
                    for coordinate, orient in ((x, "x"), (y, "y")):
                        q = quantiles[coordinate]
                        if coordinate >= 4 and np.isfinite(q).all() and (q > 0).all():
                            low, _, high = np.log(q)
                            if orient == "x":
                                axis.plot(
                                    [low, high],
                                    [reference[y]] * 2,
                                    color="#d18b1b",
                                    linewidth=2,
                                )
                            else:
                                axis.plot(
                                    [reference[x]] * 2,
                                    [low, high],
                                    color="#d18b1b",
                                    linewidth=2,
                                )
                axis.text(
                    0.96,
                    0.9,
                    f"pair FIT n={int(pair_count[x, y])}",
                    transform=axis.transAxes,
                    ha="right",
                    fontsize=7,
                )
            if y == 5:
                axis.set_xlabel(LABELS[x], fontsize=10)
            else:
                axis.tick_params(labelbottom=False)
            if x == 0 and y:
                axis.set_ylabel(LABELS[y], fontsize=10)
            elif y != x:
                axis.tick_params(labelleft=False)
            axis.tick_params(labelsize=7)
    independence = (
        "Conditional independent coordinates"
        if kind != "mdn"
        else "Shared mixture component; conditional diagonal components; sparse pair support annotated"
    )
    title = f"Gaia {int(selection['source_id'][row])} — {model_name}\n{independence}; {len(draws)} joint draws; contours ≈68/95% model mass in plotted coordinates"
    figure.suptitle(title, fontsize=13, y=0.98)
    figure.text(
        0.08,
        0.027,
        "Red: eligible reported-label reference and quoted measurement error (where known). Gold: StarHorse 16–84% catalogue posterior span, NOT measurement error.\nDESI alpha below FeH −2: diagnostic only. Contours are learned conditional predictions, NOT conformal coverage or a physical posterior.\nSix-target empirical dependence remains weakly supported where pair FIT counts are small; no labels are imputed.",
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.09, top=0.92, wspace=0.08, hspace=0.08
    )
    figure.savefig(png, dpi=140)
    figure.savefig(pdf)
    plt.close(figure)


def validate_descriptor(descriptor, selection_sha, prepared_sha, evidence):
    if (
        evidence.get("selection_role") != "tune"
        or evidence.get("test_used_for_selection") is not False
        or descriptor["run_id"] not in evidence.get("selected_run_ids", [])
    ):
        raise ValueError("Explicit TUNE-only model-selection evidence required")
    manifest = json.loads(Path(descriptor["manifest"]).read_text())
    bound(descriptor["manifest"], descriptor["manifest_sha256"])
    bound(descriptor["checkpoint"], descriptor["checkpoint_sha256"])
    bound(descriptor["independent_report"], descriptor["independent_report_sha256"])
    report = json.loads(Path(descriptor["independent_report"]).read_text())
    if report.get("status") != "complete_strict_pass":
        raise ValueError(
            "Full independent numerical PASS required, not producer/pilot scores"
        )
    bindings = report["bindings"]
    if (
        bindings.get("checkpoint") != descriptor["checkpoint_sha256"]
        or bindings.get("input") != prepared_sha
        or bindings.get("source") != descriptor["expected_source_sha"]
    ):
        raise ValueError("Independent proof does not bind this checkpoint/source/input")
    if (
        manifest.get("status") != "complete"
        or manifest.get("source_sha") != descriptor["expected_source_sha"]
        or manifest.get("checkpoint_sha256") != descriptor["checkpoint_sha256"]
        or manifest.get("arm", {}).get("kind") != descriptor["kind"]
    ):
        raise ValueError("Model producer descriptor changed")
    if descriptor.get("state_format") not in {"joint_full", "frozen_head"}:
        raise ValueError("Explicit selected checkpoint state format required")
    return dict(
        selection_sha256=selection_sha,
        checkpoint_sha256=descriptor["checkpoint_sha256"],
        descriptor=descriptor,
        sampler_sha256=sha(__file__),
        head_module_sha256=sha(
            Path(__file__).resolve().parents[1]
            / "src/masked_stellar_autoencoder/training/final_six_recipes.py"
        ),
        independent_report_sha256=sha(descriptor["independent_report"]),
    )


def load_model_and_head(descriptor, device, feature_names):
    checkpoint = torch_load_trusted(
        descriptor["checkpoint"], map_location="cpu", weights_only=False
    )
    if descriptor["state_format"] == "frozen_head":
        bound(descriptor["encoder_checkpoint"], descriptor["encoder_checkpoint_sha256"])
        encoder = torch_load_trusted(
            descriptor["encoder_checkpoint"], map_location="cpu", weights_only=False
        )
        disk = dict(checkpoint, model_state_dict=encoder["model_state_dict"])
    else:
        disk = checkpoint
    if list(checkpoint["target_names"]) != [
        "teff",
        "logg",
        "fe_h",
        "alpha",
        "age",
        "distance",
    ] or list(checkpoint["feature_columns"]) != list(feature_names):
        raise ValueError("Selected checkpoint target/feature layout differs")
    model = (
        architecture.load_model(
            yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
            disk,
            device,
        )
        .double()
        .eval()
    )
    state = checkpoint["head_state_dict"]
    key = (
        "heads.0.input.weight"
        if descriptor["kind"] == "factorized_nsf"
        else "head.input.weight"
    )
    latent_dim = state[key].shape[1]
    head = (
        SparseSixHead(descriptor["kind"], latent_dim=latent_dim)
        .double()
        .to(device)
        .eval()
    )
    head.load_state_dict(state, strict=True)
    center, scale = [
        torch.tensor(checkpoint[name], dtype=torch.float64, device=device)
        for name in ("label_center", "label_scale")
    ]
    if (
        center.shape != (6,)
        or scale.shape != (6,)
        or not torch.isfinite(center).all()
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError("Selected FIT label standardization invalid")
    return model, head, center, scale


def run(config, output):
    if (
        os.environ.get("MSA_EXPERIMENT_CPU_ONLY") == "1"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Actual predictive sampling executes on CANFAR headless CUDA only"
        )
    if not 2 <= len(config.get("models", [])) <= 3:
        raise ValueError(
            "WAIT: provide two or three explicitly TUNE-selected verified descriptors"
        )
    if config["draws"] < 8192:
        raise ValueError("At least 8192 joint predictive samples required")
    bound(config["selection"], config["selection_sha256"])
    bound(config["selection_manifest"], config["selection_manifest_sha256"])
    selection_manifest = json.loads(Path(config["selection_manifest"]).read_text())
    if (
        selection_manifest["status"]
        != "representative_metalpoor_corner_selection_complete"
    ):
        raise ValueError("Fixed representative selection incomplete")
    bound(config["selection_evidence"], config["selection_evidence_sha256"])
    evidence = json.loads(Path(config["selection_evidence"]).read_text())
    with np.load(config["selection"], allow_pickle=False) as archive:
        selection = {key: archive[key] for key in archive.files}
    output.mkdir(parents=True, exist_ok=True)
    bindings_path = output / "bindings.json"
    bindings = dict(
        policy=config,
        sampler_sha256=sha(__file__),
        selection_sha256=sha(config["selection"]),
        model_bindings=[
            validate_descriptor(
                item,
                config["selection_sha256"],
                selection_manifest["bindings"]["prepared_sha256"],
                evidence,
            )
            for item in config["models"]
        ],
    )
    if bindings_path.exists() and json.loads(bindings_path.read_text()) != bindings:
        raise ValueError("Changed cache bindings; preserve completed samples/figures")
    bindings_path.write_text(json.dumps(bindings, indent=2) + "\n")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    receipts = []
    for descriptor in config["models"]:
        directory = output / descriptor["run_id"]
        directory.mkdir(exist_ok=True)
        model = head = center = scale = None
        for row, source_id in enumerate(selection["source_id"]):
            path = directory / f"{int(source_id)}-samples.npz"
            receipt_path = path.with_suffix(".json")
            if path.exists() or receipt_path.exists():
                if not path.exists() or not receipt_path.exists():
                    raise ValueError(
                        "Incomplete saved sample cache; preserve diagnostic"
                    )
                receipt = json.loads(receipt_path.read_text())
                bound(path, receipt["samples_sha256"])
                receipts.append(receipt)
                continue
            if model is None:
                model, head, center, scale = load_model_and_head(
                    descriptor, device, selection["feature_names"]
                )
            seed = int(config["seed"]) + int.from_bytes(
                hashlib.sha256(str(int(source_id)).encode()).digest()[:4], "little"
            )
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                latent = model.encoder(
                    torch.tensor(
                        selection["input"][row : row + 1],
                        device=device,
                        dtype=torch.float64,
                    )
                )
                standardized = draw_standardized(
                    head, latent, config["draws"], generator=generator
                )
                native, plotted = native_and_plot_samples(standardized, center, scale)
            np.savez_compressed(
                path,
                source_id=np.array(source_id),
                samples_native=native.cpu().numpy(),
                samples_plot_coordinates=plotted.cpu().numpy(),
            )
            receipt = dict(
                run_id=descriptor["run_id"],
                source_id=int(source_id),
                seed=seed,
                draws=config["draws"],
                samples_sha256=sha(path),
                sampling="intrinsic learned model distribution; quoted measurement errors not added to parameter draws",
                dependence="shared MDN component"
                if descriptor["kind"] == "mdn"
                else "independent conditional coordinates; independent NSF base draws",
            )
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            receipts.append(receipt)
        del model, head
        torch.cuda.empty_cache()
    for row, source_id in enumerate(selection["source_id"]):
        arrays = {}
        for descriptor in config["models"]:
            with np.load(
                output / descriptor["run_id"] / f"{int(source_id)}-samples.npz",
                allow_pickle=False,
            ) as a:
                arrays[descriptor["run_id"]] = a["samples_plot_coordinates"]
        combined = np.concatenate(list(arrays.values()))
        low, high = np.quantile(combined, [0.002, 0.998], axis=0)
        reference = selection["value"][row].copy()
        valid = selection["quality"][row] & np.isfinite(reference)
        reference[4:][valid[4:]] = np.log(reference[4:][valid[4:]])
        low[valid] = np.minimum(low[valid], reference[valid])
        high[valid] = np.maximum(high[valid], reference[valid])
        margin = np.maximum((high - low) * 0.07, 1e-7)
        limits = list(zip(low - margin, high + margin, strict=True))
        for descriptor in config["models"]:
            stem = output / descriptor["run_id"] / str(int(source_id))
            png, pdf, receipt = (
                stem.with_suffix(".png"),
                stem.with_suffix(".pdf"),
                stem.with_suffix(".figure.json"),
            )
            if receipt.exists():
                old = json.loads(receipt.read_text())
                bound(png, old["png_sha256"])
                bound(pdf, old["pdf_sha256"])
                continue
            if png.exists() or pdf.exists():
                raise ValueError("Incomplete figure cache; preserve original files")
            plot_corner(
                arrays[descriptor["run_id"]],
                selection,
                row,
                descriptor["run_id"],
                descriptor["kind"],
                limits,
                png,
                pdf,
            )
            receipt.write_text(
                json.dumps(
                    dict(
                        png_sha256=sha(png),
                        pdf_sha256=sha(pdf),
                        axes_limits=limits,
                        samples_preserved=True,
                        visual_inspection_pass=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
    report = dict(
        status="predictive_samples_figures_complete_visual_review_pending",
        model_descriptors=config["models"],
        receipts=receipts,
        bindings=bindings,
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
        source_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        torch_version=torch.__version__,
        matplotlib_version=matplotlib.__version__,
        physical_posterior_or_conformal_claim=False,
        visual_inspection_pass=False,
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
