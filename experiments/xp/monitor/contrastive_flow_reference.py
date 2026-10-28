"""Independent grid-major likelihood enumeration for completed flow artifacts."""

import json
import time

import numpy as np
import torch

from masked_stellar_autoencoder.training.prediction_experiment_heads import (
    build_contrastive_flow,
)


def grid_mle(head, latent, grid):
    """Enumerate label-major pairs; production enumerates star-major pairs."""
    with torch.no_grad():
        summary = head.summary(latent)
        context = grid[:, None, None].expand(-1, len(summary), 1).reshape(-1, 1)
        target = summary[None].expand(len(grid), -1, -1).reshape(-1, summary.shape[1])
        scores = head.flow_loss.distribution(context).log_prob(target)
        scores = scores.reshape(len(grid), len(summary)).detach()
        assert torch.isfinite(scores).all()
        return grid[scores.argmax(dim=0)], scores


def replay(model, parent_model, saved, paired, initial, final, arm, directory):
    yc, ys = float(paired["label_center"]), float(paired["label_scale"])
    labels = ((paired["fit_target"] - yc) / ys).astype("f4")
    spec = arm["contrastive_flow"]
    low, high = (
        float(labels.min()) - spec["grid_margin_standardized"],
        float(labels.max()) + spec["grid_margin_standardized"],
    )
    grid = torch.linspace(low, high, spec["grid_points"], dtype=torch.float32)
    np.testing.assert_allclose(
        grid.numpy(), np.linspace(low, high, spec["grid_points"]), atol=1e-6, rtol=1e-6
    )
    declared = json.loads((directory / "contrastive-flow-grid.json").read_text())
    np.testing.assert_allclose(
        [grid[0], grid[-1]],
        [declared["min_standardized"], declared["max_standardized"]],
        atol=1e-6,
        rtol=1e-6,
    )
    assert declared["points"] == len(grid) == 201
    torch.manual_seed(42)
    head = build_contrastive_flow(spec).eval()
    outputs = {}
    started = time.monotonic()
    for stage, encoder, values in [
        ("initial", parent_model, initial),
        ("final", model, final),
    ]:
        stage_path = directory / f"cpu-flow-stage-{stage}.json"
        if stage_path.exists():
            previous = json.loads(stage_path.read_text())
            assert previous["rows"] == len(paired["test_input"])
            if stage == "initial":
                outputs[stage] = previous
                # Diagnose previously recorded mismatches without repeating the
                # completed full initial pass or loosening its failed gate.
                rows = np.array([1419, 1525, 1697, 3552, 3817])
                with torch.no_grad():
                    latent = encoder.encoder(
                        torch.from_numpy(paired["test_input"][rows])
                    )
                    point, scores = grid_mle(head, latent, grid)
                    original_layout = head.standardized_mle(latent, grid)
                selected = values["standardized_mle"][rows]
                index = np.abs(grid.numpy()[:, None] - selected).argmin(0)
                gaps = (
                    scores.max(0).values.numpy()
                    - scores.numpy()[index, np.arange(len(rows))]
                )
                (directory / "initial-grid-discrepancy-diagnostic.json").write_text(
                    json.dumps(
                        dict(
                            rows=rows.tolist(),
                            cpu_grid_major=point.tolist(),
                            cpu_production_layout=original_layout.tolist(),
                            saved_gpu=selected.tolist(),
                            cpu_log_likelihood_gap_at_saved_gpu_choice=gaps.tolist(),
                            note="Five previously mismatched rows only; failed full initial gate remains failed.",
                        ),
                        indent=2,
                    )
                    + "\n"
                )
                continue
        if stage == "final":
            head.load_state_dict(saved["head_state_dict"], strict=True)
        points, boundaries = [], 0
        progress = directory / "cpu-flow-progress.json"
        for start in range(0, len(paired["test_input"]), 128):
            with torch.no_grad():
                latent = encoder.encoder(
                    torch.from_numpy(paired["test_input"][start : start + 128])
                )
                point, scores = grid_mle(head, latent, grid)
            points.append(point.numpy())
            boundaries += int(((point == grid[0]) | (point == grid[-1])).sum())
            if start % 4096 == 0:
                progress.write_text(
                    json.dumps(
                        dict(
                            status="running",
                            stage=stage,
                            rows=min(start + 128, len(paired["test_input"])),
                            total_rows=len(paired["test_input"]),
                            elapsed_s=time.monotonic() - started,
                        )
                    )
                    + "\n"
                )
        output = np.concatenate(points)
        np.save(directory / f"cpu-flow-{stage}-mle.npy", output)
        matches = np.isclose(output, values["standardized_mle"], rtol=5e-5, atol=5e-5)
        outputs[stage] = dict(
            status="passed" if matches.all() else "failed",
            rows=len(output),
            mismatches=int((~matches).sum()),
            max_absolute_difference=float(
                abs(output - values["standardized_mle"]).max()
            ),
            boundary_count=boundaries,
            rtol=5e-5,
            atol=5e-5,
        )
        stage_path.write_text(json.dumps(outputs[stage], indent=2) + "\n")
        if stage == "final":
            final_output = output
    report = dict(
        status="passed"
        if all(r["status"] == "passed" for r in outputs.values())
        else "failed",
        independent_grid_major_likelihood_enumeration=True,
        initial_full_cpu_replay=outputs["initial"],
        final_full_cpu_replay=outputs["final"],
        grid_points=201,
        grid_min_standardized=float(grid[0]),
        grid_max_standardized=float(grid[-1]),
        estimand="grid maximum of p(summary|FeH), not a calibrated physical posterior",
    )
    return final_output, report
