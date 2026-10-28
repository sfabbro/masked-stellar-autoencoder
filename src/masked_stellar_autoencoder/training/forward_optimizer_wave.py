"""Per-object FP64 forward inversion and explicitly approximate flow-prior MAP.

No claimed normalized posterior: a prior changes the MLE to a MAP; inverse
Hessians describe a local Laplace approximation only. Quoted label errors
enter prior fitting, whereas XP errors enter the likelihood.
"""

import math

import torch


def measurement_nodes(count, dimensions=3, *, device="cpu", seed=314159):
    """Fixed antithetic Sobol normals with exact prefix reuse."""
    if count < 2 or count % 2:
        raise ValueError("Use a positive even node count")
    u = (
        torch.quasirandom.SobolEngine(dimensions, scramble=True, seed=seed)
        .draw(count // 2, dtype=torch.float64)
        .clamp(1e-12, 1 - 1e-12)
    )
    z = math.sqrt(2) * torch.erfinv(2 * u - 1)
    return torch.stack((z, -z), dim=1).flatten(0, 1).to(device)


def observed_prior_log_prob(prior, labels, errors, nodes):
    """QMC convolution p(y_obs)=E_normal[p_prior(y_obs + sigma*z)]."""
    if not torch.all(torch.isfinite(errors) & (errors > 0)):
        raise ValueError("Finite positive quoted label errors required")
    latent = labels[:, None, :] + errors[:, None, :] * nodes[None]
    return torch.logsumexp(prior.log_prob(latent), dim=1) - math.log(len(nodes))


def chunked_observed_prior_log_prob(prior, labels, errors, nodes, chunk=256):
    """Same QMC integral, bounded memory and stable log-sum-exp accumulation."""
    if chunk < 1 or not len(nodes):
        raise ValueError("Positive chunk and nonempty nodes required")
    if labels.shape != errors.shape or not torch.isfinite(labels).all():
        raise ValueError("Finite matching label/error arrays required")
    if not torch.all(torch.isfinite(errors) & (errors > 0)):
        raise ValueError("Finite positive quoted label errors required")
    total = None
    for start in range(0, len(nodes), chunk):
        x = labels[:, None, :] + errors[:, None, :] * nodes[None, start : start + chunk]
        value = torch.logsumexp(prior.log_prob(x), dim=1)
        total = value if total is None else torch.logaddexp(total, value)
    return total - math.log(len(nodes))


def adaptive_prior_convergence(
    prior, labels, errors, max_nodes=16384, tolerance=0.01, resume_report=None
):
    """TUNE-only nested-QMC resolution and independent-scramble stability gate.

    Two successive refinements AND a second scramble must pass the same fixed
    absolute log-density tolerance. A finite-node check is not an exact integral.
    """
    if max_nodes < 512 or max_nodes & (max_nodes - 1) or tolerance != 0.01:
        raise ValueError(
            "Power-of-two bounded cap and unchanged .01 tolerance required"
        )
    previous = None
    consecutive = 0
    levels = []
    count = 64
    if resume_report is not None:
        old_cap = resume_report.get("nodes", 0)
        if (
            resume_report.get("status") != "failed_at_bounded_cap"
            or resume_report.get("tolerance") != tolerance
            or old_cap < 512
            or old_cap >= max_nodes
            or old_cap & (old_cap - 1)
        ):
            raise ValueError(
                "Only a source-bound failed lower-resolution gate can resume"
            )
        levels = [dict(row) for row in resume_report["levels"]]
        if [row["nodes"] for row in levels] != [
            2**i for i in range(6, old_cap.bit_length())
        ]:
            raise ValueError("Incomplete predecessor quadrature schedule")
        for row in reversed(levels):
            delta = row.get("maximum_refinement_delta")
            if delta is None or delta > tolerance:
                break
            consecutive += 1
        # Old producer did not preserve vectors: recompute only its final
        # baseline, never all earlier resolutions, to compare the NEW level.
        with torch.no_grad():
            previous = chunked_observed_prior_log_prob(
                prior,
                labels,
                errors,
                measurement_nodes(old_cap, labels.shape[-1], device=labels.device),
            )
        count = old_cap * 2
    with torch.no_grad():
        while count <= max_nodes:
            nodes = measurement_nodes(count, labels.shape[-1], device=labels.device)
            value = chunked_observed_prior_log_prob(prior, labels, errors, nodes)
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite prior convolution")
            delta = None if previous is None else float((value - previous).abs().max())
            consecutive = (
                consecutive + 1 if delta is not None and delta <= tolerance else 0
            )
            row = dict(nodes=count, maximum_refinement_delta=delta)
            if consecutive >= 2:
                alternate = measurement_nodes(
                    count, labels.shape[-1], device=labels.device, seed=271828
                )
                other = chunked_observed_prior_log_prob(
                    prior, labels, errors, alternate
                )
                scramble_delta = float((value - other).abs().max())
                row["independent_scramble_delta"] = scramble_delta
                if torch.isfinite(other).all() and scramble_delta <= tolerance:
                    levels.append(row)
                    return dict(
                        status="converged_finite_qmc_only",
                        nodes=count,
                        tolerance=tolerance,
                        levels=levels,
                        independent_scramble_seed=271828,
                    )
            levels.append(row)
            previous = value
            count *= 2
    return dict(
        status="failed_at_bounded_cap",
        nodes=max_nodes,
        tolerance=tolerance,
        levels=levels,
    )


def objective(decoder, labels, observed, variance, log_prior=None):
    residual = observed - decoder(labels)
    nll = 0.5 * (
        residual.square() / variance + variance.log() + math.log(2 * math.pi)
    ).sum(-1)
    return nll if log_prior is None else nll - log_prior(labels)


def solve_one(
    decoder,
    observed,
    variance,
    starts,
    *,
    method="lbfgs",
    log_prior=None,
    max_iter=150,
    tolerance=1e-9,
):
    """Independent strong-Wolfe optimizers per object and per initialization.

    This deliberately avoids a single batch-wide LBFGS line search / stopping
    rule. Select by the same likelihood or MAP objective, never by test labels.
    """
    if observed.dtype != torch.float64 or starts.dtype != torch.float64:
        raise TypeError("Entire scoring path must use float64")
    if method not in {"lbfgs", "newton"}:
        raise ValueError(method)
    if not torch.all(torch.isfinite(variance) & (variance > 0)):
        raise ValueError("Likelihood variance must be finite and positive")
    candidates = []
    for initial in starts:
        y = initial.detach().clone().requires_grad_(True)
        if method == "lbfgs":
            optimizer = torch.optim.LBFGS(
                [y],
                max_iter=max_iter,
                history_size=20,
                tolerance_grad=tolerance,
                tolerance_change=1e-14,
                line_search_fn="strong_wolfe",
            )

            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = objective(decoder, y, observed, variance, log_prior)
                (grad,) = torch.autograd.grad(loss, y)
                y.grad = grad
                return loss

            optimizer.step(closure)
        else:
            for _ in range(max_iter):
                loss = objective(decoder, y, observed, variance, log_prior)
                (grad,) = torch.autograd.grad(loss, y)
                if float(grad.norm()) < tolerance:
                    break
                hess = torch.autograd.functional.hessian(
                    lambda z: objective(decoder, z, observed, variance, log_prior), y
                )
                damping = max(1e-6, -float(torch.linalg.eigvalsh(hess).min()) + 1e-6)
                step = torch.linalg.solve(
                    hess + damping * torch.eye(len(y), device=y.device, dtype=y.dtype),
                    grad,
                )
                with torch.no_grad():
                    for power in range(20):
                        proposal = y - step * (0.5**power)
                        if (
                            objective(decoder, proposal, observed, variance, log_prior)
                            <= loss
                        ):
                            y.copy_(proposal)
                            break
                    else:
                        break
        loss = objective(decoder, y, observed, variance, log_prior)
        (gradient,) = torch.autograd.grad(loss, y)
        candidates.append((float(loss.detach()), y.detach(), float(gradient.norm())))
    best = min(candidates, key=lambda item: item[0])
    hess = torch.autograd.functional.hessian(
        lambda z: objective(decoder, z, observed, variance, log_prior), best[1]
    )
    eigenvalues = torch.linalg.eigvalsh(hess)
    return dict(
        prediction=best[1],
        objective=best[0],
        gradient_norm=best[2],
        hessian=hess.detach(),
        hessian_min_eigenvalue=float(eigenvalues.min()),
        starts=len(starts),
        local_stationary=best[2] <= 1e-5,
    )


def laplace_importance_reference(
    decoder, observed, variance, location, hessian, log_prior, counts=(256, 512, 1024)
):
    """Bounded independent normalization/ESS diagnosis, never a global proof.

    A local Gaussian importance proposal may miss other modes. Stable values
    on nested nodes are evidence only within this proposal's visited support.
    """
    covariance = torch.linalg.inv(hessian)
    proposal = torch.distributions.MultivariateNormal(
        location, covariance_matrix=covariance
    )
    chol = torch.linalg.cholesky(covariance)
    result = {}
    for count in counts:
        latent = (
            location
            + measurement_nodes(count, len(location), device=location.device) @ chol.T
        )
        logs = -objective(
            decoder, latent, observed, variance, log_prior
        ) - proposal.log_prob(latent)
        weights = torch.softmax(logs, dim=0)
        mean = weights @ latent
        result[str(count)] = dict(
            log_normalizer=float((torch.logsumexp(logs, 0) - math.log(count)).detach()),
            effective_samples=float(weights.square().sum().reciprocal().detach()),
            posterior_mean=mean.detach().cpu().tolist(),
        )
    return result


def defensive_flow_posterior(
    decoder, observed, variance, location, hessian, prior, *, count=4096, seed=42
):
    """Self-normalized defensive importance approximation, not exact inference.

    Half the proposal is the unconditional joint flow prior, ensuring global
    support even when the other half is a local Laplace approximation.
    """
    if count % 2 or count < 2:
        raise ValueError("Even importance sample count required")
    eigenvalues, vectors = torch.linalg.eigh(hessian)
    covariance = (vectors * eigenvalues.clamp_min(1e-3).reciprocal()) @ vectors.T
    local = torch.distributions.MultivariateNormal(
        location, covariance_matrix=covariance
    )
    devices = [location.device.index or 0] if location.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        global_draw = prior.sample((count // 2,))
        local_draw = local.sample((count // 2,))
    samples = torch.cat([global_draw, local_draw], dim=0)
    log_prior = prior.log_prob(samples)
    log_proposal = torch.logaddexp(log_prior, local.log_prob(samples)) - math.log(2)
    logs = (
        -objective(decoder, samples, observed, variance, prior.log_prob) - log_proposal
    )
    weights = torch.softmax(logs, dim=0)
    mean = weights @ samples
    centered = samples - mean
    covariance_posterior = torch.einsum("n,ni,nj->ij", weights, centered, centered)
    quantiles = []
    for dimension in range(samples.shape[1]):
        values, order = samples[:, dimension].sort()
        cumulative = weights[order].cumsum(0)
        at = torch.searchsorted(
            cumulative,
            torch.tensor(
                [0.025, 0.05, 0.5, 0.95, 0.975],
                dtype=weights.dtype,
                device=weights.device,
            ),
        ).clamp_max(count - 1)
        quantiles.append(values[at])
    return dict(
        mean=mean.detach(),
        covariance=covariance_posterior.detach(),
        quantiles=torch.stack(quantiles).detach(),
        log_normalizer=float((torch.logsumexp(logs, 0) - math.log(count)).detach()),
        effective_samples=float(weights.square().sum().reciprocal().detach()),
        samples=count,
        sample_values=samples.detach(),
        normalized_weights=weights.detach(),
    )


def solve_library_gauss_newton(
    regressor, observed, sigma, starts, *, max_iter=150, tolerance=1e-9
):
    """Same per-object starts/dtype/budget, actual library damped GN solver."""
    if observed.dtype != torch.float64 or starts.dtype != torch.float64:
        raise TypeError("Entire scoring path must use float64")
    regressor.device = observed.device
    decoder = regressor.decoder_
    variance = sigma.square() + regressor.noise_variance_.to(observed).double()
    candidates = []
    for initial in starts:
        prediction = regressor.predict(
            observed[None],
            sigma_x=sigma[None],
            method="gauss_newton",
            init=initial,
            max_iter=max_iter,
            tol=tolerance,
            chunk_size=1,
        )[0].detach()
        value = objective(decoder, prediction, observed, variance)
        candidates.append((float(value.detach()), prediction))
    value, prediction = min(candidates, key=lambda item: item[0])
    gradient = torch.autograd.functional.jacobian(
        lambda y: objective(decoder, y, observed, variance), prediction
    )
    hessian = torch.autograd.functional.hessian(
        lambda y: objective(decoder, y, observed, variance), prediction
    )
    norm = float(gradient.norm())
    return dict(
        prediction=prediction,
        objective=value,
        gradient_norm=norm,
        hessian=hessian.detach(),
        hessian_min_eigenvalue=float(torch.linalg.eigvalsh(hessian).min()),
        starts=len(starts),
        local_stationary=norm <= 1e-5,
    )
