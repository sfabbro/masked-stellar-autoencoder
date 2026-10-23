"""Per-object FP64 forward inversion and explicitly approximate flow-prior MAP.

No claimed normalized posterior: a prior changes the MLE to a MAP; inverse
Hessians describe a local Laplace approximation only. Quoted label errors
enter prior fitting, whereas XP errors enter the likelihood.
"""

import math

import torch


def measurement_nodes(count, dimensions=3, *, device="cpu"):
    """Fixed antithetic Sobol normals with exact prefix reuse."""
    if count < 2 or count % 2:
        raise ValueError("Use a positive even node count")
    u = (
        torch.quasirandom.SobolEngine(dimensions, scramble=True, seed=314159)
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
