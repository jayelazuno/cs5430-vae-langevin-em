import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def temporarily_freeze_parameters(model):
    """
    During Langevin sampling we need gradients with respect to z,
    not with respect to model parameters.

    This context manager temporarily freezes model parameters
    to reduce unnecessary graph tracking.
    """
    old_requires_grad = []

    for parameter in model.parameters():
        old_requires_grad.append(parameter.requires_grad)
        parameter.requires_grad_(False)

    try:
        yield
    finally:
        for parameter, old_value in zip(model.parameters(), old_requires_grad):
            parameter.requires_grad_(old_value)


def log_standard_normal(z):
    """
    Compute log N(z; 0, I), ignoring the additive constant.

    Shape:
        z: [batch_size, latent_dim]
        returns: [batch_size]
    """
    return -0.5 * torch.sum(z.pow(2), dim=1)


def bernoulli_log_likelihood_from_logits(x, logits):
    """
    Compute log p_theta(x | z) for Bernoulli likelihood.

    The decoder returns logits, not probabilities.
    Therefore:

        log p(x | z) = - BCEWithLogits(logits, x)

    Shape:
        x:      [batch_size, input_dim]
        logits: [batch_size, input_dim]
        returns: [batch_size]
    """
    per_pixel_bce = F.binary_cross_entropy_with_logits(
        logits,
        x,
        reduction="none",
    )

    return -torch.sum(per_pixel_bce, dim=1)


def posterior_log_prob(model, x, z):
    """
    Compute unnormalized log p_theta(z | x).

    log p_theta(z | x)
      = log p(z) + log p_theta(x | z) - log p_theta(x)

    Since log p_theta(x) does not depend on z, we ignore it.

    Shape:
        x: [batch_size, 1, 28, 28] or [batch_size, 784]
        z: [batch_size, latent_dim]
        returns: [batch_size]
    """
    batch_size = x.size(0)
    x_flat = x.view(batch_size, -1)

    logits = model.decode(z)

    log_prior = log_standard_normal(z)
    log_likelihood = bernoulli_log_likelihood_from_logits(x_flat, logits)

    return log_prior + log_likelihood


def langevin_posterior_sample(
    model,
    x,
    step_size=1e-3,
    n_steps=100,
    burn_in=20,
    n_samples=1,
    init_z=None,
    clamp_z=None,
):
    """
    Draw approximate samples from p_theta(z | x) using unadjusted Langevin dynamics.

    Update:
        z_{t+1} = z_t + eta * grad_z log p_theta(z_t | x)
                        + sqrt(2 eta) * epsilon_t

    Args:
        model: VAE model with a decode(z) method.
        x: input batch, shape [batch_size, 1, 28, 28].
        step_size: Langevin step size eta.
        n_steps: total Langevin transitions.
        burn_in: number of initial transitions before collecting samples.
        n_samples: number of posterior samples to return.
        init_z: optional initial latent tensor [batch_size, latent_dim].
        clamp_z: optional absolute clamp value for numerical stability.

    Returns:
        samples: tensor with shape [n_samples, batch_size, latent_dim]
    """
    if n_steps <= burn_in:
        raise ValueError("n_steps must be greater than burn_in.")

    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    device = x.device
    batch_size = x.size(0)
    latent_dim = model.latent_dim

    if init_z is None:
        z = torch.randn(batch_size, latent_dim, device=device)
    else:
        z = init_z.detach().clone().to(device)

    samples = []

    # Collect samples after burn-in. We thin the chain so that
    # we get approximately n_samples samples over the remaining trajectory.
    available_steps = n_steps - burn_in
    thinning = max(1, available_steps // n_samples)

    noise_scale = math.sqrt(2.0 * step_size)

    model.eval()

    with temporarily_freeze_parameters(model):
        for step in range(n_steps):
            z = z.detach().requires_grad_(True)

            log_prob = posterior_log_prob(model, x, z)
            total_log_prob = torch.sum(log_prob)

            grad_z = torch.autograd.grad(total_log_prob, z)[0]

            with torch.no_grad():
                noise = torch.randn_like(z)
                z = z + step_size * grad_z + noise_scale * noise

                if clamp_z is not None:
                    z = torch.clamp(z, min=-clamp_z, max=clamp_z)

            should_collect = (
                step >= burn_in
                and len(samples) < n_samples
                and ((step - burn_in) % thinning == 0)
            )

            if should_collect:
                samples.append(z.detach().clone())

    # If thinning did not collect enough because of integer rounding,
    # repeat the final state.
    while len(samples) < n_samples:
        samples.append(z.detach().clone())

    return torch.stack(samples, dim=0)
