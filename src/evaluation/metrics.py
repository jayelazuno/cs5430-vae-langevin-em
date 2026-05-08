import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def encoder_mean_reconstruction_loss(model, loader, device, max_batches=None):
    """
    Deterministic reconstruction diagnostic using z = mu_phi(x).

    This is useful for comparing VAE and EM with the same encoder initialization.
    """
    model.eval()

    total_loss = 0.0
    n_examples = 0

    for batch_idx, (x, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        mu, _ = model.encode(x)
        logits = model.decode(mu)

        loss = F.binary_cross_entropy_with_logits(
            logits,
            x_flat,
            reduction="sum",
        )

        total_loss += loss.item()
        n_examples += batch_size

    return total_loss / n_examples


def diagonal_gaussian_log_prob(z, mu, logvar):
    """
    log q_phi(z | x) for diagonal Gaussian encoder.

    Shape:
        z:      [S, B, latent_dim]
        mu:     [B, latent_dim]
        logvar: [B, latent_dim]
        returns [S, B]
    """
    return -0.5 * torch.sum(
        math.log(2.0 * math.pi)
        + logvar.unsqueeze(0)
        + (z - mu.unsqueeze(0)).pow(2) / torch.exp(logvar).unsqueeze(0),
        dim=2,
    )


def standard_normal_log_prob(z):
    """
    log p(z) for standard normal prior.

    Shape:
        z: [S, B, latent_dim]
        returns [S, B]
    """
    latent_dim = z.size(-1)

    return -0.5 * (
        latent_dim * math.log(2.0 * math.pi)
        + torch.sum(z.pow(2), dim=2)
    )


def bernoulli_log_likelihood(model, x, z):
    """
    log p_theta(x | z) for Bernoulli decoder.

    Shape:
        x: [B, 1, 28, 28]
        z: [S, B, latent_dim]
        returns [S, B]
    """
    n_samples, batch_size, latent_dim = z.shape
    x_flat = x.view(batch_size, -1)

    z_flat = z.reshape(n_samples * batch_size, latent_dim)

    x_repeated = (
        x_flat.unsqueeze(0)
        .expand(n_samples, batch_size, x_flat.size(1))
        .reshape(n_samples * batch_size, x_flat.size(1))
    )

    logits = model.decode(z_flat)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        x_repeated,
        reduction="none",
    )

    log_likelihood = -torch.sum(bce, dim=1)

    return log_likelihood.view(n_samples, batch_size)


def importance_sampling_log_likelihood(
    model,
    loader,
    device,
    n_samples=100,
    max_batches=None,
):
    """
    Approximate log p_theta(x) using importance sampling:

        p_theta(x) = E_{q_phi(z|x)} [ p_theta(x,z) / q_phi(z|x) ]

    Estimate:

        log p_theta(x)
        approx logsumexp_s[
            log p_theta(x|z_s) + log p(z_s) - log q_phi(z_s|x)
        ] - log S

    This uses the model encoder as the proposal distribution for both VAE and EM.
    """
    model.eval()

    log_likelihoods = []

    for batch_idx, (x, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        batch_size = x.size(0)

        with torch.no_grad():
            mu, logvar = model.encode(x)
            std = torch.exp(0.5 * logvar)

            eps = torch.randn(
                n_samples,
                batch_size,
                model.latent_dim,
                device=device,
            )

            z = mu.unsqueeze(0) + std.unsqueeze(0) * eps

            log_px_given_z = bernoulli_log_likelihood(model, x, z)
            log_pz = standard_normal_log_prob(z)
            log_qz_given_x = diagonal_gaussian_log_prob(z, mu, logvar)

            log_weights = log_px_given_z + log_pz - log_qz_given_x
            batch_log_px = torch.logsumexp(log_weights, dim=0) - math.log(n_samples)

        log_likelihoods.append(batch_log_px.detach().cpu())

    log_likelihoods = torch.cat(log_likelihoods, dim=0)

    return {
        "mean_log_likelihood": log_likelihoods.mean().item(),
        "std_log_likelihood": log_likelihoods.std().item(),
        "n_examples": len(log_likelihoods),
    }
