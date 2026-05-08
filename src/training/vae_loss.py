import torch
import torch.nn.functional as F


def vae_loss(x, recon_logits, mu, logvar):
    """
    Negative ELBO for Bernoulli VAE on MNIST.

    loss = reconstruction_loss + KL(q_phi(z|x) || p(z))

    We minimize negative ELBO, so lower is better.
    """
    batch_size = x.size(0)
    x_flat = x.view(batch_size, -1)

    # Negative log p_theta(x | z), using Bernoulli likelihood.
    # Sum over pixels, then average over batch.
    reconstruction_loss = F.binary_cross_entropy_with_logits(
        recon_logits,
        x_flat,
        reduction="sum",
    )

    # KL divergence between diagonal Gaussian q_phi(z|x)
    # and standard normal prior p(z) = N(0, I).
    kl_divergence = -0.5 * torch.sum(
        1.0 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = (reconstruction_loss + kl_divergence) / batch_size
    reconstruction_loss = reconstruction_loss / batch_size
    kl_divergence = kl_divergence / batch_size

    return total_loss, reconstruction_loss, kl_divergence
