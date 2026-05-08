from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.data.mnist import get_mnist_loaders
from src.models.vae import VAE
from src.sampling.langevin import langevin_posterior_sample
from src.utils.reproducibility import get_device


def freeze_encoder(model):
    """
    In approximate EM, the encoder is not part of the generative model.

    We keep it only as a convenient initialization for Langevin sampling.
    The M-step updates the decoder/generator p_theta(x | z).
    """
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fc_mu.parameters():
        parameter.requires_grad_(False)
    for parameter in model.fc_logvar.parameters():
        parameter.requires_grad_(False)


def decoder_nll_from_posterior_samples(model, x, z_samples):
    """
    Approximate negative M-step objective.

    z_samples has shape:
        [n_samples, batch_size, latent_dim]

    We minimize:

        - 1/S sum_s log p_theta(x | z_s)

    For Bernoulli MNIST likelihood, this is binary cross entropy with logits.
    """
    n_samples, batch_size, latent_dim = z_samples.shape

    x_flat = x.view(batch_size, -1)

    z_flat = z_samples.reshape(n_samples * batch_size, latent_dim)
    x_repeated = (
        x_flat.unsqueeze(0)
        .expand(n_samples, batch_size, x_flat.size(1))
        .reshape(n_samples * batch_size, x_flat.size(1))
    )

    logits = model.decode(z_flat)

    loss = F.binary_cross_entropy_with_logits(
        logits,
        x_repeated,
        reduction="sum",
    )

    # Average per image, not per pixel.
    return loss / (n_samples * batch_size)


@torch.no_grad()
def evaluate_encoder_mean_reconstruction(model, loader, device, max_batches=None):
    """
    Simple validation diagnostic.

    This is not the EM objective. It measures how well the current decoder
    reconstructs images from the fixed VAE encoder mean.
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
        ) / batch_size

        total_loss += loss.item() * batch_size
        n_examples += batch_size

    return total_loss / n_examples


def train_em(cfg):
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_mnist_loaders(
        root=cfg["data"]["root"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        val_fraction=cfg["data"]["val_fraction"],
        seed=cfg["project"]["seed"],
    )

    model = VAE(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"]["latent_dim"],
    ).to(device)

    dataset_name = cfg["project"]["dataset"].lower()
    latent_dim = cfg["model"]["latent_dim"]

    vae_checkpoint_path = (
        Path(cfg["output"]["checkpoint_dir"])
        / f"vae_{dataset_name}_z{latent_dim}.pt"
    )

    if not vae_checkpoint_path.exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found: {vae_checkpoint_path}. "
            "Train the VAE first before running EM."
        )

    checkpoint = torch.load(vae_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    freeze_encoder(model)

    # M-step updates only the decoder/generator.
    optimizer = Adam(
        model.decoder.parameters(),
        lr=cfg["em_training"]["lr"],
    )

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    table_dir = Path(cfg["output"]["table_dir"])

    em_checkpoint_path = checkpoint_dir / f"em_{dataset_name}_z{latent_dim}.pt"
    history_path = table_dir / "em_training_history.csv"

    max_train_batches = cfg["em_training"].get("max_train_batches", None)
    max_val_batches = cfg["em_training"].get("max_val_batches", None)

    best_val_recon = float("inf")
    history = []

    for epoch in range(1, cfg["em_training"]["epochs"] + 1):
        model.train()

        total_m_step_loss = 0.0
        n_examples = 0

        for batch_idx, (x, _) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

            x = x.to(device)
            batch_size = x.size(0)

            # Use the fixed VAE encoder only to initialize the chain.
            with torch.no_grad():
                init_mu, _ = model.encode(x)

            # Approximate E-step:
            # sample z ~ p_{theta_old}(z | x) using current decoder.
            z_samples = langevin_posterior_sample(
                model=model,
                x=x,
                step_size=cfg["langevin"]["step_size"],
                n_steps=cfg["langevin"]["n_steps"],
                burn_in=cfg["langevin"]["burn_in"],
                n_samples=cfg["langevin"]["n_samples"],
                init_z=init_mu,
                clamp_z=8.0,
            )

            # EM nuance:
            # Treat E-step samples as fixed during the M-step.
            z_samples = z_samples.detach()

            # Approximate M-step:
            # several gradient steps on decoder using the same sampled z values.
            model.train()
            final_loss = None

            for _ in range(cfg["em_training"]["m_steps_per_batch"]):
                loss = decoder_nll_from_posterior_samples(
                    model=model,
                    x=x,
                    z_samples=z_samples,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                final_loss = loss

            total_m_step_loss += final_loss.item() * batch_size
            n_examples += batch_size

        train_m_step_loss = total_m_step_loss / n_examples

        val_encoder_recon = evaluate_encoder_mean_reconstruction(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_m_step_nll": train_m_step_loss,
            "val_encoder_mean_reconstruction_loss": val_encoder_recon,
        }

        history.append(row)
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"EM Epoch {epoch:03d} | "
            f"train M-step NLL {train_m_step_loss:.3f} | "
            f"val encoder-mean recon {val_encoder_recon:.3f}"
        )

        if val_encoder_recon < best_val_recon:
            best_val_recon = val_encoder_recon
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "best_val_encoder_mean_reconstruction_loss": best_val_recon,
                    "epoch": epoch,
                    "initialized_from": str(vae_checkpoint_path),
                },
                em_checkpoint_path,
            )

    checkpoint = torch.load(em_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    return model, loaders, history, em_checkpoint_path
