from pathlib import Path

import pandas as pd
import torch
from torch.optim import Adam

from src.data.mnist import get_mnist_loaders
from src.models.vae import VAE
from src.training.vae_loss import vae_loss
from src.utils.reproducibility import get_device


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_examples = 0

    for x, _ in loader:
        x = x.to(device)
        batch_size = x.size(0)

        with torch.set_grad_enabled(training):
            recon_logits, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(x, recon_logits, mu, logvar)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        n_examples += batch_size

    return {
        "loss": total_loss / n_examples,
        "reconstruction_loss": total_recon / n_examples,
        "kl_divergence": total_kl / n_examples,
    }


def train_vae(cfg):
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

    optimizer = Adam(
        model.parameters(),
        lr=cfg["vae_training"]["lr"],
    )

    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    table_dir = Path(cfg["output"]["table_dir"])

    dataset_name = cfg["project"]["dataset"].lower()
    latent_dim = cfg["model"]["latent_dim"]
    checkpoint_path = checkpoint_dir / f"vae_{dataset_name}_z{latent_dim}.pt"
    history_path = table_dir / "vae_training_history.csv"

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, cfg["vae_training"]["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reconstruction_loss": train_metrics["reconstruction_loss"],
            "train_kl_divergence": train_metrics["kl_divergence"],
            "val_loss": val_metrics["loss"],
            "val_reconstruction_loss": val_metrics["reconstruction_loss"],
            "val_kl_divergence": val_metrics["kl_divergence"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {row['train_loss']:.3f} | "
            f"val loss {row['val_loss']:.3f} | "
            f"val recon {row['val_reconstruction_loss']:.3f} | "
            f"val KL {row['val_kl_divergence']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        pd.DataFrame(history).to_csv(history_path, index=False)

    # Reload best checkpoint before returning.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    return model, loaders, history, checkpoint_path
