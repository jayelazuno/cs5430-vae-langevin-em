from pathlib import Path

import torch

from src.models.vae import VAE


def load_model_from_checkpoint(cfg, checkpoint_name, device):
    model = VAE(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"]["latent_dim"],
    ).to(device)

    checkpoint_path = Path(cfg["output"]["checkpoint_dir"]) / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint_path
