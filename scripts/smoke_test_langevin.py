import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.data.mnist import get_mnist_loaders
from src.models.vae import VAE
from src.sampling.langevin import (
    langevin_posterior_sample,
    posterior_log_prob,
)
from src.utils.config import load_config, ensure_output_dirs
from src.utils.reproducibility import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mnist_latent10.yaml",
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(cfg["project"]["seed"])

    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader = get_mnist_loaders(
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

    checkpoint_path = (
        Path(cfg["output"]["checkpoint_dir"])
        / f"vae_{cfg['project']['dataset'].lower()}_z{cfg['model']['latent_dim']}.pt"
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x, y = next(iter(test_loader))
    x = x[: args.n_images].to(device)

    with torch.no_grad():
        mu, logvar = model.encode(x)

        encoder_logits = model.decode(mu)
        encoder_recon = torch.sigmoid(encoder_logits).view(-1, 1, 28, 28)

    initial_log_prob = posterior_log_prob(model, x, mu).mean().item()

    posterior_samples = langevin_posterior_sample(
        model=model,
        x=x,
        step_size=cfg["langevin"]["step_size"],
        n_steps=cfg["langevin"]["n_steps"],
        burn_in=cfg["langevin"]["burn_in"],
        n_samples=1,
        init_z=mu,
        clamp_z=8.0,
    )

    z_langevin = posterior_samples[-1]

    final_log_prob = posterior_log_prob(model, x, z_langevin).mean().item()

    with torch.no_grad():
        langevin_logits = model.decode(z_langevin)
        langevin_recon = torch.sigmoid(langevin_logits).view(-1, 1, 28, 28)

    comparison = torch.cat(
        [
            x.cpu(),
            encoder_recon.cpu(),
            langevin_recon.cpu(),
        ],
        dim=0,
    )

    output_path = Path(cfg["output"]["figure_dir"]) / "langevin_posterior_smoke_test.png"
    save_image(comparison, output_path, nrow=args.n_images)

    print(f"Mean unnormalized posterior log-prob at encoder mean: {initial_log_prob:.3f}")
    print(f"Mean unnormalized posterior log-prob after Langevin: {final_log_prob:.3f}")
    print(f"Saved Langevin smoke-test figure to: {output_path}")
    print("")
    print("Rows in the saved image:")
    print("  row 1: original images")
    print("  row 2: decoder reconstruction from encoder mean")
    print("  row 3: decoder reconstruction from Langevin posterior sample")


if __name__ == "__main__":
    main()
