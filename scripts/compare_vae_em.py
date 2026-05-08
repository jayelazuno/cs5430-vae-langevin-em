import argparse
from pathlib import Path

import pandas as pd

from src.data.mnist import get_mnist_loaders
from src.evaluation.latent_viz import (
    collect_encoder_mean_latents,
    collect_langevin_latents,
    save_tsne_plot,
)
from src.evaluation.metrics import (
    encoder_mean_reconstruction_loss,
    importance_sampling_log_likelihood,
)
from src.evaluation.model_loading import load_model_from_checkpoint
from src.utils.config import load_config, ensure_output_dirs
from src.utils.reproducibility import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mnist_em_medium.yaml",
    )
    parser.add_argument(
        "--likelihood_samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_test_batches",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--tsne_points",
        type=int,
        default=2000,
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

    dataset_name = cfg["project"]["dataset"].lower()
    latent_dim = cfg["model"]["latent_dim"]

    vae_name = f"vae_{dataset_name}_z{latent_dim}.pt"
    em_name = f"em_{dataset_name}_z{latent_dim}.pt"

    vae_model, vae_path = load_model_from_checkpoint(cfg, vae_name, device)
    em_model, em_path = load_model_from_checkpoint(cfg, em_name, device)

    print(f"Loaded VAE checkpoint: {vae_path}")
    print(f"Loaded EM checkpoint: {em_path}")

    rows = []

    for model_name, model in [("VAE", vae_model), ("Approximate EM", em_model)]:
        recon_loss = encoder_mean_reconstruction_loss(
            model=model,
            loader=test_loader,
            device=device,
            max_batches=args.max_test_batches,
        )

        likelihood_result = importance_sampling_log_likelihood(
            model=model,
            loader=test_loader,
            device=device,
            n_samples=args.likelihood_samples,
            max_batches=args.max_test_batches,
        )

        rows.append(
            {
                "model": model_name,
                "encoder_mean_reconstruction_loss": recon_loss,
                "approx_mean_test_log_likelihood": likelihood_result["mean_log_likelihood"],
                "approx_std_test_log_likelihood": likelihood_result["std_log_likelihood"],
                "n_test_examples": likelihood_result["n_examples"],
                "importance_samples": args.likelihood_samples,
            }
        )

        print("")
        print(model_name)
        print(f"  encoder-mean reconstruction loss: {recon_loss:.3f}")
        print(
            "  approximate mean test log-likelihood: "
            f"{likelihood_result['mean_log_likelihood']:.3f}"
        )

    table_dir = Path(cfg["output"]["table_dir"])
    figure_dir = Path(cfg["output"]["figure_dir"])

    comparison_path = table_dir / "vae_em_test_metrics.csv"
    pd.DataFrame(rows).to_csv(comparison_path, index=False)
    print("")
    print(f"Saved comparison metrics to: {comparison_path}")

    print("")
    print("Generating t-SNE plots...")

    z_vae, y_vae = collect_encoder_mean_latents(
        model=vae_model,
        loader=test_loader,
        device=device,
        n_points=args.tsne_points,
    )

    save_tsne_plot(
        z=z_vae,
        y=y_vae,
        output_path=figure_dir / "vae_encoder_mean_tsne.png",
        title="VAE latent space: encoder mean",
    )

    z_em_encoder, y_em_encoder = collect_encoder_mean_latents(
        model=em_model,
        loader=test_loader,
        device=device,
        n_points=args.tsne_points,
    )

    save_tsne_plot(
        z=z_em_encoder,
        y=y_em_encoder,
        output_path=figure_dir / "em_encoder_mean_tsne.png",
        title="EM model latent space: encoder mean initialization",
    )

    z_em_langevin, y_em_langevin = collect_langevin_latents(
        model=em_model,
        loader=test_loader,
        device=device,
        n_points=args.tsne_points,
        step_size=cfg["langevin"]["step_size"],
        n_steps=cfg["langevin"]["n_steps"],
        burn_in=cfg["langevin"]["burn_in"],
    )

    save_tsne_plot(
        z=z_em_langevin,
        y=y_em_langevin,
        output_path=figure_dir / "em_langevin_posterior_tsne.png",
        title="EM latent space: Langevin posterior samples",
    )

    print("Saved t-SNE plots:")
    print(figure_dir / "vae_encoder_mean_tsne.png")
    print(figure_dir / "em_encoder_mean_tsne.png")
    print(figure_dir / "em_langevin_posterior_tsne.png")


if __name__ == "__main__":
    main()
