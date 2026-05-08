import argparse
from pathlib import Path

from src.training.train_vae import train_vae
from src.training.train_em import train_em
from src.evaluation.generation import save_reconstructions, save_prior_samples
from src.utils.config import load_config, ensure_output_dirs
from src.utils.reproducibility import set_seed, get_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mnist_latent10.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_vae",
        choices=["train_vae", "train_em"],
        help="Pipeline mode to run.",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    set_seed(cfg["project"]["seed"])

    device = get_device()
    figure_dir = Path(cfg["output"]["figure_dir"])
    sample_dir = Path(cfg["output"]["sample_dir"])

    if args.mode == "train_vae":
        model, loaders, history, checkpoint_path = train_vae(cfg)

        recon_path = figure_dir / "vae_reconstructions.png"
        sample_path = sample_dir / "vae_prior_samples.png"

        save_reconstructions(
            model=model,
            data_loader=loaders["test"],
            device=device,
            output_path=recon_path,
            n=cfg["evaluation"]["n_reconstructions"],
        )

        save_prior_samples(
            model=model,
            device=device,
            output_path=sample_path,
            latent_dim=cfg["model"]["latent_dim"],
            n=cfg["evaluation"]["n_generated_samples"],
        )

        print(f"Saved best VAE checkpoint to: {checkpoint_path}")
        print(f"Saved VAE reconstructions to: {recon_path}")
        print(f"Saved VAE prior samples to: {sample_path}")

    elif args.mode == "train_em":
        model, loaders, history, checkpoint_path = train_em(cfg)

        recon_path = figure_dir / "em_reconstructions.png"
        sample_path = sample_dir / "em_prior_samples.png"

        save_reconstructions(
            model=model,
            data_loader=loaders["test"],
            device=device,
            output_path=recon_path,
            n=cfg["evaluation"]["n_reconstructions"],
        )

        save_prior_samples(
            model=model,
            device=device,
            output_path=sample_path,
            latent_dim=cfg["model"]["latent_dim"],
            n=cfg["evaluation"]["n_generated_samples"],
        )

        print(f"Saved best EM checkpoint to: {checkpoint_path}")
        print(f"Saved EM reconstructions to: {recon_path}")
        print(f"Saved EM prior samples to: {sample_path}")


if __name__ == "__main__":
    main()
