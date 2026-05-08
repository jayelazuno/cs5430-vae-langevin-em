from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    history_path = Path("output/tables/vae_training_history.csv")
    figure_dir = Path("output/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)

    # Plot total negative ELBO
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train")
    plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Negative ELBO")
    plt.title("VAE Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "vae_negative_elbo_curve.png", dpi=300)
    plt.close()

    # Plot reconstruction loss
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_reconstruction_loss"], marker="o", label="Train")
    plt.plot(df["epoch"], df["val_reconstruction_loss"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("VAE Reconstruction Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "vae_reconstruction_loss_curve.png", dpi=300)
    plt.close()

    # Plot KL divergence
    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_kl_divergence"], marker="o", label="Train")
    plt.plot(df["epoch"], df["val_kl_divergence"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("VAE KL Divergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "vae_kl_divergence_curve.png", dpi=300)
    plt.close()

    print("Saved:")
    print(figure_dir / "vae_negative_elbo_curve.png")
    print(figure_dir / "vae_reconstruction_loss_curve.png")
    print(figure_dir / "vae_kl_divergence_curve.png")


if __name__ == "__main__":
    main()
