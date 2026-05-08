from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    table_path = Path("output/tables/vae_em_test_metrics.csv")
    figure_dir = Path("output/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table_path)

    # Reconstruction loss: lower is better
    plt.figure(figsize=(6, 5))
    plt.bar(df["model"], df["encoder_mean_reconstruction_loss"])
    plt.ylabel("Encoder-mean reconstruction loss")
    plt.title("VAE vs Approximate EM: Reconstruction")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "vae_em_reconstruction_comparison.png", dpi=300)
    plt.close()

    # Approximate test log-likelihood: higher is better
    plt.figure(figsize=(6, 5))
    plt.bar(df["model"], df["approx_mean_test_log_likelihood"])
    plt.ylabel("Approximate mean test log-likelihood")
    plt.title("VAE vs Approximate EM: Test Likelihood")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "vae_em_likelihood_comparison.png", dpi=300)
    plt.close()

    print("Saved:")
    print(figure_dir / "vae_em_reconstruction_comparison.png")
    print(figure_dir / "vae_em_likelihood_comparison.png")


if __name__ == "__main__":
    main()
