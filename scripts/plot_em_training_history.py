from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    history_path = Path("output/tables/em_training_history.csv")
    figure_dir = Path("output/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)

    plt.figure(figsize=(7, 5))
    plt.plot(df["epoch"], df["train_m_step_nll"], marker="o", label="Train M-step NLL")
    plt.xlabel("Epoch")
    plt.ylabel("Negative log p_theta(x | z)")
    plt.title("Approximate EM M-step Objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "em_m_step_nll_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(
        df["epoch"],
        df["val_encoder_mean_reconstruction_loss"],
        marker="o",
        label="Validation encoder-mean reconstruction",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("Approximate EM Validation Diagnostic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "em_validation_reconstruction_curve.png", dpi=300)
    plt.close()

    print("Saved:")
    print(figure_dir / "em_m_step_nll_curve.png")
    print(figure_dir / "em_validation_reconstruction_curve.png")


if __name__ == "__main__":
    main()
