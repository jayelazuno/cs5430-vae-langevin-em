from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE

from src.sampling.langevin import langevin_posterior_sample


@torch.no_grad()
def collect_encoder_mean_latents(model, loader, device, n_points=2000):
    model.eval()

    latents = []
    labels = []

    for x, y in loader:
        x = x.to(device)
        mu, _ = model.encode(x)

        latents.append(mu.cpu())
        labels.append(y.cpu())

        if sum(t.size(0) for t in latents) >= n_points:
            break

    z = torch.cat(latents, dim=0)[:n_points]
    y = torch.cat(labels, dim=0)[:n_points]

    return z.numpy(), y.numpy()


def collect_langevin_latents(
    model,
    loader,
    device,
    n_points=2000,
    step_size=1e-3,
    n_steps=50,
    burn_in=20,
):
    """
    Collect one Langevin posterior sample per image.

    This is slower than encoder means but better represents the EM posterior idea.
    """
    model.eval()

    latents = []
    labels = []

    for x, y in loader:
        x = x.to(device)

        with torch.no_grad():
            mu, _ = model.encode(x)

        z_samples = langevin_posterior_sample(
            model=model,
            x=x,
            step_size=step_size,
            n_steps=n_steps,
            burn_in=burn_in,
            n_samples=1,
            init_z=mu,
            clamp_z=8.0,
        )

        z = z_samples[-1]

        latents.append(z.cpu())
        labels.append(y.cpu())

        if sum(t.size(0) for t in latents) >= n_points:
            break

    z = torch.cat(latents, dim=0)[:n_points]
    y = torch.cat(labels, dim=0)[:n_points]

    return z.numpy(), y.numpy()


def save_tsne_plot(z, y, output_path, title):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=123,
    )

    z_2d = tsne.fit_transform(z)

    df = pd.DataFrame(
        {
            "tsne1": z_2d[:, 0],
            "tsne2": z_2d[:, 1],
            "label": y.astype(str),
        }
    )

    plt.figure(figsize=(7, 6))

    for digit in sorted(df["label"].unique()):
        sub = df[df["label"] == digit]
        plt.scatter(
            sub["tsne1"],
            sub["tsne2"],
            s=8,
            alpha=0.7,
            label=digit,
        )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title(title)
    plt.legend(title="Digit", markerscale=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
