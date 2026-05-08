import torch
from torchvision.utils import save_image


@torch.no_grad()
def save_reconstructions(model, data_loader, device, output_path, n=16):
    model.eval()

    x, _ = next(iter(data_loader))
    x = x[:n].to(device)

    recon_logits, _, _ = model(x)
    recon = torch.sigmoid(recon_logits).view(-1, 1, 28, 28)

    comparison = torch.cat([x.cpu(), recon.cpu()], dim=0)

    # First row: original images.
    # Second row: reconstructions.
    save_image(comparison, output_path, nrow=n)


@torch.no_grad()
def save_prior_samples(model, device, output_path, latent_dim=10, n=64):
    model.eval()

    z = torch.randn(n, latent_dim, device=device)
    sample_logits = model.decode(z)
    samples = torch.sigmoid(sample_logits).view(-1, 1, 28, 28)

    save_image(samples.cpu(), output_path, nrow=8)
