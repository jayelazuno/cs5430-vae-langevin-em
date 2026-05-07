
# CS5430 Final Project: VAE, Langevin Sampling, and Approximate EM

This repository contains code and analysis for the CS5430 Machine Learning final project.

## Project goal

We compare standard variational autoencoder training with an approximate EM procedure for a latent-variable generative model. The approximate E-step uses Langevin sampling to draw from the true posterior p_theta(z | x), and the approximate M-step trains the decoder/generator using sampled latent variables.

## Main tasks

1. Train a VAE on MNIST or CIFAR.
2. Use latent dimension z <= 10.
3. Derive and implement Langevin sampling for p_theta(z | x).
4. Perform minibatch approximate EM.
5. Compare VAE-trained and EM-trained models using:
   - latent representations
   - t-SNE visualization
   - approximate test log-likelihood
   - generated samples/reconstructions
6. Write final report and provide a driver file to reproduce main results.

## Repository structure

```text
input/
  raw/                 # downloaded datasets, not committed
  processed/           # processed tensors or cached subsets, not committed

output/
  checkpoints/         # saved model weights
  figures/             # generated plots
  logs/                # training logs
  tables/              # metrics tables
  samples/             # generated/reconstructed images

analysis/
  notebooks/           # exploratory notebooks
  derivations/         # math derivations and notes

configs/               # YAML/JSON experiment configs
scripts/               # command-line helper scripts

src/
  data/                # dataloaders and preprocessing
  models/              # VAE encoder/decoder/model classes
  training/            # VAE and EM training loops
  sampling/            # Langevin sampler
  evaluation/          # likelihood, t-SNE, plots, metrics
  utils/               # reproducibility, logging, device handling

reports/
  figures/             # final-report figures

tests/                 # small unit tests

main.py                # driver file for reproducing main results
