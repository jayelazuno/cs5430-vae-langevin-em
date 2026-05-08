#!/bin/bash

set -e

echo "Running CS5430 VAE + Langevin Approximate EM final project pipeline"

echo "1. Training baseline VAE"
python main.py --config configs/mnist_latent10.yaml --mode train_vae

echo "2. Plotting VAE training curves"
python scripts/plot_vae_training_history.py

echo "3. Running approximate EM"
python main.py --config configs/mnist_em_medium.yaml --mode train_em

echo "4. Plotting EM training curves"
python scripts/plot_em_training_history.py

echo "5. Comparing VAE and EM"
python scripts/compare_vae_em.py \
  --config configs/mnist_em_medium.yaml \
  --likelihood_samples 500 \
  --max_test_batches 80 \
  --tsne_points 2000

echo "6. Plotting final comparison metrics"
python scripts/plot_vae_em_metrics.py

echo "Done. Main outputs are in output/figures, output/tables, and output/samples."
