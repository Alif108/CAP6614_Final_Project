# Diffusion Transformer (DiT) Reproduction/Application

This is a codebase for a compute-limited term-project reproduction/application of **Scalable Diffusion Models with Transformers**. The full paper trains large latent DiT models on ImageNet with very large compute; this project instead targets reproducing the main expected trends on Imagenette using a pretrained VAE and smaller DiT configs.

## What this codebase is designed to show

The goal is not to match the paper's ImageNet FID. Instead, it produces evidence for these expected behaviors:

1. A paper-faithful latent DiT pipeline: image -> VAE latent -> DiT denoiser -> sampled latent -> VAE decode.
2. Training progress: validation denoising loss and samples improve with more steps.
3. Patch-size scaling: smaller latent patches increase token count/FLOPs and should improve quality at higher cost.
4. Model-size scaling: DiT-S/4 should outperform the tiny model when trained long enough.
5. Classifier-free guidance and DDIM-step sweeps can be visualized from trained checkpoints.
6. Efficiency benchmarking: params, token count, FLOPs, latency, throughput, memory.

## Experiment plan

Default planned runs:

| Run | Purpose | Target steps |
|---|---:|---:|
| DiT-Tiny/8 | patch-size baseline, fewest tokens | 10k |
| DiT-Tiny/4 | patch-size middle | 10k |
| DiT-Tiny/2 | patch-size most tokens, slowest tiny run | 10k |
| DiT-S/4 | stronger paper-style run | 20k |

These are still far smaller than the paper, but they are enough to produce credible trends from the paper.

## Notebook order

1. `00_setup_and_data.ipynb`  
   Installs dependencies, downloads Imagenette, loads the Stable-Diffusion VAE, caches 32x32x4 latents, and saves visual reconstructions.

2. `01_train_latent_dit.ipynb`  
   Trains the four runs by default. Saves checkpoints, histories, validation losses, and progress samples.

3. `02_sample_and_visualize.ipynb`  
   Loads trained checkpoints, samples images, makes CFG/step sweeps, and saves grids.

4. `03_evaluate.ipynb`  
   Generates or reuses sample folders, computes FID/KID where available, validation loss, and creates plots/tables.

5. `04_efficiency_and_scaling_plots.ipynb`  
   Benchmarks model sizes and patch sizes; plots compute/quality tradeoffs.

**Run these notebooks in order to regenrate the results.**

## Expected directories

The notebooks create:

```text
data/
  imagenette2-320/
  imagenette_latents/
checkpoints/
runs/
samples/
results/
figures/
```

## Important limitations

- This is **not** a full ImageNet reproduction.
- This does **not** train DiT-XL/2.
- FID-1K/FID-2K on Imagenette is **not comparable** to the paper's ImageNet FID-50K.
