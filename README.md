# Deep Learning Final Project: Gated Attention Mechanisms

Implementation and analysis of gated attention mechanisms in transformer-based language models.

## Overview

This project explores two gating variants for transformer attention:
- **Headwise Gating**: Per-head scalar gating mechanism
- **Elementwise Gating**: Per-element feature gating mechanism

Models were trained on OpenWebText dataset (~500M tokens) using 80M parameter architectures on H100 GPUs.

## Repository Contents

- `gated_attention.ipynb` - Main implementation notebook with model definitions, training pipeline, and visualizations
- `RESULTS_ANALYSIS.md` - Comprehensive results analysis and findings
- `plot_training_results.ipynb` - Training metrics visualization
- `results/` - Training outputs and plots

## Model Checkpoints

**Note:** Model checkpoints are too large to include in this repository.

**Download checkpoints from Google Drive:**  
🔗 [Google Drive Link](https://drive.google.com/drive/folders/13uR8Wa3Z0JlAK1r_lOwm7lR-oxhvyaCo?usp=drive_link)

Checkpoints include:
- `Baseline_final.pt` - Baseline model (standard attention)
- `Elementwise_Gating_final.pt` - Elementwise gating model
- Intermediate checkpoints saved every 500 steps

## Requirements

- PyTorch
- transformers (GPT2Tokenizer)
- datasets (Hugging Face)
- matplotlib
- CUDA-capable GPU (training used H100 80GB)

## Usage

1. Download checkpoints from Google Drive link above
2. Place checkpoints in `gated_attention_checkpoints_v2/` directory
3. Open `gated_attention.ipynb` to explore the implementation

## Key Results

- Gating mechanisms add minimal overhead (<0.01% parameters)
- Competitive performance with baseline attention
- Successful training on 500M tokens
- Detailed attention pattern visualizations included

## Citation

If you use this code or findings in your research, please cite this repository.