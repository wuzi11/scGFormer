# scGFormer

A lightweight graph learning framework for single-cell transcriptomics.  
Supports 5-fold cross-validation, optional dynamic graph (kNN), contrastive learning, and multiple long-tail friendly loss functions.

---
## Installation

We recommend **Python ≥ 3.9** and **PyTorch ≥ 2.0**.  
Please install PyTorch and PyG packages (`torch-scatter`, `torch-sparse`, `torch-geometric`) according to your CUDA version:

- PyTorch: https://pytorch.org  
- PyG: https://pytorch-geometric.readthedocs.io

## Implement

python main.py \
  --data_dir ./data \
  --dataset Baron \
  --use_HVG --use_knn \
  --hidden_channels 128 \
  --num_heads 8 \
  --loss ce \
  --epochs 30 \
  --seed 42

  ## Dataset
