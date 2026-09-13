---
license: other
tags:
- reinforcement-learning
- robot
- g1
- bfmzero
---

# BFMZero - 23DoF Checkpoint (Final)

Trained checkpoint for BFMZero (FBcprAuxAgent / FBcprAuxModel) with **23 degrees of freedom**.

## Training Status

- **Steps**: 384,000,000
- **Config**: Residual MLP with z_dim=256, hidden_dim=2048, 4 hidden layers
- **Device**: CUDA

## Files

| File | Description |
|------|-------------|
| `model_384000000.safetensors` | Model weights (SafeTensors format) |
| `optimizers_384000000.pth` | Optimizer states |
| `config.json` | Model architecture configuration |
| `config.yaml` | Full training configuration |
| `init_kwargs.json` | Initialization parameters |
| `train_status.json` | Training step counter |

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

# Download model weights
model_path = hf_hub_download(
    repo_id="Kennyp-Chen/bfmzero-23dof",
    filename="model_384000000.safetensors"
)
```
