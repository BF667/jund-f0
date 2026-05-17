# JUND-F0: Joint Unvoiced/Voiced Detection and F0 Estimation

A PyTorch implementation of **JUND-F0**, a novel deep learning framework that jointly solves voiced/unvoiced (V/UV) detection and fundamental frequency (F0) estimation in a single unified model.

**Reference:** *JUND-F0: A Novel Deep Learning Framework for Joint Unvoiced/Voiced Detection And F0 Estimation* (ICASSP 2026) — Y. Chen, R. Feng, Y.L. Liu, Y. Hu, J. Yuan

---

## Key Features

- **Shared Encoder + Dual Heads**: A shared CNN backbone extracts features for both V/UV classification and F0 regression
- **Self-Attention**: Captures long-range temporal dependencies in pitch contours
- **Depthwise Separable Convolutions**: Efficient architecture with minimal parameters
- **Joint Loss**: Combines V/UV BCE loss, F0 L1 loss, and FFE auxiliary loss
- **F0 Frame Error (FFE)**: Combined metric that captures both V/UV and F0 errors
- **EMA Weight Averaging**: More stable evaluation with exponential moving average
- **Colab-Ready**: Mixed precision (AMP) training for fast training on Colab GPUs
- **VCTK Support**: Automatic download and F0 pseudo-label generation

## Architecture

```
Input: Mel Spectrogram (80 bins × T frames)
  │
  ├─ Shared Encoder
  │   ├─ Input Projection (Conv1d)
  │   ├─ Residual Block 1 (kernel=3, dilation=1)
  │   ├─ Residual Block 2 (kernel=5, dilation=1)
  │   ├─ Residual Block 3 (kernel=5, dilation=2)
  │   ├─ Residual Block 4 (kernel=5, dilation=4)
  │   └─ Multi-Head Self-Attention (4 heads)
  │
  ├─ V/UV Head → Voiced/Unvoiced Probability
  └─ F0 Head → F0 in Hz [50, 800]
```

## Quick Start

### Installation

```bash
git clone https://github.com/your-repo/jund-f0.git
cd jund-f0
pip install -e .
```

### Train on Colab

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Select GPU runtime (T4 or better)
3. Run all cells — VCTK will be downloaded and F0 labels generated automatically

### Train Locally

```bash
# Download VCTK and train with default settings
python -m jund_f0.scripts.train --data_dir ./data/vctk --max_steps 100000

# Train with custom parameters
python -m jund_f0.scripts.train \
    --data_dir ./data/vctk \
    --batch_size 32 \
    --learning_rate 0.002 \
    --encoder_channels 128 \
    --use_self_attention \
    --max_steps 200000
```

### Inference

```python
from jund_f0.infer import JUNDF0Inference

# Load model
inferencer = JUNDF0Inference(model_path="best_model.pt")

# Extract F0 from audio file
result = inferencer.extract_f0("audio.wav")
print(f"F0: {result['f0']}")
print(f"V/UV: {result['vuv']}")
```

### RVC Integration

```python
from jund_f0.infer import create_rvc_compatible_extractor

# Create RVC-compatible F0 extractor
f0_extractor = create_rvc_compatible_extractor("best_model.pt")

# Use in RVC pipeline
import numpy as np
audio = np.random.randn(16000).astype(np.float32)
f0 = f0_extractor(audio, sample_rate=16000)
```

## Metrics

| Metric | Description |
|--------|-------------|
| **RPA** | Raw Pitch Accuracy — % of voiced frames with F0 error < 20% |
| **RCA** | Raw Chroma Accuracy — like RPA but ignores octave errors |
| **GPE** | Gross Pitch Error — % of voiced frames with F0 error > 20% |
| **VDE** | Voicing Decision Error — % of frames with wrong V/UV |
| **FFE** | F0 Frame Error — combined: frame is wrong if V/UV wrong OR GPE |

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Audio sample rate |
| `n_mels` | 80 | Mel spectrogram bins |
| `hop_length` | 160 | Frame hop (10ms at 16kHz) |
| `encoder_channels` | 64 | CNN channel width |
| `encoder_blocks` | 4 | Number of residual blocks |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 1e-3 | Peak learning rate |
| `max_steps` | 100000 | Total training steps |
| `segment_length` | 512 | Frames per training sample (~3.2s) |
| `use_amp` | True | Mixed precision training |
| `use_ema` | True | EMA weight averaging |

## Project Structure

```
jund-f0/
├── jund_f0/
│   ├── __init__.py          # Package init
│   ├── model.py             # JUND-F0 model architecture
│   ├── dataset.py           # VCTK dataset loader
│   ├── train.py             # Training pipeline
│   ├── metrics.py           # Evaluation metrics (RPA, RCA, GPE, VDE, FFE)
│   ├── infer.py             # Inference pipeline + RVC integration
│   └── config.py            # Configuration management
├── scripts/
│   ├── train.py             # Training CLI
│   ├── infer.py             # Inference CLI
│   └── test_install.py      # Installation test
├── configs/
│   └── default.yaml         # Default training config
├── notebooks/
│   └── train_colab.ipynb    # Colab training notebook
├── pyproject.toml           # Python package config
└── README.md
```

## Citation

```bibtex
@inproceedings{chen2026jundf0,
  title={JUND-F0: A Novel Deep Learning Framework for Joint Unvoiced/Voiced Detection And F0 Estimation},
  author={Chen, Y. and Feng, R. and Liu, Y.L. and Hu, Y. and Yuan, J.},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing},
  year={2026}
}
```

## License

MIT License
