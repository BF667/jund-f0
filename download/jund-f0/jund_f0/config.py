"""
Configuration management for JUND-F0
"""

import yaml
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from .model import JUNDF0Config


@dataclass
class TrainConfig:
    """Training configuration."""
    # Data
    data_dir: str = "./data/vctk"
    label_method: str = "pyin"  # "pyin" or "crepe"
    segment_length: int = 512

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100000
    grad_clip_norm: float = 5.0

    # Evaluation
    eval_interval: int = 2000
    save_interval: int = 5000
    log_interval: int = 100

    # System
    num_workers: int = 4
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.999

    # Output
    output_dir: str = "./runs/jund-f0"

    # Resume
    resume_from: Optional[str] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "jund-f0"


def save_config(model_config: JUNDF0Config, train_config: TrainConfig, path: str):
    """Save configuration to YAML file."""
    config = {
        "model": asdict(model_config),
        "train": asdict(train_config),
    }
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def load_config(path: str) -> tuple:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    model_config = JUNDF0Config(**config.get("model", {}))
    train_config = TrainConfig(**config.get("train", {}))

    return model_config, train_config
