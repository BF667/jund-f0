#!/usr/bin/env python3
"""
JUND-F0 Training Script

Usage:
    # Train with default settings
    python -m jund_f0.scripts.train --data_dir ./data/vctk

    # Train with custom config
    python -m jund_f0.scripts.train --config configs/default.yaml

    # Resume training
    python -m jund_f0.scripts.train --resume_from runs/jund-f0/best_model.pt
"""

import argparse
import logging
import sys

from jund_f0.model import JUNDF0Config
from jund_f0.train import Trainer
from jund_f0.config import TrainConfig, save_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train JUND-F0 model")

    # Model config
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=160)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--encoder_channels", type=int, default=64)
    parser.add_argument("--encoder_blocks", type=int, default=4)
    parser.add_argument("--use_self_attention", action="store_true", default=True)
    parser.add_argument("--no_self_attention", action="store_true")
    parser.add_argument("--f0_min", type=float, default=50.0)
    parser.add_argument("--f0_max", type=float, default=800.0)

    # Training config
    parser.add_argument("--data_dir", type=str, default="./data/vctk")
    parser.add_argument("--output_dir", type=str, default="./runs/jund-f0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--grad_clip_norm", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--segment_length", type=int, default=512)
    parser.add_argument("--label_method", type=str, default="pyin")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="jund-f0")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build model config
    model_config = JUNDF0Config(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        f0_min=args.f0_min,
        f0_max=args.f0_max,
        encoder_channels=args.encoder_channels,
        encoder_blocks=args.encoder_blocks,
        use_self_attention=not args.no_self_attention,
    )

    # Build training config
    train_config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        segment_length=args.segment_length,
        label_method=args.label_method,
        use_amp=not args.no_amp,
        use_ema=not args.no_ema,
        resume_from=args.resume_from,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )

    # Save config
    save_config(model_config, train_config, f"{args.output_dir}/config.yaml")

    # Create trainer and start training
    trainer = Trainer(
        config=model_config,
        data_dir=train_config.data_dir,
        output_dir=train_config.output_dir,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        eval_interval=train_config.eval_interval,
        save_interval=train_config.save_interval,
        log_interval=train_config.log_interval,
        grad_clip_norm=train_config.grad_clip_norm,
        use_amp=train_config.use_amp,
        use_ema=train_config.use_ema,
        label_method=train_config.label_method,
        segment_length=train_config.segment_length,
        resume_from=train_config.resume_from,
        use_wandb=train_config.use_wandb,
        wandb_project=train_config.wandb_project,
    )

    trainer.train()


if __name__ == "__main__":
    main()
