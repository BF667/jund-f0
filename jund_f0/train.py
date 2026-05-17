"""
Training Pipeline for JUND-F0

Supports:
- Single GPU and multi-GPU (DDP) training
- Mixed precision training (AMP) for Colab T4/V100/A100
- Learning rate scheduling with warmup
- Gradient clipping
- Checkpoint saving and resuming
- TensorBoard / W&B logging
- Cosine annealing with warm restarts
- EMA (Exponential Moving Average) of model weights
"""

import os
import time
import logging
import json
import math
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from .model import JUNDF0, JUNDF0Config
from .dataset import create_dataloaders
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class CosineAnnealingWarmup:
    """Cosine annealing scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        cycle_length: Optional[int] = None,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.cycle_length = cycle_length or total_steps
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = lr

    def _get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lrs[0] * self.current_step / max(1, self.warmup_steps)

        # Cosine annealing
        progress = (self.current_step - self.warmup_steps) % self.cycle_length
        progress = progress / self.cycle_length
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        return self.min_lr + (self.base_lrs[0] - self.min_lr) * cosine

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class EMA:
    """Exponential Moving Average of model parameters for more stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replace model parameters with EMA averages."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    Training manager for JUND-F0 model.

    Handles the complete training loop including:
    - Forward/backward pass with mixed precision
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Checkpoint management
    - Evaluation and metric logging
    - EMA weight averaging
    """

    def __init__(
        self,
        config: JUNDF0Config,
        data_dir: str = "./data/vctk",
        output_dir: str = "./runs/jund-f0",
        batch_size: int = 16,
        num_workers: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        eval_interval: int = 2000,
        save_interval: int = 5000,
        log_interval: int = 100,
        grad_clip_norm: float = 5.0,
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        label_method: str = "pyin",
        segment_length: int = 512,
        resume_from: Optional[str] = None,
        use_wandb: bool = False,
        wandb_project: str = "jund-f0",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.grad_clip_norm = grad_clip_norm
        self.use_amp = use_amp
        self.max_steps = max_steps

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Model
        self.model = JUNDF0(config).to(self.device)
        n_params = self.model.count_parameters()
        logger.info(f"Model parameters: {n_params:,}")

        # DataLoaders
        self.train_loader, self.val_loader = create_dataloaders(
            root_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            segment_length=segment_length,
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            hop_length=config.hop_length,
            label_method=label_method,
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            min_lr=1e-6,
        )

        # AMP
        self.scaler = GradScaler() if use_amp else None

        # EMA
        self.ema = EMA(self.model, ema_decay) if use_ema else None

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config=vars(config))
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed. Disabling W&B logging.")
                self.use_wandb = False

        # Training state
        self.global_step = 0
        self.best_ffe = float("inf")

        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Max steps: {self.max_steps}")
        logger.info(f"  Eval interval: {self.eval_interval}")
        logger.info(f"  Save interval: {self.save_interval}")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info(f"  EMA: {self.ema is not None}")

        self.model.train()
        start_time = time.time()
        data_iter = iter(self.train_loader)

        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Move to device
            mel = batch["mel"].to(self.device)
            vuv = batch["vuv"].to(self.device)
            f0 = batch["f0"].to(self.device)

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(mel, vuv_label=vuv, f0_label=f0)
                    loss = outputs["loss"]
            else:
                outputs = self.model(mel, vuv_label=vuv, f0_label=f0)
                loss = outputs["loss"]

            # Backward pass
            self.optimizer.zero_grad()
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )
                self.optimizer.step()

            # Update scheduler
            self.scheduler.step()

            # Update EMA
            if self.ema is not None:
                self.ema.update()

            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.log_interval / elapsed
                lr = self.scheduler.get_lr()

                log_msg = (
                    f"Step {self.global_step}/{self.max_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"VUV: {outputs['vuv_loss'].item():.4f} | "
                    f"F0: {outputs['f0_loss'].item():.4f} | "
                    f"FFE: {outputs['ffe'].item():.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )
                logger.info(log_msg)

                # TensorBoard
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/vuv_loss", outputs["vuv_loss"].item(), self.global_step)
                self.writer.add_scalar("train/f0_loss", outputs["f0_loss"].item(), self.global_step)
                self.writer.add_scalar("train/ffe", outputs["ffe"].item(), self.global_step)
                self.writer.add_scalar("train/lr", lr, self.global_step)

                if self.use_wandb:
                    self.wandb.log({
                        "train/loss": loss.item(),
                        "train/vuv_loss": outputs["vuv_loss"].item(),
                        "train/f0_loss": outputs["f0_loss"].item(),
                        "train/ffe": outputs["ffe"].item(),
                        "train/lr": lr,
                    }, step=self.global_step)

                start_time = time.time()

            # Evaluation
            if self.global_step % self.eval_interval == 0:
                self._evaluate()

            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self._save_checkpoint()

        # Final save
        self._save_checkpoint(is_final=True)
        self._evaluate()
        logger.info("Training complete!")

    def _evaluate(self):
        """Run evaluation on validation set."""
        logger.info("Running evaluation...")

        # Use EMA weights for evaluation
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()

        all_f0_pred = []
        all_f0_label = []
        all_vuv_pred = []
        all_vuv_label = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                mel = batch["mel"].to(self.device)
                vuv = batch["vuv"].to(self.device)
                f0 = batch["f0"].to(self.device)

                outputs = self.model(mel, vuv_label=vuv, f0_label=f0)
                total_loss += outputs["loss"].item()
                n_batches += 1

                # Get predictions
                vuv_prob = outputs["vuv_prob"]
                f0_pred = outputs["f0_pred"]

                # V/UV decision
                vuv_decision = (vuv_prob > 0.5).float()

                all_f0_pred.append(f0_pred.cpu())
                all_f0_label.append(f0.cpu())
                all_vuv_pred.append(vuv_decision.cpu())
                all_vuv_label.append(vuv.cpu())

        # Compute metrics
        f0_pred_all = torch.cat(all_f0_pred, dim=0)
        f0_label_all = torch.cat(all_f0_label, dim=0)
        vuv_pred_all = torch.cat(all_vuv_pred, dim=0)
        vuv_label_all = torch.cat(all_vuv_label, dim=0)

        metrics = compute_all_metrics(
            f0_pred_all, f0_label_all, vuv_pred_all, vuv_label_all
        )

        avg_loss = total_loss / max(1, n_batches)
        metrics["val_loss"] = avg_loss

        # Log
        logger.info(
            f"  Val Loss: {avg_loss:.4f} | "
            f"RPA: {metrics['rpa']:.4f} | "
            f"RCA: {metrics['rca']:.4f} | "
            f"GPE: {metrics['gpe']:.4f} | "
            f"VDE: {metrics['vde']:.4f} | "
            f"FFE: {metrics['ffe']:.4f}"
        )

        self.writer.add_scalar("val/loss", avg_loss, self.global_step)
        self.writer.add_scalar("val/rpa", metrics["rpa"], self.global_step)
        self.writer.add_scalar("val/rca", metrics["rca"], self.global_step)
        self.writer.add_scalar("val/gpe", metrics["gpe"], self.global_step)
        self.writer.add_scalar("val/vde", metrics["vde"], self.global_step)
        self.writer.add_scalar("val/ffe", metrics["ffe"], self.global_step)

        if self.use_wandb:
            self.wandb.log({
                "val/loss": avg_loss,
                "val/rpa": metrics["rpa"],
                "val/rca": metrics["rca"],
                "val/gpe": metrics["gpe"],
                "val/vde": metrics["vde"],
                "val/ffe": metrics["ffe"],
            }, step=self.global_step)

        # Save best model
        if metrics["ffe"] < self.best_ffe:
            self.best_ffe = metrics["ffe"]
            self._save_checkpoint(is_best=True)
            logger.info(f"  New best FFE: {self.best_ffe:.4f}")

        # Restore original weights
        if self.ema is not None:
            self.ema.restore()

        self.model.train()

    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler": self.scheduler.current_step,
            "best_ffe": self.best_ffe,
            "config": vars(self.config),
        }

        if self.ema is not None:
            checkpoint["ema_shadow"] = self.ema.shadow

        if is_best:
            path = self.output_dir / "best_model.pt"
        elif is_final:
            path = self.output_dir / "final_model.pt"
        else:
            path = self.output_dir / f"checkpoint_step{self.global_step}.pt"

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_ffe = checkpoint.get("best_ffe", float("inf"))

        if self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]

        logger.info(
            f"Resumed from step {self.global_step}, best FFE: {self.best_ffe:.4f}"
        )
