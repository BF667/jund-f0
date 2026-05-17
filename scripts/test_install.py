#!/usr/bin/env python3
"""
Quick test to verify JUND-F0 installation and model architecture.
"""

import sys
import torch

def test_model():
    from jund_f0.model import JUNDF0, JUNDF0Config

    config = JUNDF0Config()
    model = JUNDF0(config)

    # Count parameters
    n_params = model.count_parameters()
    print(f"Model parameters: {n_params:,}")

    # Forward pass test
    batch_size = 2
    n_frames = 32
    mel = torch.randn(batch_size, config.n_mels, n_frames)
    vuv = torch.randint(0, 2, (batch_size, n_frames, 1)).float()
    f0 = torch.abs(torch.randn(batch_size, n_frames, 1)) * 200 + 100

    outputs = model(mel, vuv_label=vuv, f0_label=f0)

    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"VUV loss: {outputs['vuv_loss'].item():.4f}")
    print(f"F0 loss: {outputs['f0_loss'].item():.4f}")
    print(f"FFE: {outputs['ffe'].item():.4f}")
    print(f"VUV prob shape: {outputs['vuv_prob'].shape}")
    print(f"F0 pred shape: {outputs['f0_pred'].shape}")

    # Inference test
    vuv_pred, f0_pred = model.predict(mel)
    print(f"VUV pred shape: {vuv_pred.shape}")
    print(f"F0 pred shape: {f0_pred.shape}")

    print("\nModel test PASSED!")


def test_metrics():
    from jund_f0.metrics import compute_all_metrics

    # Create synthetic predictions
    f0_pred = torch.tensor([[[100.0], [200.0], [0.0], [300.0], [150.0]]])
    f0_label = torch.tensor([[[105.0], [190.0], [0.0], [310.0], [0.0]]])

    metrics = compute_all_metrics(f0_pred, f0_label)

    print(f"RPA: {metrics['rpa']:.4f}")
    print(f"RCA: {metrics['rca']:.4f}")
    print(f"GPE: {metrics['gpe']:.4f}")
    print(f"VDE: {metrics['vde']:.4f}")
    print(f"FFE: {metrics['ffe']:.4f}")

    print("\nMetrics test PASSED!")


if __name__ == "__main__":
    print("=" * 50)
    print("JUND-F0 Installation Test")
    print("=" * 50)

    test_model()
    print()
    test_metrics()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
