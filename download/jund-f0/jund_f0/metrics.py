"""
Evaluation Metrics for F0 Estimation

Implements standard MIR evaluation metrics:
- RPA: Raw Pitch Accuracy
- RCA: Raw Chroma Accuracy
- GPE: Gross Pitch Error
- VDE: Voicing Decision Error
- FFE: F0 Frame Error (combines GPE and VDE)

Reference: "A Comparative Evaluation of Pitch Tracking Methods"
(Naylor & Bunch, 2009) and the NIST Speech Quality Assessment package.
"""

import numpy as np
import torch
from typing import Optional, Tuple


def _to_numpy(*tensors: torch.Tensor) -> Tuple[np.ndarray, ...]:
    """Convert tensors to numpy arrays."""
    result = []
    for t in tensors:
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        result.append(t)
    return tuple(result)


def compute_rpa(
    f0_pred: torch.Tensor,
    f0_label: torch.Tensor,
    vuv_pred: Optional[torch.Tensor] = None,
    vuv_label: Optional[torch.Tensor] = None,
    tol: float = 0.2,
) -> float:
    """
    Compute Raw Pitch Accuracy (RPA).

    RPA = (number of correctly pitched voiced frames) / (total voiced frames in reference)

    A frame is considered correctly pitched if the relative F0 error is within
    the tolerance threshold (default 20%).

    Args:
        f0_pred: Predicted F0 values in Hz
        f0_label: Ground truth F0 values in Hz
        vuv_pred: Predicted V/UV (if None, inferred from f0_pred > 0)
        vuv_label: Ground truth V/UV (if None, inferred from f0_label > 0)
        tol: Relative tolerance threshold (default 0.2 = 20%)

    Returns:
        RPA score in [0, 1]
    """
    f0_pred, f0_label = _to_numpy(f0_pred, f0_label)
    f0_pred = f0_pred.flatten()
    f0_label = f0_label.flatten()

    if vuv_pred is not None:
        vuv_pred = vuv_pred.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_pred = f0_pred > 0

    if vuv_label is not None:
        vuv_label = vuv_label.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_label = f0_label > 0

    # Both predicted and reference are voiced
    both_voiced = vuv_pred & vuv_label

    if both_voiced.sum() == 0:
        return 0.0

    # Relative error on both-voiced frames
    f0_pred_v = f0_pred[both_voiced]
    f0_label_v = f0_label[both_voiced]

    rel_error = np.abs(f0_pred_v - f0_label_v) / np.maximum(f0_label_v, 1e-8)

    correct = rel_error <= tol
    rpa = correct.sum() / both_voiced.sum()

    return float(rpa)


def compute_rca(
    f0_pred: torch.Tensor,
    f0_label: torch.Tensor,
    vuv_pred: Optional[torch.Tensor] = None,
    vuv_label: Optional[torch.Tensor] = None,
    tol: float = 0.2,
) -> float:
    """
    Compute Raw Chroma Accuracy (RCA).

    RCA is similar to RPA but ignores octave errors. It computes the
    chroma (pitch class) by taking F0 modulo the nearest octave.

    A frame is correct if its chroma-relative F0 is within tolerance.

    Args:
        f0_pred: Predicted F0 values in Hz
        f0_label: Ground truth F0 values in Hz
        vuv_pred: Predicted V/UV
        vuv_label: Ground truth V/UV
        tol: Relative tolerance (default 20%)

    Returns:
        RCA score in [0, 1]
    """
    f0_pred, f0_label = _to_numpy(f0_pred, f0_label)
    f0_pred = f0_pred.flatten()
    f0_label = f0_label.flatten()

    if vuv_pred is not None:
        vuv_pred = vuv_pred.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_pred = f0_pred > 0

    if vuv_label is not None:
        vuv_label = vuv_label.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_label = f0_label > 0

    both_voiced = vuv_pred & vuv_label

    if both_voiced.sum() == 0:
        return 0.0

    f0_pred_v = f0_pred[both_voiced]
    f0_label_v = f0_label[both_voiced]

    # Chroma: convert to log2 space and take fractional part
    log_pred = np.log2(np.maximum(f0_pred_v, 1e-8))
    log_label = np.log2(np.maximum(f0_label_v, 1e-8))

    # Relative error in octaves, ignoring octave shifts
    octave_error = np.abs(log_pred - log_label)
    chroma_error = np.minimum(octave_error % 1.0, 1.0 - octave_error % 1.0)

    # Convert chroma error back to relative error
    rel_error = np.abs(2 ** chroma_error - 1.0)

    correct = rel_error <= tol
    rca = correct.sum() / both_voiced.sum()

    return float(rca)


def compute_gpe(
    f0_pred: torch.Tensor,
    f0_label: torch.Tensor,
    vuv_pred: Optional[torch.Tensor] = None,
    vuv_label: Optional[torch.Tensor] = None,
    tol: float = 0.2,
) -> float:
    """
    Compute Gross Pitch Error (GPE).

    GPE = (number of frames with relative F0 error > tol) / (total voiced frames in reference)

    Only considers frames where both prediction and reference are voiced.

    Args:
        f0_pred: Predicted F0 values in Hz
        f0_label: Ground truth F0 values in Hz
        vuv_pred: Predicted V/UV
        vuv_label: Ground truth V/UV
        tol: Relative tolerance (default 20%)

    Returns:
        GPE score in [0, 1]
    """
    rpa = compute_rpa(f0_pred, f0_label, vuv_pred, vuv_label, tol)
    return 1.0 - rpa


def compute_vde(
    f0_pred: torch.Tensor,
    f0_label: torch.Tensor,
    vuv_pred: Optional[torch.Tensor] = None,
    vuv_label: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Voicing Decision Error (VDE).

    VDE = (number of frames with incorrect V/UV decision) / (total frames)

    Args:
        f0_pred: Predicted F0 values in Hz
        f0_label: Ground truth F0 values in Hz
        vuv_pred: Predicted V/UV
        vuv_label: Ground truth V/UV

    Returns:
        VDE score in [0, 1]
    """
    f0_pred, f0_label = _to_numpy(f0_pred, f0_label)
    f0_pred = f0_pred.flatten()
    f0_label = f0_label.flatten()

    if vuv_pred is not None:
        vuv_pred = vuv_pred.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_pred = f0_pred > 0

    if vuv_label is not None:
        vuv_label = vuv_label.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_label = f0_label > 0

    n_frames = len(vuv_pred)
    if n_frames == 0:
        return 0.0

    errors = vuv_pred != vuv_label
    vde = errors.sum() / n_frames

    return float(vde)


def compute_ffe(
    f0_pred: torch.Tensor,
    f0_label: torch.Tensor,
    vuv_pred: Optional[torch.Tensor] = None,
    vuv_label: Optional[torch.Tensor] = None,
    tol: float = 0.2,
) -> float:
    """
    Compute F0 Frame Error (FFE).

    FFE is the proposed combined metric that considers a frame as erroneous if:
    1. The voicing decision is wrong (VDE component), OR
    2. The voicing decision is correct but the F0 error exceeds the tolerance (GPE component)

    This is the primary evaluation metric for JUND-F0 as it captures both
    V/UV detection and F0 estimation quality in a single number.

    Args:
        f0_pred: Predicted F0 values in Hz
        f0_label: Ground truth F0 values in Hz
        vuv_pred: Predicted V/UV
        vuv_label: Ground truth V/UV
        tol: Relative tolerance (default 20%)

    Returns:
        FFE score in [0, 1]
    """
    f0_pred, f0_label = _to_numpy(f0_pred, f0_label)
    f0_pred = f0_pred.flatten()
    f0_label = f0_label.flatten()

    if vuv_pred is not None:
        vuv_pred = vuv_pred.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_pred = f0_pred > 0

    if vuv_label is not None:
        vuv_label = vuv_label.detach().cpu().numpy().flatten().astype(bool)
    else:
        vuv_label = f0_label > 0

    n_frames = len(vuv_pred)
    if n_frames == 0:
        return 0.0

    # V/UV error
    vuv_error = vuv_pred != vuv_label

    # GPE on correctly voiced frames
    both_voiced = vuv_pred & vuv_label
    gpe_frames = np.zeros(n_frames, dtype=bool)

    if both_voiced.sum() > 0:
        f0_pred_v = f0_pred[both_voiced]
        f0_label_v = f0_label[both_voiced]
        rel_error = np.abs(f0_pred_v - f0_label_v) / np.maximum(f0_label_v, 1e-8)
        gpe_indices = np.where(both_voiced)[0]
        gpe_frames[gpe_indices] = rel_error > tol

    # FFE: frame is erroneous if V/UV is wrong OR GPE
    ffe_frames = vuv_error | gpe_frames
    ffe = ffe_frames.sum() / n_frames

    return float(ffe)


def compute_all_metrics(
    f0_pred: torch.Tensor,
    f0_label: torch.Tensor,
    vuv_pred: Optional[torch.Tensor] = None,
    vuv_label: Optional[torch.Tensor] = None,
    tol: float = 0.2,
) -> dict:
    """
    Compute all F0 estimation metrics.

    Returns:
        Dictionary with RPA, RCA, GPE, VDE, FFE scores
    """
    return {
        "rpa": compute_rpa(f0_pred, f0_label, vuv_pred, vuv_label, tol),
        "rca": compute_rca(f0_pred, f0_label, vuv_pred, vuv_label, tol),
        "gpe": compute_gpe(f0_pred, f0_label, vuv_pred, vuv_label, tol),
        "vde": compute_vde(f0_pred, f0_label, vuv_pred, vuv_label),
        "ffe": compute_ffe(f0_pred, f0_label, vuv_pred, vuv_label, tol),
    }
