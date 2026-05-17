"""
JUND-F0: Joint Unvoiced/Voiced Detection and F0 Estimation
A novel deep learning framework that jointly solves voiced/unvoiced detection
and fundamental frequency (F0) estimation in a single model.

Reference: "JUND-F0: A Novel Deep Learning Framework for Joint Unvoiced/Voiced
Detection And F0 Estimation" (ICASSP 2026)
Authors: Y. Chen, R. Feng, Y.L. Liu, Y. Hu, J. Yuan
"""

__version__ = "0.1.0"

from .model import JUNDF0, JUNDF0Config
from .dataset import VCTKDataset
from .metrics import compute_rpa, compute_rca, compute_ffe, compute_gpe, compute_vde
from .infer import JUNDF0Inference

__all__ = [
    "JUNDF0",
    "JUNDF0Config",
    "VCTKDataset",
    "compute_rpa",
    "compute_rca",
    "compute_ffe",
    "compute_gpe",
    "compute_vde",
    "JUNDF0Inference",
]
