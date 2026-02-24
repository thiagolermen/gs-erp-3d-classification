"""
Neural network architectures for ERP-ViT 3D Classification.

Sub-packages
------------
blocks    — HSDC and SWHDC distortion-correction blocks.
backbones — Full model architectures (backbone + block + head).
"""

from src.models.classifier import ClassificationHead
from src.models.blocks.hsdc import HSDCBlock
from src.models.blocks.swhdc import SWHDCBlock

__all__ = [
    "ClassificationHead",
    "HSDCBlock",
    "SWHDCBlock",
]
