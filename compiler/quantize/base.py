"""Abstract base class for quantization algorithms.

Provides the shared interface for activation-aware quantization
pipelines: config validation, calibration (activation statistics
collection), and weight quantization.

Subclasses:
  - AWQQuantizer     — W4A16 salient-channel weight quantization
  - SmoothQuantCalibrator — W8A8 SmoothQuant calibration + quantization

Reference: design-phase2.md §2.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseQuantizer(ABC):
    """Abstract base class for quantizers.

    Subclasses must implement:
      - calibrate():  Collect activation statistics over a calibration set.
      - quantize():   Apply quantization to model weights.

    Args:
        model: PyTorch nn.Module to quantize.
        config: Algorithm-specific configuration dict.
    """

    def __init__(self, model: nn.Module, config: dict[str, Any]) -> None:
        self.model = model
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:  # noqa: B027
        """Validate configuration parameters.

        Override in subclasses to enforce algorithm-specific
        constraints (e.g., value ranges, required keys).
        """
        pass  # optional hook — subclasses may override

    @abstractmethod
    def calibrate(
        self,
        dataloader: list[tuple[Any, ...]] | None = None,
        num_samples: int = 512,
    ) -> None:
        """Collect activation statistics for calibration.

        Args:
            dataloader: Calibration data batches.  If None, uses a
                single random input.
            num_samples: Maximum number of calibration samples.
        """

    @abstractmethod
    def quantize(self) -> None:
        """Apply quantization to model weights."""
