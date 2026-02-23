"""TrendSignal dataclass — output of quantum trend detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.config.constants import Scenario, TrendDirection


@dataclass
class TrendSignal:
    """Result of a single quantum trend detection inference.

    Attributes
    ----------
    direction:
        Predicted market trend (LONG / SHORT / NEUTRAL).
    confidence:
        Probability confidence in ``[0, 1]``.
    scenario:
        Classified trading scenario (HOLD / DCA / REVERSAL).
    timestamp:
        When the signal was generated.
    features:
        Raw feature values used for this prediction.
    model_version:
        Identifier of the quantum model that produced this signal.
    """

    direction: TrendDirection
    confidence: float
    scenario: Scenario
    timestamp: datetime
    features: dict[str, float] = field(default_factory=dict)
    model_version: str = "v0.1.0"
