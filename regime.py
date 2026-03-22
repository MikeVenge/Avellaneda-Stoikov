"""
Regime classifier — determines which operational mode the strategy runs in.

Modes:
  1. MARKET_MAKING (default): A-S quoting around external fair value
  2. DIRECTIONAL: take position when fair value diverges from market
  3. DEFENSIVE: pull quotes, flatten, wait

Switches based on: VPIN, Hawkes branching ratio, Kyle's λ, fair value divergence.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Mode(Enum):
    MARKET_MAKING = "market_making"
    DIRECTIONAL = "directional"
    DEFENSIVE = "defensive"


@dataclass
class RegimeSignals:
    """All signals needed for regime classification."""
    vpin: float                     # 0-1, higher = more informed flow
    branching_ratio: float          # 0-1, higher = self-exciting cascade
    kyle_lambda: float              # price impact coefficient
    kyle_r_squared: float           # significance of lambda estimate
    fair_value: float               # BTC-derived probability
    market_mid: float               # Polymarket mid price
    spread: float                   # current Polymarket spread
    t_remaining: float              # minutes until expiry
    btc_flow_confirms: bool = False # does BTC order flow confirm the λ signal?


@dataclass
class RegimeDecision:
    mode: Mode
    reason: str
    confidence: float     # 0-1
    direction: Optional[int] = None   # +1 (buy YES), -1 (sell YES), None
    size_scalar: float = 1.0          # multiply default size


class RegimeClassifier:
    """
    State machine that determines the current operational mode.

    Thresholds calibrated for 5-minute BTC binary contracts.
    """

    def __init__(
        self,
        vpin_warning: float = 0.60,
        vpin_danger: float = 0.75,
        vpin_critical: float = 0.85,
        branching_warning: float = 0.70,
        branching_danger: float = 0.85,
        lambda_threshold: float = 0.005,
        divergence_threshold: float = 1.5,  # multiples of spread
        min_time_for_mm: float = 0.5,       # minutes
    ):
        self.vpin_warning = vpin_warning
        self.vpin_danger = vpin_danger
        self.vpin_critical = vpin_critical
        self.branching_warning = branching_warning
        self.branching_danger = branching_danger
        self.lambda_threshold = lambda_threshold
        self.divergence_threshold = divergence_threshold
        self.min_time_for_mm = min_time_for_mm

    def classify(self, signals: RegimeSignals) -> RegimeDecision:
        """Classify current regime from signals."""

        divergence = signals.fair_value - signals.market_mid
        div_in_spreads = (
            abs(divergence) / signals.spread if signals.spread > 0 else 0
        )

        # --- DEFENSIVE triggers (highest priority) ---

        if signals.vpin > self.vpin_critical:
            return RegimeDecision(
                mode=Mode.DEFENSIVE,
                reason=f"VPIN critical ({signals.vpin:.2f} > {self.vpin_critical})",
                confidence=0.95,
            )

        if signals.branching_ratio > self.branching_danger:
            return RegimeDecision(
                mode=Mode.DEFENSIVE,
                reason=f"Hawkes cascade ({signals.branching_ratio:.2f} > {self.branching_danger})",
                confidence=0.90,
            )

        if signals.t_remaining < self.min_time_for_mm:
            return RegimeDecision(
                mode=Mode.DEFENSIVE,
                reason=f"Too close to expiry ({signals.t_remaining:.1f} min)",
                confidence=0.99,
            )

        # --- DIRECTIONAL triggers ---

        # Large fair value divergence → arb opportunity
        if div_in_spreads > self.divergence_threshold:
            direction = 1 if divergence > 0 else -1
            # Higher confidence if Kyle's λ confirms
            conf = 0.70
            if (
                signals.kyle_lambda > self.lambda_threshold
                and signals.kyle_r_squared > 0.10
                and signals.btc_flow_confirms
            ):
                conf = 0.85

            return RegimeDecision(
                mode=Mode.DIRECTIONAL,
                reason=(
                    f"Fair value divergence: {div_in_spreads:.1f}x spread "
                    f"(FV={signals.fair_value:.3f} vs mid={signals.market_mid:.3f})"
                ),
                confidence=conf,
                direction=direction,
                size_scalar=min(div_in_spreads / self.divergence_threshold, 2.0),
            )

        # Kyle's λ spike with BTC flow confirmation → follow informed money
        if (
            signals.kyle_lambda > self.lambda_threshold * 2
            and signals.kyle_r_squared > 0.15
            and signals.btc_flow_confirms
        ):
            # Infer direction from whether fair_value > market_mid
            direction = 1 if divergence > 0 else -1
            return RegimeDecision(
                mode=Mode.DIRECTIONAL,
                reason=(
                    f"High Kyle's λ ({signals.kyle_lambda:.4f}) "
                    f"with confirming BTC flow"
                ),
                confidence=0.65,
                direction=direction,
                size_scalar=0.7,  # conservative — signal is indirect
            )

        # --- MARKET MAKING (default) ---

        # Adjust size based on adverse conditions
        size_scalar = 1.0
        if signals.vpin > self.vpin_warning:
            size_scalar *= 0.6
        if signals.branching_ratio > self.branching_warning:
            size_scalar *= 0.7
        if signals.kyle_lambda > self.lambda_threshold:
            size_scalar *= 0.8

        return RegimeDecision(
            mode=Mode.MARKET_MAKING,
            reason="Normal conditions — quoting",
            confidence=0.80,
            size_scalar=size_scalar,
        )
