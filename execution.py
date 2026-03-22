"""
Almgren-Chriss optimal execution for directional entries/exits.

Adapted for 5-minute BTC binary contracts where execution windows
are 10-30 seconds and position sizes are $200-$1000.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class ExecutionSlice:
    time: float       # seconds from start
    size: float       # $ to trade in this slice
    remaining: float  # $ remaining after this slice


@dataclass
class ExecutionPlan:
    slices: List[ExecutionSlice]
    expected_cost: float
    implementation_shortfall_pct: float
    kappa: float
    urgency: str

    def __repr__(self):
        lines = [
            f"ExecutionPlan: {len(self.slices)} slices, "
            f"urgency={self.urgency}, κ={self.kappa:.3f}",
            f"  Expected cost: ${self.expected_cost:.2f} "
            f"({self.implementation_shortfall_pct:.3f}%)",
        ]
        for s in self.slices:
            lines.append(
                f"  t={s.time:5.1f}s  trade=${s.size:8.1f}  "
                f"remaining=${s.remaining:8.1f}"
            )
        return "\n".join(lines)


def almgren_chriss_schedule(
    total_size: float,
    T_seconds: float,
    n_slices: int,
    sigma: float,
    eta: float,
    gamma_temp: float,
    risk_aversion: float,
) -> ExecutionPlan:
    """
    Compute the optimal execution schedule via Almgren-Chriss.

    Args:
        total_size:    total $ to execute
        T_seconds:     execution window in seconds
        n_slices:      number of child orders
        sigma:         price volatility (per second)
        eta:           permanent impact coefficient
        gamma_temp:    temporary impact coefficient
        risk_aversion: λ (0 = ignore risk → TWAP; high = front-load)

    Returns:
        ExecutionPlan with optimal trade sizes and timing.
    """
    tau = T_seconds / n_slices

    # Urgency parameter
    kappa_sq = risk_aversion * sigma**2 / max(eta, 1e-12)
    kappa = np.sqrt(max(kappa_sq, 1e-12))

    times = np.linspace(0, T_seconds, n_slices + 1)

    # Optimal remaining position at each time
    sinh_kT = np.sinh(kappa * T_seconds)
    if abs(sinh_kT) < 1e-12:
        # Low urgency → approximately TWAP
        remaining = total_size * (1 - times / T_seconds)
    else:
        remaining = total_size * np.sinh(kappa * (T_seconds - times)) / sinh_kT

    # Trade sizes
    trade_sizes = -np.diff(remaining)

    # Expected costs
    temp_impact = gamma_temp * np.sum(trade_sizes**2) / tau
    perm_impact = 0.5 * eta * total_size**2
    total_cost = temp_impact + perm_impact

    slices = [
        ExecutionSlice(
            time=times[i],
            size=trade_sizes[i],
            remaining=remaining[i],
        )
        for i in range(n_slices)
    ]

    if kappa > 2:
        urgency = "high"
    elif kappa < 0.5:
        urgency = "low"
    else:
        urgency = "moderate"

    return ExecutionPlan(
        slices=slices,
        expected_cost=total_cost,
        implementation_shortfall_pct=(total_cost / total_size) * 100
        if total_size > 0
        else 0.0,
        kappa=kappa,
        urgency=urgency,
    )


def quick_execution_plan(
    total_size: float,
    fair_value_divergence: float,
    t_remaining_seconds: float,
) -> ExecutionPlan:
    """
    Simplified execution planner for the 5-min market.

    Heuristics:
      - < $200  → single fill
      - $200-$1000 → 2-3 splits over 10-20s
      - > $1000 → 4-5 splits
      - Large divergence → front-load (high risk aversion)
    """
    # Determine slices and window
    if total_size < 200:
        n_slices = 1
        window = min(3.0, t_remaining_seconds * 0.5)
    elif total_size < 1000:
        n_slices = 3
        window = min(15.0, t_remaining_seconds * 0.3)
    else:
        n_slices = 5
        window = min(25.0, t_remaining_seconds * 0.3)

    # Risk aversion from divergence magnitude
    # Large divergence → high urgency (divergence closing fast)
    risk_aversion = 1e-6 + abs(fair_value_divergence) * 1e-4

    # Conservative defaults for binary contract
    sigma = 0.001      # per-second vol
    eta = 0.0005       # permanent impact
    gamma_temp = 0.002  # temporary impact

    return almgren_chriss_schedule(
        total_size=total_size,
        T_seconds=max(window, 1.0),
        n_slices=max(n_slices, 1),
        sigma=sigma,
        eta=eta,
        gamma_temp=gamma_temp,
        risk_aversion=risk_aversion,
    )
