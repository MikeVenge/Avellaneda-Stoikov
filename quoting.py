"""
Avellaneda-Stoikov quoting engine adapted for 5-minute BTC binary contracts.

Key adaptations vs. standard A-S:
  - Uses BTC-spot-derived fair value instead of Polymarket mid
  - Time-decay-aware spread schedule (wide early, tight late)
  - VPIN-triggered spread widening
  - Inventory limits that tighten as expiry approaches
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Quote:
    bid: float
    ask: float
    reservation_price: float
    half_spread: float
    spread_multiplier: float  # from VPIN / regime
    time_remaining: float
    inventory: float


@dataclass
class Fill:
    side: str       # "buy" or "sell"
    price: float
    size: float
    timestamp: float


@dataclass
class MMState:
    inventory: float = 0.0      # positive = long YES tokens ($)
    cash: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    fills: List[Fill] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    quote_history: List[Quote] = field(default_factory=list)

    @property
    def mark_to_market(self) -> float:
        """Needs external mid price — use last quote's reservation price."""
        if self.quote_history:
            mid = self.quote_history[-1].reservation_price
            return self.inventory * mid + self.cash
        return self.cash


class AvellanedaStoikovEngine:
    """
    Market maker for 5-minute BTC binary contracts.

    Parameters tuned for the ultra-short horizon:
      gamma:  risk aversion (higher = more inventory penalty)
      kappa:  order arrival intensity
      sigma:  per-period volatility of the contract price
      T:      total contract duration in minutes
      max_inventory: hard position limit ($)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        kappa: float = 8.0,
        sigma: float = 0.06,
        T_minutes: float = 5.0,
        max_inventory: float = 500.0,
        min_spread: float = 0.005,
        max_spread: float = 0.15,
    ):
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.T = T_minutes
        self.max_inventory = max_inventory
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.state = MMState()

    def reset(self):
        """Reset state for a new 5-minute contract."""
        self.state = MMState()

    # --- Core A-S formulas (time in minutes) ---

    def reservation_price(
        self, fair_value: float, inventory: float, t_remaining: float
    ) -> float:
        """
        r = fair_value - q · γ · σ² · (T - t)
        Positive inventory → shade price down (want to sell).
        """
        return fair_value - inventory * self.gamma * self.sigma**2 * t_remaining

    def base_half_spread(self, t_remaining: float) -> float:
        """
        δ = γσ²(T-t) + (1/γ)·ln(1 + γ/κ)
        Returns the half-spread (distance from reservation price to each side).
        """
        risk_term = self.gamma * self.sigma**2 * t_remaining
        execution_term = (1.0 / self.gamma) * np.log(1 + self.gamma / self.kappa)
        return risk_term + execution_term

    # --- Inventory limits scaled by time remaining ---

    def inventory_limit(self, t_remaining: float) -> float:
        """Tighten inventory limit as expiry approaches."""
        if t_remaining > 3.0:
            return self.max_inventory
        elif t_remaining > 1.0:
            return self.max_inventory * 0.6
        elif t_remaining > 0.5:
            return self.max_inventory * 0.3
        else:
            return self.max_inventory * 0.1

    # --- Spread multiplier from VPIN ---

    @staticmethod
    def vpin_spread_multiplier(vpin: float) -> float:
        """Scale spread based on adverse selection signal."""
        if vpin > 0.85:
            return np.inf  # pull quotes
        elif vpin > 0.75:
            return 2.0
        elif vpin > 0.60:
            return 1.5
        else:
            return 1.0

    # --- Spread multiplier from Hawkes branching ratio ---

    @staticmethod
    def hawkes_spread_multiplier(branching_ratio: float) -> float:
        if branching_ratio > 0.85:
            return np.inf  # pull quotes
        elif branching_ratio > 0.70:
            return 1.8
        elif branching_ratio > 0.50:
            return 1.3
        else:
            return 1.0

    # --- Main quoting function ---

    def quote(
        self,
        fair_value: float,
        t_remaining: float,
        vpin: float = 0.0,
        branching_ratio: float = 0.0,
    ) -> Optional[Quote]:
        """
        Compute bid/ask for the current state.

        Returns None if quotes should be pulled (VPIN too high, too close
        to expiry, etc.).
        """
        # Pull quotes in last 10 seconds
        if t_remaining < 10 / 60:  # 10 seconds in minutes
            return None

        # VPIN + Hawkes multiplier
        vm = self.vpin_spread_multiplier(vpin)
        hm = self.hawkes_spread_multiplier(branching_ratio)
        spread_mult = max(vm, hm)

        if spread_mult == np.inf:
            return None  # pull quotes

        inv = self.state.inventory
        r = self.reservation_price(fair_value, inv, t_remaining)
        hs = self.base_half_spread(t_remaining) * spread_mult

        # Clamp spread
        hs = np.clip(hs, self.min_spread, self.max_spread / 2)

        bid = r - hs
        ask = r + hs

        # Enforce [0.01, 0.99] and ask > bid
        bid = np.clip(bid, 0.01, 0.98)
        ask = np.clip(ask, 0.02, 0.99)
        if ask <= bid:
            ask = bid + 0.01

        # Aggressive inventory unwind near limits
        inv_limit = self.inventory_limit(t_remaining)
        if abs(inv) > inv_limit * 0.8:
            if inv > 0:
                # Long → desperately want to sell → lower ask aggressively
                ask = min(ask, fair_value - 0.005)
                ask = max(ask, 0.02)
            else:
                # Short → desperately want to buy → raise bid aggressively
                bid = max(bid, fair_value + 0.005)
                bid = min(bid, 0.98)

        q = Quote(
            bid=round(bid, 4),
            ask=round(ask, 4),
            reservation_price=round(r, 4),
            half_spread=round(hs, 4),
            spread_multiplier=spread_mult,
            time_remaining=t_remaining,
            inventory=inv,
        )
        self.state.quote_history.append(q)
        return q

    # --- Fill handling ---

    def fill_bid(self, price: float, size: float, timestamp: float = 0.0):
        """Our bid was hit — we bought YES tokens."""
        inv_limit = self.inventory_limit(
            self.state.quote_history[-1].time_remaining
            if self.state.quote_history
            else self.T
        )
        # Reject if would exceed inventory limit
        if self.state.inventory + size > inv_limit:
            size = max(0, inv_limit - self.state.inventory)
            if size <= 0:
                return

        self.state.inventory += size
        self.state.cash -= price * size
        self.state.n_buys += 1
        self.state.fills.append(Fill("buy", price, size, timestamp))

    def fill_ask(self, price: float, size: float, timestamp: float = 0.0):
        """Our ask was lifted — we sold YES tokens."""
        inv_limit = self.inventory_limit(
            self.state.quote_history[-1].time_remaining
            if self.state.quote_history
            else self.T
        )
        if self.state.inventory - size < -inv_limit:
            size = max(0, inv_limit + self.state.inventory)
            if size <= 0:
                return

        self.state.inventory -= size
        self.state.cash += price * size
        self.state.n_sells += 1
        self.state.fills.append(Fill("sell", price, size, timestamp))

    def settle(self, outcome: float) -> float:
        """
        Settle the contract. outcome = 1.0 (YES) or 0.0 (NO).
        Returns final P&L.
        """
        return self.state.inventory * outcome + self.state.cash
