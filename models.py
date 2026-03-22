"""
Core quantitative models for 5-minute BTC prediction market trading.

Models:
  - Kyle's Lambda: information asymmetry detection
  - Hawkes Process: order flow regime detection
  - VPIN: adverse selection early warning
  - Digital Option Fair Value: BTC-derived contract pricing
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress, norm
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Kyle's Lambda — estimate price impact coefficient from trade data
# ---------------------------------------------------------------------------

@dataclass
class KyleLambdaResult:
    lam: float          # price impact per unit signed volume
    r_squared: float
    std_error: float
    p_value: float

    @property
    def informed_trading(self) -> bool:
        """High lambda + significant R² → informed traders are active."""
        return self.lam > 0.002 and self.r_squared > 0.10


def estimate_kyle_lambda(
    prices: np.ndarray,
    volumes: np.ndarray,
    signs: np.ndarray,
) -> KyleLambdaResult:
    """
    Δp_t = λ · Q_t + ε_t
    where Q_t = sign_t · volume_t (signed order flow).
    """
    signed_vol = volumes * signs
    dp = np.diff(prices)
    # align: signed_vol[i] corresponds to the trade that caused dp[i]
    sv = signed_vol[1:]  # drop first (no prior price change)
    mask = dp != 0
    if mask.sum() < 10:
        return KyleLambdaResult(0.0, 0.0, np.inf, 1.0)
    slope, _, r, p, se = linregress(sv[mask], dp[mask])
    return KyleLambdaResult(lam=slope, r_squared=r**2, std_error=se, p_value=p)


# ---------------------------------------------------------------------------
# 2. Hawkes Process — self-exciting point process for order flow
# ---------------------------------------------------------------------------

@dataclass
class HawkesResult:
    mu: float       # baseline intensity
    alpha: float    # excitation
    beta: float     # decay
    log_likelihood: float

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta if self.beta > 0 else 0.0

    @property
    def avg_intensity(self) -> float:
        br = self.branching_ratio
        return self.mu / (1 - br) if br < 1 else np.inf


def _hawkes_neg_ll(params, times, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e12
    n = len(times)
    if n < 2:
        return 1e12

    # Recursive O(n) computation
    R = np.zeros(n)
    for i in range(1, n):
        R[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1 + R[i - 1])

    integral = mu * T + (alpha / beta) * np.sum(
        1 - np.exp(-beta * (T - times))
    )
    log_terms = np.log(mu + alpha * R)
    return -(np.sum(log_terms) - integral)


def fit_hawkes(event_times: np.ndarray, T: float, n_starts: int = 8) -> HawkesResult:
    """MLE fit of a univariate Hawkes process."""
    best = None
    for _ in range(n_starts):
        x0 = [
            np.random.uniform(0.1, 3.0),
            np.random.uniform(0.1, 1.0),
            np.random.uniform(1.0, 6.0),
        ]
        res = minimize(
            _hawkes_neg_ll, x0, args=(event_times, T),
            method="Nelder-Mead",
            options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
        )
        if best is None or res.fun < best.fun:
            best = res
    mu, alpha, beta = best.x
    return HawkesResult(mu=mu, alpha=alpha, beta=beta, log_likelihood=-best.fun)


def simulate_hawkes(mu: float, alpha: float, beta: float, T: float) -> np.ndarray:
    """Ogata's thinning algorithm for Hawkes simulation."""
    times = []
    t = 0.0
    while t < T:
        lam_bar = mu + alpha * sum(
            np.exp(-beta * (t - ti)) for ti in times
        ) if times else mu
        t += np.random.exponential(1 / max(lam_bar, mu))
        if t > T:
            break
        lam_t = mu + alpha * sum(
            np.exp(-beta * (t - ti)) for ti in times if ti < t
        )
        if np.random.uniform() < lam_t / max(lam_bar, 1e-12):
            times.append(t)
    return np.array(times)


# ---------------------------------------------------------------------------
# 3. VPIN — Volume-synchronized Probability of Informed Trading
# ---------------------------------------------------------------------------

def compute_vpin(
    buy_volumes: np.ndarray,
    sell_volumes: np.ndarray,
    bucket_size: int = 30,
) -> np.ndarray:
    """Rolling VPIN over non-overlapping buckets of trades."""
    n = len(buy_volumes) // bucket_size
    vpins = np.empty(n)
    for i in range(n):
        s = i * bucket_size
        vb = buy_volumes[s : s + bucket_size].sum()
        vs = sell_volumes[s : s + bucket_size].sum()
        total = vb + vs
        vpins[i] = abs(vb - vs) / total if total > 0 else 0.0
    return vpins


def rolling_vpin(
    buy_volumes: np.ndarray,
    sell_volumes: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Tick-by-tick rolling VPIN (overlapping window)."""
    n = len(buy_volumes)
    vpins = np.full(n, np.nan)
    for i in range(window, n):
        vb = buy_volumes[i - window : i].sum()
        vs = sell_volumes[i - window : i].sum()
        total = vb + vs
        vpins[i] = abs(vb - vs) / total if total > 0 else 0.0
    return vpins


# ---------------------------------------------------------------------------
# 4. Digital Option Fair Value — BTC-derived probability
# ---------------------------------------------------------------------------

def digital_option_fair_value(
    spot: float,
    strike: float,
    sigma: float,
    tau: float,        # time to expiry in years (5 min ≈ 9.51e-6)
    direction: str = "above",
) -> float:
    """
    P(BTC > strike at expiry) via Black-Scholes digital call pricing.
    direction='above' → call digital, 'below' → put digital.
    """
    if tau <= 0:
        if direction == "above":
            return 1.0 if spot > strike else 0.0
        return 1.0 if spot < strike else 0.0

    d2 = (np.log(spot / strike) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    # For a digital call (above), fair value = Φ(d2)
    # For a digital put (below), fair value = 1 - Φ(d2)
    # Note: using d2 not d1 for the digital (cash-or-nothing) payoff
    # (risk-neutral, ignoring drift for short horizon)
    prob = norm.cdf(d2)
    return prob if direction == "above" else 1 - prob


def fair_value_with_drift(
    spot: float,
    strike: float,
    sigma: float,
    tau: float,
    drift: float = 0.0,
    direction: str = "above",
) -> float:
    """
    Fair value including drift term (e.g., from funding rate or momentum signal).
    """
    if tau <= 0:
        if direction == "above":
            return 1.0 if spot > strike else 0.0
        return 1.0 if spot < strike else 0.0

    d2 = (np.log(spot / strike) + (drift + 0.5 * sigma**2) * tau) / (
        sigma * np.sqrt(tau)
    )
    prob = norm.cdf(d2)
    return prob if direction == "above" else 1 - prob


# ---------------------------------------------------------------------------
# 5. Realized Volatility — fast rolling estimators
# ---------------------------------------------------------------------------

def realized_vol_1min(prices: np.ndarray, dt_seconds: float = 1.0) -> float:
    """
    1-minute realized volatility from tick prices.
    Returns annualized σ.
    """
    log_returns = np.diff(np.log(prices))
    if len(log_returns) < 2:
        return 0.0
    var_per_tick = np.var(log_returns, ddof=1)
    ticks_per_year = 365.25 * 24 * 3600 / dt_seconds
    return np.sqrt(var_per_tick * ticks_per_year)


def realized_vol_window(
    prices: np.ndarray, window: int = 60, dt_seconds: float = 1.0
) -> np.ndarray:
    """Rolling realized vol over a window of ticks."""
    n = len(prices)
    vols = np.full(n, np.nan)
    lr = np.diff(np.log(prices))
    ticks_per_year = 365.25 * 24 * 3600 / dt_seconds
    for i in range(window, len(lr)):
        chunk = lr[i - window : i]
        vols[i + 1] = np.sqrt(np.var(chunk, ddof=1) * ticks_per_year)
    return vols
