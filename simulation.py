"""
Full simulation harness for the 5-minute BTC prediction market strategy.

Generates synthetic:
  - BTC spot price path (GBM with jumps + mean-reverting vol)
  - Polymarket order book (lagged fair value + noise + informed traders)
  - Trade arrivals via Hawkes process

Runs the full strategy stack:
  models → regime classifier → quoting engine / execution → settlement
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from models import (
    estimate_kyle_lambda,
    fit_hawkes,
    simulate_hawkes,
    rolling_vpin,
    digital_option_fair_value,
    realized_vol_window,
)
from quoting import AvellanedaStoikovEngine
from execution import quick_execution_plan
from regime import RegimeClassifier, RegimeSignals, Mode


# ---------------------------------------------------------------------------
# Synthetic market data generation
# ---------------------------------------------------------------------------

@dataclass
class BTCPath:
    """Simulated BTC spot price at 1-second resolution."""
    times: np.ndarray       # seconds from contract start
    prices: np.ndarray      # BTC/USD
    returns: np.ndarray     # log returns


def generate_btc_path(
    s0: float = 85000.0,
    T_seconds: int = 300,
    annual_vol: float = 0.60,
    jump_intensity: float = 0.01,   # jumps per second
    jump_size_std: float = 0.002,
    seed: Optional[int] = None,
) -> BTCPath:
    """
    Generate a realistic BTC price path with stochastic vol and jumps.
    GBM + Poisson jumps.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1.0 / (365.25 * 24 * 3600)  # 1 second in years
    sigma = annual_vol
    n = T_seconds

    prices = np.empty(n + 1)
    prices[0] = s0

    for i in range(1, n + 1):
        # Diffusion
        dW = np.random.normal(0, 1) * np.sqrt(dt)
        diffusion = -0.5 * sigma**2 * dt + sigma * dW

        # Jump
        jump = 0.0
        if np.random.random() < jump_intensity:
            jump = np.random.normal(0, jump_size_std)

        prices[i] = prices[i - 1] * np.exp(diffusion + jump)

    times = np.arange(n + 1, dtype=float)
    returns = np.diff(np.log(prices))

    return BTCPath(times=times, prices=prices, returns=returns)


@dataclass
class PolymarketBook:
    """Simulated Polymarket order book state at each second."""
    times: np.ndarray
    mids: np.ndarray        # Polymarket mid price (lagged fair value + noise)
    spreads: np.ndarray     # Polymarket spread
    fair_values: np.ndarray # true fair value from BTC spot


def generate_polymarket_book(
    btc_path: BTCPath,
    strike: float,
    direction: str = "above",
    lag_seconds: int = 2,
    noise_std: float = 0.008,
    base_spread: float = 0.03,
    seed: Optional[int] = None,
) -> PolymarketBook:
    """
    Generate Polymarket book that lags the BTC-derived fair value.

    The lag is the key edge source — our strategy reads BTC spot in real time
    while Polymarket participants react 1-3 seconds later.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(btc_path.prices)
    annual_vol = 0.60
    T_years = 300.0 / (365.25 * 24 * 3600)

    fair_values = np.empty(n)
    mids = np.empty(n)
    spreads = np.empty(n)

    for i in range(n):
        t_remaining_years = max(T_years * (1 - i / (n - 1)), 1e-10)
        fv = digital_option_fair_value(
            btc_path.prices[i], strike, annual_vol, t_remaining_years, direction
        )
        fair_values[i] = fv

        # Polymarket mid lags by lag_seconds + noise
        lag_idx = max(0, i - lag_seconds)
        t_remaining_lagged = max(T_years * (1 - lag_idx / (n - 1)), 1e-10)
        lagged_fv = digital_option_fair_value(
            btc_path.prices[lag_idx], strike, annual_vol, t_remaining_lagged, direction
        )
        mids[i] = np.clip(lagged_fv + np.random.normal(0, noise_std), 0.02, 0.98)

        # Spread narrows toward expiry
        time_frac = i / (n - 1)
        spreads[i] = base_spread * (1.0 - 0.6 * time_frac) + np.random.exponential(0.005)

    return PolymarketBook(
        times=btc_path.times,
        mids=mids,
        spreads=spreads,
        fair_values=fair_values,
    )


# ---------------------------------------------------------------------------
# Synthetic trade flow (Hawkes-driven)
# ---------------------------------------------------------------------------

@dataclass
class TradeFlow:
    """Simulated trade arrivals on Polymarket."""
    timestamps: np.ndarray   # seconds
    sizes: np.ndarray        # $ volume
    signs: np.ndarray        # +1 buy, -1 sell
    prices: np.ndarray       # execution prices


def generate_trade_flow(
    poly_book: PolymarketBook,
    hawkes_mu: float = 0.5,
    hawkes_alpha: float = 0.3,
    hawkes_beta: float = 1.0,
    informed_fraction: float = 0.15,
    seed: Optional[int] = None,
) -> TradeFlow:
    """
    Generate trade flow using a Hawkes process for arrival times,
    with a fraction of trades being informed (trading toward fair value).
    """
    if seed is not None:
        np.random.seed(seed)

    T = float(poly_book.times[-1])
    event_times = simulate_hawkes(hawkes_mu, hawkes_alpha, hawkes_beta, T)

    n_trades = len(event_times)
    sizes = np.random.exponential(50, n_trades)  # avg $50 per trade
    signs = np.zeros(n_trades)
    prices = np.zeros(n_trades)

    for i, t in enumerate(event_times):
        idx = min(int(t), len(poly_book.mids) - 1)
        mid = poly_book.mids[idx]
        fv = poly_book.fair_values[idx]
        spread = poly_book.spreads[idx]

        if np.random.random() < informed_fraction:
            # Informed trade: trade toward fair value
            signs[i] = 1.0 if fv > mid else -1.0
        else:
            # Noise trade: random direction with slight bias
            signs[i] = np.random.choice([-1.0, 1.0])

        # Execution price
        if signs[i] > 0:
            prices[i] = mid + spread / 2  # buy at ask
        else:
            prices[i] = mid - spread / 2  # sell at bid

    return TradeFlow(
        timestamps=event_times,
        sizes=sizes,
        signs=signs,
        prices=prices,
    )


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

@dataclass
class ContractResult:
    """Result of a single 5-minute contract."""
    pnl: float
    n_fills: int
    n_buys: int
    n_sells: int
    avg_spread: float
    max_inventory: float
    mode_history: List[str]
    final_inventory: float
    settlement: float      # 0 or 1
    btc_start: float
    btc_end: float
    strike: float


def run_single_contract(
    btc_s0: float = 85000.0,
    strike: Optional[float] = None,
    direction: str = "above",
    seed: Optional[int] = None,
    verbose: bool = False,
    # Tunable parameters
    gamma: float = 2.0,
    kappa: float = 8.0,
    sigma: float = 0.06,
    max_inventory: float = 500.0,
    fill_prob: float = 0.15,
    div_threshold: float = 1.5,
) -> ContractResult:
    """
    Run the full strategy on one 5-minute contract.

    Steps each second:
      1. Update BTC spot → compute fair value
      2. Compute signals (Kyle's λ, Hawkes BR, VPIN) from recent trades
      3. Classify regime
      4. If MM mode: quote via A-S engine, check for fills
      5. If Directional: enter/exit via Almgren-Chriss
      6. If Defensive: flatten
      7. At T=0: settle
    """
    rng = np.random.RandomState(seed)
    if seed is not None:
        np.random.seed(seed)

    # Strike defaults to ATM
    if strike is None:
        strike = btc_s0

    # Generate synthetic data
    btc = generate_btc_path(s0=btc_s0, seed=seed)
    poly = generate_polymarket_book(btc, strike, direction, seed=seed)
    trades = generate_trade_flow(poly, seed=seed)

    # Initialize components with tunable params
    engine = AvellanedaStoikovEngine(
        gamma=gamma, kappa=kappa, sigma=sigma, T_minutes=5.0,
        max_inventory=max_inventory,
    )
    classifier = RegimeClassifier(divergence_threshold=div_threshold)

    mode_history = []
    trade_cursor = 0  # index into trades
    n_trades_total = len(trades.timestamps)

    # Accumulators for signal estimation
    recent_prices = []
    recent_volumes = []
    recent_signs = []
    recent_buy_vols = []
    recent_sell_vols = []
    spreads_seen = []

    T_seconds = 300

    for t in range(T_seconds):
        t_remaining_min = (T_seconds - t) / 60.0
        prev_trade_cursor = trade_cursor

        # --- Collect trades that arrived this second ---
        while (
            trade_cursor < n_trades_total
            and trades.timestamps[trade_cursor] < t + 1
        ):
            p = trades.prices[trade_cursor]
            v = trades.sizes[trade_cursor]
            s = trades.signs[trade_cursor]
            recent_prices.append(p)
            recent_volumes.append(v)
            recent_signs.append(s)
            if s > 0:
                recent_buy_vols.append(v)
                recent_sell_vols.append(0.0)
            else:
                recent_buy_vols.append(0.0)
                recent_sell_vols.append(v)
            trade_cursor += 1

        # --- Compute signals (need at least some history) ---
        vpin_val = 0.0
        branching_ratio = 0.0
        kyle_lam = 0.0
        kyle_r2 = 0.0

        if len(recent_prices) > 30:
            # VPIN
            bv = np.array(recent_buy_vols[-30:])
            sv = np.array(recent_sell_vols[-30:])
            total = bv.sum() + sv.sum()
            if total > 0:
                vpin_val = abs(bv.sum() - sv.sum()) / total

        if len(recent_prices) > 50:
            # Kyle's lambda
            kl = estimate_kyle_lambda(
                np.array(recent_prices[-50:]),
                np.array(recent_volumes[-50:]),
                np.array(recent_signs[-50:]),
            )
            kyle_lam = kl.lam
            kyle_r2 = kl.r_squared

        # Hawkes branching ratio — approximate from trade clustering
        # (full MLE fit is too expensive per tick; use empirical estimate)
        if t % 30 == 0 and trade_cursor > 20:
            ts = trades.timestamps[:trade_cursor]
            recent_ts = ts[ts > max(0, t - 120)]
            if len(recent_ts) > 10:
                # Empirical clustering: fraction of inter-arrival times < median
                iat = np.diff(recent_ts)
                if len(iat) > 5:
                    med_iat = np.median(iat)
                    # Short inter-arrivals = clustering = high branching ratio
                    branching_ratio = np.clip(
                        (iat < med_iat * 0.5).mean() * 1.5, 0.0, 0.95
                    )

        fair_value = poly.fair_values[t]
        market_mid = poly.mids[t]
        spread = poly.spreads[t]
        spreads_seen.append(spread)

        # --- Classify regime ---
        signals = RegimeSignals(
            vpin=vpin_val,
            branching_ratio=branching_ratio,
            kyle_lambda=kyle_lam,
            kyle_r_squared=kyle_r2,
            fair_value=fair_value,
            market_mid=market_mid,
            spread=spread,
            t_remaining=t_remaining_min,
            btc_flow_confirms=False,  # simplified
        )
        decision = classifier.classify(signals)
        mode_history.append(decision.mode.value)

        if verbose and t % 30 == 0:
            print(
                f"  t={t:3d}s  mode={decision.mode.value:14s}  "
                f"FV={fair_value:.3f}  mid={market_mid:.3f}  "
                f"VPIN={vpin_val:.2f}  BR={branching_ratio:.2f}  "
                f"inv={engine.state.inventory:.0f}"
            )

        # --- Act based on regime ---

        # Count trades this second (for fill probability)
        trades_this_sec = trade_cursor - prev_trade_cursor
        filled_this_sec = False  # at most one fill per second per side

        if decision.mode == Mode.MARKET_MAKING:
            quote = engine.quote(
                fair_value=fair_value,
                t_remaining=t_remaining_min,
                vpin=vpin_val,
                branching_ratio=branching_ratio,
            )
            if quote is not None and trades_this_sec > 0:
                # Realistic fill model: we get filled only if a trade
                # crosses our level, and we only capture a small slice
                for i in range(prev_trade_cursor, trade_cursor):
                    if filled_this_sec:
                        break
                    tp = trades.prices[i]
                    ts_sign = trades.signs[i]

                    # Fill probability: decays with distance from mid
                    # (our quotes may be away from where flow lands)
                    fill_prob_adj = fill_prob * decision.size_scalar

                    if ts_sign > 0 and tp >= quote.ask and rng.random() < fill_prob_adj:
                        fill_size = min(rng.exponential(8), 25)
                        engine.fill_ask(quote.ask, fill_size, float(t))
                        filled_this_sec = True

                    elif ts_sign < 0 and tp <= quote.bid and rng.random() < fill_prob_adj:
                        fill_size = min(rng.exponential(8), 25)
                        engine.fill_bid(quote.bid, fill_size, float(t))
                        filled_this_sec = True

        elif decision.mode == Mode.DIRECTIONAL and decision.direction is not None:
            # Take position toward fair value — max once per 5 seconds
            if t % 5 == 0:
                target_size = 30 * decision.size_scalar
                current_inv = engine.state.inventory

                if decision.direction > 0 and current_inv < target_size:
                    buy_size = min(target_size - current_inv, 10)
                    buy_price = market_mid + spread * 0.4
                    engine.fill_bid(buy_price, buy_size, float(t))
                elif decision.direction < 0 and current_inv > -target_size:
                    sell_size = min(target_size + current_inv, 10)
                    sell_price = market_mid - spread * 0.4
                    engine.fill_ask(sell_price, sell_size, float(t))

        elif decision.mode == Mode.DEFENSIVE:
            # Flatten inventory gradually — max once per 3 seconds
            inv = engine.state.inventory
            if abs(inv) > 5 and t % 3 == 0:
                unwind = min(abs(inv) * 0.3, 15)
                if inv > 0:
                    engine.fill_ask(market_mid - spread * 0.3, unwind, float(t))
                else:
                    engine.fill_bid(market_mid + spread * 0.3, unwind, float(t))

    # --- Settlement ---
    btc_end = btc.prices[-1]
    if direction == "above":
        outcome = 1.0 if btc_end > strike else 0.0
    else:
        outcome = 1.0 if btc_end < strike else 0.0

    pnl = engine.settle(outcome)

    return ContractResult(
        pnl=pnl,
        n_fills=engine.state.n_buys + engine.state.n_sells,
        n_buys=engine.state.n_buys,
        n_sells=engine.state.n_sells,
        avg_spread=np.mean(spreads_seen) if spreads_seen else 0,
        max_inventory=max(
            abs(f.size) for f in engine.state.fills
        ) if engine.state.fills else 0,
        mode_history=mode_history,
        final_inventory=engine.state.inventory,
        settlement=outcome,
        btc_start=btc.prices[0],
        btc_end=btc_end,
        strike=strike,
    )


# ---------------------------------------------------------------------------
# Batch runner + analytics
# ---------------------------------------------------------------------------

def run_backtest(
    n_contracts: int = 200,
    btc_s0: float = 85000.0,
    verbose: bool = False,
) -> Dict:
    """
    Run many 5-minute contract simulations and aggregate results.
    """
    results = []
    for i in range(n_contracts):
        # Slight randomization of starting conditions
        s0 = btc_s0 * (1 + np.random.normal(0, 0.001))
        # Strike near ATM with small offset
        strike = s0 * (1 + np.random.normal(0, 0.0005))
        direction = np.random.choice(["above", "below"])

        r = run_single_contract(
            btc_s0=s0,
            strike=strike,
            direction=direction,
            seed=i * 7 + 42,
            verbose=verbose and i < 3,
        )
        results.append(r)

        if (i + 1) % 50 == 0:
            pnls_so_far = [x.pnl for x in results]
            print(
                f"  [{i+1}/{n_contracts}] "
                f"mean P&L=${np.mean(pnls_so_far):.2f}  "
                f"win%={np.mean([p > 0 for p in pnls_so_far])*100:.0f}%"
            )

    pnls = np.array([r.pnl for r in results])
    fills = np.array([r.n_fills for r in results])
    inventories = np.array([abs(r.final_inventory) for r in results])

    # Mode distribution
    all_modes = []
    for r in results:
        all_modes.extend(r.mode_history)
    mode_counts = {m: all_modes.count(m) for m in set(all_modes)}
    total_ticks = len(all_modes)
    mode_pcts = {m: c / total_ticks * 100 for m, c in mode_counts.items()}

    # Adverse selection: % of fills where P&L was negative
    losing_contracts = sum(1 for p in pnls if p < 0)

    sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0

    return {
        "n_contracts": n_contracts,
        "total_pnl": float(np.sum(pnls)),
        "mean_pnl": float(np.mean(pnls)),
        "median_pnl": float(np.median(pnls)),
        "std_pnl": float(np.std(pnls)),
        "sharpe": float(sharpe),
        "win_rate": float(np.mean(pnls > 0)),
        "max_drawdown": float(np.min(np.minimum.accumulate(np.cumsum(pnls)))),
        "avg_fills": float(np.mean(fills)),
        "avg_final_inventory": float(np.mean(inventories)),
        "mode_distribution": mode_pcts,
        "pnls": pnls,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("POLYMARKET 5-MIN BTC STRATEGY BACKTEST")
    print("=" * 70)
    print()

    # Run a single verbose contract first
    print("--- Single contract (verbose) ---")
    single = run_single_contract(
        btc_s0=85000.0,
        strike=85000.0,
        direction="above",
        seed=42,
        verbose=True,
    )
    print(f"\n  Result: P&L=${single.pnl:.2f}, fills={single.n_fills}, "
          f"settlement={'YES' if single.settlement == 1 else 'NO'}")
    print(f"  BTC: {single.btc_start:.0f} → {single.btc_end:.0f} "
          f"(strike={single.strike:.0f})")
    print()

    # Batch backtest
    print("--- Batch backtest (200 contracts) ---")
    stats = run_backtest(n_contracts=200, verbose=False)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Contracts:           {stats['n_contracts']}")
    print(f"  Total P&L:          ${stats['total_pnl']:>10.2f}")
    print(f"  Mean P&L:           ${stats['mean_pnl']:>10.2f}")
    print(f"  Median P&L:         ${stats['median_pnl']:>10.2f}")
    print(f"  Std P&L:            ${stats['std_pnl']:>10.2f}")
    print(f"  Sharpe ratio:        {stats['sharpe']:>10.3f}")
    print(f"  Win rate:            {stats['win_rate']*100:>9.1f}%")
    print(f"  Max drawdown:       ${stats['max_drawdown']:>10.2f}")
    print(f"  Avg fills/contract:  {stats['avg_fills']:>10.1f}")
    print(f"  Avg final inventory: ${stats['avg_final_inventory']:>9.1f}")
    print()
    print("  Mode distribution:")
    for mode, pct in sorted(stats["mode_distribution"].items()):
        print(f"    {mode:20s} {pct:5.1f}%")

    # P&L distribution
    pnls = stats["pnls"]
    print()
    print("  P&L percentiles:")
    for q in [5, 25, 50, 75, 95]:
        print(f"    {q:3d}th: ${np.percentile(pnls, q):>10.2f}")

    # Cumulative P&L
    cum_pnl = np.cumsum(pnls)
    print()
    print(f"  Cumulative P&L range: ${cum_pnl.min():.2f} to ${cum_pnl.max():.2f}")
    print(f"  Final cumulative:     ${cum_pnl[-1]:.2f}")
