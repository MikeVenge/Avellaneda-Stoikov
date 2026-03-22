"""
Parameter sweep to find optimal strategy configuration.
"""

import numpy as np
from itertools import product
from simulation import run_single_contract
import time


def sweep_params():
    param_grid = {
        "gamma": [0.5, 1.0, 2.0, 4.0, 8.0],
        "kappa": [5.0, 10.0, 20.0],
        "max_inventory": [100, 300],
        "fill_prob": [0.10, 0.20],
        "div_threshold": [1.0, 2.0],
    }

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    n_contracts = 60
    print(f"Sweeping {len(combos)} configs × {n_contracts} contracts")
    print("=" * 80)

    # Pre-generate seeds and market conditions
    seeds = []
    conditions = []
    for i in range(n_contracts):
        np.random.seed(i * 7 + 42)
        s0 = 85000 * (1 + np.random.normal(0, 0.001))
        strike = s0 * (1 + np.random.normal(0, 0.0005))
        direction = np.random.choice(["above", "below"])
        seeds.append(i * 7 + 42)
        conditions.append((s0, strike, direction))

    results = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        pnls = []
        fills_total = 0

        for i in range(n_contracts):
            s0, strike, direction = conditions[i]
            r = run_single_contract(
                btc_s0=s0, strike=strike, direction=direction,
                seed=seeds[i],
                gamma=params["gamma"],
                kappa=params["kappa"],
                max_inventory=params["max_inventory"],
                fill_prob=params["fill_prob"],
                div_threshold=params["div_threshold"],
            )
            pnls.append(r.pnl)
            fills_total += r.n_fills

        pnl_arr = np.array(pnls)
        mean_pnl = pnl_arr.mean()
        std_pnl = pnl_arr.std()
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
        win_rate = (pnl_arr > 0).mean()
        avg_fills = fills_total / n_contracts

        results.append({
            **params,
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
            "sharpe": sharpe,
            "win_rate": win_rate,
            "avg_fills": avg_fills,
        })

        if (idx + 1) % 10 == 0:
            print(
                f"  [{idx+1}/{len(combos)}] "
                f"γ={params['gamma']:.1f} κ={params['kappa']:.0f} "
                f"inv={params['max_inventory']:.0f} "
                f"fp={params['fill_prob']:.2f} "
                f"div={params['div_threshold']:.1f} → "
                f"mean=${mean_pnl:.2f} sharpe={sharpe:.3f} win={win_rate*100:.0f}%"
            )

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print()
    print("TOP 15 CONFIGURATIONS BY SHARPE:")
    print("=" * 90)
    fmt = "{:>5} {:>5} {:>5} {:>5} {:>5} | {:>8} {:>8} {:>6} {:>6}"
    print(fmt.format("γ", "κ", "inv", "fp", "div", "Mean$", "Sharpe", "Win%", "Fills"))
    print("-" * 72)
    for r in results[:15]:
        print(fmt.format(
            f"{r['gamma']:.1f}", f"{r['kappa']:.0f}", f"{r['max_inventory']:.0f}",
            f"{r['fill_prob']:.2f}", f"{r['div_threshold']:.1f}",
            f"${r['mean_pnl']:.2f}", f"{r['sharpe']:.3f}",
            f"{r['win_rate']*100:.0f}%", f"{r['avg_fills']:.0f}",
        ))

    print()
    print("WORST 5:")
    print("-" * 72)
    for r in results[-5:]:
        print(fmt.format(
            f"{r['gamma']:.1f}", f"{r['kappa']:.0f}", f"{r['max_inventory']:.0f}",
            f"{r['fill_prob']:.2f}", f"{r['div_threshold']:.1f}",
            f"${r['mean_pnl']:.2f}", f"{r['sharpe']:.3f}",
            f"{r['win_rate']*100:.0f}%", f"{r['avg_fills']:.0f}",
        ))

    # Best config
    best = results[0]
    print()
    print("BEST CONFIGURATION:")
    for k, v in best.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    t0 = time.time()
    sweep_params()
    print(f"\nCompleted in {time.time()-t0:.1f}s")
