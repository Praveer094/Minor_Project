#!/usr/bin/env python3
"""
optimizer.py  -  choose which 50 m cells to repair within a budget.

Problem (a 0/1 knapsack):
    for every cell i:  x_i = 1 if we repair it, 0 if not
    maximise   sum( benefit_i * x_i )         benefit = risk x importance
    subject to sum( cost_i    * x_i ) <= budget

Methods compared on the SAME budget
    1. Optimizer            exact solution (Google OR-Tools CP-SAT)
    2. Worst-first greedy   repair the highest-risk cells first (what people do by default)
    3. Value-per-rupee greedy   best benefit/cost first (a stronger baseline)
    4. Random               average of many random orders

Usage
-----
  python optimizer.py --demo                       # made-up data, 4 roads x 30 cells
  python optimizer.py --demo --budget 4            # budget in lakh rupees
  python optimizer.py --demo --sweep               # also compare budgets 2..20 lakh
  python optimizer.py --cells my_cells.csv --budget 10 --sweep

Cells file (CSV) needs columns:   cell_id, risk, cost_rs
Optional columns:                 road, importance, lat, lon
    risk        0-10 condition score from the model
    cost_rs     repair cost in rupees (damaged area m^2 x PWD/CPWD Schedule of Rates)
    importance  weight for busy roads (default 1)

Needs:  pip install ortools pandas numpy matplotlib
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from ortools.sat.python import cp_model
except ImportError:
    sys.exit("OR-Tools is missing. Run:  pip install ortools")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAKH = 100_000


# --------------------------------------------------------------------------- #
def make_demo(seed=42, cells_per_road=30):
    """Made-up cells. The repair rate is a PLACEHOLDER: replace with the real
    PWD/CPWD Schedule of Rates figure before showing real results."""
    rng = np.random.default_rng(seed)
    roads = [("Road 1", 1.5), ("Road 2", 1.0), ("Road 3", 1.2), ("Road 4", 0.8)]
    rate_per_m2 = 900          # placeholder rupees per m^2
    fixed = 5000               # placeholder crew/mobilisation cost per cell
    rows, cid = [], 0
    for name, importance in roads:
        level = rng.uniform(2, 7)
        for k in range(cells_per_road):
            level = float(np.clip(level + rng.normal(0, 1.1), 0.3, 9.8))  # damage clusters along a road
            area = max(1.0, rng.gamma(2.0, 1.0 + level * 1.6))            # worse cell -> bigger patch
            rows.append(dict(cell_id=cid, road=name, risk=round(level, 2),
                             cost_rs=int(round(fixed + area * rate_per_m2, -2)),
                             importance=importance))
            cid += 1
    return pd.DataFrame(rows)


def load_cells(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "cost" in df.columns and "cost_rs" not in df.columns:
        df = df.rename(columns={"cost": "cost_rs"})
    for need in ("cell_id", "risk", "cost_rs"):
        if need not in df.columns:
            sys.exit(f"Cells file needs a '{need}' column. Found: {list(df.columns)}")
    if "importance" not in df.columns:
        df["importance"] = 1.0
    df = df.dropna(subset=["risk", "cost_rs"])
    if (df["cost_rs"] <= 0).any():
        sys.exit("All costs must be above zero.")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
def solve_optimal(benefit, cost, budget, time_limit=20.0):
    n = len(benefit)
    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"x{i}") for i in range(n)]
    model.Add(sum(int(cost[i]) * x[i] for i in range(n)) <= int(budget))
    model.Maximize(sum(int(round(benefit[i] * 1000)) * x[i] for i in range(n)))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return np.zeros(n, dtype=bool), "no solution"
    chosen = np.array([solver.Value(v) == 1 for v in x])
    return chosen, ("optimal" if status == cp_model.OPTIMAL else "feasible (time limit)")


def greedy(order, cost, budget):
    chosen = np.zeros(len(cost), dtype=bool)
    spent = 0
    for i in order:
        if spent + cost[i] <= budget:
            chosen[i] = True
            spent += cost[i]
    return chosen


def random_baseline(benefit, cost, budget, trials=500, seed=0):
    rng = np.random.default_rng(seed)
    vals, costs, counts = [], [], []
    for _ in range(trials):
        c = greedy(rng.permutation(len(cost)), cost, budget)
        vals.append(benefit[c].sum())
        costs.append(cost[c].sum())
        counts.append(c.sum())
    return float(np.mean(vals)), float(np.mean(costs)), float(np.mean(counts))


def run_all(df, budget):
    benefit = (df["risk"] * df["importance"]).to_numpy(float)
    cost = df["cost_rs"].to_numpy(float)
    risk = df["risk"].to_numpy(float)

    opt, status = solve_optimal(benefit, cost, budget)
    worst = greedy(np.argsort(-risk), cost, budget)
    ratio = greedy(np.argsort(-(benefit / cost)), cost, budget)
    r_val, r_cost, r_n = random_baseline(benefit, cost, budget)

    rows = [
        ("Optimizer (OR-Tools)", benefit[opt].sum(), cost[opt].sum(), int(opt.sum())),
        ("Worst-first greedy", benefit[worst].sum(), cost[worst].sum(), int(worst.sum())),
        ("Value-per-rupee greedy", benefit[ratio].sum(), cost[ratio].sum(), int(ratio.sum())),
        ("Random (avg of 500)", r_val, r_cost, r_n),
    ]
    best = rows[0][1] if rows[0][1] > 0 else 1.0
    table = pd.DataFrame([{"method": m, "benefit_removed": round(b, 2),
                           "pct_of_optimizer": round(100 * b / best, 1),
                           "cost_rs": int(round(c)), "cells_repaired": round(n, 1)}
                          for m, b, c, n in rows])
    return table, opt, status


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Budget-constrained repair optimizer.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cells", help="cells CSV (cell_id, risk, cost_rs, ...)")
    src.add_argument("--demo", action="store_true", help="use made-up sample cells")
    ap.add_argument("--budget", type=float, default=10.0, help="budget in LAKH rupees (default 10)")
    ap.add_argument("--sweep", action="store_true", help="also compare budgets from 2 to 20 lakh")
    ap.add_argument("--out", default="optimizer_out", help="output folder")
    args = ap.parse_args()

    df = make_demo() if args.demo else load_cells(args.cells)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = args.budget * LAKH

    total_cost = df["cost_rs"].sum()
    print(f"\nCells: {len(df)}   |   cost to repair ALL: Rs {total_cost / LAKH:.1f} lakh   "
          f"|   budget: Rs {args.budget:.1f} lakh")
    if args.demo:
        print("(demo data: costs use a placeholder rate, not the real Schedule of Rates)")

    table, chosen, status = run_all(df, budget)
    print(f"\nResult at Rs {args.budget:.1f} lakh  (solver status: {status})\n")
    print(table.to_string(index=False))

    gain = table.loc[0, "benefit_removed"] - table.loc[1, "benefit_removed"]
    print(f"\nOptimizer removes {gain:.1f} more benefit than worst-first greedy "
          f"({gain / max(table.loc[1, 'benefit_removed'], 1e-9) * 100:.1f}% more).")

    sel = df.copy()
    sel["repair"] = chosen
    sel["benefit"] = (sel["risk"] * sel["importance"]).round(2)
    sel.to_csv(out / "selected_cells.csv", index=False)
    table.to_csv(out / "comparison.csv", index=False)

    # bar chart at the chosen budget
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = ["#0b7285", "#e8590c", "#f08c00", "#868e96"]
    ax.bar(table["method"], table["benefit_removed"], color=colors)
    for i, v in enumerate(table["benefit_removed"]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="risk removed (benefit)", title=f"Same budget, Rs {args.budget:.1f} lakh")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    plt.close(fig)

    if args.sweep:
        budgets = np.arange(2, 20.5, 2)
        res = {"Optimizer": [], "Worst-first greedy": [], "Value-per-rupee greedy": [], "Random": []}
        for b in budgets:
            t, _, _ = run_all(df, b * LAKH)
            for k, v in zip(res, t["benefit_removed"]):
                res[k].append(v)
        sweep = pd.DataFrame(res, index=budgets)
        sweep.index.name = "budget_lakh"
        sweep.round(2).to_csv(out / "budget_sweep.csv")
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for (k, v), c in zip(res.items(), colors):
            ax.plot(budgets, v, marker="o", label=k, color=c)
        ax.set(xlabel="budget (lakh rupees)", ylabel="risk removed", title="Benefit vs budget")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "budget_sweep.png", dpi=140)
        plt.close(fig)
        print(f"Sweep saved: {out / 'budget_sweep.png'}")

    print(f"\nSaved in {out}/ :  selected_cells.csv  comparison.csv  comparison.png")


if __name__ == "__main__":
    main()
