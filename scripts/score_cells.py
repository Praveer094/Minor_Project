#!/usr/bin/env python3
"""
score_cells.py  -  turn the joined features into  risk (0-10)  and  repair cost (Rs)  per cell.

Reads every features_*.csv written by build_cells.py (one per road), scores all roads
together, and writes the file optimizer.py wants.

Usage
-----
  python score_cells.py --features out/features_*.csv --out out
  python score_cells.py --features out/features_road1.csv out/features_road2.csv \
        --importance "Road 1=1.5" --importance "Road 2=1.0" --out out

Outputs
-------
  out/cells_all.csv         every cell with risk, cost, reason  (for the map / dashboard)
  out/cells_repairable.csv  cell_id, road, risk, cost_rs, importance, lat, lon   <- input of optimizer.py
                            (cells with risk below --min-risk are left out: nothing to repair)

!! PLACEHOLDERS - replace before showing real results !!
  * RISK weights below are a hand-made starting formula. Replace with the model trained on your
    labelled cells (XGBoost) once the two-person labels exist, or tune against them.
  * AREA_M2 is a typical patch size per damage type, because the camera has no scale.
  * RATE_PER_M2 / FIXED_COST_RS are placeholders. Use the real PWD/CPWD Schedule of Rates.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- risk formula (0-10) -------------------------------------------------- #
W_SENSOR = 0.5                     # share of the risk that comes from the sensors
W_VISION = 0.5                     # share that comes from the camera
VISION_POINTS = {"n_pothole": 3.0, "n_alligator": 1.5, "n_longitudinal": 0.7,
                 "n_transverse": 0.7, "n_other": 0.3}
VISION_FULL = 4.0                  # this many points in one 50 m cell = maximum camera score
DEPTH_FULL_CM = 8.0                # a hole this deep = maximum depth score
BUMPS_FULL = 3                     # this many bumps in a cell = maximum bump score

# ---- cost model (PLACEHOLDERS) -------------------------------------------- #
AREA_M2 = {"n_pothole": 0.30, "n_alligator": 1.00, "n_longitudinal": 0.30,
           "n_transverse": 0.30, "n_other": 0.20}
RATE_PER_M2 = 900                  # Rs per m^2 of patch repair (placeholder)
FIXED_COST_RS = 5000               # crew + mobilisation per repaired cell (placeholder)
MIN_AREA_M2 = 1.0                  # even a small defect needs a minimum patch


def main():
    ap = argparse.ArgumentParser(description="Score cells: risk and repair cost.")
    ap.add_argument("--features", nargs="+", required=True, help="features CSV(s) (globs allowed)")
    ap.add_argument("--importance", action="append", default=[], metavar='"ROAD=VALUE"',
                    help='traffic weight per road, e.g. --importance "Road 1=1.5" (default 1)')
    ap.add_argument("--min-risk", type=float, default=2.0,
                    help="cells below this risk are not offered to the optimizer (default 2)")
    ap.add_argument("--out", default="out", help="output folder")
    args = ap.parse_args()

    files = sorted({f for pat in args.features for f in glob.glob(pat)})
    if not files:
        sys.exit("No features files found.")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Scoring {len(df)} cells from {len(files)} road file(s): {sorted(df['road'].unique())}")

    # ---- sensor score (0-1). roughness is rank-scaled across ALL roads so they are comparable
    p95 = np.nanpercentile(df["roughness_idx"], 95)
    rn = np.clip(df["roughness_idx"] / p95, 0, 1) if p95 > 0 else pd.Series(0.0, index=df.index)
    depth = np.clip(df["max_depth_cm"].fillna(0) / DEPTH_FULL_CM, 0, 1)
    bumps = np.clip(df["n_bumps"] / BUMPS_FULL, 0, 1)
    df["sensor_score"] = (0.5 * rn + 0.3 * depth + 0.2 * bumps).round(3)

    # ---- camera score (0-1)
    pts = sum(df[c] * w for c, w in VISION_POINTS.items())
    df["vision_score"] = np.clip(pts / VISION_FULL, 0, 1).round(3)

    df["risk"] = (10 * (W_SENSOR * df["sensor_score"] + W_VISION * df["vision_score"])).round(2)

    # ---- cost
    area = sum(df[c] * a for c, a in AREA_M2.items())
    area = np.where(area > 0, np.maximum(area, MIN_AREA_M2), 0.0)
    # a bump/roughness with no camera damage still needs some patching in proportion to risk
    area = np.where((area == 0) & (df["risk"] >= args.min_risk), MIN_AREA_M2, area)
    df["est_area_m2"] = np.round(area, 2)
    df["cost_rs"] = (FIXED_COST_RS + df["est_area_m2"] * RATE_PER_M2).round(-2).astype(int)

    # ---- importance
    imp = {k.strip(): float(v) for k, _, v in (s.partition("=") for s in args.importance)}
    df["importance"] = df["road"].map(imp).fillna(1.0)

    # ---- short human-readable reason (simple stand-in for SHAP)
    def reason(r):
        bits = []
        if r["n_pothole"]:
            bits.append(f"{int(r['n_pothole'])} pothole(s) seen")
        if r["n_alligator"]:
            bits.append(f"{int(r['n_alligator'])} alligator crack(s)")
        if r["n_longitudinal"] + r["n_transverse"]:
            bits.append(f"{int(r['n_longitudinal'] + r['n_transverse'])} crack(s)")
        if pd.notna(r["max_depth_cm"]) and r["max_depth_cm"] >= 3:
            bits.append(f"hole {r['max_depth_cm']:.0f} cm deep")
        if r["n_bumps"]:
            bits.append(f"{int(r['n_bumps'])} bump(s)")
        if r["roughness_idx"] >= p95:
            bits.append("very rough ride")
        elif r["sensor_score"] >= 0.4:
            bits.append("rough ride")
        if r["confirmed"]:
            bits.append("camera + sensor agree")
        return ", ".join(bits) if bits else "no damage found"
    df["reason"] = df.apply(reason, axis=1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cells_all.csv", index=False)

    rep = df[df["risk"] >= args.min_risk].copy()
    rep = rep.rename(columns={"cell_uid": "cell_id_global"})
    rep = rep[["cell_id_global", "road", "risk", "cost_rs", "importance", "lat", "lon", "reason"]]
    rep = rep.rename(columns={"cell_id_global": "cell_id"})
    rep.to_csv(out / "cells_repairable.csv", index=False)

    print(f"risk: min {df['risk'].min():.1f}  mean {df['risk'].mean():.1f}  max {df['risk'].max():.1f}")
    print(f"{len(rep)} of {len(df)} cells have risk >= {args.min_risk}; "
          f"cost to fix them all: Rs {rep['cost_rs'].sum() / 100000:.1f} lakh")
    print(f"Saved: {out / 'cells_all.csv'}   {out / 'cells_repairable.csv'}")


if __name__ == "__main__":
    main()
