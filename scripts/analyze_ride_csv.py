#!/usr/bin/env python3
"""
analyze_ride_csv.py  -  check and analyze a ride_XXX.csv from the ESP32 SD card.

Usage
-----
  python analyze_ride_csv.py ride_001.csv
  python analyze_ride_csv.py ride_001.csv --cell 50 --bump-thr 3.0
  python analyze_ride_csv.py ride_001.csv --map ax=accel_x --map dist=us_cm   # if your headers differ

Needs:  pip install pandas numpy scipy matplotlib

What it does
------------
1. HEALTH CHECK   rows, duration, sample rate, GPS fix %, gravity ~9.8, taps found
2. SIGNALS        - dynamic vibration  = |acceleration| minus its 1-second running median
                    (removes gravity, works whatever way the sensor is mounted)
                  - roughness per cell = RMS of that vibration, divided by speed
                  - pothole depth      = ultrasonic distance minus the road baseline
3. 50 m CELLS     distance travelled = speed integrated over time (falls back to GPS positions)
4. OUTPUT         <csv name>_analysis/cells.csv, events.csv, ride_overview.png

Column names are matched automatically (ms, gps_valid, lat, lon, speed, ax..gz, dist).
If a name is not recognised the script lists your columns; fix it with --map name=yourcolumn.
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks
except ImportError:
    sys.exit("SciPy is missing. Run:  pip install scipy")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.80665

ALIASES = {
    "ms":        ["millis", "ms", "time_ms", "t_ms", "timestamp_ms", "timestamp"],
    "gps_valid": ["gps_valid", "gpsvalid", "valid", "fix"],
    "lat":       ["lat", "latitude"],
    "lon":       ["lon", "lng", "longitude"],
    "speed":     ["speed_kmph", "speed_kmh", "speed_kph", "speed", "kmph", "speed_mps"],
    "ax":        ["ax", "acc_x", "accel_x", "accx"],
    "ay":        ["ay", "acc_y", "accel_y", "accy"],
    "az":        ["az", "acc_z", "accel_z", "accz"],
    "dist":      ["dist_cm", "distance_cm", "dist", "distance", "range_cm", "us_cm"],
}
REQUIRED = ["ax", "ay", "az"]


# --------------------------------------------------------------------------- #
def clean_name(c):
    return str(c).lower().split("(")[0].strip().replace(" ", "_")


def map_columns(df, overrides):
    names = {clean_name(c): c for c in df.columns}
    found = {}
    for key, options in ALIASES.items():
        for o in options:
            if o in names:
                found[key] = names[o]
                break
    for item in overrides:
        k, _, v = item.partition("=")
        if v not in df.columns:
            sys.exit(f"--map {item}: column '{v}' not found. Your columns: {list(df.columns)}")
        found[k.strip()] = v
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        sys.exit(f"Could not find columns for {missing}.\nYour columns are: {list(df.columns)}\n"
                 f"Use --map, for example:  --map ax=<your column>")
    return found


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def mad(x):
    x = np.asarray(x, dtype=float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def find_tap_groups(t, dyn, rate):
    """Return a list of (time_of_first_tap, [3 tap times]) for 3 sharp spikes close together."""
    a = np.abs(dyn)
    thr = max(np.median(a) + 8 * mad(a), 2.5)
    peaks, _ = find_peaks(a, height=thr, distance=max(1, int(0.12 * rate)))
    pt = t[peaks]
    groups, i = [], 0
    while i + 2 < len(pt):
        trio = pt[i:i + 3]
        if trio[2] - trio[0] <= 3.0 and np.all(np.diff(trio) <= 1.5):
            groups.append(list(np.round(trio, 2)))
            i += 3
        else:
            i += 1
    return groups, thr


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Analyze a ride CSV from the ESP32 rig.")
    ap.add_argument("csv", help="ride_XXX.csv")
    ap.add_argument("--cell", type=float, default=50.0, help="cell length in metres (default 50)")
    ap.add_argument("--rate", type=float, default=100.0,
                    help="assumed sample rate if there is no time column (default 100)")
    ap.add_argument("--speed-unit", choices=["kmph", "mps"], default=None,
                    help="speed column unit (default: kmph, or mps if the header says so)")
    ap.add_argument("--bump-thr", type=float, default=3.0,
                    help="vibration spike counted as a bump above this, m/s^2 (default 3)")
    ap.add_argument("--depth-thr", type=float, default=3.0,
                    help="ultrasonic dip counted as a hole above this, cm (default 3)")
    ap.add_argument("--map", action="append", default=[], metavar="NAME=COLUMN",
                    help="manual column mapping, e.g. --map ax=accel_x")
    ap.add_argument("--out", default=None, help="output folder")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    if len(df) < 50:
        sys.exit(f"Only {len(df)} rows in the file: too short to analyze.")
    col = map_columns(df, args.map)

    out_dir = Path(args.out) if args.out else path.with_name(path.stem + "_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    num = lambda key: pd.to_numeric(df[col[key]], errors="coerce") if key in col else None

    # ---- time axis ------------------------------------------------------- #
    ms = num("ms")
    if ms is not None and ms.notna().all() and ms.is_monotonic_increasing and ms.iloc[-1] > ms.iloc[0]:
        t = ((ms - ms.iloc[0]) / 1000.0).to_numpy()
        time_note = f"from '{col['ms']}' column"
    else:
        t = np.arange(len(df)) / args.rate
        time_note = f"no usable time column, assumed {args.rate:.0f} Hz"
    duration = float(t[-1] - t[0])
    rate = len(df) / duration if duration > 0 else args.rate

    # ---- acceleration ---------------------------------------------------- #
    ax_, ay_, az_ = (num(k).to_numpy(float) for k in ("ax", "ay", "az"))
    mag = np.sqrt(ax_ ** 2 + ay_ ** 2 + az_ ** 2)
    grav = float(np.nanmedian(mag))
    unit_note = ""
    if 0.5 < grav < 1.5:           # values are in g, not m/s^2
        mag *= G
        grav *= G
        unit_note = " (values looked like g, converted to m/s^2)"
    win = max(3, int(rate * 1.0)) | 1
    baseline_mag = pd.Series(mag).rolling(win, center=True, min_periods=1).median().to_numpy()
    dyn = mag - baseline_mag

    # ---- GPS ------------------------------------------------------------- #
    lat = num("lat").to_numpy(float) if "lat" in col else np.full(len(df), np.nan)
    lon = num("lon").to_numpy(float) if "lon" in col else np.full(len(df), np.nan)
    valid = np.isfinite(lat) & np.isfinite(lon) & (lat != 0) & (lon != 0)
    if "gps_valid" in col:
        valid &= (num("gps_valid").fillna(0).to_numpy() > 0)
    gps_pct = 100.0 * valid.mean()

    # ---- speed and distance travelled ------------------------------------ #
    speed_ms = np.zeros(len(df))
    have_speed = "speed" in col
    if have_speed:
        unit = args.speed_unit or ("mps" if "mps" in clean_name(col["speed"]) else "kmph")
        sp = num("speed").to_numpy(float)
        sp = np.where(valid & np.isfinite(sp), sp, np.nan)
        sp = pd.Series(sp).ffill().fillna(0).to_numpy()
        speed_ms = sp / 3.6 if unit == "kmph" else sp
    dt = np.diff(t, prepend=t[0])
    dist_src = "integrated GPS speed"
    cum = np.cumsum(speed_ms * dt)
    if (not have_speed) or cum[-1] < 1.0:
        # fallback: sum of straight-line steps between GPS fixes
        dist_src = "GPS positions (haversine)"
        la = pd.Series(np.where(valid, lat, np.nan)).ffill().bfill().to_numpy()
        lo = pd.Series(np.where(valid, lon, np.nan)).ffill().bfill().to_numpy()
        if np.isfinite(la).all() and valid.any():
            step = haversine_m(la[:-1], lo[:-1], la[1:], lo[1:])
            step = np.where(step < 0.5, 0.0, step)       # ignore standing-still jitter
            cum = np.concatenate([[0.0], np.cumsum(step)])
        else:
            dist_src = "none (no GPS): cells cannot be built"
            cum = np.zeros(len(df))

    # ---- taps ------------------------------------------------------------ #
    groups, tap_thr = find_tap_groups(t, dyn, rate)
    start_tap = next((g for g in groups if g[0] <= 90), None)
    end_tap = next((g for g in reversed(groups) if g[0] >= duration - 90), None)
    if end_tap is not None and start_tap is not None and end_tap == start_tap:
        end_tap = None
    # PIPELINE PATCH: save tap times so build_cells.py can sync the video to this ride
    import json
    (out_dir / "taps.json").write_text(json.dumps(
        {"start_tap": [float(x) for x in start_tap] if start_tap else None,
         "end_tap": [float(x) for x in end_tap] if end_tap else None,
         "duration_s": round(float(duration), 2)}))

    # ---- ultrasonic ------------------------------------------------------ #
    events = []
    dist10 = None
    if "dist" in col:
        d = num("dist").to_numpy(float)
        d = np.where((d > 2) & (d < 400), d, np.nan)
        b10 = np.floor(t * 10).astype(int)
        first = pd.DataFrame({"b": b10, "t": t, "d": d, "cum": cum}).groupby("b").first()
        first["base"] = first["d"].rolling(51, center=True, min_periods=5).median()
        first["depth"] = first["d"] - first["base"]
        dist10 = first
        mask = (first["depth"] >= args.depth_thr).to_numpy()
        run_id = np.cumsum(np.r_[True, mask[1:] != mask[:-1]])
        for rid in np.unique(run_id[mask]):
            seg = first[(run_id == rid) & mask]
            if len(seg) >= 2:                       # at least 0.2 s long
                events.append({"type": "depth", "t_s": float(seg["t"].iloc[0]),
                               "along_m": float(seg["cum"].mean()),
                               "value": float(seg["depth"].max()), "unit": "cm"})

    # ---- bumps from vibration ------------------------------------------- #
    tap_times = [x for g in ([start_tap] if start_tap else []) + ([end_tap] if end_tap else []) for x in g]
    bump_idx, _ = find_peaks(np.abs(dyn), height=args.bump_thr, distance=max(1, int(0.25 * rate)))
    for i in bump_idx:
        if any(abs(t[i] - tt) < 0.4 for tt in tap_times):
            continue                                 # a deliberate tap, not road damage
        events.append({"type": "bump", "t_s": float(t[i]), "along_m": float(cum[i]),
                       "value": float(abs(dyn[i])), "unit": "m/s2"})
    ev = pd.DataFrame(events, columns=["type", "t_s", "along_m", "value", "unit"])
    ev = ev.sort_values("t_s").round(3)
    ev.to_csv(out_dir / "events.csv", index=False)

    # ---- 50 m cells ------------------------------------------------------ #
    cell_id = np.floor(cum / args.cell).astype(int)
    # PIPELINE PATCH: the deliberate taps are not road roughness, so blank them before the cell RMS
    dyn_cells = dyn.copy()
    for tt in tap_times:
        dyn_cells[np.abs(t - tt) < 0.5] = 0.0
    work = pd.DataFrame({"cell_id": cell_id, "t": t, "dyn": dyn_cells, "speed_ms": speed_ms,
                         "lat": np.where(valid, lat, np.nan), "lon": np.where(valid, lon, np.nan),
                         "cum": cum})
    cells = work.groupby("cell_id").agg(
        t_start_s=("t", "min"), t_end_s=("t", "max"),
        length_m=("cum", lambda s: s.max() - s.min()),
        lat=("lat", "mean"), lon=("lon", "mean"),
        mean_speed_kmph=("speed_ms", lambda s: s.mean() * 3.6),
        rms_vibration=("dyn", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        max_vibration=("dyn", lambda s: float(np.max(np.abs(s)))),
    ).reset_index()
    cells["roughness_idx"] = cells["rms_vibration"] / np.maximum(cells["mean_speed_kmph"] / 3.6, 2.0)
    if len(ev):
        for typ, name_n, name_v, fn in (("bump", "n_bumps", None, None),
                                         ("depth", "n_holes", "max_depth_cm", "max")):
            e = ev[ev["type"] == typ].copy()
            e["cell_id"] = np.floor(e["along_m"] / args.cell).astype(int)
            cells[name_n] = cells["cell_id"].map(e.groupby("cell_id").size()).fillna(0).astype(int)
            if name_v:
                cells[name_v] = cells["cell_id"].map(e.groupby("cell_id")["value"].max())
    else:
        cells["n_bumps"], cells["n_holes"], cells["max_depth_cm"] = 0, 0, np.nan
    cells = cells.round(3)
    cells.to_csv(out_dir / "cells.csv", index=False)

    # ---- plot ------------------------------------------------------------ #
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=False)
    axs[0].plot(t, dyn, lw=0.5, color="#1b6ca8")
    for g in ([start_tap] if start_tap else []) + ([end_tap] if end_tap else []):
        for x in g:
            axs[0].axvline(x, color="#d9480f", lw=1, alpha=0.8)
    axs[0].set(title="Dynamic vibration (orange lines = detected taps)", ylabel="m/s²", xlabel="time (s)")
    axs[1].plot(t, speed_ms * 3.6, color="#2b8a3e")
    axs[1].set(title="Speed", ylabel="km/h", xlabel="time (s)")
    if dist10 is not None:
        axs[2].plot(dist10["t"], dist10["d"], lw=0.8, color="#6741d9", label="distance")
        axs[2].plot(dist10["t"], dist10["base"], lw=0.8, color="#868e96", label="road baseline")
        dd = ev[ev["type"] == "depth"]
        axs[2].scatter(dd["t_s"], np.interp(dd["t_s"], dist10["t"], dist10["d"]), color="#d9480f", zorder=3, label="hole")
        axs[2].legend(loc="upper right", fontsize=8)
    axs[2].set(title="Ultrasonic distance to road", ylabel="cm", xlabel="time (s)")
    axs[3].bar(cells["cell_id"], cells["roughness_idx"], color="#e8590c")
    axs[3].set(title=f"Roughness index per {args.cell:.0f} m cell (vibration RMS ÷ speed)",
               ylabel="index", xlabel="cell number")
    fig.tight_layout()
    fig.savefig(out_dir / "ride_overview.png", dpi=130)
    plt.close(fig)

    # ---- printed report -------------------------------------------------- #
    def line(ok, label, detail):
        print(f"  [{'PASS' if ok else 'CHECK'}] {label:<26} {detail}")

    print(f"\nFile: {path.name}")
    print(f"Columns recognised: " + ", ".join(f"{k}={v}" for k, v in col.items()))
    print("\nHEALTH CHECK")
    line(len(df) >= 1000, "Rows", f"{len(df):,} over {duration:.1f} s  ({time_note})")
    line(80 <= rate <= 120, "Sample rate", f"{rate:.1f} Hz (target about 100)")
    line(abs(grav - G) < 1.0, "Gravity", f"median |a| = {grav:.2f} m/s²{unit_note} (should be about 9.8)")
    line(gps_pct >= 80, "GPS valid rows", f"{gps_pct:.0f}%")
    if "dist" in col:
        ok_d = np.isfinite(pd.to_numeric(df[col['dist']], errors='coerce').where(lambda s: (s > 2) & (s < 400))).mean()
        line(ok_d > 0.8, "Ultrasonic readings", f"{100 * ok_d:.0f}% in valid range (2-400 cm)")
    else:
        line(False, "Ultrasonic readings", "no distance column found")
    line(start_tap is not None, "Start taps (3 spikes)", f"at {start_tap} s" if start_tap else "not found in first 90 s")
    line(end_tap is not None, "End taps (3 spikes)", f"at {end_tap} s" if end_tap else "not found in last 90 s")
    if start_tap and end_tap:
        print(f"        time between tap groups: {end_tap[0] - start_tap[0]:.1f} s")

    print("\nRIDE SUMMARY")
    print(f"  Distance travelled  : {cum[-1]:.0f} m  ({dist_src})")
    if have_speed:
        print(f"  Speed               : mean {speed_ms[speed_ms > 0.5].mean() * 3.6 if (speed_ms > 0.5).any() else 0:.1f} km/h, "
              f"max {speed_ms.max() * 3.6:.1f} km/h")
    print(f"  {args.cell:.0f} m cells         : {len(cells)}")
    print(f"  Bumps (>{args.bump_thr} m/s²)    : {int((ev['type'] == 'bump').sum()) if len(ev) else 0}")
    print(f"  Holes (>{args.depth_thr} cm dip)   : {int((ev['type'] == 'depth').sum()) if len(ev) else 0}")
    if len(cells):
        worst = cells.sort_values("roughness_idx", ascending=False).head(3)
        print("  Roughest cells      : " + ", ".join(f"#{int(c)} ({r:.2f})" for c, r in zip(worst['cell_id'], worst['roughness_idx'])))

    print(f"\nSaved in {out_dir}/ :  cells.csv   events.csv   ride_overview.png")


if __name__ == "__main__":
    main()
