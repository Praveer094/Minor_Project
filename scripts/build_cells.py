#!/usr/bin/env python3
"""
build_cells.py  -  JOIN the sensor analysis and the camera detections, cell by cell.

This is the glue between analyze_ride_csv.py (sensor side) and detect_frames.py (camera side).

What it does
------------
1. SYNC     finds the 3 taps in the video's audio and matches them to the 3 taps that
            analyze_ride_csv.py found in the sensor log:
                 sensor_time = video_time + offset
            (if the taps cannot be found, pass the offset yourself with --offset)
2. PLACE    every detection gets a sensor time, so it falls into one of the 50 m cells
            (cells.csv from analyze_ride_csv.py knows the time range of each cell)
3. DEDUP    the same pothole is visible in many consecutive frames (5 per second), so
            detections of the same class closer than --instance-gap seconds count as ONE
4. MERGE    one row per cell: sensor features + camera counts + "confirmed" flag

Usage
-----
  python build_cells.py --road "Road 1" --analysis ride_001_analysis \
        --frames-index frames_road1/frames_index.csv --detections detections.csv \
        --video road1.mp4 --out out/features_road1.csv
  # if the audio taps are not found:   add  --offset 2.35

Needs:  pip install pandas numpy scipy     and  ffmpeg  (only for the automatic audio sync)
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import find_peaks


# --------------------------------------------------------------------------- #
#  sync
# --------------------------------------------------------------------------- #
def mad(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def trios(times):
    """Groups of 3 spikes within 3 s of each other (same rule as analyze_ride_csv.py)."""
    out, i = [], 0
    while i + 2 < len(times):
        t = times[i:i + 3]
        if t[2] - t[0] <= 3.0 and np.all(np.diff(t) <= 1.5):
            out.append([float(x) for x in t])
            i += 3
        else:
            i += 1
    return out


def video_taps(video):
    """Find (start_taps, end_taps) in the audio track of the video, in video seconds."""
    with tempfile.TemporaryDirectory() as d:
        wav = Path(d) / "a.wav"
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-vn",
                            "-ac", "1", "-ar", "16000", str(wav)], capture_output=True, text=True)
        if r.returncode != 0 or not wav.exists():
            raise RuntimeError("ffmpeg could not read the audio (is ffmpeg installed? "
                               f"does the video have sound?)  {r.stderr.strip()[:200]}")
        sr, x = wavfile.read(wav)
    x = x.astype(float)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.abs(np.diff(x, prepend=x[0]))           # emphasise sharp clicks over rumble/voice
    env = pd.Series(x).rolling(int(sr * 0.005), min_periods=1).max().to_numpy()
    thr = max(np.median(env) + 12 * mad(env), 0.2 * env.max())
    peaks, _ = find_peaks(env, height=thr, distance=int(0.12 * sr))
    t = peaks / sr
    groups = trios(t)
    dur = len(x) / sr
    start = next((g for g in groups if g[0] <= 90), None)
    end = next((g for g in reversed(groups) if g[0] >= dur - 90), None)
    if start is not None and end is not None and start == end:
        end = None
    return start, end


def compute_offset(analysis_dir, video, manual):
    if manual is not None:
        print(f"Using manual offset {manual:+.3f} s")
        return manual
    tp = Path(analysis_dir) / "taps.json"
    if not tp.exists():
        sys.exit("taps.json missing: re-run analyze_ride_csv.py (patched version), or pass --offset.")
    s = json.loads(tp.read_text())
    if not s["start_tap"]:
        sys.exit("No start taps in the sensor log. Pass --offset manually.")
    if not video:
        sys.exit("Pass --video (for the audio taps) or --offset.")
    v_start, v_end = video_taps(video)
    if not v_start:
        sys.exit("No 3 taps found in the video audio. Pass --offset manually.")
    off = float(np.mean(np.array(s["start_tap"]) - np.array(v_start)))
    print(f"Start taps: sensor {s['start_tap']}  video {[round(x, 2) for x in v_start]}  -> offset {off:+.3f} s")
    if s["end_tap"] and v_end:
        off_end = float(np.mean(np.array(s["end_tap"]) - np.array(v_end)))
        print(f"End taps  : offset {off_end:+.3f} s   drift over ride = {off_end - off:+.3f} s "
              f"({'OK' if abs(off_end - off) < 1.0 else 'CHECK: clocks drifted or wrong taps matched'})")
    else:
        print("End taps not found in both streams: drift check skipped.")
    return off


# --------------------------------------------------------------------------- #
#  camera -> cells
# --------------------------------------------------------------------------- #
def class_group(name):
    n = str(name).lower()
    for key in ("pothole", "alligator", "longitudinal", "transverse"):
        if key in n:
            return key
    return "other"


def detections_to_cells(det, idx, cells, offset, gap):
    """Return per-cell camera counts (after de-duplication) and how many detections were used."""
    cols = ["n_pothole", "n_alligator", "n_longitudinal", "n_transverse", "n_other", "max_pothole_conf"]
    out = pd.DataFrame(0.0, index=cells["cell_id"], columns=cols)
    if det.empty:
        return out, 0, 0
    d = det.merge(idx[["frame_file", "video_time_s"]], on="frame_file", how="left")
    if d["video_time_s"].isna().any():
        print(f"  warning: {int(d['video_time_s'].isna().sum())} detections have no entry in frames_index.csv (dropped)")
        d = d.dropna(subset=["video_time_s"])
    d["t_sensor"] = d["video_time_s"] + offset
    d["group"] = d["class_name"].map(class_group)

    # de-duplicate: same class, consecutive frames -> one instance
    d = d.sort_values(["group", "t_sensor"])
    new = (d["group"] != d["group"].shift()) | (d["t_sensor"].diff() > gap)
    d["instance"] = new.cumsum()
    inst = d.loc[d.groupby("instance")["confidence"].idxmax()]      # best frame of each instance

    starts = cells["t_start_s"].to_numpy()
    t_end_all = float(cells["t_end_s"].max())
    pos = np.searchsorted(starts, inst["t_sensor"].to_numpy(), side="right") - 1
    inside = (inst["t_sensor"].to_numpy() >= starts[0] - 1.0) & (inst["t_sensor"].to_numpy() <= t_end_all + 1.0)
    pos = np.clip(pos, 0, len(cells) - 1)
    inst = inst.assign(cell_id=cells["cell_id"].to_numpy()[pos])[inside]
    dropped = int((~inside).sum())

    for (cid, grp), g in inst.groupby(["cell_id", "group"]):
        out.loc[cid, f"n_{grp}"] += len(g)
        if grp == "pothole":
            out.loc[cid, "max_pothole_conf"] = max(out.loc[cid, "max_pothole_conf"], g["confidence"].max())
    return out, len(inst), dropped


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Join sensor cells and camera detections.")
    ap.add_argument("--road", required=True, help='road name, e.g. "Road 1"')
    ap.add_argument("--analysis", required=True, help="folder written by analyze_ride_csv.py")
    ap.add_argument("--frames-index", required=True, help="frames_index.csv from extract_frames.py")
    ap.add_argument("--detections", required=True, help="detections.csv from detect_frames.py")
    ap.add_argument("--video", help="the ride video (for the audio-tap sync)")
    ap.add_argument("--offset", type=float, default=None,
                    help="seconds to ADD to video time to get sensor time (skips the audio sync)")
    ap.add_argument("--instance-gap", type=float, default=1.0,
                    help="same-class detections closer than this (s) count as one damage (default 1.0)")
    ap.add_argument("--out", required=True, help="output features CSV")
    args = ap.parse_args()

    ad = Path(args.analysis)
    cells = pd.read_csv(ad / "cells.csv").sort_values("t_start_s").reset_index(drop=True)
    idx = pd.read_csv(args.frames_index)
    det = pd.read_csv(args.detections)

    offset = compute_offset(ad, args.video, args.offset)
    cam, n_inst, dropped = detections_to_cells(det, idx, cells, offset, args.instance_gap)
    print(f"Detections: {len(det)} boxes -> {n_inst} separate damages, {dropped} outside the ride (ignored)")

    f = cells.merge(cam.reset_index(), on="cell_id", how="left")
    for c in cam.columns:
        f[c] = f[c].fillna(0)
    for c in ("n_bumps", "n_holes"):
        if c not in f:
            f[c] = 0
    # sensor and camera agree: a pothole seen by the camera AND felt/measured by a sensor
    f["confirmed"] = (f["n_pothole"] > 0) & ((f["n_bumps"] > 0) | (f["n_holes"] > 0))
    f.insert(0, "road", args.road)
    f.insert(1, "cell_uid", [f"{args.road}_{int(c)}" for c in f["cell_id"]])
    f["video_offset_s"] = round(offset, 3)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    f.to_csv(args.out, index=False)
    print(f"{len(f)} cells -> {args.out}   "
          f"(cells with a camera pothole: {int((f['n_pothole'] > 0).sum())}, confirmed by sensors: {int(f['confirmed'].sum())})")


if __name__ == "__main__":
    main()
