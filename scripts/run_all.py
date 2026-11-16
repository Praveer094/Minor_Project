#!/usr/bin/env python3
"""
run_all.py  -  one command per step of the pipeline.

  1) process ONE ride  (run once per road / ride)
       python run_all.py ride --road "Road 1" --csv rides/road1/ride_001.csv --video rides/road1/road1.mp4
       (add --offset 2.35 if the audio-tap sync fails;  --fps 5 for frames per second)

  2) decide  (after all 4 roads are processed)
       python run_all.py decide --budget 10 --importance "Road 1=1.5"
       python run_all.py decide --budget 10 --sweep

Folders created next to this script's parent:
  work/<road>/analysis/  frames/  frames_index.csv  detections.csv
  out/features_<road>.csv  cells_all.csv  cells_repairable.csv  optimizer/...

Each step is skipped if its output already exists (use --redo to force), because
frame extraction and detection are the slow parts.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable


def run(cmd):
    print("\n>> " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        sys.exit(f"Step failed (exit {r.returncode}). Fix it and re-run; finished steps are skipped.")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def cmd_ride(a):
    w = ROOT / "work" / slug(a.road)
    ana, frames = w / "analysis", w / "frames"
    idx, det = w / "frames_index.csv", w / "detections.csv"
    w.mkdir(parents=True, exist_ok=True)

    print("== 1/4 sensor analysis")
    if a.redo or not (ana / "cells.csv").exists():
        run([PY, HERE / "analyze_ride_csv.py", a.csv, "--out", ana])
    else:
        print("   skipped (done)")

    print("== 2/4 extract frames")
    if a.redo or not (frames / "frames_index.csv").exists():
        run([PY, HERE / "extract_frames.py", "--video", a.video, "--fps", a.fps, "--out", frames])
    else:
        print("   skipped (done)")
    if not idx.exists() or a.redo:
        idx.write_bytes((frames / "frames_index.csv").read_bytes())

    print("== 3/4 detect damage on frames")
    if a.redo or not det.exists():
        run([PY, HERE / "detect_frames.py", "--frames", frames, "--weights", a.weights,
             "--conf", a.conf, "--out", det, "--annotate"])
    else:
        print("   skipped (done)")

    print("== 4/4 join sensors + camera into cells")
    cmd = [PY, HERE / "build_cells.py", "--road", a.road, "--analysis", ana, "--frames-index", idx,
           "--detections", det, "--video", a.video, "--out", ROOT / "out" / f"features_{slug(a.road)}.csv"]
    if a.offset is not None:
        cmd += ["--offset", a.offset]
    run(cmd)
    print(f"\nDone with {a.road}. Next road, or:  python run_all.py decide --budget 10")


def cmd_decide(a):
    out = ROOT / "out"
    feats = sorted(out.glob("features_*.csv"))
    if not feats:
        sys.exit("No out/features_*.csv yet. Run the 'ride' step for at least one road.")
    cmd = [PY, HERE / "score_cells.py", "--features", *feats, "--out", out]
    for imp in a.importance:
        cmd += ["--importance", imp]
    run(cmd)
    cmd = [PY, HERE / "optimizer.py", "--cells", out / "cells_repairable.csv",
           "--budget", a.budget, "--out", out / "optimizer"]
    if a.sweep:
        cmd.append("--sweep")
    run(cmd)
    print(f"\nResults: {out / 'optimizer'}/  (selected_cells.csv, comparison.csv, comparison.png)")


def main():
    ap = argparse.ArgumentParser(description="Run the Infrastructure Forensics pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("ride", help="process one road's ride")
    r.add_argument("--road", required=True)
    r.add_argument("--csv", required=True, help="ride_XXX.csv from the SD card")
    r.add_argument("--video", required=True)
    r.add_argument("--weights", default=str(ROOT / "models" / "infra_forensics_best.pt"))
    r.add_argument("--fps", type=float, default=5.0)
    r.add_argument("--conf", type=float, default=0.25)
    r.add_argument("--offset", type=float, default=None)
    r.add_argument("--redo", action="store_true")
    r.set_defaults(fn=cmd_ride)

    d = sub.add_parser("decide", help="score cells and run the optimizer")
    d.add_argument("--budget", type=float, default=10.0, help="lakh rupees")
    d.add_argument("--importance", action="append", default=[], help='"Road 1=1.5"')
    d.add_argument("--sweep", action="store_true")
    d.set_defaults(fn=cmd_decide)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
