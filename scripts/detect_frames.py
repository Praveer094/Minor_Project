#!/usr/bin/env python3
"""
detect_frames.py  -  run the trained CNN on EVERY frame in a folder (batch version of detect_frame.py).

Usage
-----
  python detect_frames.py --frames frames_road1 --weights infra_forensics_best.pt --out detections.csv
  python detect_frames.py --frames frames_road1 --annotate        # also save pictures with boxes

Output  detections.csv  (one row per detected box; frames with no damage have no row)
  frame_file, class_id, class_name, confidence, x1, y1, x2, y2,
  x_center_norm, y_center_norm, width_norm, height_norm
frame_file is the same name that extract_frames.py wrote in frames_index.csv, which is
how build_cells.py later finds the video time of each detection.

Needs:  pip install ultralytics opencv-python
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Ultralytics is missing. Run:  pip install ultralytics")

HEADER = ["frame_file", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
          "x_center_norm", "y_center_norm", "width_norm", "height_norm"]


def main():
    ap = argparse.ArgumentParser(description="Run the damage CNN on a folder of frames.")
    ap.add_argument("--frames", required=True, help="folder of JPG frames (from extract_frames.py)")
    ap.add_argument("--weights", default="infra_forensics_best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="images per model call")
    ap.add_argument("--out", default="detections.csv")
    ap.add_argument("--annotate", action="store_true",
                    help="save frames with boxes drawn (only frames that have detections)")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not files:
        sys.exit(f"No images found in {frames_dir}")
    if not Path(args.weights).exists():
        sys.exit(f"Model file not found: {args.weights}\n"
                 "Download infra_forensics_best.pt from the Kaggle notebook's Output tab.")

    model = YOLO(args.weights)
    ann_dir = None
    if args.annotate:
        import cv2
        ann_dir = frames_dir.parent / (frames_dir.name + "_annotated")
        ann_dir.mkdir(exist_ok=True)

    n_det, n_frames_hit = 0, 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        for i in range(0, len(files), args.batch):
            chunk = files[i:i + args.batch]
            results = model.predict([str(p) for p in chunk], conf=args.conf,
                                    imgsz=args.imgsz, verbose=False)
            for p, r in zip(chunk, results):
                if len(r.boxes) == 0:
                    continue
                n_frames_hit += 1
                for b in r.boxes:
                    cid = int(b.cls[0])
                    x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
                    xc, yc, bw, bh = (float(v) for v in b.xywhn[0])
                    wr.writerow([p.name, cid, model.names[cid], round(float(b.conf[0]), 4),
                                 round(x1), round(y1), round(x2), round(y2),
                                 round(xc, 4), round(yc, 4), round(bw, 4), round(bh, 4)])
                    n_det += 1
                if ann_dir is not None:
                    import cv2
                    cv2.imwrite(str(ann_dir / p.name), r.plot())
            print(f"  {min(i + args.batch, len(files))}/{len(files)} frames", end="\r")
    print(f"\nDone: {len(files)} frames checked, {n_frames_hit} had damage, {n_det} boxes -> {out}")


if __name__ == "__main__":
    main()
