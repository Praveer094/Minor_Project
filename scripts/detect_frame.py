#!/usr/bin/env python3
"""
detect_frame.py  -  feed ONE image (or one video frame) to the trained CNN.

Usage
-----
  # one image
  python detect_frame.py --image road.jpg

  # one frame taken straight from a video at 125.4 seconds
  python detect_frame.py --video road1.mp4 --at 125.4

  # options
  python detect_frame.py --image road.jpg --weights infra_forensics_best.pt --conf 0.25

Needs:  pip install ultralytics opencv-python
Keep infra_forensics_best.pt (from the Kaggle results zip) in the same folder.

What it prints, per detection
-----------------------------
  class name, confidence, the box in pixels (x1 y1 x2 y2), and the same box in
  YOLO label format:  class_id  x_center  y_center  width  height
  (all four numbers are fractions of the image size, 0 to 1)

It also saves the picture with the boxes drawn on it and a small CSV.
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is missing. Run:  pip install opencv-python")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Ultralytics is missing. Run:  pip install ultralytics")


def load_image(args):
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Could not read image: {args.image}")
        return img, Path(args.image).stem
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(args.at * fps)))
    ok, img = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame at {args.at}s.")
    return img, f"{Path(args.video).stem}_t{args.at:.3f}s"


def main():
    ap = argparse.ArgumentParser(description="Run the damage-detection CNN on one frame.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to a JPG/PNG image")
    src.add_argument("--video", help="path to a video (use with --at)")
    ap.add_argument("--at", type=float, default=0.0,
                    help="video time in seconds (with --video)")
    ap.add_argument("--weights", default="infra_forensics_best.pt",
                    help="trained model file")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="minimum confidence to keep a detection (default 0.25)")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (training used 640)")
    ap.add_argument("--out", default="detect_out", help="output folder")
    args = ap.parse_args()

    if not Path(args.weights).exists():
        sys.exit(f"Model file not found: {args.weights}\n"
                 "Copy infra_forensics_best.pt from the Kaggle results zip into this folder.")

    img, name = load_image(args)
    h, w = img.shape[:2]

    model = YOLO(args.weights)
    result = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"\nImage: {name}  ({w} x {h})   model: {args.weights}   conf >= {args.conf}")
    if len(result.boxes) == 0:
        print("No damage detected.")
    else:
        print(f"{'class':<22}{'conf':>6}   {'box (x1 y1 x2 y2 px)':<26}YOLO format")
        for b in result.boxes:
            cid = int(b.cls[0])
            cname = model.names[cid]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            xc, yc, bw, bh = (float(v) for v in b.xywhn[0])
            print(f"{cname:<22}{conf:>6.2f}   "
                  f"{x1:>5.0f} {y1:>5.0f} {x2:>5.0f} {y2:>5.0f}      "
                  f"{cid} {xc:.4f} {yc:.4f} {bw:.4f} {bh:.4f}")
            rows.append([cid, cname, round(conf, 4), round(x1), round(y1),
                         round(x2), round(y2), round(xc, 4), round(yc, 4),
                         round(bw, 4), round(bh, 4)])

    annotated = result.plot()  # BGR image with boxes drawn
    img_path = out_dir / f"{name}_detected.jpg"
    cv2.imwrite(str(img_path), annotated)

    csv_path = out_dir / f"{name}_detections.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["class_id", "class_name", "confidence", "x1", "y1", "x2", "y2",
                     "x_center_norm", "y_center_norm", "width_norm", "height_norm"])
        wr.writerows(rows)

    print(f"\nSaved picture: {img_path}")
    print(f"Saved table:   {csv_path}")


if __name__ == "__main__":
    main()
