"""
Infrastructure Forensics - run the trained road-damage model on a ride video.

Reads the video directly, takes a few frames per second, runs YOLO on each,
and writes every detection to a CSV with its time in the video.
It also saves a sample of frames with boxes drawn, for checking and slides.

Usage:
    python run_detection.py --video road1.mp4
Optional:
    --model infra_forensics_best.pt   trained weights (default: this name, same folder)
    --fps 5                           frames analysed per second of video
    --conf 0.25                       minimum confidence to keep a detection
    --out detections_road1.csv        output file (default: detections_<video name>.csv)
"""
import argparse, csv, os, sys, time

try:
    import cv2
    from ultralytics import YOLO
    import torch
except ImportError:
    sys.exit("Missing packages. Run:  pip install ultralytics opencv-python")


def pick_device():
    if torch.cuda.is_available():
        return 0                      # NVIDIA GPU (Windows/Linux)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"                  # Apple M1/M2/M3/M4
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="infra_forensics_best.pt")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-samples", type=int, default=40, help="how many annotated frames to save")
    a = ap.parse_args()

    if not os.path.exists(a.video):
        sys.exit(f"Video not found: {a.video}")
    if not os.path.exists(a.model):
        sys.exit(f"Model not found: {a.model}  (put infra_forensics_best.pt in this folder)")

    name = os.path.splitext(os.path.basename(a.video))[0]
    out_csv = a.out or f"detections_{name}.csv"
    sample_dir = f"samples_{name}"
    os.makedirs(sample_dir, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"Could not open video: {a.video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(video_fps / a.fps))
    device = pick_device()
    model = YOLO(a.model)
    names = model.names

    print(f"Video: {a.video} | {video_fps:.1f} fps | {total} frames | analysing every {step}th frame")
    print(f"Model: {a.model} | device: {device} | confidence >= {a.conf}")

    fields = ["video", "frame", "video_time_s", "class_id", "class_name", "confidence",
              "x1", "y1", "x2", "y2", "img_w", "img_h"]
    n_frames = n_det = saved = 0
    t0 = time.time()
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        idx = -1
        while True:
            ok = cap.grab()                     # move to next frame without decoding (fast)
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            ok, frame = cap.retrieve()          # decode only the frames we analyse
            if not ok:
                continue
            n_frames += 1
            h, wid = frame.shape[:2]
            r = model.predict(frame, conf=a.conf, imgsz=640, device=device, verbose=False)[0]
            t = idx / video_fps
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                k = int(k)
                w.writerow([name, idx, f"{t:.3f}", k, names[k], f"{c:.3f}",
                            int(x1), int(y1), int(x2), int(y2), wid, h])
                n_det += 1
            if len(r.boxes) and saved < a.save_samples and n_frames % 10 == 0:
                cv2.imwrite(os.path.join(sample_dir, f"{name}_{t:08.2f}s.jpg"), r.plot())
                saved += 1
            if n_frames % 100 == 0:
                done = f"{100 * idx / total:.0f}%" if total else f"{t:.0f}s of video"
                print(f"  {done} | {n_frames} frames analysed | {n_det} detections | {time.time() - t0:.0f}s elapsed")
    cap.release()
    print(f"\nDone in {time.time() - t0:.0f}s: {n_frames} frames, {n_det} detections")
    print(f"CSV: {out_csv}")
    print(f"Sample images: {sample_dir}/ ({saved} saved)")


if __name__ == "__main__":
    main()
