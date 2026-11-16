#!/usr/bin/env python3
"""
extract_frames.py  -  turn a ride video into JPG frames.

Two modes
---------
1) Sample the whole video (default: 5 frames per second):
       python extract_frames.py --video road1.mp4
       python extract_frames.py --video road1.mp4 --fps 2 --start 30 --end 300

2) Grab ONE frame at a given video time (seconds):
       python extract_frames.py --video road1.mp4 --at 125.4

Output
------
frames_<video name>/        the JPG files, each named with its video time,
                            e.g. road1_t0125.400s.jpg
frames_<video name>/frames_index.csv
                            frame_file, frame_number, video_time_s
                            (this video_time_s is what you later match to
                             the sensor log after the 3-tap sync)

Install once:  pip install opencv-python
"""
import argparse
import csv
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    sys.exit("OpenCV is missing. Run:  pip install opencv-python")


def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not fps or fps <= 0:
        sys.exit("Could not read the video's frame rate.")
    return cap, fps, total


def save_frame(img, out_dir, stem, t_s, quality):
    name = f"{stem}_t{t_s:09.3f}s.jpg"
    cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return name


def extract_single(cap, fps, total, t_s, out_dir, stem, quality):
    frame_no = int(round(t_s * fps))
    if frame_no < 0 or (total > 0 and frame_no >= total):
        sys.exit(f"Time {t_s}s is outside the video (length {total / fps:.1f}s).")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    if not ok:
        sys.exit(f"Could not read a frame at {t_s}s.")
    name = save_frame(img, out_dir, stem, frame_no / fps, quality)
    print(f"Saved single frame: {out_dir / name}")


def extract_many(cap, fps, total, target_fps, start_s, end_s, out_dir, stem, quality):
    step = max(1, int(round(fps / target_fps)))
    first = int(round(start_s * fps))
    last = total if end_s is None else min(total, int(round(end_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    rows = []
    n = first
    while n < last:
        # grab() only moves to the next frame; retrieve() decodes it.
        # Skipping decode on unused frames makes this much faster.
        if not cap.grab():
            break
        if (n - first) % step == 0:
            ok, img = cap.retrieve()
            if ok:
                t_s = n / fps
                name = save_frame(img, out_dir, stem, t_s, quality)
                rows.append((name, n, round(t_s, 3)))
                if len(rows) % 100 == 0:
                    print(f"  {len(rows)} frames saved (video time {t_s:.1f}s)")
        n += 1

    with open(out_dir / "frames_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_file", "frame_number", "video_time_s"])
        w.writerows(rows)
    print(f"Done: {len(rows)} frames -> {out_dir}")
    print(f"Index file: {out_dir / 'frames_index.csv'}")


def main():
    ap = argparse.ArgumentParser(description="Extract frames from a ride video.")
    ap.add_argument("--video", required=True, help="path to the video file")
    ap.add_argument("--fps", type=float, default=5.0,
                    help="frames to keep per second of video (default 5)")
    ap.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="end time in seconds")
    ap.add_argument("--at", type=float, default=None,
                    help="save ONE frame at this video time (seconds) and stop")
    ap.add_argument("--out", default=None, help="output folder")
    ap.add_argument("--quality", type=int, default=92, help="JPG quality 1-100")
    args = ap.parse_args()

    video = Path(args.video)
    stem = video.stem
    out_dir = Path(args.out) if args.out else Path(f"frames_{stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap, fps, total = open_video(video)
    print(f"Video: {video.name}  |  {fps:.2f} fps  |  {total} frames  |  {total / fps:.1f} s")

    if args.at is not None:
        extract_single(cap, fps, total, args.at, out_dir, stem, args.quality)
    else:
        if args.fps <= 0 or args.fps > fps:
            sys.exit(f"--fps must be between 0 and the video's {fps:.1f} fps.")
        extract_many(cap, fps, total, args.fps, args.start, args.end,
                     out_dir, stem, args.quality)
    cap.release()


if __name__ == "__main__":
    main()
