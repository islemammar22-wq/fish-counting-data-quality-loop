"""Compare predicted counts with ground-truth counts and flag the worst images.

Ground truth count = number of lines in each YOLO label file.
Output: errors_<tag>.csv sorted by absolute error (largest first).
The top rows are the images to re-check: often the label is wrong, not the model.
"""
import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def gt_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True, help="folder of validation images")
    p.add_argument("--labels", required=True, help="folder of matching YOLO .txt labels")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--tag", default="round1")
    p.add_argument("--flag", type=int, default=15, help="how many worst images to print")
    args = p.parse_args()

    model = YOLO(args.weights)
    images = sorted(f for f in Path(args.images).iterdir() if f.suffix.lower() in IMG_EXT)

    rows = []
    for img in images:
        result = model.predict(str(img), conf=args.conf, iou=args.iou, verbose=False)[0]
        pred = len(result.boxes)
        gt = gt_count(Path(args.labels) / f"{img.stem}.txt")
        rows.append((img.name, gt, pred, pred - gt, abs(pred - gt)))

    rows.sort(key=lambda r: -r[4])
    mae = sum(r[4] for r in rows) / max(len(rows), 1)

    out = Path(f"errors_{args.tag}.csv")
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "predicted", "error", "abs_error"])
        w.writerows(rows)

    print(f"[{args.tag}] images: {len(rows)}  MAE: {mae:.2f}")
    print(f"Saved {out}. Worst {args.flag} images to re-check:")
    for r in rows[: args.flag]:
        print(f"  {r[0]}: gt={r[1]} pred={r[2]} error={r[3]:+d}")


if __name__ == "__main__":
    main()
