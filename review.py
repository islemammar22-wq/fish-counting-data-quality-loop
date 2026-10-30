"""Draw ground-truth (green) and predicted (red) boxes on the worst images.

How to read the pictures:
- green only  -> a labeled fish the model missed (model error, or a bad label)
- red only    -> the model found a fish that has no label (often a MISSING LABEL)
- both overlap -> the model and the label agree

Usage (Windows):
python review.py --weights runs\\detect\\runs\\round1\\weights\\best.pt ^
    --images data\\valid\\images --labels data\\valid\\labels ^
    --csv errors_round1.csv --top 15 --out review_round1
"""
import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

GREEN = (0, 200, 0)
RED = (0, 0, 255)


def read_gt(label_path: Path, w: int, h: int):
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--csv", required=True, help="errors_<tag>.csv from evaluate.py")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--out", default="review")
    args = p.parse_args()

    model = YOLO(args.weights)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="") as f:
        rows = list(csv.DictReader(f))[: args.top]

    for row in rows:
        name = row["image"]
        img_path = Path(args.images) / name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read {img_path}, skipping")
            continue
        h, w = img.shape[:2]

        gt_boxes = read_gt(Path(args.labels) / f"{Path(name).stem}.txt", w, h)
        result = model.predict(img, conf=args.conf, iou=args.iou, verbose=False)[0]
        pred_boxes = [tuple(int(v) for v in b) for b in result.boxes.xyxy.tolist()]

        for (x1, y1, x2, y2) in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        for (x1, y1, x2, y2) in pred_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2)

        caption = f"label={len(gt_boxes)} (green)  pred={len(pred_boxes)} (red)"
        # Put the caption in a white strip ABOVE the image so it never hides the top edge.
        banner_h = 30
        img = cv2.copyMakeBorder(
            img, banner_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(img, caption, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        short = name.split(".rf.")[0]
        out_path = out_dir / f"{short}_review.jpg"
        cv2.imwrite(str(out_path), img)
        print(f"{out_path}: label={len(gt_boxes)} pred={len(pred_boxes)}")

    print(f"Done. Open the folder: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
