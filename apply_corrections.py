"""Apply approved label corrections to a YOLO dataset (single-class, class id 0).

corrections.csv has a header and one row per image:

    split,image,action
    valid,1281,add
    valid,541,remove

  split   train | valid | test
  image   the number at the start of the file name (1281 for 1281_jpg.rf.xxxx.jpg)
  action  add     -> add the model's boxes that have NO matching label
          remove  -> remove the labels that have NO matching model box

Safety:
- Dry run by default: it only prints what it would change. Add --apply to write.
- Original label files are copied to <split>/labels_backup/ before the first change.
- If more than --max-change boxes would change in one image, that image is skipped
  so you can fix it by hand.
- Always re-check the changed images with review.py afterwards.

Usage (Windows):
    python apply_corrections.py --weights runs\\detect\\runs\\round1\\weights\\best.pt ^
        --data data --csv corrections.csv --conf 0.25 --iou 0.5
    (add --apply when the dry run looks right)
"""
import argparse
import csv
import shutil
from pathlib import Path


def to_xyxy(b):
    cx, cy, w, h = b
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def iou(a, b):
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def read_labels(path: Path):
    """Return a list of (original_line, (cx, cy, w, h))."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                rows.append((line.strip(), tuple(float(v) for v in parts[1:5])))
    return rows


def unmatched(boxes, others, thr):
    """Indices of boxes that have no box in `others` with IoU >= thr."""
    return [i for i, a in enumerate(boxes) if not any(iou(a, b) >= thr for b in others)]


def process(row, data_dir: Path, predict_fn, thr, max_change, apply):
    split, num, action = (v.strip() for v in row)
    if action not in ("add", "remove"):
        return f"{split}/{num}: unknown action '{action}', skipped"

    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"
    matches = sorted(img_dir.glob(f"{num}_jpg.rf.*"))
    if len(matches) != 1:
        return f"{split}/{num}: expected exactly 1 image, found {len(matches)}, skipped"

    img = matches[0]
    lbl = lbl_dir / f"{img.stem}.txt"
    gt_rows = read_labels(lbl)
    gt_boxes = [r[1] for r in gt_rows]
    preds = predict_fn(img)

    if action == "remove":
        drop = unmatched(gt_boxes, preds, thr)
        if not drop:
            return f"{split}/{num}: nothing to remove (every label has a matching prediction)"
        if len(drop) > max_change:
            return f"{split}/{num}: {len(drop)} labels would be removed (limit {max_change}), skipped, fix by hand"
        kept = [r[0] for i, r in enumerate(gt_rows) if i not in set(drop)]
        new_lines = kept
        msg = f"{split}/{num}: remove {len(drop)} label(s)"
    else:
        extra = unmatched(preds, gt_boxes, thr)
        if not extra:
            return f"{split}/{num}: nothing to add (every prediction has a matching label)"
        if len(extra) > max_change:
            return f"{split}/{num}: {len(extra)} labels would be added (limit {max_change}), skipped, fix by hand"
        added = [
            "0 {:.6f} {:.6f} {:.6f} {:.6f}".format(*preds[i]) for i in extra
        ]
        new_lines = [r[0] for r in gt_rows] + added
        msg = f"{split}/{num}: add {len(extra)} label(s)"

    if not apply:
        return msg + "  [dry run]"

    backup_dir = lbl_dir.parent / "labels_backup"
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / lbl.name
    if lbl.exists() and not backup.exists():
        shutil.copy2(lbl, backup)
    lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
    return msg + "  [written]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True, help="dataset folder containing train/ valid/ test/")
    p.add_argument("--csv", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU for predictions")
    p.add_argument("--match-iou", type=float, default=0.3, help="IoU to call a box matched")
    p.add_argument("--max-change", type=int, default=2)
    p.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = p.parse_args()

    from ultralytics import YOLO  # imported here so the logic above can be tested alone

    model = YOLO(args.weights)

    def predict_fn(path):
        result = model.predict(str(path), conf=args.conf, iou=args.iou, verbose=False)[0]
        return [tuple(b) for b in result.boxes.xywhn.tolist()]

    with open(args.csv, newline="") as f:
        rows = [(r["split"], r["image"], r["action"]) for r in csv.DictReader(f)]

    for row in rows:
        print(process(row, Path(args.data), predict_fn, args.match_iou, args.max_change, args.apply))

    if not args.apply:
        print("Dry run only. Add --apply to write the changes.")


if __name__ == "__main__":
    main()
