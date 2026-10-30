"""Fine-tune a pretrained YOLO detector on a counting dataset (YOLO format)."""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to data.yaml")
    p.add_argument("--base", default="yolov8n.pt", help="pretrained weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="round1", help="e.g. round1, round2 after label fixes")
    args = p.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )
    print(f"Weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
