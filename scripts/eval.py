#!/usr/bin/env python3
"""
Evaluate a trained YOLO model on val or test split.

For val split: Runs COCO evaluation locally and prints metrics.
For test split: Saves predictions in COCO JSON format for server submission.

Usage:
    python scripts/eval.py --weights best.pt --data configs/data/fisheye_coco.yaml --split val
    python scripts/eval.py --weights best.pt --data configs/data/fisheye_coco.yaml --split test --save-json
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

import yaml


def _load_data_config(data_path: str) -> dict:
    with open(data_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_yolov7(args):
    """Run YOLOv7 test.py for evaluation."""
    test_py = Path("third_party/yolov7/test.py").resolve()

    data_cfg = _load_data_config(args.data)
    data_root = Path(data_cfg["path"])

    # Determine annotation file based on split
    if args.split == "test":
        ann_file = data_root / "annotations" / "image_info_test2017.json"
    else:
        ann_file = data_root / "annotations" / f"instances_{args.split}2017.json"

    argv = [
        str(test_py),
        "--weights",
        args.weights,
        "--data",
        args.data,
        "--img-size",
        str(args.imgsz),
        "--batch-size",
        str(args.batch),
        "--device",
        args.device,
        "--task",
        args.split,
    ]

    if args.save_json:
        argv.append("--save-json")

    if args.project:
        argv.extend(["--project", args.project])
    if args.name:
        argv.extend(["--name", args.name])

    sys.argv = argv
    runpy.run_path(str(test_py), run_name="__main__")


def _run_yolov8(args):
    """Run YOLOv8 validation."""
    from ultralytics import YOLO

    model = YOLO(args.weights)

    results = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=args.save_json,
        project=args.project,
        name=args.name,
    )

    # Print summary
    if hasattr(results, "box"):
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({args.split} split)")
        print(f"{'='*50}")
        print(f"mAP50-95: {results.box.map:.4f}")
        print(f"mAP50:    {results.box.map50:.4f}")
        print(f"mAP75:    {results.box.map75:.4f}")
        print(f"{'='*50}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model on val or test split."
    )
    parser.add_argument("--weights", required=True, help="Model weights path")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--backbone",
        choices=["yolov7", "yolov8"],
        default="yolov8",
        help="YOLO backbone version",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="0", help="CUDA device")
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save predictions in COCO JSON format",
    )
    parser.add_argument("--project", default="runs/eval", help="Output project dir")
    parser.add_argument("--name", default="exp", help="Output name")
    args = parser.parse_args()

    # For test split, always save JSON since local eval is not possible
    if args.split == "test" and not args.save_json:
        print("Note: Test split has no ground truth. Enabling --save-json for submission.")
        args.save_json = True

    if args.backbone == "yolov7":
        _run_yolov7(args)
    else:
        _run_yolov8(args)


if __name__ == "__main__":
    main()
