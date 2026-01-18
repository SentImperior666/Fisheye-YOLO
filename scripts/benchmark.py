#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from fisheye_yolo.eval import evaluate_coco


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple prediction files on COCO.")
    parser.add_argument("--ann", required=True, help="COCO ground truth annotation JSON")
    parser.add_argument("--pred", required=True, nargs="+", help="Prediction JSON files (COCO format)")
    parser.add_argument("--out", default="benchmark_results.json", help="Output JSON summary")
    args = parser.parse_args()

    results = {}
    for pred in args.pred:
        stats = evaluate_coco(args.ann, pred)
        results[pred] = [float(x) for x in stats]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
