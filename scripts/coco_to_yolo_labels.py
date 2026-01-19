#!/usr/bin/env python3
"""Convert COCO JSON annotations to YOLO txt labels."""
import argparse
import json
from pathlib import Path

from tqdm import tqdm


def coco_to_yolo(bbox, img_w, img_h):
    """Convert COCO bbox [x, y, w, h] to YOLO format [xc, yc, w, h] normalized."""
    x, y, w, h = bbox
    xc = (x + w / 2) / img_w
    yc = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return xc, yc, w_norm, h_norm


def main():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to YOLO labels.")
    parser.add_argument("--ann-file", required=True, help="COCO annotation JSON file")
    parser.add_argument("--output-dir", required=True, help="Output directory for YOLO labels")
    parser.add_argument("--category-map", default=None, help="Category map JSON (optional)")
    args = parser.parse_args()

    with open(args.ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build category ID to index mapping
    if args.category_map:
        with open(args.category_map, "r", encoding="utf-8") as f:
            cat_id_to_idx = json.load(f)
        # Convert string keys to int if needed
        cat_id_to_idx = {int(k): v for k, v in cat_id_to_idx.items()}
    else:
        categories = sorted(data.get("categories", []), key=lambda c: c["id"])
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}

    # Build image ID to info mapping
    img_id_to_info = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    anns_by_image = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Write YOLO labels
    for img_id, img_info in tqdm(img_id_to_info.items(), desc="Writing labels"):
        img_w = img_info["width"]
        img_h = img_info["height"]
        file_stem = Path(img_info["file_name"]).stem

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cat_idx = cat_id_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue
            xc, yc, w, h = coco_to_yolo(ann["bbox"], img_w, img_h)
            lines.append(f"{cat_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        label_path = out_dir / f"{file_stem}.txt"
        label_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(img_id_to_info)} label files to {out_dir}")


if __name__ == "__main__":
    main()
