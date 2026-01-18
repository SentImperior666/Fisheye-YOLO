#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2

from fisheye_yolo.data.warp import (
    warp_bbox_xyxy_pinhole_to_fisheye,
    warp_pinhole_to_fisheye,
    xyxy_to_yolo,
)


def _parse_size(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("out-size must be 'H,W'")
    return int(parts[0]), int(parts[1])


def _load_coco(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _bbox_xywh_to_xyxy(bbox):
    x, y, w, h = bbox
    return x, y, x + w, y + h


def main():
    parser = argparse.ArgumentParser(description="Create Fisheye-COCO from COCO-2017.")
    parser.add_argument("--coco-root", required=True, help="COCO root directory")
    parser.add_argument("--split", default="train2017", help="Dataset split (train2017/val2017)")
    parser.add_argument("--output-root", required=True, help="Output directory for Fisheye-COCO")
    parser.add_argument("--out-size", type=_parse_size, default=(640, 640), help="Output size H,W")
    parser.add_argument("--fov-src", type=float, default=90.0, help="Source pinhole FOV in degrees")
    parser.add_argument("--fov-fisheye", type=float, default=180.0, help="Fisheye FOV in degrees")
    parser.add_argument("--samples-per-edge", type=int, default=9, help="BBox boundary samples per edge")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for debugging")
    parser.add_argument("--write-yolo-labels", action="store_true", help="Also write YOLO txt labels")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    split = args.split
    ann_path = coco_root / "annotations" / f"instances_{split}.json"
    data = _load_coco(ann_path)

    out_root = Path(args.output_root)
    out_images = out_root / "images" / split
    out_labels = out_root / "labels" / split
    out_annotations = out_root / "annotations"
    out_images.mkdir(parents=True, exist_ok=True)
    out_annotations.mkdir(parents=True, exist_ok=True)
    if args.write_yolo_labels:
        out_labels.mkdir(parents=True, exist_ok=True)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    _write_json(out_root / "category_map.json", cat_id_to_idx)

    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    new_ann_id = 1

    h_out, w_out = args.out_size
    img_count = 0
    for info in data["images"]:
        if args.max_images is not None and img_count >= args.max_images:
            break
        img_count += 1

        img_path = coco_root / split / info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        warped = warp_pinhole_to_fisheye(
            image,
            out_size=(h_out, w_out),
            fov_fisheye_deg=args.fov_fisheye,
            fov_src_deg=args.fov_src,
        )
        cv2.imwrite(str(out_images / info["file_name"]), warped)

        new_images.append(
            {
                "id": info["id"],
                "file_name": info["file_name"],
                "width": w_out,
                "height": h_out,
            }
        )

        yolo_lines = []
        for ann in anns_by_image.get(info["id"], []):
            x1, y1, x2, y2 = _bbox_xywh_to_xyxy(ann["bbox"])
            warped_bbox = warp_bbox_xyxy_pinhole_to_fisheye(
                (x1, y1, x2, y2),
                src_size=(info["height"], info["width"]),
                dst_size=(h_out, w_out),
                fov_src_deg=args.fov_src,
                fov_fisheye_deg=args.fov_fisheye,
                samples_per_edge=args.samples_per_edge,
            )
            if warped_bbox is None:
                continue
            wx1, wy1, wx2, wy2 = warped_bbox
            new_w = wx2 - wx1
            new_h = wy2 - wy1
            new_annotations.append(
                {
                    "id": new_ann_id,
                    "image_id": info["id"],
                    "category_id": ann["category_id"],
                    "bbox": [wx1, wy1, new_w, new_h],
                    "area": float(new_w * new_h),
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            new_ann_id += 1

            if args.write_yolo_labels:
                xc, yc, bw, bh = xyxy_to_yolo(wx1, wy1, wx2, wy2, w_out, h_out)
                yolo_lines.append(f"{cat_id_to_idx[ann['category_id']]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if args.write_yolo_labels:
            label_path = out_labels / f"{Path(info['file_name']).stem}.txt"
            label_path.write_text("\n".join(yolo_lines), encoding="utf-8")

    out_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }
    out_json = out_annotations / f"instances_fisheye_{split}.json"
    _write_json(out_json, out_data)


if __name__ == "__main__":
    main()
