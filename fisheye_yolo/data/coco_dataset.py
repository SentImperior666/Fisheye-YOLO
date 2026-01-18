import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    def __init__(self, images_dir: str, ann_path: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.ann_path = Path(ann_path)
        self.transforms = transforms

        with self.ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.images = sorted(data["images"], key=lambda x: x["id"])
        self.categories = {c["id"]: c for c in data.get("categories", [])}
        self.anns_by_image = {}
        for ann in data.get("annotations", []):
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_path = self.images_dir / info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.anns_by_image.get(info["id"], [])
        boxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(int(ann["category_id"]))
            areas.append(float(ann.get("area", w * h)))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([info["id"]]),
            "area": areas,
            "iscrowd": torch.zeros(len(anns), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0

        return image, target
