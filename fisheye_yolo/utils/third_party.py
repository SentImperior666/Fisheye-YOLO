import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_lieconv_on_path() -> None:
    lieconv_path = _repo_root() / "third_party" / "LieConv"
    if lieconv_path.exists():
        path_str = str(lieconv_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def ensure_yolo_on_path() -> None:
    root = _repo_root()
    for rel in ("third_party/yolov7", "third_party/yolov8"):
        path = root / rel
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def ensure_third_party_on_path() -> None:
    ensure_lieconv_on_path()
    ensure_yolo_on_path()
