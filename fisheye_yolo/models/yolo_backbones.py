from pathlib import Path

from fisheye_yolo.utils.third_party import ensure_yolo_on_path

ensure_yolo_on_path()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_yolov7_cfg() -> str:
    return str(_repo_root() / "third_party" / "yolov7" / "cfg" / "training" / "yolov7.yaml")


def default_yolov8_cfg() -> str:
    return str(
        _repo_root()
        / "third_party"
        / "yolov8"
        / "ultralytics"
        / "cfg"
        / "models"
        / "v8"
        / "yolov8n.yaml"
    )


def build_yolov7_model(cfg_path: str, ch=3, nc=80, anchors=None):
    from models.yolo import Model  # type: ignore

    return Model(cfg_path, ch=ch, nc=nc, anchors=anchors)


def build_yolov8_model(cfg_path: str, ch=3, nc=80):
    from ultralytics.nn.tasks import DetectionModel  # type: ignore

    return DetectionModel(cfg=cfg_path, ch=ch, nc=nc)
