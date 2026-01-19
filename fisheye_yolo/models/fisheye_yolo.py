from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch.nn as nn

from fisheye_yolo.geometry.fisheye_camera import FisheyeCameraModel
from fisheye_yolo.layers import FisheyeConv
from fisheye_yolo.models.yolo_backbones import (
    build_yolov7_model,
    build_yolov8_model,
    default_yolov7_cfg,
    default_yolov8_cfg,
)


@dataclass
class FisheyeLayerConfig:
    liftsamples: int = 4
    mc_samples: int = 32
    fill: float = 1 / 4
    knn: bool = True
    max_replace: Optional[int] = None
    # Scalability parameters
    mode: Literal["patch", "sparse", "full"] = "patch"
    patch_size: int = 16
    patch_overlap: float = 0.5
    neighborhood_radius: int = 5


def _replace_convs(module, conv_cls, camera: FisheyeCameraModel, cfg: FisheyeLayerConfig, count):
    for name, child in module.named_children():
        if isinstance(child, conv_cls):
            conv = child.conv
            replacement = FisheyeConv(
                conv.in_channels,
                conv.out_channels,
                k=conv.kernel_size[0],
                s=conv.stride[0],
                p=conv.padding[0],
                g=conv.groups,
                act=child.act,
                camera=camera,
                liftsamples=cfg.liftsamples,
                mc_samples=cfg.mc_samples,
                fill=cfg.fill,
                knn=cfg.knn,
                mode=cfg.mode,
                patch_size=cfg.patch_size,
                patch_overlap=cfg.patch_overlap,
                neighborhood_radius=cfg.neighborhood_radius,
            )
            setattr(module, name, replacement)
            count[0] += 1
            if cfg.max_replace is not None and count[0] >= cfg.max_replace:
                return
        else:
            _replace_convs(child, conv_cls, camera, cfg, count)
            if cfg.max_replace is not None and count[0] >= cfg.max_replace:
                return


class FisheyeYOLO(nn.Module):
    def __init__(
        self,
        backbone: str = "yolov7",
        cfg_path: Optional[str] = None,
        ch: int = 3,
        nc: int = 80,
        camera: Optional[FisheyeCameraModel] = None,
        use_fisheye_stem: bool = False,
        replace_convs: bool = True,
        fisheye_cfg: Optional[FisheyeLayerConfig] = None,
    ):
        super().__init__()
        if fisheye_cfg is None:
            fisheye_cfg = FisheyeLayerConfig()
        if replace_convs and camera is None:
            raise ValueError("camera is required when replace_convs=True")

        if backbone == "yolov7":
            cfg_path = cfg_path or default_yolov7_cfg()
            self.model = build_yolov7_model(cfg_path, ch=ch, nc=nc)
        elif backbone == "yolov8":
            cfg_path = cfg_path or default_yolov8_cfg()
            self.model = build_yolov8_model(cfg_path, ch=ch, nc=nc)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = backbone
        self.camera = camera
        self.fisheye_cfg = fisheye_cfg

        self.fisheye_stem = None
        if use_fisheye_stem:
            if camera is None:
                raise ValueError("camera is required when use_fisheye_stem=True")
            self.fisheye_stem = FisheyeConv(
                ch,
                ch,
                s=1,
                camera=camera,
                liftsamples=fisheye_cfg.liftsamples,
                mc_samples=fisheye_cfg.mc_samples,
                fill=fisheye_cfg.fill,
                knn=fisheye_cfg.knn,
                mode=fisheye_cfg.mode,
                patch_size=fisheye_cfg.patch_size,
                patch_overlap=fisheye_cfg.patch_overlap,
                neighborhood_radius=fisheye_cfg.neighborhood_radius,
            )

        if replace_convs:
            count = [0]
            if backbone == "yolov7":
                from models.common import Conv as Y7Conv  # type: ignore

                _replace_convs(self.model, Y7Conv, camera, fisheye_cfg, count)
            else:
                from ultralytics.nn.modules.conv import Conv as Y8Conv  # type: ignore

                _replace_convs(self.model, Y8Conv, camera, fisheye_cfg, count)

    def forward(self, x, *args, **kwargs):
        if self.fisheye_stem is not None:
            x = self.fisheye_stem(x)
        return self.model(x, *args, **kwargs)


def apply_fisheye_convs(model, backbone: str, camera: FisheyeCameraModel, fisheye_cfg: FisheyeLayerConfig):
    count = [0]
    if backbone == "yolov7":
        from models.common import Conv as Y7Conv  # type: ignore

        _replace_convs(model, Y7Conv, camera, fisheye_cfg, count)
    elif backbone == "yolov8":
        from ultralytics.nn.modules.conv import Conv as Y8Conv  # type: ignore

        _replace_convs(model, Y8Conv, camera, fisheye_cfg, count)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
