#!/usr/bin/env python3
import argparse
import runpy
import sys
from pathlib import Path

from fisheye_yolo.geometry import FisheyeCameraModel
from fisheye_yolo.layers import FisheyeConv
from fisheye_yolo.models import FisheyeLayerConfig, apply_fisheye_convs
from fisheye_yolo.utils.third_party import ensure_yolo_on_path

ensure_yolo_on_path()


def _build_camera(args):
    return FisheyeCameraModel(
        fx=args.camera_fx,
        fy=args.camera_fy,
        cx=args.camera_cx,
        cy=args.camera_cy,
        model=args.camera_model,
        flip_y=not args.no_flip_y,
    )


def _fisheye_cfg(args):
    return FisheyeLayerConfig(
        liftsamples=args.liftsamples,
        mc_samples=args.mc_samples,
        fill=args.fill,
        knn=not args.no_knn,
        max_replace=args.max_replace,
    )


def _run_yolov7(args, fisheye: bool):
    train_py = Path("third_party/yolov7/train.py").resolve()
    if fisheye:
        import models.common as y7_common  # type: ignore

        camera = _build_camera(args)
        cfg = _fisheye_cfg(args)

        class FisheyeConvWrapper(FisheyeConv):
            def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
                super().__init__(
                    c1,
                    c2,
                    k=k,
                    s=s,
                    p=p,
                    g=g,
                    act=act,
                    camera=camera,
                    liftsamples=cfg.liftsamples,
                    mc_samples=cfg.mc_samples,
                    fill=cfg.fill,
                    knn=cfg.knn,
                )

        y7_common.Conv = FisheyeConvWrapper

    argv = [
        str(train_py),
        "--data",
        args.data,
        "--cfg",
        args.cfg,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch),
        "--img-size",
        str(args.imgsz),
        str(args.imgsz),
        "--device",
        args.device,
        "--project",
        args.project,
        "--name",
        args.name,
    ]
    if args.weights:
        argv += ["--weights", args.weights]
    sys.argv = argv
    runpy.run_path(str(train_py), run_name="__main__")


def _run_yolov8(args, fisheye: bool):
    from ultralytics import YOLO  # type: ignore

    model = YOLO(args.cfg)
    if fisheye:
        camera = _build_camera(args)
        cfg = _fisheye_cfg(args)
        apply_fisheye_convs(model.model, "yolov8", camera, cfg)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


def main():
    parser = argparse.ArgumentParser(description="Train standard or Fisheye YOLO models.")
    parser.add_argument("--mode", choices=["standard", "fisheye"], default="standard")
    parser.add_argument("--backbone", choices=["yolov7", "yolov8"], default="yolov8")
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--cfg", required=True, help="Model YAML or config path")
    parser.add_argument("--weights", default="", help="Optional weights path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="exp")

    parser.add_argument("--camera-fx", type=float, default=430.0)
    parser.add_argument("--camera-fy", type=float, default=430.0)
    parser.add_argument("--camera-cx", type=float, default=320.0)
    parser.add_argument("--camera-cy", type=float, default=320.0)
    parser.add_argument("--camera-model", default="equidistant")
    parser.add_argument("--no-flip-y", action="store_true")
    parser.add_argument("--liftsamples", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--fill", type=float, default=0.25)
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument("--max-replace", type=int, default=None)
    args = parser.parse_args()

    fisheye = args.mode == "fisheye"
    if args.backbone == "yolov7":
        _run_yolov7(args, fisheye=fisheye)
    else:
        _run_yolov8(args, fisheye=fisheye)


if __name__ == "__main__":
    main()
