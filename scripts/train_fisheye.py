#!/usr/bin/env python3
"""
Custom training script for Fisheye-YOLO models.

This script uses a YAML config with FisheyeConv layers directly,
ensuring the model is built with fisheye-equivariant convolutions from the start.
"""
import argparse

from fisheye_yolo.utils.third_party import ensure_yolo_on_path
from fisheye_yolo.utils import set_global_fisheye_camera

ensure_yolo_on_path()


def count_conv_types(model):
    """Count Conv and FisheyeConv layers in the model."""
    from ultralytics.nn.modules.conv import Conv as UltralyticsConv
    from fisheye_yolo.layers import FisheyeConv
    
    conv_count = 0
    fisheye_count = 0
    
    for module in model.modules():
        if isinstance(module, FisheyeConv):
            fisheye_count += 1
        elif isinstance(module, UltralyticsConv):
            conv_count += 1
    
    return conv_count, fisheye_count


def main():
    parser = argparse.ArgumentParser(description="Train Fisheye-YOLO models with FisheyeConv layers.")
    
    # Model and data
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--model", default="configs/models/yolov8n-fisheye.yaml", help="Model YAML path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/fisheye")
    parser.add_argument("--name", default="exp")
    
    # Camera parameters
    parser.add_argument("--camera-fx", type=float, default=320.0)
    parser.add_argument("--camera-fy", type=float, default=320.0)
    parser.add_argument("--camera-cx", type=float, default=320.0)
    parser.add_argument("--camera-cy", type=float, default=320.0)
    parser.add_argument("--camera-model", default="equisolid")
    parser.add_argument("--no-flip-y", action="store_true")
    
    # Fisheye layer parameters
    parser.add_argument("--liftsamples", type=int, default=4)
    parser.add_argument("--mc-samples", type=int, default=32)
    parser.add_argument("--fill", type=float, default=0.25)
    parser.add_argument("--no-knn", action="store_true")
    parser.add_argument(
        "--fisheye-mode", 
        choices=["patch", "sparse", "full"], 
        default="patch",
        help="LieConv processing mode: 'patch' (default), 'sparse', or 'full'"
    )
    parser.add_argument("--patch-size", type=int, default=32)
    
    args = parser.parse_args()
    
    # IMPORTANT: Set global camera BEFORE importing YOLO
    # This allows FisheyeConv layers to access camera parameters during model construction
    set_global_fisheye_camera(
        fx=args.camera_fx,
        fy=args.camera_fy,
        cx=args.camera_cx,
        cy=args.camera_cy,
        model=args.camera_model,
        flip_y=not args.no_flip_y,
        liftsamples=args.liftsamples,
        mc_samples=args.mc_samples,
        fill=args.fill,
        knn=not args.no_knn,
        mode=args.fisheye_mode,
        patch_size=args.patch_size,
    )
    
    # Now import and build YOLO model - FisheyeConv will use global camera
    from ultralytics import YOLO
    
    print(f"Loading model from: {args.model}")
    yolo = YOLO(args.model)
    
    # Count layer types to verify FisheyeConv is being used
    conv_count, fisheye_count = count_conv_types(yolo.model)
    print(f"Model layers: {conv_count} Conv, {fisheye_count} FisheyeConv")
    
    if fisheye_count == 0:
        print("WARNING: No FisheyeConv layers found! Check your model YAML.")
        print("Make sure to use a *-fisheye.yaml config (e.g., configs/models/yolov8n-fisheye.yaml)")
    
    # Train with fisheye-safe augmentations
    yolo.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        # Disable problematic augmentations for fisheye
        pretrained=False,
        amp=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        perspective=0.0,
        shear=0.0,
    )


if __name__ == "__main__":
    main()
