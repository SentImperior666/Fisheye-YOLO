# Fisheye-YOLO

Fisheye-YOLO replaces standard shift-equivariant convolutions with fisheye-equivariant layers derived from a fisheye
camera model and SO(3) group actions (via LieConv). The project provides:

- Vendored YOLOv7 and YOLOv8 architectures.
- Fisheye-equivariant layers built on LieConv and a fisheye camera model.
- Fisheye-COCO dataset generation (images + annotations warped consistently).
- Training, inference, evaluation, and benchmarking scripts.
- Dockerized, reproducible environment with TensorBoard logging.

## Theory in short

Fisheye warps are not linear in pixel space, so translation equivariance in the image plane is a poor inductive bias.
Instead, the warp can be modeled as an SO(3) action on rays:

```
u' = π( R · π⁻¹(u) )
```

where `π` and `π⁻¹` are fisheye projection/unprojection. We lift pixel coordinates to rays and run SO(3)-equivariant
LieConv kernels over these lifted coordinates.

## Repo layout

- `third_party/`: vendored YOLOv7, YOLOv8, and LieConv sources.
- `fisheye_yolo/`: core package (geometry, layers, models, data, eval).
- `scripts/`: dataset creation, training, inference, benchmarking.
- `configs/`: dataset/model configuration YAMLs.
- `docker/`: CUDA Docker environment.

## Docker setup (required)

All scripts should run inside Docker.

```bash
docker compose -f docker/docker-compose.yaml build
docker compose -f docker/docker-compose.yaml run --rm fisheye-yolo bash
```

Inside the container, `/workspace` is the repo root and `/data` is mounted for datasets.

## Create Fisheye-COCO

```bash
python scripts/create_fisheye_coco.py \
  --coco-root /data/coco \
  --split train2017 \
  --output-root /data/fisheye_coco \
  --out-size 640,640 \
  --fov-src 90 \
  --fov-fisheye 180 \
  --samples-per-edge 9 \
  --write-yolo-labels
```

This produces:
- `images/train2017` (fisheye-warped images)
- `labels/train2017` (YOLO txt labels, optional)
- `annotations/instances_fisheye_train2017.json`

## Training

### Standard YOLO (v7/v8)

```bash
python scripts/train.py \
  --mode standard \
  --backbone yolov8 \
  --data configs/data/coco.yaml \
  --cfg third_party/yolov8/ultralytics/cfg/models/v8/yolov8n.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

### Fisheye-YOLO

```bash
python scripts/train.py \
  --mode fisheye \
  --backbone yolov8 \
  --data configs/data/fisheye_coco.yaml \
  --cfg third_party/yolov8/ultralytics/cfg/models/v8/yolov8n.yaml \
  --camera-fx 430 --camera-fy 430 --camera-cx 320 --camera-cy 320 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

YOLOv7 is supported by swapping `--backbone yolov7` and providing the YOLOv7 cfg path.

TensorBoard logs are written to `runs/train/` by YOLO training scripts.

## Inference

```bash
python scripts/infer.py \
  --mode standard \
  --backbone yolov8 \
  --weights /path/to/weights.pt \
  --source /path/to/images
```

Use `--mode fisheye` to run the fisheye-equivariant variant.

## Evaluation and benchmarking

To evaluate COCO-format predictions:

```bash
python scripts/benchmark.py \
  --ann /data/fisheye_coco/annotations/instances_fisheye_val2017.json \
  --pred /path/to/preds_model_a.json /path/to/preds_model_b.json \
  --out /workspace/runs/benchmarks/fisheye_coco.json
```

## Notes on compute

Fisheye-equivariant layers based on LieConv are more expensive than standard 2D convolutions.
Use smaller image sizes or reduce the number of fisheye layers (via `--max-replace`) for prototyping.
