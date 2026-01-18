from pathlib import Path

import cv2


def draw_detections(image_bgr, detections, color=(0, 255, 0), score_thresh=0.25):
    for det in detections:
        score = det.get("score", 1.0)
        if score < score_thresh:
            continue
        x, y, w, h = det["bbox"]
        p1 = int(x), int(y)
        p2 = int(x + w), int(y + h)
        cv2.rectangle(image_bgr, p1, p2, color, 2)
        cv2.putText(
            image_bgr,
            f"{det.get('category_id', -1)}:{score:.2f}",
            (p1[0], max(p1[1] - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return image_bgr


def save_visualization(image_path, detections, output_path, score_thresh=0.25):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = draw_detections(image, detections, score_thresh=score_thresh)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
