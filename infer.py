"""
infer.py

Quick sanity-check inference script. Loads the custom-trained weights and
runs predictions on a few validation images, saving annotated bounding box
outputs to the default runs/detect/predict folder for visual inspection.

Run from the project root with your venv active:
    python infer.py
"""

import os
import glob
import random
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = "/home/pradeep/campus_waste_robot"
WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "runs", "detect", "tricascade_baseline_v1-2", "weights", "best.pt"
)
VAL_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "val")

NUM_SAMPLE_IMAGES = 3
VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    if not os.path.isfile(WEIGHTS_PATH):
        print(f"[ERROR] Trained weights not found at: {WEIGHTS_PATH}")
        print("Run train.py first to generate best.pt.")
        return

    if not os.path.isdir(VAL_IMAGES_DIR):
        print(f"[ERROR] Validation image directory not found: {VAL_IMAGES_DIR}")
        return

    all_val_images = [
        f for f in glob.glob(os.path.join(VAL_IMAGES_DIR, "*"))
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
    ]

    if not all_val_images:
        print(f"[ERROR] No images found in: {VAL_IMAGES_DIR}")
        return

    sample_count = min(NUM_SAMPLE_IMAGES, len(all_val_images))
    sample_images = random.sample(all_val_images, sample_count)

    print("=" * 70)
    print(f"Loading trained model: {WEIGHTS_PATH}")
    model = YOLO(WEIGHTS_PATH)
    print(f"Running inference on {sample_count} sample validation image(s):")
    for img in sample_images:
        print(f"  - {os.path.basename(img)}")
    print("=" * 70)

    # save=True writes annotated images to the default runs/detect/predict folder
    results = model.predict(
        source=sample_images,
        save=True,
        conf=0.25,
    )

    print("-" * 70)
    for result in results:
        image_name = os.path.basename(result.path)
        num_detections = len(result.boxes)
        print(f"[{image_name}] -> {num_detections} detection(s)")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            print(f"    - {class_name} (confidence: {confidence:.2f})")

    print("-" * 70)
    print("[DONE] Annotated images saved under: runs/detect/predict/")
    print("Open that folder in VS Code's file explorer to visually verify the bounding boxes.")
    print("-" * 70)


if __name__ == "__main__":
    main()