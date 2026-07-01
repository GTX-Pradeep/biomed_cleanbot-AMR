"""
train.py

Baseline YOLOv8n training script for the Campus Waste Robot project.
Trains on the TriCascade 80/20 split produced by prepare_data.py.

Run from the project root with your venv active:
    python train.py
"""

import os
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

PROJECT_ROOT = "/home/pradeep/campus_waste_robot"
DATA_YAML = os.path.join(PROJECT_ROOT, "data.yaml")

MODEL_WEIGHTS = "yolov8n.pt"
EPOCHS = 30
IMG_SIZE = 640
BATCH_SIZE = 4 
RUN_NAME = "tricascade_baseline_v1"


def main():
    if not os.path.isfile(DATA_YAML):
        print(f"[ERROR] data.yaml not found at: {DATA_YAML}")
        print("Make sure data.yaml is saved in the project root before running this script.")
        return

    print("=" * 70)
    print("Campus Waste Robot - YOLOv8n Baseline Training")
    print("=" * 70)
    print(f"  Data config : {DATA_YAML}")
    print(f"  Base model  : {MODEL_WEIGHTS}")
    print(f"  Epochs      : {EPOCHS}")
    print(f"  Image size  : {IMG_SIZE}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Run name    : {RUN_NAME}")
    print("=" * 70)

    # Load the pretrained YOLOv8 nano checkpoint
    model = YOLO(MODEL_WEIGHTS)

    # Train — ultralytics defaults to saving under runs/detect/<name>
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name=RUN_NAME,
        workers=2,
    )

    best_weights_path = os.path.join(
        PROJECT_ROOT, "runs", "detect", RUN_NAME, "weights", "best.pt"
    )
    print("-" * 70)
    print("[DONE] Training complete.")
    print(f"  Best weights saved to: {best_weights_path}")
    print(f"  Training plots/metrics in: runs/detect/{RUN_NAME}/")
    print("-" * 70)


if __name__ == "__main__":
    main()