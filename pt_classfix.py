import torch
from ultralytics import YOLO
import cv2

#PT_PATH = "/home/jackson/yolov8ntraining_1/runs/finetune_force_gpu_maxutil_10-30/weights/best_10-30.pt"
#OUT_PATH = "/home/jackson/yolov8ntraining_1/best_10-30_fixed_11-28.pt"

#PT_PATH = "/home/jackson/yolov8ntraining_1/yolov8s-obb.pt"
PT_PATH = "/home/jackson/yolov8ntraining_1/yolov8s-obb.pt"
OUT_PATH = "/home/jackson/yolov8ntraining_1/yolov8s-obb_12-04.pt"

NEW_NC = 7
NEW_NAMES = ['Battery', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

ckpt = torch.load(PT_PATH, map_location="cpu")

# Ultralytics keeps the model architecture YAML in:
# ckpt['model'].yaml
model = ckpt["model"]

print("Before edit — nc:", model.yaml.get("nc"))

model.yaml["nc"] = NEW_NC # mod yaml
model.yaml["names"] = NEW_NAMES

# also update ultralytics overrides if present
if "nc" in ckpt.get("overrides", {}):
    ckpt["overrides"]["nc"] = NEW_NC
    ckpt["overrides"]["names"] = NEW_NAMES

print("After edit — nc:", model.yaml.get("nc"))

torch.save(ckpt, OUT_PATH)

print("Saved updated checkpoint to:", OUT_PATH)
