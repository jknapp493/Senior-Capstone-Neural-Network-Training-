#!/usr/bin/env python3
"""
Fine-tune YOLOv8 using TWO merged datasets.
Works with:
  - dataset_yolo_detect_finetune_10-25
  - dataset_yolo_detect_finetune_11-28
Ensures all paths exist and combined YAML is valid.
"""

import os
import torch
import subprocess
import gc
from ultralytics import YOLO
import yaml

# ----------------------------------------
# CONFIG — YOUR TWO DATASETS
# ----------------------------------------
DATASET1 = "/home/jackson/yolov8ntraining_1/dataset_yolo_detect_finetune_10-25"
DATASET2 = "/home/jackson/yolov8ntraining_1/dataset_yolo_detect_finetune_11-28"

COMBINED_YAML = "/home/jackson/yolov8ntraining_1/combined_finetune_data_11-29.yaml"

#CHKPT = "/home/jackson/yolov8ntraining_1/runs/finetune_force_gpu_maxutil_11-28_REFINED/weights/best.pt"
CHKPT = "/home/jackson/yolov8ntraining_1/yolov8m_12-04.pt"
FALLBACK = "yolov8m.pt"

EPOCHS = 150 # Hyperparams 
IMGSZ = 640
BATCH = 10
WORKERS = 8
CACHE = True
AMP = False
PROJECT = "/home/jackson/yolov8ntraining_1"
NAME = "12-04_combinedtestmodel"

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

print("🔍 Loading dataset YAML files...") #Parse dual yamls

y1 = load_yaml(os.path.join(DATASET1, "data.yaml"))
y2 = load_yaml(os.path.join(DATASET2, "data.yaml"))

print("YAML 1 classes:", y1["names"])
print("YAML 2 classes:", y2["names"])

if y1["names"] != y2["names"]:
    raise ValueError("❌ ERROR: Dataset1 and Dataset2 class name lists differ!")

nc = len(y1["names"])

required_dirs = [ # Path verify
    os.path.join(DATASET1, "images/train"),
    os.path.join(DATASET1, "images/val"),
    os.path.join(DATASET2, "images/train"),
    os.path.join(DATASET2, "images/val"),
]

for d in required_dirs:
    if not os.path.isdir(d):
        raise RuntimeError(f"❌ Missing directory: {d}")
    else:
        print(f"Found {d}")

combined = { # Build new yaml
    "train": [
        os.path.join(DATASET1, "images/train"),
        os.path.join(DATASET2, "images/train"),
    ],
    "val": [
        os.path.join(DATASET1, "images/val"),
        os.path.join(DATASET2, "images/val"),
    ],
    "nc": nc,
    "names": y1["names"],
}

with open(COMBINED_YAML, "w") as f:
    yaml.dump(combined, f)

print(f"Combined YAML saved to {COMBINED_YAML}")
print("Combined YAML Contents")
print(yaml.dump(combined))

if not torch.cuda.is_available(): # GPU 
    raise SystemError("❌ CUDA not available!")

device_name = torch.cuda.get_device_name(0)
print(f"GPU: {device_name}")
print(f"CUDA: {torch.version.cuda}")

DEVICE = torch.device("cuda:0")
torch.cuda.empty_cache()
gc.collect()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print("⚡ Preallocating VRAM...")
dummy = torch.randn(8, 3, IMGSZ, IMGSZ, device=DEVICE)
del dummy
torch.cuda.empty_cache()

model_path = CHKPT if os.path.exists(CHKPT) else FALLBACK # Model Loading
print(f"📦 Loading model: {model_path}")
model = YOLO(model_path)
model.to(DEVICE)

#if model.model.nc != nc: # Class check override
#    raise ValueError(
#        f"❌ Class mismatch: model={model.model.nc}, dataset={nc}"
#    )

print("\n🚀 Starting training...\n")

try: # Expected hyperparams 
    results = model.train(
        data=COMBINED_YAML,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        lr0=0.0003,
        patience=0,
        batch=BATCH,
        device="cuda:0",
        workers=WORKERS,
        cache=CACHE,
        amp=AMP,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        pretrained=True,
        resume=False
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU OOM — reducing batch size")
        torch.cuda.empty_cache()
        results = model.train(
            data=COMBINED_YAML,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            lr0=0.0001,
            batch=max(1, BATCH // 2),
            device="cuda:0",
            workers=WORKERS,
            cache=CACHE,
            amp=AMP,
            project=PROJECT,
            name=NAME + "_lowmem",
            exist_ok=True,
            pretrained=True,
            resume=False,
        )
    else:
        raise

print("\n Training Completed")
print(f"Output at: {os.path.join(PROJECT, NAME)}")

