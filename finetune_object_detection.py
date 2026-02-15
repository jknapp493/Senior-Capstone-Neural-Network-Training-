#!/usr/bin/env python3
"""
finetune_force_gpu_fixed.py
Fine-tune YOLOv8 on GTX1070 using GPU (CUDA forced) — with class safety & VRAM checks.
"""
#!/usr/bin/env python3
"""
finetune_force_gpu_maxutil.py
Fine-tune YOLOv8 on GTX1070 — optimized for higher GPU utilization.
"""

import os
import torch
import subprocess
import gc
from ultralytics import YOLO

DATA_YAML = "/home/jackson/yolov8ntraining_1/dataset_yolo_detect_finetune_10-25/data.yaml"
CHKPT = "/home/jackson/yolov8ntraining_1/yolov8s_12-01.pt"#yolov8s model
#CHKPT = "/home/jackson/yolov8ntraining_1/best_10-30_fixed_11-28.pt"  # pretrained model
FALLBACK = "yolov8n.pt"

EPOCHS = 250
IMGSZ = 960              # slightly higher resolution
BATCH = 8                 
WORKERS = 8
CACHE = True
AMP = False               # full precision therefore more VRAM usage
PROJECT = "/home/jackson/yolov8ntraining_1/runs"
NAME = "12-01_Refine"

if not torch.cuda.is_available():
    raise SystemError("❌ CUDA not available!")

device_name = torch.cuda.get_device_name(0)
print(f"Using GPU: {device_name}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: cuda:0")

DEVICE = torch.device("cuda:0")
torch.cuda.empty_cache()
gc.collect()

# cuDNN tuning
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Warm-up to allocate memory early
print("Preallocating VRAM with dummy tensor...")
dummy = torch.randn(12, 3, IMGSZ, IMGSZ, device=DEVICE)
del dummy
torch.cuda.empty_cache()
print("Warm-up done.")

# Show VRAM snapshot
try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    print("nvidia-smi not available.")

model_path = CHKPT if os.path.exists(CHKPT) else FALLBACK
print(f"Loading model from: {model_path}")
model = YOLO(model_path)
model.to(DEVICE)

data_nc = None
with open(DATA_YAML, "r") as f:
    for line in f:
        if "nc:" in line:
            data_nc = int(line.split(":")[1].strip())
            break

if data_nc is None:
    raise ValueError("❌ Could not find 'nc:' entry in data.yaml")

#if model.model.nc != data_nc:
#    raise ValueError(
#        f"Class mismatch: model has {model.model.nc} but data.yaml defines {data_nc}."
#    )
#print(f"Class count check passed ({data_nc} classes)")


print("\nStarting fine-tuning with maximum GPU usage...\n")

try:
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        patience=0, #No time out from learning 
        lr0=0.0002, #0.0003, reduce learning rate to be more smooth over many epochs 
        imgsz=IMGSZ,
        batch=BATCH,
        device="cuda:0",
        workers=WORKERS,
        cache=CACHE,
        amp=AMP,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        pretrained=True,
        resume=False,
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        print("Out of memory — reducing batch size and retrying")
        torch.cuda.empty_cache()
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            patience=0, #No time out from learning 
            lr0= 0.0001, #0.0003, reduce learning rate to be more smooth over many epochs 
            imgsz=IMGSZ,
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

print("\nFinished fine-tuning successfully!")
print(f"Results saved under: {os.path.join(PROJECT, NAME)}")

print("\nGPU utilization (after training):") # GPU Evaluation
try:
    subprocess.run(["nvidia-smi"], check=False)
except FileNotFoundError:
    pass

