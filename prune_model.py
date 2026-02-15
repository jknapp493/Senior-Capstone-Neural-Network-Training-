import torch
import torch.nn as nn
from ultralytics import YOLO

ORIGINAL_MODEL = "/home/jackson/yolov8ntraining_1/runs/finetune_force_gpu_maxutil_10-30/weights/last.pt"
PRUNE_RATIO = 0.30
OUTPUT_PATH = "/home/jackson/yolov8ntraining_1/11-15_pruned.pt"

print("Loading original YOLO model")
yolo = YOLO(ORIGINAL_MODEL)
model = yolo.model  # raw nn.Module

print("Collect BatchNorm gamma values")

bn_weights = []
bn_layers = []

for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        bn_weights.append(module.weight.data.abs().clone())
        bn_layers.append(module)

bn_weights = torch.cat(bn_weights)
num_prune = int(PRUNE_RATIO * bn_weights.numel())

print(f"Pruning {PRUNE_RATIO*100:.0f}%  => {num_prune} channels")

# find global threshold
threshold, _ = torch.kthvalue(bn_weights, num_prune)

def prune_conv_bn_pair(conv, bn, mask):
    idx = torch.where(mask)[0].long()

    # prune conv out channels
    conv.weight.data = conv.weight.data[idx, :, :, :].clone()
    if conv.bias is not None:
        conv.bias.data = conv.bias.data[idx].clone()

    # prune BN
    bn.weight.data = bn.weight.data[idx].clone()
    bn.bias.data = bn.bias.data[idx].clone()
    bn.running_mean = bn.running_mean[idx].clone()
    bn.running_var = bn.running_var[idx].clone()

    return idx


print("Applying mask pruning layer-by-layer")

mask_list = []
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        mask = module.weight.data.abs() > threshold
        # must keep minimum channels (avoid collapse)
        if mask.sum() < 2:
            _, top_idx = torch.topk(module.weight.data.abs(), 2)
            mask[top_idx] = True
        mask_list.append(mask)

# apply pruning
new_model = model.cpu()
bn_id = 0
for m in new_model.modules():
    if isinstance(m, nn.Sequential):
        # for YOLO-specific blocks
        pass
    if isinstance(m, nn.BatchNorm2d):
        mask = mask_list[bn_id]
        prev = list(new_model.modules())[list(new_model.modules()).index(m)-1]
        if isinstance(prev, nn.Conv2d):
            prune_conv_bn_pair(prev, m, mask)
        bn_id += 1

print("Weight pruning done")

# -------------------------------------------------
# 4) REBUILD ULTRALYTICS-COMPATIBLE CHECKPOINT
# -------------------------------------------------
print("Rebuilding")

ckpt = {
    "model": new_model.float(),  # must be FP32
    "ema": None,
    "updates": 0,
    "optimizer": None,
    "train_results": None,
    "epoch": -1,
}

torch.save(ckpt, OUTPUT_PATH)

print(f"model saved:\n{OUTPUT_PATH}")

