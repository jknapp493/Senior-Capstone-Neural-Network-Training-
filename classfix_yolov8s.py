import torch
from ultralytics import YOLO

# load the checkpoint (weights_only=False so that the entire model is loaded)
ckpt = torch.load("yolov8s.pt", map_location="cpu")
model = ckpt['model']          # DetectionModel
model.yaml["nc"] = 7           # new number of classes
model.yaml["names"] = ['Battery','Cardboard','Glass','Metal','Paper','Plastic','Trash']

# update the detect layer’s nc and rebuild its internal buffers
detect = model.model[-1]       # Detect head
detect.nc = 7
detect.no = detect.nc + 5      # (5 = x,y,w,h,objectness)
detect.initialize_biases()     # reinitialise biases for the new number of classes

# store updated model back into checkpoint and save
ckpt['model'] = model
torch.save(ckpt, "yolov8s_custom.pt")
print("[INFO] Saved custom model with 7 classes.")

