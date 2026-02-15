#!/usr/bin/env python3
"""
prepare_via_to_yolo.py

Converts VIA-format CSV annotations stored in:
/home/jackson/yolov8ntraining_1/test_box_training/train/<ClassName>/*.csv
to a YOLOv8 dataset at:
/home/jackson/yolov8ntraining_1/dataset_yolo_detect

Annotated images are symlinked from the archive:
/home/jackson/yolov8ntraining_1/archive/garbage-dataset/<ClassName>/filename

Produces a train/val split and data.yaml.
"""

import csv, json, random, os, sys
from pathlib import Path
from PIL import Image

CLASS_NAMES = ["Battery", "Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"] # conf
CSV_ROOT = Path("/home/jackson/yolov8ntraining_1/test_box_training/train2")
ARCHIVE_ROOT = Path("/home/jackson/yolov8ntraining_1/archive/garbage-dataset-2")
OUT_ROOT = Path("/home/jackson/yolov8ntraining_1/dataset_yolo_detect_finetune_11-28")
VAL_RATIO = 0.10   # 10% validation
RANDOM_SEED = 42

IMG_TRAIN = OUT_ROOT / "images" / "train"
IMG_VAL = OUT_ROOT / "images" / "val"
LBL_TRAIN = OUT_ROOT / "labels" / "train"
LBL_VAL = OUT_ROOT / "labels" / "val"
for d in (IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL):
    d.mkdir(parents=True, exist_ok=True)

def class_folder_candidate(class_name): # folder retrieval
    candidates = [class_name, class_name.lower(), class_name.capitalize()]
    return candidates

def find_image_path(class_name, filename):
    for cand in class_folder_candidate(class_name):
        p = ARCHIVE_ROOT / cand / filename
        if p.exists():
            return p
    for p in ARCHIVE_ROOT.rglob(filename): # root recursive search 
        return p
    return None

# parse all CSVs and collect annotations per image
annotations = {}  # image_abs_path -> list of (class_id, x, y, w, h)
images_seen = []  # keep order for split

for cls in CLASS_NAMES:
    folder = CSV_ROOT / cls
    if not folder.exists():
        folder = CSV_ROOT / cls.lower()
    if not folder.exists():
        print(f"[WARN] CSV folder not found for class '{cls}': checked {CSV_ROOT / cls} and {CSV_ROOT / cls.lower()}")
        continue

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"[INFO] No CSVs in {folder}")
        continue
    for csv_file in csv_files:
        print(f"[INFO] Scanning CSV: {csv_file}")
        with open(csv_file, newline='', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                fname = row.get("filename") or row.get("image") or row.get("image_filename")
                if not fname:
                    continue
                # skip empty
                rc = row.get("region_count","0")
                try:
                    if int(rc) == 0:
                        continue
                except:
                    pass
                rsa = row.get("region_shape_attributes","")
                if not rsa or rsa.strip() in ("{}", ""):
                    continue
                # parse JSON in region_shape_attributes
                try:
                    shape = json.loads(rsa)
                except Exception as e:
                    print(f"[WARN] malformed shape JSON for {fname} in {csv_file}: {e}")
                    continue
                # require rectangle
                if shape.get("name") not in ("rect","rectangle"):
                    continue
                # find image path in archive
                img_path = find_image_path(cls, fname)
                if img_path is None:
                    print(f"[WARN] image {fname} not found in archive for class {cls}. skipping.")
                    continue
                # load image size
                try:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                except Exception as e:
                    print(f"[WARN] cannot open image {img_path}: {e}")
                    continue
                # read bbox in pixels (VIA uses x,y,width,height)
                try:
                    x = float(shape["x"])
                    y = float(shape["y"])
                    w = float(shape["width"])
                    h = float(shape["height"])
                except Exception as e:
                    print(f"[WARN] invalid bbox for {fname}: {e}")
                    continue
                # normalize to YOLO xywh (center)
                xc = (x + w/2.0) / img_w
                yc = (y + h/2.0) / img_h
                wn = w / img_w
                hn = h / img_h
                class_id = CLASS_NAMES.index(cls)
                img_key = str(img_path.resolve())
                if img_key not in annotations:
                    annotations[img_key] = []
                    images_seen.append(img_key)
                annotations[img_key].append((class_id, xc, yc, wn, hn))

print(f"[INFO] Found {len(annotations)} images with annotations across classes.")

# split into train/val
random.seed(RANDOM_SEED)
random.shuffle(images_seen)
n_val = max(1, int(len(images_seen) * VAL_RATIO)) if images_seen else 0
val_set = set(images_seen[:n_val])
train_set = set(images_seen[n_val:])
print(f"[INFO] Train images: {len(train_set)}, Val images: {len(val_set)}")

# helper: write label file and symlink image
def ensure_symlink(src_path: Path, dst_path: Path):
    try:
        if dst_path.exists():
            return
        dst_path.symlink_to(src_path)
    except Exception:
        # fallback to copy if symlink not allowed
        import shutil
        shutil.copy2(src_path, dst_path)

# write files
for img_abs in images_seen:
    boxes = annotations[img_abs]
    src = Path(img_abs)
    basename = src.name
    stem = src.stem
    if img_abs in val_set:
        dst_img = IMG_VAL / basename
        dst_lbl = LBL_VAL / (stem + ".txt")
    else:
        dst_img = IMG_TRAIN / basename
        dst_lbl = LBL_TRAIN / (stem + ".txt")
    # symlink/copy image
    ensure_symlink(src, dst_img)
    # write labels (overwrite if exists)
    with open(dst_lbl, "w") as lf:
        for (cid, xc, yc, wn, hn) in boxes:
            lf.write(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

print(f"[INFO] Wrote labels and symlinks under {OUT_ROOT}")

# create data.yaml
yaml_path = OUT_ROOT / "data.yaml"
names_list = CLASS_NAMES
yaml_text = f"""path: {OUT_ROOT}
train: images/train
val: images/val
nc: {len(names_list)}
names: {names_list}
"""
yaml_path.write_text(yaml_text)
print(f"[INFO] Wrote data.yaml at {yaml_path}")

print("data.yaml in", OUT_ROOT)

