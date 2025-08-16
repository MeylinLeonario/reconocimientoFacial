from PIL import Image, ImageFile
import os, pathlib
ImageFile.LOAD_TRUNCATED_IMAGES = True

INPUT_DIR = "dataset"
MID_DIR   = "dataset_fixed"
OUT_DIR   = "dataset_fixed_clean"

os.makedirs(MID_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# 1) dataset -> dataset_fixed (RGB 8-bit)
for f in os.listdir(INPUT_DIR):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        path_in  = os.path.join(INPUT_DIR, f)
        path_out = os.path.join(MID_DIR,  os.path.splitext(f)[0] + ".jpg")
        with Image.open(path_in) as im:
            im.convert("RGB").save(path_out, "JPEG", quality=95)
        print("✔ (1) ->", f)

# 2) dataset_fixed -> dataset_fixed_clean (garantiza que no quedó nada raro)
for f in os.listdir(MID_DIR):
    if f.lower().endswith((".jpg", ".jpeg", ".png")):
        path_in  = os.path.join(MID_DIR, f)
        path_out = os.path.join(OUT_DIR, os.path.splitext(f)[0] + ".jpg")
        with Image.open(path_in) as im:
            im.convert("RGB").save(path_out, "JPEG", quality=95)
        print("✔ (2) ->", f)

print("Listo. Usa carpeta:", OUT_DIR)
