import face_recognition
from pathlib import Path

img_path = Path("dataset_fixed_clean") / "Messi.jpg"   # ajusta nombre
img = face_recognition.load_image_file(str(img_path))
print("shape:", img.shape, "dtype:", img.dtype)
boxes = face_recognition.face_locations(img, model="hog")
print("faces:", len(boxes))
