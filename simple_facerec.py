import os
import cv2
import numpy as np
print(np.__version__)

import face_recognition
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # permite archivos parcialmente truncados

def to_rgb_uint8(image):
    """
    Asegura RGB uint8 contiguo.
    - Acepta BGR (OpenCV), RGB, RGBA, escala de grises, 16-bit, etc.
    - Devuelve np.ndarray shape (H,W,3), dtype=uint8, C-contiguous.
    """
    if image is None:
        raise ValueError("Imagen vacía (None)")

    # Si es ruta, ábrela con PIL
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            return np.ascontiguousarray(arr)

    arr = np.asarray(image)

    # Si viene en float/16-bit -> escala a 8-bit
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            # normaliza por rango y castea
            arr = (arr.astype(np.float32) / arr.max() * 255.0).astype(np.uint8)

    # Manejo de dimensiones/canales
    if arr.ndim == 2:  # gris
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:  # RGBA/BGRA
            # Si viene de OpenCV probablemente es BGRA
            try:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            except cv2.error:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.shape[2] == 3:
            # ¿BGR o RGB? Si viene de OpenCV: BGR->RGB
            # Heurística: si lo llama detect_known_faces pasará BGR
            pass
        else:
            raise ValueError(f"Canales no soportados: {arr.shape}")

    # Garantiza contigüidad
    arr = np.ascontiguousarray(arr)
    return arr

class SimpleFacerec:
    def __init__(self, frame_resizing: float = 0.25, tolerance: float = 0.6, model: str = "hog"):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing
        self.tolerance = tolerance
        self.model = model  # 'hog' o 'cnn'

    def load_encoding_images(self, images_path: str):
        self.known_face_encodings.clear()
        self.known_face_names.clear()

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [f for f in os.listdir(images_path) if f.lower().endswith(valid_exts)]
        files.sort()
        if not files:
            print(f"[SimpleFacerec] No se encontraron imágenes en: {images_path}")
            return

        for filename in files:
            path = os.path.join(images_path, filename)
            try:
                # Fuerza RGB8
                img_rgb = to_rgb_uint8(path)

                #DEBUG 1
                print(f"[DEBUG] {filename} -> shape={img_rgb.shape}, dtype={img_rgb.dtype}, "
                  f"contiguous={img_rgb.flags['C_CONTIGUOUS']}")
                
                # (opcional) baja tamaño si es gigante
                h, w = img_rgb.shape[:2]
                if max(h, w) > 1600:
                    scale = 1600.0 / max(h, w)
                    img_rgb = cv2.resize(img_rgb, (int(w*scale), int(h*scale)))

                boxes = face_recognition.face_locations(img_rgb, model=self.model)
                if not boxes:
                    print(f"[SimpleFacerec] Sin cara detectable en: {filename}")
                    continue

                encs = face_recognition.face_encodings(img_rgb, boxes)
                if not encs:
                    print(f"[SimpleFacerec] No se pudo codificar: {filename}")
                    continue

                self.known_face_encodings.append(encs[0])
                self.known_face_names.append(os.path.splitext(filename)[0])
                print(f"[SimpleFacerec] OK: {filename}")

            except Exception as e:
                print(f"[SimpleFacerec] ERROR con '{filename}': {e}")

        print(f"[SimpleFacerec] Cargados: {self.known_face_names}")

    def detect_known_faces(self, frame_bgr):
        # 1) Valida frame de cámara
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Frame de cámara vacío (ret=False o MSMF devolvió frame inválido)")

        if frame_bgr.dtype != np.uint8:
            frame_bgr = (frame_bgr.astype(np.float32).clip(0, 255)).astype(np.uint8)

        # 2) Downscale para velocidad
        small = cv2.resize(frame_bgr, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)

        # 3) BGR -> RGB uint8 contiguo
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb_small = np.ascontiguousarray(rgb_small)

        # 4) face_recognition
        locations_small = face_recognition.face_locations(rgb_small, model=self.model)
        encodings = face_recognition.face_encodings(rgb_small, locations_small)

        names = []
        for enc in encodings:
            if not self.known_face_encodings:
                names.append("Unknown")
                continue
            dists = face_recognition.face_distance(self.known_face_encodings, enc)
            best_idx = int(np.argmin(dists))
            names.append(self.known_face_names[best_idx] if dists[best_idx] <= self.tolerance else "Unknown")

        # 5) Reescala cajas
        scale = 1.0 / self.frame_resizing
        locations = []
        for (top, right, bottom, left) in locations_small:
            locations.append((int(top*scale), int(right*scale), int(bottom*scale), int(left*scale)))

        return locations, names
