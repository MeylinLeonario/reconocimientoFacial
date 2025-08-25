import os
import cv2
import glob
import numpy as np
import face_recognition

# ----------------- CONFIG -----------------
DATASETS_DIR = "dataset"   # Carpeta con subcarpetas por persona
TOLERANCIA = 0.5            # 0.4-0.6 suele ir bien (más bajo = más estricto)
DOWNSCALE = 0.25            # Para acelerar (procesa a 1/4 de tamaño)
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ------------------------------------------

def cargar_encodings(datasets_dir):
    known_encodings = []
    known_names = []

    personas = [d for d in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir, d))]
    if not personas:
        raise RuntimeError(f"No se encontraron subcarpetas en '{datasets_dir}'.")

    for persona in personas:
        folder = os.path.join(datasets_dir, persona)
        rutas = glob.glob(os.path.join(folder, "*.*"))  # cualquier formato común

        for ruta in rutas:
            img_bgr = cv2.imread(ruta)
            if img_bgr is None:
                print(f"[AVISO] No pude leer: {ruta}")
                continue

            # 🔑 Forzar a 3 canales RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            boxes = face_recognition.face_locations(img_rgb, model="hog")
            if not boxes:
                print(f"[AVISO] Sin cara detectable en: {ruta}")
                continue

            encs = face_recognition.face_encodings(img_rgb, boxes)
            if encs:
                known_encodings.append(encs[0])
                known_names.append(persona)

    if not known_encodings:
        raise RuntimeError("No se generaron encodings. Revisa que las imágenes tengan caras claras.")
    return known_encodings, known_names

def main():
    print("[INFO] Cargando base de rostros…")
    known_encodings, known_names = cargar_encodings(DATASETS_DIR)
    print(f"[INFO] Cargados {len(known_encodings)} encodings de {len(set(known_names))} persona(s).")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    print("Presiona ESC para salir.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] La cámara no entregó un frame válido. Reintentando…")
            continue

        # Reducimos tamaño para acelerar
        small = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Localizar y codificar caras en el frame
        boxes = face_recognition.face_locations(rgb_small, model="hog")  # 'cnn' si tienes dlib con CUDA
        encs = face_recognition.face_encodings(rgb_small, boxes)

        names_in_frame = []
        for enc in encs:
            # Distancias a todos los conocidos
            dists = face_recognition.face_distance(known_encodings, enc)
            if len(dists) == 0:
                names_in_frame.append("Desconocido")
                continue

            best_idx = np.argmin(dists)
            best_dist = dists[best_idx]
            name = known_names[best_idx] if best_dist <= TOLERANCIA else "Desconocido"
            names_in_frame.append(name)

        # Dibujar cajas (escalamos de vuelta a tamaño original)
        for (top, right, bottom, left), name in zip(boxes, names_in_frame):
            top = int(top / DOWNSCALE)
            right = int(right / DOWNSCALE)
            bottom = int(bottom / DOWNSCALE)
            left = int(left / DOWNSCALE)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Etiqueta con fondo
            label = f"{name}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            cv2.rectangle(frame, (left, bottom), (left + tw + 6, bottom + th + 10), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, label, (left + 3, bottom + th + 3), FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Reconocimiento en vivo", frame)
        k = cv2.waitKey(1)
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
