import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec(frame_resizing=0.25, tolerance=0.6, model="hog")
sfr.load_encoding_images("dataset_fixed_clean/")  # usa el folder normalizado


if not sfr.known_face_encodings:
    raise RuntimeError("No hay encodings cargados. Revisa que las imágenes tengan una cara detectable.")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

print("Presiona ESC para salir.")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[WARN] La cámara no entregó un frame válido (MSMF). Reintentando…")
        continue  # o haz break si pasa demasiado

    locations, names = sfr.detect_known_faces(frame)

    for (t, r, b, l), name in zip(locations, names):
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, name, (l, max(0, t-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

