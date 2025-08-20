import argparse
import glob
import os
import time
from datetime import datetime

import cv2
import numpy as np


def load_models(fd_model_path: str, fr_model_path: str):
    """
    Carga YuNet (detección) y SFace (reconocimiento) desde rutas .onnx.
    """
    if not os.path.exists(fd_model_path):
        raise FileNotFoundError(
            f"No encontré el modelo de detección: {fd_model_path}\n"
            "Descárgalo desde OpenCV Zoo (YuNet) y verifica la ruta."
        )
    if not os.path.exists(fr_model_path):
        raise FileNotFoundError(
            f"No encontré el modelo de reconocimiento: {fr_model_path}\n"
            "Descárgalo desde OpenCV Zoo (SFace) y verifica la ruta."
        )

    detector = cv2.FaceDetectorYN.create(
        fd_model_path, "", (320, 320), 0.9, 0.3, 5000
    )
    recognizer = cv2.FaceRecognizerSF.create(fr_model_path, "")

    if detector is None:
        raise RuntimeError("Falló la creación de FaceDetectorYN (revisa tu OpenCV).")
    if recognizer is None:
        raise RuntimeError("Falló la creación de FaceRecognizerSF (revisa tu OpenCV).")

    return detector, recognizer


def detect_faces(detector, img):
    """
    Corre YuNet y devuelve un arreglo Nx15 (o None si no hay caras).
    Formato: [x, y, w, h, 10 coords de 5 landmarks, score]
    """
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    result = detector.detect(img)
    faces = result[1] if isinstance(result, tuple) else result
    return faces


def pick_largest_face(faces):
    """
    Devuelve el índice de la cara con mayor área del bbox.
    """
    if faces is None or len(faces) == 0:
        return None
    areas = [(i, faces[i][2] * faces[i][3]) for i in range(faces.shape[0])]
    return max(areas, key=lambda t: t[1])[0]


def build_db_from_folder(detector, recognizer, dataset_dir: str):
    """
    Estructura esperada:
      dataset/
        PersonaA/*.jpg|png
        PersonaB/*.jpg|png
    Devuelve (names, features) donde features es (N, D).
    """
    people_dirs = [d for d in sorted(glob.glob(os.path.join(dataset_dir, "*")))
                   if os.path.isdir(d)]
    if not people_dirs:
        raise RuntimeError(f"No hay carpetas dentro de {dataset_dir}")

    names, feats = [], []
    for pdir in people_dirs:
        person = os.path.basename(pdir)
        imgs = sorted(glob.glob(os.path.join(pdir, "*.*")))
        person_feats = []

        for path in imgs:
            img = cv2.imread(path)
            if img is None:
                print(f"[WARN] No pude leer {path}")
                continue

            faces = detect_faces(detector, img)
            if faces is None or len(faces) == 0:
                print(f"[WARN] No se detectó rostro en {path}")
                continue

            idx = pick_largest_face(faces)
            aligned = recognizer.alignCrop(img, faces[idx])  # alinea con landmarks
            feat = recognizer.feature(aligned)               # embedding
            feat = np.asarray(feat, dtype=np.float32).reshape(-1)
            person_feats.append(feat)

        if len(person_feats) == 0:
            print(f"[WARN] {person}: sin embeddings (carpetas/fotos problemáticas).")
            continue

        mean_feat = np.mean(np.stack(person_feats, axis=0), axis=0)  # robustez
        names.append(person)
        feats.append(mean_feat)

        print(f"[OK] {person}: {len(person_feats)} fotos -> 1 embedding promedio.")

    if len(names) == 0:
        raise RuntimeError("No se generaron embeddings. Revisa tu dataset.")

    features = np.stack(feats, axis=0).astype(np.float32)
    return names, features


def save_db(names, features, out_path="faces_db.npz"):
    np.savez_compressed(out_path, names=np.array(names), features=features)
    print(f"[OK] Base guardada en {out_path} (personas={len(names)}, dim={features.shape[1]})")


def load_db(path="faces_db.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré la base {path}. Corre primero --mode enroll.")
    data = np.load(path, allow_pickle=True)
    names = list(data["names"])
    features = data["features"].astype(np.float32)
    return names, features


def match_feature(recognizer, feat, db_features, db_names, cosine_thresh: float):
    """
    Compara 'feat' contra toda la base usando similitud coseno de SFace.
    Retorna (best_name, best_score) o ("Unknown", score) si no supera el umbral.
    """
    best_name, best_score = "Unknown", -1.0
    for i in range(db_features.shape[0]):
        score = recognizer.match(feat, db_features[i], cv2.FaceRecognizerSF_FR_COSINE)
        if score > best_score:
            best_score = float(score)
            best_name = db_names[i]
    if best_score >= cosine_thresh:
        return best_name, best_score
    else:
        return "Unknown", best_score


def draw_face_box(img, face_row, label=None):
    x, y, w, h = face_row[:4].astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Landmarks: re, le, nt, rcm, lcm
    pts = face_row[4:14].reshape(-1, 2).astype(int)
    for (px, py) in pts:
        cv2.circle(img, (px, py), 2, (255, 0, 255), -1)
    if label:
        cv2.rectangle(img, (x, y - 22), (x + w, y), (0, 255, 0), -1)
        cv2.putText(img, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def run_realtime(detector, recognizer, db_path="faces_db.npz",
                 camera=0, downscale=1.0, log_csv=None, cosine_thresh=0.363):
    names, features = load_db(db_path)
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)  # CAP_DSHOW ayuda en Windows
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    print("Presiona ESC para salir.")
    last_log = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[WARN] Frame inválido, continuando…")
            continue

        frame_show = cv2.resize(frame, None, fx=downscale, fy=downscale,
                                interpolation=cv2.INTER_LINEAR) if downscale != 1.0 else frame

        faces = detect_faces(detector, frame_show)
        if faces is not None and len(faces) > 0:
            for i in range(faces.shape[0]):
                aligned = recognizer.alignCrop(frame_show, faces[i])
                feat = recognizer.feature(aligned)
                feat = np.asarray(feat, dtype=np.float32).reshape(-1)

                name, score = match_feature(recognizer, feat, features, names, cosine_thresh)

                label = f"{name} ({score:.3f})" if name != "Unknown" else f"Unknown ({score:.3f})"
                draw_face_box(frame_show, faces[i], label=label)

                # Log (simple): evita spamear muchas filas por la misma persona
                if log_csv and name != "Unknown":
                    tnow = time.time()
                    last_t = last_log.get(name, 0)
                    if tnow - last_t > 5:  # cada 5 s como mínimo
                        with open(log_csv, "a", encoding="utf-8") as f:
                            f.write(f"{datetime.now().isoformat(timespec='seconds')},{name},{score:.3f}\n")
                        last_log[name] = tnow

        cv2.imshow("YuNet + SFace (ESC para salir)", frame_show)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Reconocimiento facial con YuNet (detección) + SFace (reconocimiento)."
    )
    parser.add_argument("--mode", choices=["enroll", "run"], required=True,
                        help="enroll: crear base | run: reconocer por webcam")
    parser.add_argument("--dataset", type=str,
                        help="Carpeta del dataset (para --mode enroll)")
    parser.add_argument("--db", type=str, default="faces_db.npz",
                        help="Ruta del archivo .npz con embeddings")
    parser.add_argument("--fd_model", type=str, required=True,
                        help="Ruta al modelo YuNet .onnx (detección)")
    parser.add_argument("--fr_model", type=str, required=True,
                        help="Ruta al modelo SFace .onnx (reconocimiento)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Índice de cámara para --mode run")
    parser.add_argument("--downscale", type=float, default=1.0,
                        help="Factor de reducción de frame (ej. 0.5)")
    parser.add_argument("--log_csv", type=str, default=None,
                        help="Archivo CSV para registrar asistencias (opcional)")
    parser.add_argument("--cosine_thresh", type=float, default=0.363,
                        help="Umbral de similitud coseno (>= coincide)")
    args = parser.parse_args()

    detector, recognizer = load_models(args.fd_model, args.fr_model)

    if args.mode == "enroll":
        if not args.dataset:
            raise SystemExit("Debes pasar --dataset=carpeta_con_fotos")
        names, feats = build_db_from_folder(detector, recognizer, args.dataset)
        save_db(names, feats, args.db)

    elif args.mode == "run":
        run_realtime(detector, recognizer, db_path=args.db, camera=args.camera,
                     downscale=args.downscale, log_csv=args.log_csv,
                     cosine_thresh=args.cosine_thresh)


if __name__ == "__main__":
    main()
