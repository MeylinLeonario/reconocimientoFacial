import os
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageFile

# Permite cargar JPG/PNG parcialmente truncados
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------- CONFIG -----------------
RUTA_DATASET = "dataset"     # dataset/<NombrePersona>/*.jpg|png
TOLERANCIA = 0.5             # 0.4-0.6 suele ir bien (más bajo = más estricto)
DOWNSCALE = 0.25             # acelera el procesamiento (1/4 del tamaño)
MODELO_LOC = "hog"           # 'hog' (CPU) o 'cnn' (si tienes dlib con CUDA)
# ------------------------------------------

def leer_imagen_simple_y_segura(ruta_img: str):
    """
    Método ultra simple para leer imágenes que funciona 100% con face_recognition
    """
    try:
        # Método 1: Solo OpenCV, más directo
        img = cv2.imread(ruta_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"No se pudo leer: {ruta_img}")
        
        # Convertir BGR a RGB y asegurar uint8
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
        
        return img_rgb
        
    except Exception:
        # Método 2: PIL como backup
        with Image.open(ruta_img) as pil_img:
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            img_array = np.array(pil_img, dtype=np.uint8)
            return np.ascontiguousarray(img_array)

def redimensionar_imagen_si_es_muy_grande(img, max_size=1000):
    """
    Redimensiona la imagen si es muy grande para evitar problemas con dlib
    """
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        # Calcular nuevo tamaño manteniendo aspecto
        if h > w:
            new_h, new_w = max_size, int(w * max_size / h)
        else:
            new_h, new_w = int(h * max_size / w), max_size
        
        print(f"[DEBUG] Redimensionando de {w}x{h} a {new_w}x{new_h}")
        img = cv2.resize(img, (new_w, new_h))
    
    return img

# ----- Utilidades de asistencia -----
def ruta_csv_hoy():
    fecha = datetime.now().strftime("%Y-%m-%d")
    return Path(f"asistencia_{fecha}.csv")

def cargar_ya_marcados_hoy():
    ya = set()
    p = ruta_csv_hoy()
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            for fila in r:
                if not fila:
                    continue
                ya.add(fila[0])
    return ya

def registrar_asistencia(nombre):
    p = ruta_csv_hoy()
    ya = cargar_ya_marcados_hoy()
    if nombre in ya or nombre == "Desconocido":
        return
    hora = datetime.now().strftime("%H:%M:%S")
    with open(p, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([nombre, hora])
    print(f"[ASISTENCIA] {nombre} @ {hora}")

# ----- Función de face_recognition MUY robusta -----
def detectar_rostros_seguro(img_rgb, modelo="hog"):
    """
    Detecta rostros con múltiples intentos y configuraciones
    """
    # Asegurar que la imagen esté en el formato correcto
    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    
    # Redimensionar si es muy grande
    img_rgb = redimensionar_imagen_si_es_muy_grande(img_rgb, max_size=800)
    
    print(f"[DEBUG] Detectando rostros en imagen: {img_rgb.shape}, dtype: {img_rgb.dtype}")
    print(f"[DEBUG] Rango de valores: [{img_rgb.min()}, {img_rgb.max()}]")
    
    # Intentar con diferentes configuraciones
    configuraciones = [
        {"model": modelo, "number_of_times_to_upsample": 0},
        {"model": "hog", "number_of_times_to_upsample": 0},
        {"model": "hog", "number_of_times_to_upsample": 1},
    ]
    
    for i, config in enumerate(configuraciones):
        try:
            print(f"[DEBUG] Intento {i+1} con configuración: {config}")
            locs = face_recognition.face_locations(img_rgb, **config)
            print(f"[DEBUG] ✓ Éxito! Rostros encontrados: {len(locs)}")
            return locs, img_rgb
        except Exception as e:
            print(f"[DEBUG] ✗ Intento {i+1} falló: {e}")
            continue
    
    # Si todo falla, intentar con imagen más pequeña
    try:
        print("[DEBUG] Intentando con imagen muy pequeña...")
        img_tiny = cv2.resize(img_rgb, (200, 200))
        img_tiny = np.ascontiguousarray(img_tiny, dtype=np.uint8)
        locs = face_recognition.face_locations(img_tiny, model="hog", number_of_times_to_upsample=0)
        # Reescalar coordenadas
        h_orig, w_orig = img_rgb.shape[:2]
        locs_scaled = []
        for top, right, bottom, left in locs:
            top = int(top * h_orig / 200)
            right = int(right * w_orig / 200)
            bottom = int(bottom * h_orig / 200)
            left = int(left * w_orig / 200)
            locs_scaled.append((top, right, bottom, left))
        print(f"[DEBUG] ✓ Éxito con imagen pequeña! Rostros: {len(locs_scaled)}")
        return locs_scaled, img_rgb
    except Exception as e:
        print(f"[DEBUG] ✗ También falló con imagen pequeña: {e}")
    
    print("[ERROR] TODOS los métodos de detección fallaron")
    return [], img_rgb

def generar_encodings_seguro(img_rgb, locs):
    """
    Genera encodings con manejo robusto de errores
    """
    try:
        encs = face_recognition.face_encodings(img_rgb, locs)
        return encs
    except Exception as e:
        print(f"[ERROR] face_encodings falló: {e}")
        # Intentar con imagen más pequeña
        try:
            img_small = redimensionar_imagen_si_es_muy_grande(img_rgb, max_size=500)
            # Reescalar localizaciones
            h_orig, w_orig = img_rgb.shape[:2]
            h_small, w_small = img_small.shape[:2]
            
            locs_scaled = []
            for top, right, bottom, left in locs:
                top = int(top * h_small / h_orig)
                right = int(right * w_small / w_orig)
                bottom = int(bottom * h_small / h_orig)
                left = int(left * w_small / w_orig)
                locs_scaled.append((top, right, bottom, left))
            
            encs = face_recognition.face_encodings(img_small, locs_scaled)
            return encs
        except Exception as e2:
            print(f"[ERROR] También falló con imagen pequeña: {e2}")
            return []

# ----- Cargar dataset con máxima robustez -----
def cargar_encodings(ruta_dataset):
    nombres_conocidos = []
    encodings_conocidos = []
    extensiones = (".jpg", ".jpeg", ".png", ".bmp")
    
    total_archivos = 0
    archivos_procesados = 0
    
    print(f"[INFO] Escaneando dataset en: {ruta_dataset}")
    
    for root, dirs, files in os.walk(ruta_dataset):
        partes = Path(root).parts
        try:
            idx = partes.index(Path(ruta_dataset).parts[-1])
            persona = partes[idx+1] if idx+1 < len(partes) else None
        except ValueError:
            persona = None
        if persona is None:
            continue

        archivos_validos = [f for f in files if f.lower().endswith(extensiones)]
        total_archivos += len(archivos_validos)
        
        for archivo in archivos_validos:
            ruta_img = os.path.join(root, archivo)
            print(f"\n{'='*50}")
            print(f"[{archivos_procesados + 1}/{total_archivos}] PROCESANDO: {persona}/{archivo}")
            print(f"{'='*50}")

            try:
                # 1. Leer imagen
                print("[1/4] Leyendo imagen...")
                img_rgb = leer_imagen_simple_y_segura(ruta_img)
                print(f"[✓] Imagen leída: {img_rgb.shape}")
                
                # 2. Detectar rostros
                print("[2/4] Detectando rostros...")
                locs, img_procesada = detectar_rostros_seguro(img_rgb, MODELO_LOC)
                
                if not locs:
                    print("[✗] Sin rostros detectados")
                    continue
                
                print(f"[✓] {len(locs)} rostro(s) detectado(s)")
                
                # 3. Generar encodings
                print("[3/4] Generando encodings...")
                encs = generar_encodings_seguro(img_procesada, locs)
                
                if not encs:
                    print("[✗] Sin encodings generados")
                    continue
                
                print(f"[✓] {len(encs)} encoding(s) generado(s)")
                
                # 4. Guardar encodings
                print("[4/4] Guardando encodings...")
                for enc in encs:
                    encodings_conocidos.append(enc)
                    nombres_conocidos.append(persona)
                
                archivos_procesados += 1
                print(f"[✓] ÉXITO: {persona} procesado correctamente")

            except Exception as e:
                print(f"[✗] ERROR CRÍTICO: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n[RESUMEN FINAL]")
    print(f"Archivos encontrados: {total_archivos}")
    print(f"Archivos procesados exitosamente: {archivos_procesados}")
    print(f"Encodings generados: {len(encodings_conocidos)}")
    print(f"Personas únicas: {len(set(nombres_conocidos))}")
    
    return np.array(encodings_conocidos), np.array(nombres_conocidos)

def main():
    print("="*60)
    print("SISTEMA DE RECONOCIMIENTO FACIAL - VERSIÓN ROBUSTA")
    print("="*60)
    
    # Verificar que el dataset existe
    if not os.path.exists(RUTA_DATASET):
        raise RuntimeError(f"La carpeta {RUTA_DATASET} no existe")
    
    print(f"[INFO] Cargando encodings desde: {RUTA_DATASET}")
    encodings_conocidos, nombres_conocidos = cargar_encodings(RUTA_DATASET)
    
    if len(encodings_conocidos) == 0:
        raise RuntimeError("No se generaron encodings. Revisa las imágenes del dataset.")

    print("\n" + "="*60)
    print("INICIANDO RECONOCIMIENTO EN TIEMPO REAL")
    print("="*60)
    print("Presiona 'q' para salir")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    ya_marcados = cargar_ya_marcados_hoy()
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        
        # Procesar solo cada N frames para mejor rendimiento
        if frame_count % 3 != 0:  # procesar cada 3 frames
            cv2.imshow("Asistencia - Reconocimiento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        try:
            # Redimensionar para acelerar
            small = cv2.resize(frame, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb_small = np.ascontiguousarray(rgb_small, dtype=np.uint8)

            # Detectar rostros en frame
            locs, _ = detectar_rostros_seguro(rgb_small, "hog")
            
            if locs:
                encs = generar_encodings_seguro(rgb_small, locs)
                
                for enc, (top, right, bottom, left) in zip(encs, locs):
                    if enc is None or len(enc) == 0:
                        continue
                        
                    nombre = "Desconocido"
                    if len(encodings_conocidos) > 0:
                        dists = face_recognition.face_distance(encodings_conocidos, enc)
                        if len(dists) > 0:
                            idx_min = np.argmin(dists)
                            if dists[idx_min] <= TOLERANCIA:
                                nombre = nombres_conocidos[idx_min]

                    # Reescalar coordenadas
                    top = int(top / DOWNSCALE)
                    right = int(right / DOWNSCALE)
                    bottom = int(bottom / DOWNSCALE)
                    left = int(left / DOWNSCALE)

                    # Dibujar rectángulo y nombre
                    color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    cv2.putText(frame, nombre, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Registrar asistencia
                    if nombre != "Desconocido" and nombre not in ya_marcados:
                        registrar_asistencia(nombre)
                        ya_marcados.add(nombre)

        except Exception as e:
            print(f"[ERROR] Frame {frame_count}: {e}")

        cv2.imshow("Asistencia - Reconocimiento Facial", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Sistema finalizado correctamente.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] {e}")
        import traceback
        traceback.print_exc()