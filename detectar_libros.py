from ultralytics import YOLO
import cv2
import os
import time
import csv
import pandas as pd
from datetime import datetime

# =========================================================
# INICIO
# =========================================================

# =========================================================
# CONFIGURACIÓN DEL SISTEMA
# =========================================================

MODELO = "modelos/yolov8s.pt"

CARPETA_CAPTURAS = "capturas"
CARPETA_RESULTADOS = "resultados"
CARPETA_IMAGENES_DETECTADAS = "resultados/imagenes_detectadas"
CARPETA_CSV = "resultados/csv"
CARPETA_EXCEL = "resultados/excel"

ARCHIVO_REALES = "reales.csv"

CONFIDENCE = 0.40
CLASE_OBJETIVO = "book"

os.makedirs(CARPETA_CAPTURAS, exist_ok=True)
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
os.makedirs(CARPETA_IMAGENES_DETECTADAS, exist_ok=True)
os.makedirs(CARPETA_CSV, exist_ok=True)
os.makedirs(CARPETA_EXCEL, exist_ok=True)

# =========================================================
# CARGA DEL MODELO YOLOv8
# =========================================================

modelo = YOLO(MODELO)

# =========================================================
# CARGA DE DATOS REALES (CON VALIDACIÓN DE CELDAS VACÍAS)
# =========================================================

datos_reales = {}

if os.path.exists(ARCHIVO_REALES):
    with open(ARCHIVO_REALES, newline="", encoding="utf-8-sig") as archivo:
        lector = csv.DictReader(archivo)
        for fila in lector:
            # Validamos que la fila no esté vacía antes de meterla al diccionario
            if fila.get("imagen") and fila.get("libros_reales") and fila["libros_reales"].strip() != "":
                datos_reales[fila["imagen"]] = int(fila["libros_reales"])

# =========================================================
# VARIABLES GENERALES
# =========================================================

resultados_tabla = []

total_reales = 0
total_detectados = 0
total_correctos_estimados = 0
total_falsos_positivos = 0
total_falsos_negativos = 0

contador_capturas = 1

historial_conteos = []

# =========================================================
# CAPTURA DE IMAGEN DEL ESTANTE
# (LECTURA DESDE CÁMARA)
# =========================================================

camara = cv2.VideoCapture(2)

if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("\n===== SISTEMA DE DETECCIÓN DE LIBROS =====")
print("Presiona ESPACIO para capturar una imagen.")
print("Presiona ESC para salir.")
print("Los resultados se guardarán en la carpeta resultados.\n")

while True:
    inicio_frame = time.time()

    ret, frame = camara.read()

    if not ret:
        print("No se pudo leer la imagen de la cámara.")
        break

    frame_mostrado = frame.copy()

    # =========================================================
    # PREPROCESAMIENTO DE IMAGEN
    # =========================================================

    frame_redimensionado = cv2.resize(frame_mostrado, (640, 480))

    # =========================================================
    # ANÁLISIS DE IMAGEN CON YOLOv8
    # =========================================================

    resultados = modelo.predict(
        source=frame_redimensionado,
        conf=CONFIDENCE,
        iou=0.3,
        verbose=False
    )

    resultado = resultados[0]

    # =========================================================
    # ¿SE DETECTARON LIBROS?
    # GENERAR RECUADROS DE DETECCIÓN
    # CONTAR LIBROS DETECTADOS
    # =========================================================

    libros_detectados = 0 # Inicializamos el conteo para este frame

    for box in resultado.boxes:
        clase_id = int(box.cls[0])
        nombre_clase = modelo.names[clase_id]

        # 1. ¿Es un libro? (Clase correcta)
        if nombre_clase == CLASE_OBJETIVO:
            
            # Extraemos coordenadas para analizar la forma de la caja
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            alto = y2 - y1
            ancho = x2 - x1

            
            if ancho >= (alto * 0.7):
                continue # Se salta este objeto, no lo cuenta NI LO DIBUJA

            # 3. Si pasó los filtros: ES UN LIBRO VERTICAL VÁLIDO
            libros_detectados += 1 # Incrementar el conteo
            
            confianza = float(box.conf[0])

            # --- DIBUJAR EL RECUADRO ROJO ---
            cv2.rectangle(
                frame_redimensionado,
                (x1, y1),
                (x2, y2),
                (0, 0, 255), # Rojo BGR
                2
            )

            # --- DIBUJAR EL TEXTO DE CONFIANZA ---
            cv2.putText(
                frame_redimensionado,
                f"{confianza:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, # Tamaño más chico para que no se encime tanto
                (0, 0, 255), # Rojo BGR
                2
            )

    fin_frame = time.time()
    tiempo_procesamiento = fin_frame - inicio_frame
    fps = 1 / tiempo_procesamiento if tiempo_procesamiento > 0 else 0

    # =========================================================
    # INTERFAZ VISUAL SIMPLE
    # =========================================================

    cv2.rectangle(frame_redimensionado, (0, 0), (640, 90), (0, 0, 0), -1)

    historial_conteos.append(libros_detectados)
    if len(historial_conteos) > 12:  
        historial_conteos.pop(0)
    
   
    conteo_estabilizado = round(sum(historial_conteos) / len(historial_conteos))

    cv2.putText(
        frame_redimensionado,
        f"Libros detectados (tiempo real): {libros_detectados} | Estabilizado: {conteo_estabilizado}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, # Lo bajé un poquito para que quepa bien el texto
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame_redimensionado,
        f"FPS: {fps:.2f} | ESPACIO: Capturar | ESC: Salir",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow("Deteccion de libros por lomo", frame_redimensionado)

    tecla = cv2.waitKey(1) & 0xFF

    # =========================================================
    # FIN
    # =========================================================

    if tecla == 27:
        break

    # =========================================================
    # CAPTURA DE IMAGEN DEL ESTANTE
    # =========================================================

    if tecla == 32:
        nombre_imagen = f"captura_{contador_capturas}.jpg"

        ruta_captura = os.path.join(CARPETA_CAPTURAS, nombre_imagen)
        ruta_detectada = os.path.join(CARPETA_IMAGENES_DETECTADAS, nombre_imagen)

        cv2.imwrite(ruta_captura, frame)
        cv2.imwrite(ruta_detectada, frame_redimensionado)

        # =========================================================
        # COMPARAR CON LIBROS REALES
        # =========================================================

        libros_reales = datos_reales.get(nombre_imagen, None)

        if libros_reales is not None:
            diferencia = libros_detectados - libros_reales
            error_absoluto = abs(diferencia)

            correctos_estimados = min(libros_reales, libros_detectados)
            falsos_positivos = max(0, libros_detectados - libros_reales)
            falsos_negativos = max(0, libros_reales - libros_detectados)

            precision = correctos_estimados / libros_detectados if libros_detectados > 0 else 0
            recall = correctos_estimados / libros_reales if libros_reales > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0 else 0
            )

            total_reales += libros_reales
            total_detectados += libros_detectados
            total_correctos_estimados += correctos_estimados
            total_falsos_positivos += falsos_positivos
            total_falsos_negativos += falsos_negativos

        else:
            diferencia = "No registrado"
            error_absoluto = "No registrado"
            precision = "No registrado"
            recall = "No registrado"
            f1_score = "No registrado"
            falsos_positivos = "No registrado"
            falsos_negativos = "No registrado"

        if libros_detectados == 0:
            estado = "No se detectaron libros"
        elif libros_reales is None:
            estado = "Sin valor real para comparar"
        elif diferencia == 0:
            estado = "Correcto"
        elif diferencia > 0:
            estado = "Detectó libros de más"
        else:
            estado = "Detectó libros de menos"

        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        resultados_tabla.append({
            "Imagen": nombre_imagen,
            "Fecha": fecha,
            "Libros reales": libros_reales if libros_reales is not None else "No registrado",
            "Libros detectados": libros_detectados,
            "Diferencia": diferencia,
            "Error absoluto": error_absoluto,
            "Falsos positivos": falsos_positivos,
            "Falsos negativos": falsos_negativos,
            "Precisión": precision if isinstance(precision, str) else round(precision * 100, 2),
            "Recall": recall if isinstance(recall, str) else round(recall * 100, 2),
            "F1 Score": f1_score if isinstance(f1_score, str) else round(f1_score * 100, 2),
            "FPS": round(fps, 2),
            "Tiempo procesamiento": round(tiempo_procesamiento, 4),
            "Resultado": estado
        })

        # =========================================================
        # GENERAR ARCHIVO CSV/XLSX
        # =========================================================

        df = pd.DataFrame(resultados_tabla)

        ruta_csv = os.path.join(CARPETA_CSV, "resultados_deteccion.csv")
        ruta_excel = os.path.join(CARPETA_EXCEL, "resultados_deteccion.xlsx")

        df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
        df.to_excel(ruta_excel, index=False)

        # =========================================================
        # MOSTRAR RESULTADOS FINALES
        # =========================================================

        print("\n===== RESULTADO DE CAPTURA =====")
        print(f"Imagen: {nombre_imagen}")
        print(f"Libros detectados: {libros_detectados}")

        if libros_reales is not None:
            print(f"Libros reales: {libros_reales}")
            print(f"Diferencia: {diferencia}")
            print(f"Error absoluto: {error_absoluto}")
            print(f"Falsos positivos: {falsos_positivos}")
            print(f"Falsos negativos: {falsos_negativos}")
            print(f"Precisión: {precision * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")
            print(f"F1 Score: {f1_score * 100:.2f}%")
        else:
            print("Libros reales: No registrado en reales.csv")
            print("Agrega esta captura al reales.csv para calcular métricas reales.")

        print(f"FPS: {fps:.2f}")
        print(f"Tiempo de procesamiento: {tiempo_procesamiento:.4f} segundos")
        print(f"Estado: {estado}")
        print(f"Imagen detectada guardada en: {ruta_detectada}")
        print(f"CSV generado en: {ruta_csv}")
        print(f"Excel generado en: {ruta_excel}")

        contador_capturas += 1

# =========================================================
# RESULTADOS GENERALES
# =========================================================

if total_reales > 0 and total_detectados > 0:
    precision_general = total_correctos_estimados / total_detectados
    recall_general = total_correctos_estimados / total_reales
    f1_general = (
        2 * (precision_general * recall_general) / (precision_general + recall_general)
        if (precision_general + recall_general) > 0 else 0
    )
    accuracy_aproximada = total_correctos_estimados / total_reales

    print("\n===== MÉTRICAS GENERALES =====")
    print(f"Total de libros reales: {total_reales}")
    print(f"Total de libros detectados: {total_detectados}")
    print(f"Detecciones correctas estimadas: {total_correctos_estimados}")
    print(f"Falsos positivos estimados: {total_falsos_positivos}")
    print(f"Falsos negativos estimados: {total_falsos_negativos}")
    print(f"Precisión general: {precision_general * 100:.2f}%")
    print(f"Recall general: {recall_general * 100:.2f}%")
    print(f"F1 Score general: {f1_general * 100:.2f}%")
    print(f"Accuracy aproximada: {accuracy_aproximada * 100:.2f}%")
else:
    print("\nNo se calcularon métricas generales porque faltan datos reales en reales.csv.")

camara.release()
cv2.destroyAllWindows()