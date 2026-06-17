import os
import csv
import pandas as pd

CARPETA_EXCEL = "resultados/excel"
CARPETA_CSV = "resultados/csv"
ARCHIVO_REALES = "reales.csv"
RUTA_EXCEL = os.path.join(CARPETA_EXCEL, "resultados_deteccion.xlsx")
RUTA_CSV = os.path.join(CARPETA_CSV, "resultados_deteccion.csv")

if not os.path.exists(RUTA_EXCEL) or not os.path.exists(ARCHIVO_REALES):
    print("Falta el archivo de resultados Excel o el archivo reales.csv.")
    exit()


datos_reales = {}
with open(ARCHIVO_REALES, newline="", encoding="utf-8-sig") as archivo:
    lector = csv.DictReader(archivo)
    for fila in lector:
        # Verificamos que la fila tenga datos y que 'libros_reales' no esté vacío
        if fila.get("imagen") and fila.get("libros_reales") and fila["libros_reales"].strip() != "":
            datos_reales[fila["imagen"]] = int(fila["libros_reales"])


df = pd.read_excel(RUTA_EXCEL)


columnas_numericas = ["Libros reales", "Diferencia", "Error absoluto", "Falsos positivos", "Falsos negativos", "Precisión", "Recall", "F1 Score"]
for col in columnas_numericas:
    if col in df.columns:
        df[col] = df[col].astype(object)


total_reales = 0
total_detectados = 0
total_correctos = 0


for idx, fila in df.iterrows():
    img = fila["Imagen"]
    if img in datos_reales:
        libros_reales = datos_reales[img]
        libros_detectados = int(fila["Libros detectados"])
        
        diferencia = libros_detectados - libros_reales
        error_absoluto = abs(diferencia)
        
        correctos = min(libros_reales, libros_detectados)
        fp = max(0, libros_detectados - libros_reales)
        fn = max(0, libros_reales - libros_detectados)
        
        precision = (correctos / libros_detectados * 100) if libros_detectados > 0 else 0
        recall = (correctos / libros_reales * 100) if libros_reales > 0 else 0
        f1 = (2 * (precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
        
        # Actualizar la fila en el DataFrame
        df.at[idx, "Libros reales"] = libros_reales
        df.at[idx, "Diferencia"] = diferencia
        df.at[idx, "Error absoluto"] = error_absoluto
        df.at[idx, "Falsos positivos"] = fp
        df.at[idx, "Falsos negativos"] = fn
        df.at[idx, "Precisión"] = round(precision, 2)
        df.at[idx, "Recall"] = round(recall, 2)
        df.at[idx, "F1 Score"] = round(f1, 2)
        
        if libros_detectados == 0: estado = "No se detectaron libros"
        elif diferencia == 0: estado = "Correcto"
        elif diferencia > 0: estado = "Detectó libros de más"
        else: estado = "Detectó libros de menos"
        df.at[idx, "Resultado"] = estado

        total_reales += libros_reales
        total_detectados += libros_detectados
        total_correctos += correctos


df.to_excel(RUTA_EXCEL, index=False)
df.to_csv(RUTA_CSV, index=False, encoding="utf-8-sig")

print("\n===== ¡REPORTE ACTUALIZADO CON ÉXITO! =====")
if total_reales > 0:
    p_gen = (total_correctos / total_detectados) * 100 if total_detectados > 0 else 0
    r_gen = (total_correctos / total_reales) * 100
    f1_gen = (2 * (p_gen * r_gen) / (p_gen + r_gen)) if (p_gen + r_gen) > 0 else 0
    print(f"Métricas Generales Calculadas:")
    print(f"-> Precisión General: {p_gen:.2f}%")
    print(f"-> Recall General: {r_gen:.2f}%")
    print(f"-> F1 Score General: {f1_gen:.2f}%")