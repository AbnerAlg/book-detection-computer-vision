# Sistema de Detección y Conteo de Libros en Estantes



Proyecto de visión por computadora desarrollado para la materia de Procesamiento de Imágenes en la carrera de Ingeniería de Software (CUUT).



##  Descripción del Proyecto

Este sistema automatiza el proceso de inventariado en bibliotecas mediante el procesamiento digital de imágenes. El algoritmo analiza tomas de estanterías, aplica técnicas de segmentación, detección de objetos y filtrado para identificar y contar con precisión el número de libros presentes en cada repisa, optimizando los tiempos de auditoría de stock.



##  Stack Tecnológico

* **Lenguaje:** Python 3.x

* **Librerías Principales:** OpenCV, [Menciona aquí si usaste YOLOv8, NumPy, Matplotlib, etc.]

* **Entorno:** Desarrollado y probado nativamente en Linux.



##  Metodología de Procesamiento

1. **Preprocesamiento:** Conversión a escala de grises, reducción de ruido (Gaussian Blur) y ajuste de contraste.

2. **Segmentación / Detección:** Aplicación de filtros morfológicos, detección de bordes o inferencia mediante modelo de detección de objetos.

3. **Lógica de Conteo:** Filtrado por área de contornos y cálculo final de unidades detectadas.

