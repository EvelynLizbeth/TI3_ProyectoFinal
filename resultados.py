from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# Cargar el modelo entrenado
best_model_path = 'C:/UNSA/TI3_YOLO/runs/detect/train/weights/best.pt'
model = YOLO(best_model_path)

# Detectar en una nueva imagen
image_path = 'C:/UNSA/TI3_YOLO/EJEMPLOS/images6.jpeg'
results = model(image_path)

# Obtener métricas de evaluación
metrics = model.val()

# Extraer métricas desde results_dict
precision = metrics.results_dict['metrics/precision(B)']
recall = metrics.results_dict['metrics/recall(B)']
map50 = metrics.results_dict['metrics/mAP50(B)']
map50_95 = metrics.results_dict['metrics/mAP50-95(B)']


# Calcular el F1-score
if precision + recall > 0:  # Evitar división por cero
    f1_score = 2 * (precision * recall) / (precision + recall)
else:
    f1_score = 0.0


# Crear un DataFrame para las métricas
metrics_data = {
    "Métrica": ['Precisión', 'Recall', 'mAP@50', 'mAP@50-95', 'F1-score'],
    "Valor": [precision, recall, map50, map50_95, f1_score]
}
metrics_df = pd.DataFrame(metrics_data)

# Imprimir las métricas en formato tabular
print(metrics_df)

# Guardar las métricas como tabla en HTML o CSV (opcional)
metrics_df.to_html("metrics.html", index=False)

# Procesar la imagen con detecciones
image = cv2.imread(image_path)  # Cargar la imagen original

# Definir el índice de la clase que deseas detectar
target_class_index = 0  # Cambiar este índice al de la clase deseada
confidence_threshold = 0.4  # Umbral de confianza (40%)

for box in results[0].boxes:
    if int(box.cls[0]) == target_class_index and box.conf[0] >= confidence_threshold:  # Filtrar por clase y confianza
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]  # Coordenadas de la caja
        label = results[0].names[int(box.cls[0])]  # Nombre de la clase detectada
        score = box.conf[0]  # Confianza de la predicción

        # Dibujar la caja de detección
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Convertir BGR a RGB para mostrar con matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Mostrar la imagen con matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')  # Ocultar los ejes
plt.title("Detección de Zorro Andino")
plt.show()
