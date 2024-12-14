from ultralytics import YOLO
import cv2

# Cargar el modelo entrenado
best_model_path = 'C:/UNSA/TI3_YOLO/runs/detect/train/weights/best.pt'
model = YOLO(best_model_path)

# Ruta al archivo de video
video_path = 'C:/UNSA/TI3_YOLO/EJEMPLOS/Zorro culpeo (Lycalopex culpaeus).mp4' #Zorro culpeo (Lycalopex culpaeus).mp4

# Leer el video
cap = cv2.VideoCapture(video_path)

# Verificar si el video se abrió correctamente
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

# Obtener la información del video (ancho, alto, frames por segundo)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Configurar el escritor para guardar el video con detecciones
out = cv2.VideoWriter('detecciones_salida.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    # Leer un frame del video
    ret, frame = cap.read()
    if not ret:
        break  # Si no hay más frames, salir del bucle

    # Realizar detección de objetos en el frame
    results = model(frame)  # Pasar el frame al modelo

    # Dibujar las cajas de detección sobre el frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]  # Coordenadas de la caja
        class_index = int(box.cls[0])  # Índice de la clase detectada
        score = box.conf[0]  # Confianza de la predicción

        # Obtener el nombre de la clase detectada
        label = results[0].names[class_index]  # Nombre de la clase detectada

        # Dibujar la caja de detección
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame con las detecciones
    cv2.imshow('Deteccion de Zorro Andino en video', frame)

    # Guardar el frame con detecciones en el video de salida
    out.write(frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
