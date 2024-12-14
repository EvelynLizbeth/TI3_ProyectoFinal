from ultralytics import YOLO

# Carga el modelo pre-entrenado (puedes elegir otro modelo como yolov8s.pt o yolov8m.pt)
model = YOLO('yolov8n.pt')

# Entrena el modelo
model.train(
    data='C:/UNSA/TI3_YOLO/Zorros.v2-zorro.yolov8/data.yaml',  # Ruta al archivo de configuración del dataset
    epochs=50,                # Número de épocas (ajustar según lo necesario)
    batch=8,                 # Tamaño del batch (reducir por  poca RAM)
    imgsz=640,                # Tamaño de las imágenes de entrada
    workers=2                 # Número de hilos para cargar datos
)