import os
from ultralytics import YOLO, settings

def train_phase_2():
    # 1. Desactivamos MLflow en los settings globales de Ultralytics
    # Esto sobreescribe cualquier intento automático de conexión
    settings.update({"mlflow": False})
    
    # 2. Variables de entorno de seguridad
    os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'
    os.environ['REPORT_TO'] = 'none'

    # 3. Cargamos el modelo
    model = YOLO("yolov8n.pt") 

    print("🚀 FASE 2: ¡MODO ULTRA-OFFLINE ACTIVADO!")
    print(f"CUDA disponible: {model.device}")

    model.train(
        data="config/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="steel_model_v2_augmented",
        device=0,         # GPU NVIDIA RTX 3050
        workers=4,
        # Parámetros de aumentación
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1
    )

if __name__ == "__main__":
    train_phase_2()