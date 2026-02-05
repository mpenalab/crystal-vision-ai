import os
from ultralytics import YOLO

# YA NO NECESITAMOS EL BYPASS DE MAGICMOCK
# Las librerías de sistema ya están instaladas

def train_custom_model():
    os.environ['YOLO_CONFIG_DIR'] = '/tmp/Ultralytics'
    
    model = YOLO("yolov8n.pt")
    
    print("🚀 ¡IGNICIÓN REAL! Iniciando entrenamiento...")
    
    model.train(
        data="config/data.yaml",
        epochs=15,
        imgsz=640,
        batch=16,
        name="steel_model_v1_final",
        device="cpu",
        workers=4 # Mantener en 0 para evitar problemas de memoria compartida en Docker
    )

if __name__ == "__main__":
    train_custom_model()