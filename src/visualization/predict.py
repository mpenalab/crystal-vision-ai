import os
from ultralytics import YOLO
import glob

def run_prediction():
    # 1. Ruta al mejor modelo entrenado
    model_path = "runs/detect/steel_model_v1_final/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encontró el modelo en {model_path}")
        return

    # 2. Cargar el modelo
    model = YOLO(model_path)

    # 3. Seleccionar imágenes de validación (que el modelo no usó para entrenar)
    # Tomaremos una de cada tipo para ver la variedad
    test_images = glob.glob("data/processed/validation/images/*.jpg")[:10]

    print(f"🧐 Procesando {len(test_images)} imágenes de inspección...")

    # 4. Ejecutar predicción
    # save=True guardará las imágenes con los cuadros dibujados
    results = model.predict(source=test_images, save=True, conf=0.3, imgsz=640)

    # 5. Informar dónde se guardaron
    # YOLO crea una carpeta 'predict' dentro de 'runs/detect'
    save_dir = results[0].save_dir
    print(f"✅ ¡Inspección completada! Resultados guardados en: {save_dir}")

if __name__ == "__main__":
    run_prediction()