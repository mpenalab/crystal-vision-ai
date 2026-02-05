import os
import glob
import random
from ultralytics import YOLO

def run_prediction_random():
    # 1. Ruta al modelo v2 (50 epochs)
    model_path = "models/saved_models/model_v2_augmented_50epochs.pt"

    # 2. Cargar modelo
    model = YOLO(model_path)

    # 3. Obtener todas las imágenes de validación
    all_test_images = glob.glob("data/processed/validation/images/*.jpg")
    
    # 4. Mezclar aleatoriamente para ver distintos defectos
    random.shuffle(all_test_images)
    test_selection = all_test_images[:15]

    print(f"🎲 Seleccionando 15 imágenes aleatorias para inspección...")

    # 5. Predicción con el umbral óptimo de tu curva F1
    results = model.predict(
        source=test_selection, 
        save=True, 
        conf=0.25, 
        imgsz=640
    )

    print(f"✅ Inspección variada lista en: {results[0].save_dir}")

if __name__ == "__main__":
    run_prediction_random()