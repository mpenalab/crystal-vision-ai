import cv2
import os
import glob
import numpy as np

def apply_industrial_filter(img):
    """Aplica filtros para resaltar defectos en acero."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def build_processed_dataset():
    input_base = "data/raw"
    output_base = "data/processed"
    
    # Buscamos las imágenes dentro de las subcarpetas de NEU-DET
    image_paths = glob.glob(os.path.join(input_base, "**/*.jpg"), recursive=True)
    
    print(f"🛠️  Extrayendo características de {len(image_paths)} imágenes...")
    
    for path in image_paths:
        # Extraer categoría (Cr, In, Pa, etc.) de la estructura de carpetas
        category = os.path.basename(os.path.dirname(path))
        target_dir = os.path.join(output_base, category)
        os.makedirs(target_dir, exist_ok=True)
        
        img = cv2.imread(path)
        if img is not None:
            # Procesamiento
            enhanced = apply_industrial_filter(img)
            final = cv2.resize(enhanced, (640, 640))
            
            # Guardar en data/processed
            cv2.imwrite(os.path.join(target_dir, os.path.basename(path)), final)

    print("✅ Dataset procesado y guardado en data/processed/")

if __name__ == "__main__":
    build_processed_dataset()