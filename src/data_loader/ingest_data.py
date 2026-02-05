import os
import subprocess

def download_from_kaggle(dataset_name):
    raw_path = "data/raw"
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    
    print(f"Descargando {dataset_name} desde Kaggle...")
    # Usaremos el dataset de defectos superficiales de acero de NEU
    subprocess.run(["kaggle", "datasets", "download", "-d", "kaustubhdikshit/neu-surface-defect-database", "-p", raw_path, "--unzip"])
    print("Descarga y extracción completada.")

if __name__ == "__main__":
    # Este dataset es estándar en la industria para control de calidad
    DATASET = "kaustubhdikshit/neu-surface-defect-database"
    download_from_kaggle(DATASET)