import xml.etree.ElementTree as ET
import os
from glob import glob

def convert_xml_to_yolo(xml_path, output_labels_dir, classes):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # El nombre del archivo txt debe coincidir con el del xml/imagen
    txt_name = os.path.basename(xml_path).replace('.xml', '.txt')
    
    with open(os.path.join(output_labels_dir, txt_name), 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes: continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            
            # Formato YOLO: class x_center y_center width height (normalizado 0-1)
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)
            
            x_center = (xmin + xmax) / (2.0 * w)
            y_center = (ymin + ymax) / (2.0 * h)
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Definimos las clases tal cual aparecen en tus XML
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

def prepare_dataset():
    # Iteramos sobre train y validation (nombres de tus carpetas raw)
    for split in ['train', 'validation']:
        print(f"Procesando {split}...")
        
        # Rutas basadas en tu descripción
        raw_images_path = f"data/raw/NEU-DET/{split}/images"
        raw_xml_path = f"data/raw/NEU-DET/{split}/annotations"
        
        # Nueva estructura organizada para YOLO
        processed_path = f"data/processed/{split}"
        os.makedirs(f"{processed_path}/images", exist_ok=True)
        os.makedirs(f"{processed_path}/labels", exist_ok=True)
        
        # 1. Convertir XMLs
        xml_files = glob(f"{raw_xml_path}/*.xml")
        for xml in xml_files:
            convert_xml_to_yolo(xml, f"{processed_path}/labels", classes)
            
        # 2. Copiar imágenes a la nueva carpeta procesada
        # Buscamos en todas las subcarpetas de imágenes (crazing, etc.)
        img_files = glob(f"{raw_images_path}/**/*.jpg", recursive=True)
        import shutil
        for img in img_files:
            shutil.copy(img, f"{processed_path}/images/{os.path.basename(img)}")

if __name__ == "__main__":
    prepare_dataset()
    print("✅ Dataset organizado: Imágenes y etiquetas listas en data/processed")