import cv2
import matplotlib.pyplot as plt
import os
import glob

def compare_processing(category="scratches"): # Sc = Scratches (Rayones)
    raw_img_path = glob.glob(f"data/raw/**/{category}*.jpg", recursive=True)[0]
    proc_img_path = os.path.join("data/processed", category, os.path.basename(raw_img_path))

    raw_img = cv2.imread(raw_img_path)
    proc_img = cv2.imread(proc_img_path)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {category}")
    plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Procesada (CLAHE + Resize)")
    plt.imshow(proc_img, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_processing()