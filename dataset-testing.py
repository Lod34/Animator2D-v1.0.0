import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from huggingface_hub import hf_hub_download

# Funzione per scaricare il dataset
def download_dataset(dataset_name, cache_dir="dataset"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    file_path = hf_hub_download(repo_id=dataset_name, filename="data.zip", cache_dir=cache_dir)
    return file_path

# Funzione per rilevare la griglia
def detect_grid(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # Contare le linee orizzontali e verticali
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(y1 - y2) < 10:  # Linea orizzontale
                horizontal_lines.append((x1, y1, x2, y2))
            elif abs(x1 - x2) < 10:  # Linea verticale
                vertical_lines.append((x1, y1, x2, y2))
    
    rows = len(horizontal_lines) + 1
    cols = len(vertical_lines) + 1
    return rows, cols

# Funzione per salvare i frame
def save_frames(image_path, rows, cols, output_dir):
    image = Image.open(image_path)
    img_width, img_height = image.size
    frame_width = img_width // cols
    frame_height = img_height // rows

    for row in range(rows):
        for col in range(cols):
            left = col * frame_width
            top = row * frame_height
            right = (col + 1) * frame_width
            bottom = (row + 1) * frame_height
            frame = image.crop((left, top, right, bottom))
            frame.save(os.path.join(output_dir, f"frame_{row}_{col}.png"))

# Funzione principale
def main():
    dataset_name = "pawkanarek/spraix_1024"
    cache_dir = "dataset"
    output_base_dir = "output_frames"
    
    download_dataset(dataset_name, cache_dir)
    image_paths = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.png')]
    
    for idx, image_path in enumerate(image_paths[:10]):  # Test su 10 immagini
        rows, cols = detect_grid(image_path)
        output_dir = os.path.join(output_base_dir, f"img_{idx}_grid_{rows}x{cols}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_frames(image_path, rows, cols, output_dir)
        print(f"Salvati frame per {image_path} con griglia {rows}x{cols}")

if __name__ == "__main__":
    main()