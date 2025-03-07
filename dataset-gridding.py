import os
import numpy as np
from PIL import Image
from datasets import load_dataset

# Funzione per scaricare il dataset
def download_dataset(dataset_name, cache_dir="dataset"):
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    return dataset

# Funzione per determinare il colore dello sfondo (es. angolo in alto a sinistra)
def get_background_color(image):
    img_array = np.array(image)
    # Prendi il colore dall'angolo in alto a sinistra (probabile sfondo)
    background_color = img_array[0, 0]
    return background_color

# Funzione per contare i frame occupati in una griglia
def count_occupied_frames(image, rows, cols, background_color, threshold=10):
    img_width, img_height = image.size
    frame_width = img_width // cols
    frame_height = img_height // rows
    
    occupied_frames = 0
    frame_positions = []
    
    for row in range(rows):
        for col in range(cols):
            left = col * frame_width
            top = row * frame_height
            right = (col + 1) * frame_width
            bottom = (row + 1) * frame_height
            frame = image.crop((left, top, right, bottom))
            frame_array = np.array(frame)
            
            # Calcola la differenza media dal colore dello sfondo
            diff_from_background = np.mean(np.abs(frame_array - background_color))
            if diff_from_background > threshold:  # Frame occupato se differisce abbastanza
                occupied_frames += 1
                frame_positions.append((row, col))
    
    return occupied_frames, frame_positions

# Funzione per rilevare la griglia
def detect_grid(image):
    img_width, img_height = image.size
    if img_width != 1024 or img_height != 1024:
        print("Errore: immagine non 1024x1024")
        return 1, 1, 1
    
    # Determina il colore dello sfondo
    background_color = get_background_color(image)
    
    # Possibili griglie da testare
    grid_options = [
        (2, 2),  # 4 frame
        (4, 4),  # 16 frame
        (5, 5),  # 25 frame
        (2, 4),  # 8 frame
        (4, 2),  # 8 frame
    ]
    
    best_grid = None
    best_frame_count = 0
    best_score = 0
    
    for rows, cols in grid_options:
        frame_count, frame_positions = count_occupied_frames(image, rows, cols, background_color)
        if frame_count > 0:
            # Calcola un punteggio basato su frame occupati e continuità
            continuity_score = sum(1 for i in range(len(frame_positions)-1) 
                                 if abs(frame_positions[i][0] - frame_positions[i+1][0]) <= 1 
                                 and abs(frame_positions[i][1] - frame_positions[i+1][1]) <= 1)
            score = frame_count + continuity_score / max(1, frame_count)
            if score > best_score and 4 <= frame_count <= 25:
                best_grid = (rows, cols)
                best_frame_count = frame_count
                best_score = score
    
    if best_grid is None:
        print("Warning: Nessuna griglia valida rilevata. Uso 1x1.")
        return 1, 1, 1
    
    return best_grid[0], best_grid[1], best_frame_count

# Funzione per salvare i frame occupati
def save_frames(image, rows, cols, frame_count, output_dir, background_color, threshold=10):
    img_width, img_height = image.size
    frame_width = img_width // cols
    frame_height = img_height // rows
    
    valid_frames = 0
    for row in range(rows):
        for col in range(cols):
            if valid_frames >= frame_count:
                break
            left = col * frame_width
            top = row * frame_height
            right = (col + 1) * frame_width
            bottom = (row + 1) * frame_height
            frame = image.crop((left, top, right, bottom))
            frame_array = np.array(frame)
            
            # Salva solo frame occupati
            if np.mean(np.abs(frame_array - background_color)) > threshold:
                frame.save(os.path.join(output_dir, f"frame_{valid_frames}.png"))
                valid_frames += 1
    
    return valid_frames

# Funzione principale
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "dataset")
    output_base_dir = os.path.join(script_dir, "output_frames_color")
    
    dataset_name = "pawkanarek/spraix_1024"
    dataset = download_dataset(dataset_name, cache_dir=cache_dir)
    images = dataset['train']['image']
    descriptions = dataset['train']['text'] if 'text' in dataset['train'].column_names else dataset['train']['description'] if 'description' in dataset['train'].column_names else [f"Descrizione non trovata" for _ in images]
    
    for idx, image_data in enumerate(images[:10]):  # Prime 10 immagini
        background_color = get_background_color(image_data)
        rows, cols, frame_count = detect_grid(image_data)
        output_dir = os.path.join(output_base_dir, f"img_{idx}_grid_{rows}x{cols}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_frames = save_frames(image_data, rows, cols, frame_count, output_dir, background_color)
        description = descriptions[idx]
        print(f"Salvati {saved_frames} frame per immagine {idx} con griglia {rows}x{cols} (frame stimati: {frame_count}) - {description}")
        
        # Salva i dettagli in un file
        with open(os.path.join(script_dir, "descriptions_color.txt"), "a") as desc_file:
            desc_file.write(f"Immagine {idx}: {description} - Griglia {rows}x{cols} - {saved_frames} frame\n")

if __name__ == "__main__":
    main()