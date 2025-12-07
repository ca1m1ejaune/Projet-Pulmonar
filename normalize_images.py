import os
import cv2

# 1. Definir les chemins 
input_base_path = r"C:\Users\callu\Projet PULMONAR"
output_base_path = r"C:\Users\callu\Projet PULMONAR_Preprocessed" # New folder location

categories = [ "Pneumonia-Bacterial"]
IMG_SIZE = 224

def create_preprocessed_dataset():
    # Creation d'un output si il n'existe pas
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"Created new directory: {output_base_path}")

    for category in categories:
        # Définition de l'input et de l'outpu
        path_in = os.path.join(input_base_path, category)
        path_out = os.path.join(output_base_path, category)
        
        # Création des souss dossiers COVID 19...
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # Vérification l'existence de l'input
        if not os.path.exists(path_in):
            print(f"Skipping {category}, folder not found.")
            continue

        print(f"Processing {category}...")
        
        # Normalissation
        count = 0
        for img_name in os.listdir(path_in):
            try:
                # Read
                img_path = os.path.join(path_in, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_array is None:
                    continue

                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                # Sauvegarde au dossier de l'output
                save_path = os.path.join(path_out, img_name)
                cv2.imwrite(save_path, new_array)
                count += 1
                
            except Exception as e:
                print(f"Error on {img_name}: {e}")
        
        print(f"-> Saved {count} images to {path_out}")

if __name__ == "__main__":
    create_preprocessed_dataset()

    print("\nDone! Your resized images are in 'Projet PULMONAR_Preprocessed'.")
