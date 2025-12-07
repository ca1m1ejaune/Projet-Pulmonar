import os
import cv2

# 1. Define paths
input_base_path = r"C:\Users\callu\Projet PULMONAR"
output_base_path = r"C:\Users\callu\Projet PULMONAR_Preprocessed" # New folder location

categories = [ "Pneumonia-Bacterial"]
IMG_SIZE = 224

def create_preprocessed_dataset():
    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
        print(f"Created new directory: {output_base_path}")

    for category in categories:
        # Define input and output paths for this specific category
        path_in = os.path.join(input_base_path, category)
        path_out = os.path.join(output_base_path, category)
        
        # Create the sub-folder (e.g., .../COVID-19) in the output directory
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        # Check if input path exists
        if not os.path.exists(path_in):
            print(f"Skipping {category}, folder not found.")
            continue

        print(f"Processing {category}...")
        
        # Process images
        count = 0
        for img_name in os.listdir(path_in):
            try:
                # Read
                img_path = os.path.join(path_in, img_name)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_array is None:
                    continue

                # Resize (We keep it 0-255 here so we can save it as an image)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                # Save to the new output folder
                save_path = os.path.join(path_out, img_name)
                cv2.imwrite(save_path, new_array)
                count += 1
                
            except Exception as e:
                print(f"Error on {img_name}: {e}")
        
        print(f"-> Saved {count} images to {path_out}")

if __name__ == "__main__":
    create_preprocessed_dataset()
    print("\nDone! Your resized images are in 'Projet PULMONAR_Preprocessed'.")