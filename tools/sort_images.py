import cv2
import numpy as np
import shutil
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# CONFIGURATION
# ==========================================
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Point this to your main 'data' folder that contains all the city folders
BASE_DATA_DIR = str(PROJECT_ROOT / "data")  
IGNORE_FOLDERS = {"train", "test", "val", "meta", "working"}      # Folders to skip at the base level
THRESHOLD_PERCENT = 75                         # If > 75% of the image is blue/green/black, move it

def is_overwhelmingly_bg_b(image_path):
    """
    Checks if an image is mostly blue, green, or black.
    Returns True if it exceeds the threshold, False otherwise.
    """
    try:
        # Read image using OpenCV (loads as BGR format)
        img = cv2.imread(str(image_path))
        if img is None:
            return False # Skip corrupted or unreadable images
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Define range for BLACK (Low Value/Brightness)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 45])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 2. Define range for GREEN (Foliage/Grass)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 3. Define range for BLUE (Water/Sky)
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 4. Define range for WHITE
        mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
        
        # Combine all masks using bitwise OR
        combined_mask = mask_black | mask_green | mask_blue | mask_white
        
        # Calculate the percentage of pixels that match the combined mask
        total_pixels = img.shape[0] * img.shape[1]
        matching_pixels = cv2.countNonZero(combined_mask)
        percentage = (matching_pixels / total_pixels) * 100
        
        return percentage >= THRESHOLD_PERCENT

    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return False

def process_image(args):
    """Worker function to process a single image and move it if necessary."""
    image_path, dest_dir = args  # Unpack the arguments
    
    if is_overwhelmingly_bg_b(image_path):
        dest_path = dest_dir / image_path.name
        try:
            shutil.move(str(image_path), str(dest_path))
            return f"Moved: {image_path.name} to {dest_dir.parent.name}/n_2"
        except Exception as e:
            return f"Failed to move {image_path.name}: {e}"
    return None

def main():
    base_path = Path(BASE_DATA_DIR)
    
    # We will build a list of tasks. Each task is a tuple: (image_path, destination_directory)
    tasks = []
    
    # 1. Iterate through all items in the base data directory
    for city_dir in base_path.iterdir():
        # Ensure it's a directory and not in our ignore list
        if city_dir.is_dir() and city_dir.name.lower() not in IGNORE_FOLDERS:
            
            n_folder = city_dir / "n"
            
            # 2. Check if the "n" folder exists inside this city folder
            if n_folder.exists() and n_folder.is_dir():
                
                # Define and create the target "n_2" directory for this specific city
                dest_dir = city_dir / "n_2"
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # 3. Gather all images from this "n" folder
                for img_path in n_folder.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        tasks.append((img_path, dest_dir))
    
    total_images = len(tasks)
    if total_images == 0:
        print("No images found in any 'n' directories. Please check your BASE_DATA_DIR path.")
        return

    print(f"Found {total_images} images across all city 'n' folders. Starting processing...")
    
    moved_count = 0
    
    # Process images in parallel
    with ProcessPoolExecutor() as executor:
        # Map the worker function to all our tasks
        results = executor.map(process_image, tasks)
        
        for i, result in enumerate(results):
            if result:
                moved_count += 1
                
            # Print a progress update every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total_images} images... Moved {moved_count} so far.")

    print("==========================================")
    print(f"Done! Filtered out {moved_count} images out of {total_images} total.")

if __name__ == '__main__':
    main()