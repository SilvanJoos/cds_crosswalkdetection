import cv2
import numpy as np
import shutil
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# ==========================================
# CONFIGURATION
# ==========================================
SOURCE_DIR = "C:\\Users\\silva\\Documents\\DeepLearning\\data\\asconalocarno\\n"      # Folder containing the 100k images
DEST_DIR = "C:\\Users\\silva\\Documents\\DeepLearning\\data\\asconalocarno\\n_2"     # Folder to move the rejected images to
THRESHOLD_PERCENT = 75                 # If > 80% of the image is blue/green/black, move it

# Create the destination directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

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
        # Hue and Saturation don't matter, we just look for dark pixels
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 45])
        mask_black = cv2.inRange(hsv, lower_black, upper_black)
        
        # 2. Define range for GREEN (Foliage/Grass)
        # Hue in OpenCV is 0-179. Green is roughly 35 to 85.
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        
        # 3. Define range for BLUE (Water/Sky)
        # Blue is roughly 90 to 130.
        lower_blue = np.array([90, 40, 40])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
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

def process_image(image_path):
    """Worker function to process a single image and move it if necessary."""
    if is_overwhelmingly_bg_b(image_path):
        dest_path = Path(DEST_DIR) / image_path.name
        try:
            shutil.move(str(image_path), str(dest_path))
            return f"Moved: {image_path.name}"
        except Exception as e:
            return f"Failed to move {image_path.name}: {e}"
    return None

def main():
    # Gather all image paths (modify extensions if you have .png or others)
    source_path = Path(SOURCE_DIR)
    image_paths = [p for p in source_path.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    total_images = len(image_paths)
    print(f"Found {total_images} images. Starting processing...")
    
    moved_count = 0
    
    # Process images in parallel to speed up the 100k iteration
    with ProcessPoolExecutor() as executor:
        # Map the worker function to all image paths
        results = executor.map(process_image, image_paths)
        
        for i, result in enumerate(results):
            if result:
                moved_count += 1
                
            # Print a progress update every 1000 images
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{total_images} images... Moved {moved_count} so far.")

    print("==========================================")
    print(f"Done! Filtered out {moved_count} images out of {total_images}.")

if __name__ == '__main__':
    main()