import shutil
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DATA_DIR = "C:\\Users\\silva\\Documents\\cds_crosswalkdetection\\data"
WORKING_DIR = os.path.join(BASE_DATA_DIR, "working")

# Folders to skip at the base level (add "working" so it doesn't process itself)
IGNORE_FOLDERS = {"train", "test", "val", "working"}  

# Target directories
TARGET_Y = Path(WORKING_DIR) / "y"
TARGET_N = Path(WORKING_DIR) / "n"

# Create the working directories if they don't exist
TARGET_Y.mkdir(parents=True, exist_ok=True)
TARGET_N.mkdir(parents=True, exist_ok=True)

def copy_image(args):
    """Worker function to copy a single image."""
    src_path, dest_dir, city_name = args
    
    # Prepend the city name to the filename to avoid collisions when merging folders
    dest_filename = f"{city_name}_{src_path.name}"
    dest_path = dest_dir / dest_filename
    
    try:
        # copy2 preserves file metadata (timestamps)
        shutil.copy2(str(src_path), str(dest_path))
        return True
    except Exception as e:
        print(f"Error copying {src_path.name}: {e}")
        return False

def main():
    base_path = Path(BASE_DATA_DIR)
    
    # List to hold all copy jobs
    tasks = []
    
    # 1. Iterate through all items in the base data directory
    for city_dir in base_path.iterdir():
        if city_dir.is_dir() and city_dir.name.lower() not in IGNORE_FOLDERS:
            city_name = city_dir.name.lower()
            
            # ==========================================
            # CONDITION 1: Copy 'y' for EVERY city
            # ==========================================
            y_folder = city_dir / "y"
            if y_folder.exists() and y_folder.is_dir():
                for img_path in y_folder.iterdir():
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # Add to tasks: (source path, destination folder, prefix)
                        tasks.append((img_path, TARGET_Y, city_name))
            
            # ==========================================
            # CONDITION 2: Copy 'n' ONLY for asconalocarno
            # ==========================================
            if city_name == "asconalocarno":
                n_folder = city_dir / "n"
                if n_folder.exists() and n_folder.is_dir():
                    for img_path in n_folder.iterdir():
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            # Add to tasks: (source path, destination folder, prefix)
                            tasks.append((img_path, TARGET_N, city_name))
    
    total_images = len(tasks)
    if total_images == 0:
        print("No images found matching your criteria. Check your paths.")
        return

    print(f"Found {total_images} files to copy. Starting transfer...")
    
    copied_count = 0
    
    # 2. Execute the copy tasks in parallel
    with ThreadPoolExecutor() as executor:
        results = executor.map(copy_image, tasks)
        
        for i, success in enumerate(results):
            if success:
                copied_count += 1
                
            # Print a progress update every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"Copied {i + 1}/{total_images} files...")

    print("==========================================")
    print(f"Done! Successfully COPIED {copied_count} out of {total_images} files to /working/")

if __name__ == '__main__':
    main()