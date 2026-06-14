import os
import io
import json
import shutil
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import torch
from torch.utils.data import DataLoader
import sys

# Setup project root for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import from local modules
from dataset import CrosswalkDataset
from model import create_model, set_seed

class DataCleanerApp:
    def __init__(self, root, review_queue, queue_file, verified_file, verified_set):
        self.root = root
        self.review_queue = review_queue
        self.queue_file = queue_file
        self.verified_file = verified_file
        self.verified_set = verified_set
        self.current_index = 0
        
        # Configure Window
        self.root.title("Crosswalk Data Cleaner - Human In The Loop")
        self.root.geometry("600x700")
        self.root.configure(padx=20, pady=20)
        
        # UI Elements
        self.info_label = tk.Label(root, text="Loading...", font=("Helvetica", 14))
        self.info_label.pack(pady=10)
        
        self.confidence_label = tk.Label(root, text="", font=("Helvetica", 12, "bold"))
        self.confidence_label.pack(pady=5)
        
        self.canvas = tk.Canvas(root, width=400, height=400, bg="grey")
        self.canvas.pack(pady=20)
        self.canvas_img_id = self.canvas.create_image(200, 200, image=None)
        
        # Buttons Frame
        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X, pady=20)
        
        self.btn_fix = tk.Button(btn_frame, text="✅ Model is RIGHT\n(Move File)", 
                                 bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), 
                                 command=self.fix_label, width=20, height=3)
        self.btn_fix.pack(side=tk.LEFT, padx=10, expand=True)
        
        self.btn_keep = tk.Button(btn_frame, text="❌ Model is WRONG\n(Keep Label)", 
                                  bg="#f44336", fg="white", font=("Helvetica", 12, "bold"), 
                                  command=self.keep_label, width=20, height=3)
        self.btn_keep.pack(side=tk.RIGHT, padx=10, expand=True)
        
        self.progress_label = tk.Label(root, text="", font=("Helvetica", 10))
        self.progress_label.pack(side=tk.BOTTOM, pady=10)
        
        self.load_current_image()

    def set_buttons_state(self, state):
        """Prevents double-clicking from crashing the script."""
        self.btn_fix.config(state=state)
        self.btn_keep.config(state=state)

    def load_current_image(self):
        self.set_buttons_state("normal")
        
        # THE FIX: Auto-skip any item that is already reviewed
        while self.current_index < len(self.review_queue) and self.review_queue[self.current_index].get('reviewed', False):
            self.current_index += 1
            
        if self.current_index >= len(self.review_queue):
            messagebox.showinfo("Done!", "You have reviewed all discrepancies!", parent=self.root)
            self.root.destroy()
            return
            
        data = self.review_queue[self.current_index]
        img_path = data['file_path']
        
        # If file is completely gone, skip to avoid UI loops
        if not os.path.exists(img_path):
            print(f"⚠️ File missing, skipping: {img_path}")
            self.review_queue[self.current_index]['reviewed'] = True
            self.save_progress()
            self.root.after(10, self.next_image)
            return
            
        actual = "Crosswalk (y)" if data['actual_label'] == 0 else "No-Crosswalk (n)"
        pred = "Crosswalk (y)" if data['predicted_label'] == 0 else "No-Crosswalk (n)"
        
        self.info_label.config(text=f"Folder Label: {actual}\nModel Guess: {pred}")
        self.confidence_label.config(text=f"Model Confidence: {data['confidence']:.1f}%")
        self.confidence_label.config(fg="red" if data['confidence'] > 90 else "black")
        self.progress_label.config(text=f"Reviewing image {self.current_index + 1} of {len(self.review_queue)}")
        
        try:
            # Safely load the image into memory
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
                
            img = Image.open(io.BytesIO(img_bytes))
            resized_img = img.resize((400, 400))
            
            self.photo = ImageTk.PhotoImage(resized_img)
            self.canvas.itemconfig(self.canvas_img_id, image=self.photo)
            self.root.update()
            
        except Exception as e:
            print(f"⚠️ Skipping corrupt image: {img_path} | Error: {e}")
            self.review_queue[self.current_index]['reviewed'] = True
            self.save_progress()
            self.root.after(10, self.next_image)

    def fix_label(self):
        self.set_buttons_state("disabled") # Prevent double-clicks
        
        try:
            data = self.review_queue[self.current_index]
            old_path = Path(data['file_path'])
            
            parent_dir = old_path.parent.parent
            new_folder = "y" if data['predicted_label'] == 0 else "n"
            new_path = parent_dir / new_folder / old_path.name
            
            # The Copy-and-Delete method bypassing Windows locks
            shutil.copy2(str(old_path), str(new_path))
            
            # Verify the copy worked
            if os.path.exists(new_path):
                print(f"✓ Copied to correct folder: {new_folder}/{old_path.name}")
                
                # Now safely try to delete the old one
                deleted = False
                for _ in range(3):
                    try:
                        os.remove(str(old_path))
                        deleted = True
                        break
                    except PermissionError:
                        time.sleep(0.2)
                
                if not deleted:
                    print(f"⚠️ Warning: Windows locked {old_path.name}. It copied fine, but you may need to delete the original manually later.")
            else:
                raise Exception("Copy failed silently.")

            self.save_progress()
            self.root.after(10, self.next_image)
            
        except Exception as e:
            messagebox.showerror("File Error", f"Could not move file.\n\nError: {e}", parent=self.root)
            self.set_buttons_state("normal")

    def keep_label(self):
        self.set_buttons_state("disabled")
        try:
            self.save_progress()
            self.root.after(10, self.next_image)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save progress.\n\nError: {e}", parent=self.root)
            self.set_buttons_state("normal")

    def next_image(self):
        self.current_index += 1
        self.load_current_image()
        
    def save_progress(self):
        data = self.review_queue[self.current_index]
        data['reviewed'] = True
        
        # PERMANENT MEMORY: Add this file to the verified list
        self.verified_set.add(data['file_path'])
        
        # Save the current queue progress
        with open(self.queue_file, 'w') as f:
            json.dump(self.review_queue, f, indent=4)
            
        # Save the permanent memory 
        with open(self.verified_file, 'w') as f:
            json.dump(list(self.verified_set), f, indent=4)

def scan_dataset(model, device, dataset, verified_set, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mismatches = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(labels)):
                if labels[i] != preds[i]:
                    idx = batch_idx * batch_size + i
                    img_path = str(dataset.images[idx])
                    
                    # PERMANENT MEMORY FIX: If a human already verified this, skip it!
                    if img_path in verified_set:
                        continue
                        
                    confidence = float(probs[i][preds[i]].cpu() * 100)
                    
                    mismatches.append({
                        "file_path": img_path,
                        "actual_label": int(labels[i].cpu()),
                        "predicted_label": int(preds[i].cpu()),
                        "confidence": confidence,
                        "reviewed": False
                    })
    return mismatches

def main():
    SEED = 42
    
    META_DIR = PROJECT_ROOT / "data" / "meta"
    META_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE = META_DIR / "review_queue.json"
    VERIFIED_FILE = META_DIR / "verified_labels.json"
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Permanent Memory
    verified_set = set()
    if os.path.exists(VERIFIED_FILE):
        with open(VERIFIED_FILE, 'r') as f:
            verified_set = set(json.load(f))
            print(f"🧠 Loaded {len(verified_set)} previously verified images from permanent memory.")
    
    # Load or Create Queue
    if os.path.exists(QUEUE_FILE):
        print(f"📂 Found existing {QUEUE_FILE.name}. Resuming previous session...")
        with open(QUEUE_FILE, 'r') as f:
            review_queue = json.load(f)
    else:
        print("🔍 Scanning entire dataset for mismatches (This may take a minute)...")
        
        model = create_model(device=device, dropout_rate=0.4, freeze_backbone=False, seed=SEED)
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            print(f"❌ No checkpoints found in {checkpoint_dir}")
            sys.exit(1)
            
        model.load_state_dict(torch.load(str(checkpoints[-1]), map_location=device))
        
        train_dataset = CrosswalkDataset(root_dir=str(PROJECT_ROOT / "data" / "train"), augment=False, seed=SEED)
        test_dataset = CrosswalkDataset(root_dir=str(PROJECT_ROOT / "data" / "test"), augment=False, seed=SEED)
        
        print(f"Scanning Train Set ({len(train_dataset)} images)...")
        train_mismatches = scan_dataset(model, device, train_dataset, verified_set)
        
        print(f"Scanning Test Set ({len(test_dataset)} images)...")
        test_mismatches = scan_dataset(model, device, test_dataset, verified_set)
        
        review_queue = train_mismatches + test_mismatches
        review_queue.sort(key=lambda x: x['confidence'], reverse=True)
        
        with open(QUEUE_FILE, 'w') as f:
            json.dump(review_queue, f, indent=4)
            
        print(f"✅ Scan complete! Found {len(review_queue)} unverified discrepancies.")
    
    pending_queue = [item for item in review_queue if not item.get('reviewed', False)]
    
    if not pending_queue:
        print("🎉 No pending images to review! Dataset is clean.")
        return
        
    print(f"🚀 Launching UI to review {len(pending_queue)} images...")
    
    root = tk.Tk()
    # The UI now naturally skips reviewed images as soon as it launches
    app = DataCleanerApp(root, review_queue, QUEUE_FILE, VERIFIED_FILE, verified_set)
    root.mainloop()

if __name__ == "__main__":
    main()