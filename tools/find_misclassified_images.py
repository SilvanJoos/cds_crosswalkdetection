import torch
import csv
from pathlib import Path
import sys

# Setup project root for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import from your local modules
from dataset import create_dataloaders, CrosswalkDataset
from model import create_model, set_seed

def main():
    # ========== CONFIGURATION ==========
    DATA_DIR = PROJECT_ROOT / "data" / "working"
    BATCH_SIZE = 32
    SEED = 42
    # ===================================
    
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔍 Starting Error Analysis on Device: {device}\n")
    
    try:
        # 1. Load the Entire Dataset (We need the raw dataset object to get the file paths)
        print("📊 Loading data from working directory...")
        dataset = CrosswalkDataset(
            root_dir=str(DATA_DIR),
            augment=False,
            seed=SEED
        )
        
        # 2. Get the loader (crucially, shuffle=False)
        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8, # or suitable number
            pin_memory=True
        )
        
        # 3. Find and load the best model
        print("\n🏗️  Building and loading model...")
        model = create_model(device=device, dropout_rate=0.4, freeze_backbone=False, seed=SEED)
        
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
        checkpoints = sorted(checkpoint_dir.glob("*.pth"))
        if not checkpoints:
            print(f"❌ No checkpoints found in {checkpoint_dir}")
            sys.exit(1)
            
        best_checkpoint = str(checkpoints[-1])
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        print(f"✓ Weights loaded from {best_checkpoint}")
        
        # 4. Run Inference
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        print("\n🧠 Running inference on the dataset...")
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                
                # Get probabilities and predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
        # 5. Analyze Errors and Write to CSV
        output_csv = PROJECT_ROOT / "eval" / "misclassified_images.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        image_paths = dataset.images  # Matches exactly because shuffle=False
        
        error_count = 0
        
        print(f"\n📝 Writing errors to {output_csv}...")
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["File_Path", "Actual_Label", "Predicted_Label", "Error_Type", "Model_Confidence"])
            
            for i in range(len(image_paths)):
                true_lbl = all_labels[i]
                pred_lbl = all_preds[i]
                
                # If the model got it wrong...
                if true_lbl != pred_lbl:
                    error_count += 1
                    
                    # Determine Error Type
                    # Class 0: Crosswalk, Class 1: No-Crosswalk
                    if true_lbl == 1 and pred_lbl == 0:
                        error_type = "False Positive (Ghost Crosswalk)"
                    else:
                        error_type = "False Negative (Missed Crosswalk)"
                        
                    # How confident was the model in its wrong answer?
                    confidence = all_probs[i][pred_lbl]
                    
                    # Human-readable labels
                    actual_str = "Crosswalk" if true_lbl == 0 else "No-Crosswalk"
                    pred_str = "Crosswalk" if pred_lbl == 0 else "No-Crosswalk"
                    
                    writer.writerow([
                        str(image_paths[i]), 
                        actual_str, 
                        pred_str, 
                        error_type, 
                        f"{confidence * 100:.2f}%"
                    ])
                    
        print(f"\n✅ Done! Found {error_count} misclassifications out of {len(image_paths)} images.")
        print(f"💡 Open '{output_csv}' to view the file paths of the tricky images.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()