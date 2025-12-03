import os
import shutil
import random
from sklearn.model_selection import train_test_split

def create_directory_structure(base_path):
    
    splits = ['train', 'validation', 'test']
    classes = ['Positive', 'Negative']
    
    for split in splits:
        for class_name in classes:
            dir_path = os.path.join(base_path, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")
    
    print("✅ Created directory structure")

def split_dataset():
    RAW_DATA_PATH = "D:/DeepLearningModels/Model1(SurfaceCrack)/data/raw"
    PROCESSED_DATA_PATH = "D:/DeepLearningModels/Model1(SurfaceCrack)/data/processed"
    
    
    if not os.path.exists(RAW_DATA_PATH):
        print(f"❌ Error: Raw data path '{RAW_DATA_PATH}' does not exist!")
        return
    
    random.seed(42)
    classes = ['Positive', 'Negative']
    
    print("🚀 Starting dataset preprocessing...")
    print(f"Source: {RAW_DATA_PATH}")
    print(f"Destination: {PROCESSED_DATA_PATH}")
    
    
    create_directory_structure(PROCESSED_DATA_PATH)
    
    
    for class_name in classes:
        print(f"\n📁 Processing {class_name} class...")
        
        class_path = os.path.join(RAW_DATA_PATH, class_name)
        if not os.path.exists(class_path):
            print(f"❌ Error: {class_path} does not exist!")
            continue
            
        # Get all image files
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"   Found {len(images)} images")
        
        if len(images) == 0:
            print(f"   ⚠️  No images found in {class_path}")
            continue
        
        # Shuffle and split (70% train, 15% validation, 15% test)
        random.shuffle(images)
        
        train_files, temp_files = train_test_split(images, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        print(f"   📊 Split: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
        
        # Copy files to respective directories
        for file in train_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(PROCESSED_DATA_PATH, 'train', class_name, file)
            shutil.copy2(src, dst)
            
        for file in val_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(PROCESSED_DATA_PATH, 'validation', class_name, file)
            shutil.copy2(src, dst)
            
        for file in test_files:
            src = os.path.join(class_path, file)
            dst = os.path.join(PROCESSED_DATA_PATH, 'test', class_name, file)
            shutil.copy2(src, dst)
    
    print(f"\n✅ Dataset splitting completed!")
    print(f"Processed data saved to: {PROCESSED_DATA_PATH}")

def verify_split():
    """Verify the split was successful"""
    PROCESSED_DATA_PATH = "D:/DeepLearningModels/Model1(SurfaceCrack)/data/processed"
    
    print("\n" + "="*50)
    print("VERIFYING DATASET SPLIT")
    print("="*50)
    
    splits = ['train', 'validation', 'test']
    classes = ['Positive', 'Negative']
    
    total_images = 0
    for split in splits:
        print(f"\n{split.upper()}:")
        split_total = 0
        for class_name in classes:
            split_path = os.path.join(PROCESSED_DATA_PATH, split, class_name)
            if os.path.exists(split_path):
                num_images = len([f for f in os.listdir(split_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                print(f"  {class_name}: {num_images} images")
                split_total += num_images
                total_images += num_images
        print(f"  TOTAL: {split_total} images")
    
    print(f"\n📈 GRAND TOTAL: {total_images} images")

if __name__ == "__main__":
    split_dataset()
    verify_split()