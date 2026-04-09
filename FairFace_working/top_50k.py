import os
import random
import shutil

# Paths
source_folder = "./train"
target_folder = "./FairFace_40k"

# Create target folder
os.makedirs(target_folder, exist_ok=True)

# Get all image files
files = [f for f in os.listdir(source_folder) if f.endswith(".jpg")]

print("Total images:", len(files))

# Shuffle randomly
random.seed(42)   # for reproducibility
random.shuffle(files)

# Select first 40k
selected_files = files[:50000]

print("Selected:", len(selected_files))

# Copy selected images
for file in selected_files:
    src = os.path.join(source_folder, file)
    dst = os.path.join(target_folder, file)
    shutil.copy(src, dst)

print("Done ✅ 50k images copied safely")