from collections import Counter
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
counter = Counter()

for f in selected_files:
    parts = f.split("_")
    gender = parts[1]
    race = parts[2]
    counter[(race, gender)] += 1

print(counter)