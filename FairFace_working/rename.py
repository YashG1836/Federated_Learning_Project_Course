from pathlib import Path
import pandas as pd

# Resolve paths relative to this script so it works from any cwd.
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "train_labels.csv"
image_folder = base_dir / "train"

# Load CSV
df = pd.read_csv(csv_path)

# Encoding maps
gender_map = {
    "Male": 0,
    "Female": 1
}

race_map = {
    "White": 0,
    "Black": 1,
    "East Asian": 2,
    "Indian": 3,
    "Latino_Hispanic": 4,
    "Middle Eastern": 5,
    "Southeast Asian": 6
}

# Loop through dataframe
for idx, row in df.iterrows():
    file_path = row['file']   # e.g. train/1.jpg
    gender = row['gender']
    race = row['race']

    # Extract filename
    filename = Path(file_path).name   # 1.jpg
    old_path = image_folder / filename

    # Skip if file doesn't exist
    if not old_path.exists():
        continue

    # Encode
    g = gender_map.get(gender, -1)
    r = race_map.get(race, -1)

    # Optional: dummy age (since FairFace has age groups, not exact)
    age = 0  

    # New filename (UTK-like)
    new_filename = f"{age}_{g}_{r}_{idx}.jpg"
    new_path = image_folder / new_filename

    # Avoid overwriting an existing file if rerun.
    if new_path.exists():
        continue

    # Rename
    old_path.rename(new_path)

print("Renaming completed")