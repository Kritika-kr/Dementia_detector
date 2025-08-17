import numpy as np
import pandas as pd
import os

# Paths
csv_path = "data/preprocessed_data.csv"
output_dir = "outputs/features"
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)

# Create synthetic filenames based on row index
filenames = [f"row_{i}" for i in range(len(df))]
np.save(os.path.join(output_dir, "filenames.npy"), np.array(filenames))

print(f"Saved filenames.npy with {len(filenames)} entries")
