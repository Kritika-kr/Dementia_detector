import numpy as np
import os

feature_files = [
    "features/features_tfidf.npy",
    "features/features_count.npy",
    "features/features_word2vec.npy",
    "features/features_embed_word2vec.npy",
    "features/features_embed_glove.npy",
    "features/features_sentence_embed.npy",
    "features/features_frequency.npy",
    "features/features_hesitation.npy",
    "features/features_linguistic.npy"
]

features = []
for file_path in feature_files:
    if os.path.exists(file_path):
        arr = np.load(file_path)
        print(f"{file_path}: shape = {arr.shape}")
        features.append(arr)
    else:
        print(f"{file_path} not found — skipping.")

# Concatenate on axis=1 (column-wise)
combined_features = np.concatenate(features, axis=1)
print(f"Combined feature shape: {combined_features.shape}")

# Save combined features
os.makedirs("outputs/features", exist_ok=True)
np.save("outputs/features/combined_features.npy", combined_features)
print("Saved")
