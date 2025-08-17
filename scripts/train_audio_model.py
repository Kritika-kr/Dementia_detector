import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

AUDIO_DIR = "data/audio data"
MODEL_PATH = "audio_model_boosted.pkl"
SAMPLE_RATE = 16000
N_MFCC = 13

# -------------------------
# Audio Augmentation
# -------------------------
def augment_audio(y, sr):
    aug_data = [y]

    # Time stretch - safe
    for rate in [0.9, 1.1]:
        try:
            aug_data.append(librosa.effects.time_stretch(y, rate))
        except Exception:
            pass

    # Pitch shift
    for steps in [-2, 2]:
        try:
            aug_data.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=steps))
        except Exception:
            pass

    # Add noise
    noise = np.random.normal(0, 0.005, y.shape)
    aug_data.append(y + noise)

    # Volume change
    aug_data.append(y * 1.2)
    aug_data.append(y * 0.8)

    return aug_data


# -------------------------
# Feature Extraction
# -------------------------
def extract_features(y, sr):
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        # Mean & std
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1),
            np.std(mfcc_delta2, axis=1)
        ])
        return features
    except Exception:
        return None


# -------------------------
# Dataset Loader
# -------------------------
def load_audio_dataset(audio_dir):
    X, y = [], []
    dementia_count, control_count = 0, 0

    print(f"🎯 Loading audio data from: {audio_dir}")

    for label_name in ["dementia", "control"]:
        label = 1 if label_name == "dementia" else 0
        folder = os.path.join(audio_dir, label_name)

        if not os.path.exists(folder):
            print(f"⚠️ Folder not found: {folder}")
            continue

        files = [f for f in os.listdir(folder) if f.lower().endswith(".mp3")]
        for file in tqdm(files, desc=f"Loading {label_name}"):
            file_path = os.path.join(folder, file)
            try:
                y_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                if len(y_audio) < sr:  # pad if < 1 sec
                    y_audio = np.pad(y_audio, (0, sr - len(y_audio)), mode='constant')

                for aug in augment_audio(y_audio, sr):
                    feats = extract_features(aug, sr)
                    if feats is not None:
                        X.append(feats)
                        y.append(label)

                if label == 1:
                    dementia_count += 1
                else:
                    control_count += 1

            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")

    X, y = np.array(X), np.array(y)
    print(f"🎵 Loaded {len(X)} samples ({dementia_count} dementia, {control_count} control)")
    return X, y


# -------------------------
# Model Training
# -------------------------
def train_audio_model(audio_dir):
    X, y = load_audio_dataset(audio_dir)

    if len(X) == 0:
        raise ValueError("No audio data loaded! Check paths & file formats.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        n_jobs=-1
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_audio_model(AUDIO_DIR)
