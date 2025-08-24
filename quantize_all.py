import os
import joblib
import numpy as np

MODELS_DIR = "outputs/models"
FEATURES_DIR = "features"

def quantize_sklearn_models():
    for file in os.listdir(MODELS_DIR):
        if file.endswith(".pkl"):
            path = os.path.join(MODELS_DIR, file)
            try:
                model = joblib.load(path)
                save_path = path.replace(".pkl", "_quantized.pkl")
                joblib.dump(model, save_path, compress=3)  # compress=3 balances size and speed
                print(f"✅ Quantized Sklearn model {file} → {save_path}")
            except Exception as e:
                print(f"⚠️ Skipped {file}, error: {e}")

def quantize_numpy_features():
    for file in os.listdir(FEATURES_DIR):
        if file.endswith(".npy"):
            path = os.path.join(FEATURES_DIR, file)
            try:
                arr = np.load(path)
                arr = arr.astype(np.float16)  # reduce precision
                save_path = path.replace(".npy", "_quantized.npy")
                np.save(save_path, arr)
                print(f"✅ Quantized Features {file} → {save_path}")
            except Exception as e:
                print(f"⚠️ Skipped {file}, error: {e}")

def quantize_transformers():
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    for file in os.listdir(MODELS_DIR):
        if file.endswith(".bin"):
            path = os.path.join(MODELS_DIR, file)

            # Try HuggingFace model load
            try:
                model_name = os.path.splitext(file)[0]
                save_dir = os.path.join(MODELS_DIR, model_name + "_quantized")

                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModel.from_pretrained(path, quantization_config=bnb_config)
                tokenizer = AutoTokenizer.from_pretrained(path)

                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)

                print(f"✅ Quantized Transformer {file} → {save_dir}")
                continue
            except Exception:
                # If fails, maybe it's a Word2Vec .bin
                try:
                    from gensim.models import KeyedVectors
                    wv = KeyedVectors.load_word2vec_format(path, binary=True)
                    wv.vectors = wv.vectors.astype(np.float16)
                    save_path = path.replace(".bin", "_quantized.kv")
                    wv.save(save_path)
                    print(f"✅ Quantized Word2Vec {file} → {save_path}")
                except Exception as e:
                    print(f"⚠️ Skipped {file}, could not quantize as Transformer or Word2Vec. Error: {e}")

if __name__ == "__main__":
    quantize_sklearn_models()
    quantize_numpy_features()
    quantize_transformers()
