import os
import numpy as np
from tqdm import tqdm
import argparse
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

def load_model():
    base = InceptionV3(weights="imagenet", include_top=True)
    model = tf.keras.Model(base.input, base.layers[-2].output)
    return model

def extract_features(model, npy_path):
    arr = np.load(npy_path)
    x = arr * 255.0
    x = tf.image.resize(x, (299,299))
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    feat = model.predict(x, verbose=0)
    return feat.squeeze()

def process_folder(model, folder_in, folder_out):
    os.makedirs(folder_out, exist_ok=True)
    for fname in tqdm(os.listdir(folder_in)):
        if not fname.lower().endswith(".npy"):
            continue
        src = os.path.join(folder_in, fname)
        outp = os.path.join(folder_out, os.path.splitext(fname)[0] + ".npy")
        if os.path.exists(outp):
            continue
        try:
            feat = extract_features(model, src)
            np.save(outp, feat)
        except Exception:
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-train", required=True)
    parser.add_argument("--input-val", required=True)
    parser.add_argument("--output-dir", default="./features")
    args = parser.parse_args()
    model = load_model()
    process_folder(model, args.input_train, os.path.join(args.output_dir, "train"))
    process_folder(model, args.input_val, os.path.join(args.output_dir, "val"))

if __name__ == "__main__":
    main()
