import os
import cv2
import numpy as np
import argparse

def preprocess_images(input_dir, output_dir, size=(299,299)):
    os.makedirs(output_dir, exist_ok=True)
    for split in ["train","val"]:
        in_path = os.path.join(input_dir, split)
        out_path = os.path.join(output_dir, split)
        os.makedirs(out_path, exist_ok=True)
        if not os.path.isdir(in_path):
            continue
        for fname in os.listdir(in_path):
            if not fname.lower().endswith((".jpg",".jpeg",".png")):
                continue
            src = os.path.join(in_path, fname)
            img = cv2.imread(src)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size)
            arr = img.astype("float32")/255.0
            np.save(os.path.join(out_path, os.path.splitext(fname)[0] + ".npy"), arr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=299)
    parser.add_argument("--height", type=int, default=299)
    args = parser.parse_args()
    preprocess_images(args.input_dir, args.output_dir, (args.width, args.height))
