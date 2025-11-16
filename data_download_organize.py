import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def organize_dataset(images_dir, captions_file, output_dir, test_size=0.2, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(captions_file)
    if "image" not in df.columns or "caption" not in df.columns:
        cols = df.columns.tolist()
        if len(cols) >= 2:
            df = df[[cols[0], cols[1]]]
            df.columns = ["image", "caption"]
        else:
            raise ValueError("captions_file must contain at least two columns")
    df["image"] = df["image"].apply(lambda x: os.path.basename(str(x)))
    images = df["image"].unique().tolist()
    train_imgs, val_imgs = train_test_split(images, test_size=test_size, random_state=seed)
    img_out = os.path.join(output_dir, "images")
    cap_out = os.path.join(output_dir, "captions")
    for p in [img_out, cap_out]:
        os.makedirs(os.path.join(p, "train"), exist_ok=True)
        os.makedirs(os.path.join(p, "val"), exist_ok=True)
    for img in train_imgs:
        src = os.path.join(images_dir, img)
        dst = os.path.join(img_out, "train", img)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    for img in val_imgs:
        src = os.path.join(images_dir, img)
        dst = os.path.join(img_out, "val", img)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    df[df["image"].isin(train_imgs)].to_csv(os.path.join(cap_out, "captions_train.csv"), index=False)
    df[df["image"].isin(val_imgs)].to_csv(os.path.join(cap_out, "captions_val.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--captions-file", required=True)
    parser.add_argument("--output-dir", default="./processed_data")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    organize_dataset(args.images_dir, args.captions_file, args.output_dir, args.test_size)
