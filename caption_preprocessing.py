import os
import pandas as pd
import numpy as np
import string
import pickle
import argparse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_caption(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return "<start> " + text + " <end>"

def build_tokenizer(captions, num_words=None, oov_token="unk"):
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token, filters="")
    tokenizer.fit_on_texts(captions)
    return tokenizer

def captions_to_sequences(tokenizer, captions, max_len=None):
    seqs = tokenizer.texts_to_sequences(captions)
    if max_len is None:
        max_len = max(len(s) for s in seqs)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post")
    return padded, max_len

def save_tokenizer(tokenizer, path):
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)

def load_captions(csv_path):
    df = pd.read_csv(csv_path)
    if "image" not in df.columns or "caption" not in df.columns:
        cols = df.columns.tolist()
        df = df[[cols[0], cols[1]]]
        df.columns = ["image","caption"]
    df["caption"] = df["caption"].apply(clean_caption)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions", required=True)
    parser.add_argument("--tokenizer-out", required=True)
    parser.add_argument("--sequences-out", required=True)
    parser.add_argument("--max-words", type=int, default=None)
    args = parser.parse_args()
    df = load_captions(args.captions)
    captions = df["caption"].tolist()
    tokenizer = build_tokenizer(captions, num_words=args.max_words)
    padded, max_len = captions_to_sequences(tokenizer, captions)
    save_tokenizer(tokenizer, args.tokenizer_out)
    np.save(args.sequences_out, padded)
    print(max_len)

if __name__ == "__main__":
    main()
