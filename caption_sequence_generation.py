import numpy as np
import argparse
import os
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_sequences_from_caption(seq, max_len):
    X, y = [], []
    for i in range(1, len(seq)):
        in_seq = seq[:i]
        out_word = seq[i]
        X.append(in_seq)
        y.append(out_word)
    X = pad_sequences(X, maxlen=max_len, padding="post")
    y = np.array(y)
    return X, y

def generate_sequences(seq_file, tokenizer_file, max_len, out_in, out_out):
    seqs = np.load(seq_file, allow_pickle=True)
    with open(tokenizer_file, "rb") as f:
        tokenizer = pickle.load(f)
    all_X, all_y = [], []
    for s in seqs:
        X, y = create_sequences_from_caption(s, max_len)
        all_X.append(X)
        all_y.append(y)
    X_all = np.vstack(all_X) if len(all_X)>0 else np.array([])
    y_all = np.concatenate(all_y) if len(all_y)>0 else np.array([])
    np.save(out_in, X_all)
    np.save(out_out, y_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-file", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--out-in", required=True)
    parser.add_argument("--out-out", required=True)
    args = parser.parse_args()
    generate_sequences(args.seq_file, args.tokenizer, args.max_len, args.out_in, args.out_out)
