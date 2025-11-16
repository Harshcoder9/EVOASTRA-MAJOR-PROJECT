import numpy as np
import argparse
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_build import build_caption_model
import os

def data_generator(feat_dir, in_seq_file, out_seq_file, batch_size=32):
    X_feats = []
    for f in sorted(os.listdir(feat_dir)):
        if f.lower().endswith(".npy"):
            X_feats.append(os.path.join(feat_dir, f))
    X_in = np.load(in_seq_file, allow_pickle=True)
    y = np.load(out_seq_file, allow_pickle=True)
    n = len(y)
    idx = np.arange(n)
    while True:
        np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_in = X_in[batch_idx]
            batch_y = y[batch_idx]
            batch_feats = []
            for j in batch_idx:
                fname = os.path.splitext(os.path.basename(X_feats[j % len(X_feats)]))[0]
                feat_path = X_feats[j % len(X_feats)]
                feat = np.load(feat_path)
                batch_feats.append(feat)
            X1 = np.array(batch_feats)
            X2 = np.array(batch_in)
            yield [X1, X2], batch_y

def train(train_feats, val_feats, train_in_seq, train_out_seq, val_in_seq, val_out_seq, tokenizer_path, max_len, epochs, batch_size, output_weights):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    model = build_caption_model(vocab_size, max_len)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    steps_per_epoch = max(1, int(np.load(train_out_seq, allow_pickle=True).shape[0] / batch_size))
    val_steps = max(1, int(np.load(val_out_seq, allow_pickle=True).shape[0] / batch_size))
    checkpoint = ModelCheckpoint(output_weights, save_weights_only=True, save_best_only=True, monitor="val_loss")
    stopper = EarlyStopping(patience=5, restore_best_weights=True)
    train_gen = data_generator(train_feats, train_in_seq, train_out_seq, batch_size)
    val_gen = data_generator(val_feats, val_in_seq, val_out_seq, batch_size)
    model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_gen, validation_steps=val_steps, callbacks=[checkpoint, stopper])
    model.save_weights(output_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feats", required=True)
    parser.add_argument("--val-feats", required=True)
    parser.add_argument("--train-in-seq", required=True)
    parser.add_argument("--train-out-seq", required=True)
    parser.add_argument("--val-in-seq", required=True)
    parser.add_argument("--val-out-seq", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-weights", required=True)
    args = parser.parse_args()
    train(args.train_feats, args.val_feats, args.train_in_seq, args.train_out_seq, args.val_in_seq, args.val_out_seq, args.tokenizer, args.max_len, args.epochs, args.batch_size, args.output_weights)
