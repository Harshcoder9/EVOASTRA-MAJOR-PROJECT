import numpy as np
import argparse
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_build import build_caption_model

def fine_tune(train_feats, train_in_seq, train_out_seq, tokenizer_path, max_len, output_weights, epochs=10, batch_size=32, lr=1e-4):
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    model = build_caption_model(vocab_size, max_len)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])
    X_in = np.load(train_in_seq, allow_pickle=True)
    y = np.load(train_out_seq, allow_pickle=True)
    n = len(y)
    def gen():
        idx = np.arange(n)
        while True:
            np.random.shuffle(idx)
            for i in range(0, n, batch_size):
                batch_idx = idx[i:i+batch_size]
                batch_in = X_in[batch_idx]
                batch_y = y[batch_idx]
                batch_feats = []
                feat_files = sorted([f for f in os.listdir(train_feats) if f.endswith(".npy")])
                for j in batch_idx:
                    feat_path = os.path.join(train_feats, feat_files[j % len(feat_files)])
                    batch_feats.append(np.load(feat_path))
                yield [np.array(batch_feats), np.array(batch_in)], np.array(batch_y)
    checkpoint = ModelCheckpoint(output_weights, save_weights_only=True, save_best_only=True, monitor="loss")
    stopper = EarlyStopping(patience=5)
    steps = max(1, n // batch_size)
    model.fit(gen(), epochs=epochs, steps_per_epoch=steps, callbacks=[checkpoint, stopper])
    model.save_weights(output_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-feats", required=True)
    parser.add_argument("--train-in-seq", required=True)
    parser.add_argument("--train-out-seq", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    fine_tune(args.train_feats, args.train_in_seq, args.train_out_seq, args.tokenizer, args.max_len, args.output, args.epochs, args.batch_size)
