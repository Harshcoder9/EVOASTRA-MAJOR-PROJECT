import numpy as np
import argparse
import pickle
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from model_build import build_caption_model
import os

def greedy_search(model, tokenizer, photo, max_len):
    in_text = "<start>"
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = tf.keras.preprocessing.sequence.pad_sequences([sequence], maxlen=max_len, padding="post")
        yhat = model.predict([np.expand_dims(photo, 0), sequence], verbose=0)
        yhat = np.argmax(yhat, axis=-1)[0]
        word = None
        for w, i in tokenizer.word_index.items():
            if i == yhat:
                word = w
                break
        if word is None:
            break
        in_text += " " + word
        if word == "<end>":
            break
    return in_text.replace("<start> ", "").replace(" <end>", "")

def evaluate_model(model, tokenizer, features, references, max_len):
    smoothie = SmoothingFunction().method4
    scores = []
    for i in range(len(features)):
        photo = features[i]
        reference = references[i]
        pred = greedy_search(model, tokenizer, photo, max_len).split()
        ref = [reference.replace("<start>","").replace("<end>","").split()]
        score = sentence_bleu(ref, pred, smoothing_function=smoothie)
        scores.append(score)
    return float(np.mean(scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--captions", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--max-len", type=int, required=True)
    args = parser.parse_args()
    with open(args.tokenizer, "rb") as f:
        tokenizer = pickle.load(f)
    vocab_size = len(tokenizer.word_index) + 1
    model = build_caption_model(vocab_size, args.max_len)
    model.load_weights(args.weights)
    features = np.load(args.features, allow_pickle=True)
    captions = np.load(args.captions, allow_pickle=True)
    score = evaluate_model(model, tokenizer, features, captions, args.max_len)
    print(score)
