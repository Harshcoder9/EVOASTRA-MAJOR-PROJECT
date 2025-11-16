import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add, RepeatVector, Reshape
from tensorflow.keras.models import Model

def build_caption_model(vocab_size, max_len, embedding_dim=256, lstm_units=256):
    image_input = Input(shape=(2048,), name="image_input")
    img_dense = Dense(embedding_dim, activation="relu")(image_input)
    img_repeat = RepeatVector(max_len)(img_dense)
    caption_input = Input(shape=(max_len,), name="caption_input")
    emb = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(caption_input)
    merged = Add()([img_repeat, emb])
    lstm = LSTM(lstm_units, return_sequences=False)(merged)
    out = Dense(vocab_size, activation="softmax")(lstm)
    model = Model(inputs=[image_input, caption_input], outputs=out)
    return model
