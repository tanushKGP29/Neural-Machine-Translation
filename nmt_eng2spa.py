"""
Neural Machine Translation: English -> Spanish (Baseline Seq2Seq)

This is the original baseline model (no attention). For the improved model
with attention and better BLEU scores, use train_improved.py instead.

Fixes applied:
  - Correct data path (spa.txt, not data/spa.txt)
  - Spanish accented characters preserved
  - Fixed batch generator edge case (last batch padding)
  - Fixed save path (local, not Google Drive)
  - Fixed vocab size (added +1 for padding index 0)
  - Consistent random seeds
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import re

RANDOM_SEED = 42

# ─── Load data ───────────────────────────────────────────────────────────────
# Use header=None so the first row ("Go." / "Ve.") is treated as data, not headers
data = pd.read_table(
    'spa.txt',
    header=None,
    names=['eng', 'spa', 'attrib'],
    usecols=[0, 1],
    sep='\t'
)
data = data.dropna()

X = data['eng'].apply(lambda x: str(x).lower())
y = data['spa'].apply(lambda x: str(x).lower())

# English: only keep a-z and spaces
X = X.apply(lambda x: re.sub(r"[^a-z\s]", " ", x))
X = X.apply(lambda x: re.sub(r"\s+", " ", x).strip())

# Spanish: keep a-z, accented characters (ñ, á, é, í, ó, ú, ü), and spaces
y = y.apply(lambda x: re.sub(r"[^a-z\s\u00f1\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc]", " ", x))
y = y.apply(lambda x: re.sub(r"\s+", " ", x).strip())

# Add START/END tokens
y = y.apply(lambda x: 'START_ ' + x + ' _END')

# ─── Build vocabularies ──────────────────────────────────────────────────────
eng_vocab, spa_vocab = set(), set()
for sent in X:
    for word in sent.split():
        eng_vocab.add(word)
for sent in y:
    for word in sent.split():
        spa_vocab.add(word)

engVocab = sorted(list(eng_vocab))
spaVocab = sorted(list(spa_vocab))

# Find maximum sentence length
max_eng_sent_length = max(len(l.split()) for l in X)
max_spa_sent_length = max(len(l.split()) for l in y)

# Word to index mappings (index 0 reserved for padding)
eng_word2idx = {word: i + 1 for i, word in enumerate(engVocab)}
spa_word2idx = {word: i + 1 for i, word in enumerate(spaVocab)}

# Index to word mappings
eng_idx2word = {i: word for word, i in eng_word2idx.items()}
spa_idx2word = {i: word for word, i in spa_word2idx.items()}

# ─── Prepare data for training ───────────────────────────────────────────────
X, y = shuffle(X, y, random_state=RANDOM_SEED)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_SEED
)

# +1 to account for padding index 0
num_encoder_tokens = len(engVocab) + 1
num_decoder_tokens = len(spaVocab) + 1

# ─── Batch generator ─────────────────────────────────────────────────────────
def generate_batch(X=x_train, y=y_train, batch_size=128):
    """Generator that yields (encoder_input, decoder_input), decoder_output batches."""
    while True:
        for i in range(0, len(X), batch_size):
            # Handle last batch which may be smaller
            actual_batch = min(batch_size, len(X) - i)
            encoder_input_data = np.zeros((actual_batch, max_eng_sent_length), dtype="float32")
            decoder_input_data = np.zeros((actual_batch, max_spa_sent_length), dtype="float32")
            decoder_output_data = np.zeros(
                (actual_batch, max_spa_sent_length, num_decoder_tokens), dtype="float32"
            )
            for j, (input_text, target_text) in enumerate(
                zip(X[i:i + actual_batch], y[i:i + actual_batch])
            ):
                for k, word in enumerate(input_text.split()):
                    if word in eng_word2idx:
                        encoder_input_data[j, k] = eng_word2idx[word]
                for k, word in enumerate(target_text.split()):
                    if word in spa_word2idx:
                        if k < len(target_text.split()) - 1:
                            decoder_input_data[j, k] = spa_word2idx[word]
                        if k > 0:
                            decoder_output_data[j, k - 1, spa_word2idx[word]] = 1
            yield ([encoder_input_data, decoder_input_data], decoder_output_data)

# ─── Model configuration ─────────────────────────────────────────────────────
train_samples = len(x_train)
val_samples = len(x_test)
batch_size = 128
epochs = 50
latent_dim = 256

# ─── Encoder ─────────────────────────────────────────────────────────────────
encoder_inputs = Input(shape=(None,))
enc_emb_layer = Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
enc_lstm_layer = LSTM(units=latent_dim, return_state=True)
encoder_outputs, h_state, c_state = enc_lstm_layer(enc_emb_layer)
encoder_states = [h_state, c_state]

# ─── Decoder ─────────────────────────────────────────────────────────────────
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim, mask_zero=True)
dec_emb = dec_emb_layer(decoder_inputs)
dec_lstm_layer = LSTM(units=latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = dec_lstm_layer(dec_emb, initial_state=encoder_states)
dec_dense = Dense(units=num_decoder_tokens, activation="softmax")
decoder_outputs = dec_dense(decoder_outputs)

# ─── Model ───────────────────────────────────────────────────────────────────
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

import os
os.makedirs('my_models', exist_ok=True)

checkpoint = ModelCheckpoint(
    'my_models/nmt_eng2spa.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

reduceLR = ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.0001
)

callbacks = [checkpoint, reduceLR]

hist = model.fit(
    generate_batch(X=x_train, y=y_train),
    steps_per_epoch=train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
    validation_data=generate_batch(X=x_test, y=y_test),
    validation_steps=val_samples // batch_size,
)

# Save final weights locally (not to Google Drive)
model.save_weights('my_models/nmt_eng2spa_final.h5')
print("Training complete. Model saved to my_models/")
