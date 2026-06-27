"""
NMT BLEU Evaluation Script
Loads the saved improved model (BiLSTM + Attention), runs inference
on test set, and computes BLEU score.

Can be run standalone to evaluate a previously trained model.
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Embedding, Input, Dropout,
    Bidirectional, Concatenate, Dot, Activation
)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

print("=" * 60)
print("NMT Evaluation Script - BLEU Score Computation")
print("=" * 60)

# ─── Configuration (must match training!) ────────────────────────────────────
LATENT_DIM   = 512
EMBED_DIM    = 256
MAX_SAMPLES  = 80_000
TEST_SIZE    = 0.10
MIN_FREQ     = 2
RANDOM_SEED  = 42
BEAM_WIDTH   = 5
NUM_EVAL     = 500
MODEL_PATH   = 'my_models/nmt_improved.keras'

np.random.seed(RANDOM_SEED)

# ─── 1. Load and preprocess data (identical to training) ─────────────────────
print("\n[1/4] Loading and preprocessing data ...")
data = pd.read_table(
    'spa.txt',
    header=None,
    names=['eng', 'spa', 'attrib'],
    usecols=[0, 1],
    sep='\t'
)
data = data.dropna()
data = data.drop_duplicates().reset_index(drop=True)
data = data.iloc[:MAX_SAMPLES]

def clean_eng(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def clean_spa(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z\s\u00f1\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

X_raw = data['eng'].apply(clean_eng)
y_raw = data['spa'].apply(clean_spa)
y_raw = y_raw.apply(lambda s: 'bos ' + s + ' eos')

mask = (X_raw.str.len() > 0) & (y_raw.str.len() > 5)
X_raw = X_raw[mask].reset_index(drop=True)
y_raw = y_raw[mask].reset_index(drop=True)

print(f"      Dataset: {len(X_raw):,} sentence pairs")

# ─── 2. Build vocabularies ───────────────────────────────────────────────────
print("[2/4] Building vocabularies ...")

def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())
    vocab = ['<PAD>', '<UNK>'] + [w for w, c in counter.most_common() if c >= min_freq]
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return vocab, word2idx, idx2word

eng_vocab, eng_w2i, eng_i2w = build_vocab(X_raw, min_freq=MIN_FREQ)
spa_vocab, spa_w2i, spa_i2w = build_vocab(y_raw, min_freq=MIN_FREQ)

num_enc_tokens = len(eng_vocab)
num_dec_tokens = len(spa_vocab)

print(f"      English vocab: {num_enc_tokens:,}")
print(f"      Spanish vocab: {num_dec_tokens:,}")

# ─── 3. Encode sequences ─────────────────────────────────────────────────────
actual_max_eng = max(len(s.split()) for s in X_raw)
actual_max_spa = max(len(s.split()) for s in y_raw)
MAX_ENG = min(25, actual_max_eng)
MAX_SPA = min(30, actual_max_spa)
DEC_SEQ_LEN = MAX_SPA - 1

def encode(sentences, word2idx, max_len):
    unk_idx = word2idx.get('<UNK>', 0)
    out = np.zeros((len(sentences), max_len), dtype=np.int32)
    for i, s in enumerate(sentences):
        for j, w in enumerate(s.split()[:max_len]):
            out[i, j] = word2idx.get(w, unk_idx)
    return out

X_enc = encode(X_raw, eng_w2i, MAX_ENG)
y_enc = encode(y_raw, spa_w2i, MAX_SPA)

# Same split as training
X_enc, y_enc = shuffle(X_enc, y_enc, random_state=RANDOM_SEED)
x_tr, x_te, y_tr, y_te = train_test_split(
    X_enc, y_enc, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

print(f"      Test set: {len(x_te):,} samples")

# ─── 4. Rebuild model and load weights ───────────────────────────────────────
print("\n[3/4] Rebuilding model and loading weights ...")

if not os.path.exists(MODEL_PATH):
    print(f"\nERROR: Model file not found at {MODEL_PATH}")
    print("Please train the model first by running: python train_improved.py")
    sys.exit(1)

# === Encoder ===
enc_input = Input(shape=(MAX_ENG,), name='enc_input')
enc_emb   = Embedding(num_enc_tokens, EMBED_DIM, mask_zero=True, name='enc_emb')(enc_input)
enc_emb   = Dropout(0.0)(enc_emb)  # No dropout at inference

enc_lstm = Bidirectional(
    LSTM(LATENT_DIM // 2, return_sequences=True, return_state=True, name='enc_lstm'),
    name='bi_enc'
)
enc_out, fwd_h, fwd_c, bwd_h, bwd_c = enc_lstm(enc_emb)

enc_h = Concatenate(name='merge_h')([fwd_h, bwd_h])
enc_c = Concatenate(name='merge_c')([fwd_c, bwd_c])
enc_h = Dense(LATENT_DIM, activation='tanh', name='proj_h')(enc_h)
enc_c = Dense(LATENT_DIM, activation='tanh', name='proj_c')(enc_c)
encoder_states = [enc_h, enc_c]

# === Decoder ===
dec_input = Input(shape=(DEC_SEQ_LEN,), name='dec_input')
dec_emb_layer = Embedding(num_dec_tokens, EMBED_DIM, mask_zero=True, name='dec_emb')
dec_emb   = dec_emb_layer(dec_input)

dec_lstm_layer = LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='dec_lstm')
dec_out_seq, _, _ = dec_lstm_layer(dec_emb, initial_state=encoder_states)

# Attention
scores   = Dot(axes=[2, 2], name='attn_scores')([dec_out_seq, enc_out])
attn_w   = Activation('softmax', name='attn_weights')(scores)
context  = Dot(axes=[2, 1], name='context')([attn_w, enc_out])
combined = Concatenate(name='combined')([dec_out_seq, context])

attn_proj = Dense(LATENT_DIM, activation='tanh', name='attn_proj')(combined)
output = Dense(num_dec_tokens, activation='softmax', name='output')(attn_proj)

model = Model([enc_input, dec_input], output)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Load trained weights
print(f"      Loading model from {MODEL_PATH} ...")
model.load_weights(MODEL_PATH)
print("      Model loaded successfully!")

# ─── 5. Build inference models ───────────────────────────────────────────────
enc_model = Model(enc_input, [enc_out, enc_h, enc_c])

# Decoder inference model (one step)
dec_state_in_h = Input(shape=(LATENT_DIM,), name='state_h_in')
dec_state_in_c = Input(shape=(LATENT_DIM,), name='state_c_in')
enc_out_in     = Input(shape=(MAX_ENG, LATENT_DIM), name='enc_out_in')

dec_in_single  = Input(shape=(1,), name='dec_single_in')
dec_emb_single = model.get_layer('dec_emb')(dec_in_single)
dec_seq_single, st_h, st_c = model.get_layer('dec_lstm')(
    dec_emb_single, initial_state=[dec_state_in_h, dec_state_in_c]
)

sc   = Dot(axes=[2, 2])([dec_seq_single, enc_out_in])
aw   = Activation('softmax')(sc)
ctx  = Dot(axes=[2, 1])([aw, enc_out_in])
comb = Concatenate()([dec_seq_single, ctx])
attn_p = model.get_layer('attn_proj')(comb)
out_single = model.get_layer('output')(attn_p)

dec_model = Model(
    [dec_in_single, dec_state_in_h, dec_state_in_c, enc_out_in],
    [out_single, st_h, st_c]
)

print("      Inference models built!\n")

# ─── Translation functions ───────────────────────────────────────────────────
bos_idx = spa_w2i.get('bos', 0)
eos_idx = spa_w2i.get('eos', 0)

def translate_greedy(enc_seq):
    enc_seq = enc_seq.reshape(1, -1)
    res = enc_model(enc_seq, training=False)
    enc_outs, h, c = res[0].numpy(), res[1].numpy(), res[2].numpy()
    target = np.array([[bos_idx]])
    decoded = []
    for _ in range(MAX_SPA):
        res_dec = dec_model([target, h, c, enc_outs], training=False)
        out_tok, h, c = res_dec[0].numpy(), res_dec[1].numpy(), res_dec[2].numpy()
        tok_id = np.argmax(out_tok[0, 0])
        if tok_id == eos_idx or tok_id == 0:
            break
        word = spa_i2w.get(tok_id, '<UNK>')
        if word not in ('<PAD>', '<UNK>'):
            decoded.append(word)
        target = np.array([[tok_id]])
    return decoded

def translate_beam(enc_seq, beam_width=BEAM_WIDTH):
    enc_seq = enc_seq.reshape(1, -1)
    res = enc_model(enc_seq, training=False)
    enc_outs, h, c = res[0].numpy(), res[1].numpy(), res[2].numpy()

    beams = [(0.0, [bos_idx], h, c)]
    completed = []

    for step in range(MAX_SPA):
        candidates = []
        for log_prob, seq, h_state, c_state in beams:
            last_tok = np.array([[seq[-1]]])
            res_dec = dec_model([last_tok, h_state, c_state, enc_outs], training=False)
            out_tok, new_h, new_c = res_dec[0].numpy(), res_dec[1].numpy(), res_dec[2].numpy()
            probs = out_tok[0, 0]
            top_k = min(beam_width, len(probs))
            top_indices = np.argsort(probs)[-top_k:][::-1]

            for idx in top_indices:
                new_log_prob = log_prob + np.log(probs[idx] + 1e-10)
                new_seq = seq + [idx]
                if idx == eos_idx or idx == 0:
                    score = new_log_prob / (len(new_seq) ** 0.6)
                    completed.append((score, new_seq, new_h, new_c))
                else:
                    candidates.append((new_log_prob, new_seq, new_h, new_c))

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        if len(completed) >= beam_width:
            break

    if not completed:
        completed = [(b[0]/max(len(b[1]),1)**0.6, b[1], b[2], b[3]) for b in beams]

    completed.sort(key=lambda x: x[0], reverse=True)
    best_seq = completed[0][1]

    decoded = []
    for tok_id in best_seq:
        if tok_id == bos_idx:
            continue
        if tok_id == eos_idx or tok_id == 0:
            break
        word = spa_i2w.get(tok_id, '<UNK>')
        if word not in ('<PAD>', '<UNK>'):
            decoded.append(word)
    return decoded

# ─── 6. Compute BLEU ─────────────────────────────────────────────────────────
print(f"[4/4] Computing BLEU on {min(NUM_EVAL, len(x_te))} test sentences (beam search) ...")
print("-" * 60)

n_eval = min(NUM_EVAL, len(x_te))
references = []
hypotheses = []
smoothie = SmoothingFunction().method4
sample_translations = []

for i in range(n_eval):
    if (i + 1) % 50 == 0:
        print(f"  Translated {i + 1}/{n_eval} sentences...")

    ref_ids = y_te[i].tolist()
    ref_words = [spa_i2w[t] for t in ref_ids
                 if t > 0 and spa_i2w.get(t, '') not in ('bos', 'eos', '<PAD>', '<UNK>')]
    references.append([ref_words])

    hyp_words = translate_beam(x_te[i])
    hypotheses.append(hyp_words)

    if i < 10:
        src_words = [eng_i2w[t] for t in x_te[i] if t > 0]
        sample_translations.append((' '.join(src_words), ' '.join(ref_words), ' '.join(hyp_words)))

# ─── Print results ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAMPLE TRANSLATIONS")
print("=" * 60)
for src, ref, hyp in sample_translations:
    print(f"  SRC: {src}")
    print(f"  REF: {ref}")
    print(f"  HYP: {hyp}")
    print()

bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),
                     smoothing_function=smoothie) * 100
bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),
                     smoothing_function=smoothie) * 100
bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0),
                     smoothing_function=smoothie) * 100
bleu_4 = corpus_bleu(references, hypotheses,
                     smoothing_function=smoothie) * 100

print("=" * 60)
print("BLEU SCORES")
print("=" * 60)
print(f"  BLEU-1: {bleu_1:.2f}")
print(f"  BLEU-2: {bleu_2:.2f}")
print(f"  BLEU-3: {bleu_3:.2f}")
print(f"  BLEU-4: {bleu_4:.2f}")
print("=" * 60)

# Use ASCII-safe characters to avoid Windows cp1252 encoding errors
if bleu_4 >= 25:
    print(f"[OK] BLEU-4 score is {bleu_4:.2f} (>= 25). Resume-worthy!")
else:
    print(f"[!!] BLEU-4 score is {bleu_4:.2f} (< 25). Model needs improvement.")
    print("     Run train_improved.py to train a better model.")

# Save results
with open('bleu_results.txt', 'w', encoding='utf-8') as f:
    f.write(f"BLEU-1: {bleu_1:.2f}\n")
    f.write(f"BLEU-2: {bleu_2:.2f}\n")
    f.write(f"BLEU-3: {bleu_3:.2f}\n")
    f.write(f"BLEU-4: {bleu_4:.2f}\n")
    f.write(f"\nSample Translations:\n")
    for src, ref, hyp in sample_translations:
        f.write(f"SRC: {src}\nREF: {ref}\nHYP: {hyp}\n\n")

print("\nResults saved to bleu_results.txt")
