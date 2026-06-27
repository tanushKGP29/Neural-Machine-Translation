# Neural Machine Translation: English to Spanish

A sequence-to-sequence (Seq2Seq) neural machine translation system that translates English sentences to Spanish using LSTM-based architectures built with TensorFlow/Keras.

This repository contains both a **Baseline Seq2Seq Model** (using a simple Encoder-Decoder architecture) and an **Improved Seq2Seq Model** equipped with a Bidirectional LSTM encoder, Luong-style attention, and Beam Search decoding.

---

## 📋 Project Overview

This project implements a neural machine translation (NMT) model that learns to translate English sentences to Spanish. The models use LSTM (Long Short-Term Memory) networks to capture sequential dependencies in both source and target languages.

### 🌟 Key Enhancements in the Improved Model:
- **Bidirectional Encoder**: Processes input sequences forward and backward to capture complete context.
- **Luong Attention Mechanism**: Uses dot-product attention to dynamically focus on relevant parts of the source sentence during decoding.
- **Large Model Capacity**: Increased embedding dimensions (256) and LSTM hidden states (512 units).
- **Beam Search Decoding**: Decodes translations by keeping track of the top $K$ (e.g., 5) candidate sequences with length normalization, rather than just choosing the single best word at each step (greedy decoding).
- **Proper Language Preservation**: Normalizes Spanish text while explicitly preserving accented characters (`ñáéíóúü`) crucial to Spanish grammar.
- **Label Smoothing & Regularization**: Implements Dropout (0.3) and label smoothing (0.1) for better generalization.

---

## 🗂️ Dataset

The project uses the **Tatoeba English-Spanish parallel corpus** (`spa.txt`):
- **Size**: 123,771 sentence pairs
- **Format**: Tab-separated values (English | Spanish | Attribution)
- **Source**: [tatoeba.org](https://tatoeba.org)

*Note: For the improved model, the dataset is restricted to a clean subset of the first 80,000 sentence pairs.*

---

## 🏗️ Models & Architectures

### 1. Baseline Model (`nmt_eng2spa.py` / `nmt_eng2spa.ipynb`)
- **Encoder**: Single-direction LSTM (256 units).
- **Decoder**: Single-direction LSTM (256 units).
- **Decoding**: Greedy decoding.
- **Model File**: Saved to `my_models/nmt_eng2spa.h5`.

### 2. Improved Model (`train_improved.py`)
- **Encoder**: Bidirectional LSTM (512 units).
- **Decoder**: LSTM (512 units) with Luong attention.
- **Decoding**: Beam Search decoding (width = 5).
- **Model File**: Saved to `my_models/nmt_improved.keras`.

---

## 🚀 Installation & Requirements

Ensure you are using **Python 3.12** (TensorFlow 2.16.2 does not fully support Python 3.13).

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Training the Baseline Model
You can run the script or step through the notebook:
```bash
python nmt_eng2spa.py
```
Or open the notebook:
```bash
jupyter notebook nmt_eng2spa.ipynb
```

### Training the Improved Model (BiLSTM + Attention)
To train the improved model:
```bash
python train_improved.py
```

### Evaluating BLEU Scores
To evaluate the translation quality of the trained models using BLEU-4 scores on the test set:
```bash
python evaluate_bleu.py
```

---

## 📁 Project Structure

```
Neural Machine Translation/
│
├── .gitignore              # Ignores large datasets and binary weights
├── requirements.txt        # Project package dependencies
├── nmt_eng2spa.py          # Baseline NMT training script
├── nmt_eng2spa.ipynb        # Baseline NMT Jupyter Notebook
├── train_improved.py       # Improved NMT with Attention training script
├── evaluate_bleu.py        # BLEU evaluation & inference script
├── test_gen.py             # Small helper script for checking batches
├── bleu_results.txt        # Exported BLEU evaluation logs
├── spa.txt                 # Dataset (untracked)
└── my_models/              # Directory for model checkpoints (untracked)
    ├── nmt_eng2spa.h5      # Baseline model weights
    └── nmt_improved.keras  # Improved model weights
```
