# Neural-Machine-Translation
# Neural Machine Translation: English to Spanish

A sequence-to-sequence neural machine translation system that translates English sentences to Spanish using LSTM-based encoder-decoder architecture with TensorFlow/Keras.

## 📋 Project Overview

This project implements a neural machine translation (NMT) model that learns to translate English sentences to Spanish. The model uses an encoder-decoder architecture with LSTM (Long Short-Term Memory) networks to capture sequential dependencies in both source and target languages.

## 🎯 Features

- **Encoder-Decoder Architecture**: LSTM-based sequence-to-sequence model
- **Word-level Translation**: Tokenization and vocabulary management for both languages
- **Training Optimization**: 
  - Learning rate reduction on plateau
  - Model checkpointing to save best performing models
  - Batch generation for efficient memory usage
- **Data Preprocessing**: Automatic text cleaning and normalization
- **START/END Tokens**: Proper sequence delimiting for decoder training

## 🗂️ Dataset

The project uses the **Tatoeba English-Spanish parallel corpus** (`spa.txt`):
- **Size**: 123,771 sentence pairs
- **Format**: Tab-separated values (English | Spanish | Attribution)
- **Source**: [tatoeba.org](https://tatoeba.org)

## 🏗️ Model Architecture

### Encoder
- **Input Layer**: Variable-length sequences
- **Embedding Layer**: Converts word indices to dense vectors (256 dimensions)
- **LSTM Layer**: Processes input sequence and outputs hidden states
- **Output**: Context vectors (h_state, c_state) passed to decoder

### Decoder
- **Input Layer**: Variable-length sequences with START token
- **Embedding Layer**: Spanish word embeddings (256 dimensions)
- **LSTM Layer**: Generates output sequence conditioned on encoder states
- **Dense Layer**: Softmax activation for word prediction
- **Output**: Spanish word probabilities at each timestep

### Model Parameters
- **Latent Dimensions**: 256
- **Batch Size**: 128
- **Epochs**: 50
- **Train/Test Split**: 85% / 15%
- **Optimizer**: Adam (default)
- **Loss Function**: Categorical Crossentropy

## 📊 Data Preprocessing

1. **Text Normalization**:
   - Convert to lowercase
   - Remove non-alphabetic characters
   - Clean whitespace

2. **Special Tokens**:
   - Add `START_` prefix to target sentences
   - Add `_END` suffix to target sentences

3. **Vocabulary Creation**:
   - Build separate vocabularies for English and Spanish
   - Create word-to-index and index-to-word mappings

4. **Sequence Padding**:
   - Pad sequences to maximum sentence length
   - English max length: 47 words
   - Spanish max length: 50 words

## 🚀 Installation

### Requirements
```bash
pip install tensorflow
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
```

## 💻 Usage

### Training the Model

1. Ensure `spa.txt` is in the project directory
2. Open the Jupyter notebook:
```bash
jupyter notebook nmt_eng2spa.ipynb
```
3. Run all cells to train the model

### Model Files
The trained model is saved as:
- `my_models/nmt_eng2spa.h5` (best model based on validation loss)

### Batch Generation
The project uses a custom generator function for memory-efficient training:
```python
generate_batch(X=x_train, y=y_train, batch_size=128)
```

## 📈 Training Details

### Callbacks
- **ModelCheckpoint**: Saves the best model based on validation loss
- **ReduceLROnPlateau**: Reduces learning rate by factor of 0.2 when validation loss plateaus

### Training Configuration
```python
callbacks = [
    ModelCheckpoint(
        filepath='my_models/nmt_eng2spa.h5',
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_delta=0.0001
    )
]
```

## 📁 Project Structure

```
neural-machine-translation/
│
├── nmt_eng2spa.ipynb      # Main training notebook
├── spa.txt                 # English-Spanish parallel corpus
├── my_models/
│   └── nmt_eng2spa.h5     # Saved model weights
└── README.md              # Project documentation
```

## 🔍 Key Components

### 1. Data Loading and Preprocessing
- Load parallel corpus from `spa.txt`
- Text cleaning and normalization
- Vocabulary creation

### 2. Model Building
- Encoder-decoder LSTM architecture
- Embedding layers for both languages
- Dense output layer with softmax

### 3. Training
- Batch generation for efficient memory usage
- Model checkpointing
- Learning rate scheduling

### 4. Inference (Future Work)
- Decoder inference mode
- Greedy decoding or beam search
- Translation evaluation (BLEU score)

## 🎓 Technical Details

### Word Embeddings
- Trainable embeddings learned from scratch
- Dimension: 256
- Mask zero padding enabled

### LSTM Configuration
- Units: 256
- Return sequences: True (decoder only)
- Return states: True (for passing context)

### Output Layer
- Dense layer with vocabulary size units
- Softmax activation
- One-hot encoded target outputs

## 📊 Expected Results

After training, the model should achieve:
- Reasonable accuracy on common phrases
- Better performance on shorter sentences
- Improved results with increased training epochs
- Validation loss convergence

## ⚠️ Known Limitations

- Word-level translation (no subword tokenization)
- Limited to vocabulary seen during training
- May struggle with long sentences
- No attention mechanism (basic seq2seq)
- Requires significant training time

