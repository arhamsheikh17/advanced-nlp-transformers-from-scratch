# Advanced NLP: Transformers from Scratch

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)

A hands-on implementation of the Transformer architecture using **Python, NumPy, TensorFlow, and Keras**. This project follows a five-module learning journey—from scaled dot-product attention to an autoregressive Portuguese-to-English translation demo.

The goal is educational: each important Transformer component is implemented and tested in a small experiment before being assembled into a complete model.

## Project Overview

This repository covers:

- Scaled dot-product attention
- Multi-head self-attention
- Attention-weight visualization
- Sinusoidal positional encoding
- Transformer encoder and decoder layers
- Residual connections and layer normalization
- Position-wise feed-forward networks
- Padding and look-ahead masks
- Teacher forcing
- Autoregressive greedy decoding
- Portuguese-to-English translation

## Learning Journey

| Module | Topic | Main Outcome |
|---|---|---|
| 1 | The Attention Revolution | Built and visualized self-attention and trained a small attention classifier |
| 2 | Positional Encoding | Added sine and cosine signals so attention could understand token order |
| 3 | Encoder and Decoder Layers | Implemented the reusable building blocks of a Transformer |
| 4 | Building the Transformer | Combined all components and trained an encoder-decoder model |
| 5 | Inference and Translation | Added autoregressive decoding to generate English text token by token |

---

## Module 1: The Attention Revolution

The first module introduces self-attention and explains how every token can directly compare itself with every other token in a sequence.

### Concepts Covered

- Query, Key, and Value vectors
- Scaled dot-product attention
- Softmax attention weights
- Multi-head attention
- Parallel processing compared with recurrent neural networks

The central attention operation is:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

Where:

- `Q` represents Query vectors.
- `K` represents Key vectors.
- `V` represents Value vectors.
- `d_k` is the dimension of the Key vectors.
- Softmax converts attention scores into weights that sum to `1.0`.

### Implementation

In this module:

- Scaled dot-product attention was implemented using NumPy.
- The attention-weight matrix was calculated for a small three-word sequence.
- Each attention row was verified to sum to `1.0`.
- Attention weights were visualized using a heatmap.
- A TensorFlow sentiment classifier was created using `MultiHeadAttention`.
- Global average pooling was used to convert sequence output into a fixed-size representation.
- A sigmoid output layer was used for binary sentiment classification.

### Training Illustration

The original Kaggle dataset was unavailable through its old dataset handle. Therefore, the Cornell Sentence Polarity dataset was uploaded and loaded through a local Kaggle Input path.

Dataset details:

- Total samples: 10,662
- Positive samples: 5,331
- Negative samples: 5,331
- Embedding model: Wiki-words
- Embedding dimension: 250
- Maximum sequence length: 59
- Training subset: 200 shuffled samples
- Projection dimension: 16
- Number of attention heads: 2
- Epochs: 5

The miniature model was created only to demonstrate that self-attention works correctly inside a TensorFlow training loop and that the training loss can decrease.

It was not intended to achieve production-level sentiment classification accuracy.

---

## Module 2: Positional Encoding

Self-attention processes every token simultaneously. Because of this, it does not naturally know the order in which tokens appear.

For example:

1. `The dog bit the man`
2. `The man bit the dog`

Both sentences contain the same words but have completely different meanings.

Without positional information, self-attention only sees the token representations and does not know which token appeared first.

### Concepts Covered

- Why self-attention is position-blind
- Fixed positional encoding
- Sine signals for even dimensions
- Cosine signals for odd dimensions
- Combining token embeddings with positional information

The positional encoding equations are:

$$
PE(pos,2i)
=
\sin
\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

$$
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}{10000^{2i/d_{model}}}
\right)
$$

Where:

- `pos` represents the token position.
- `i` represents the embedding dimension.
- `d_model` represents the total model dimension.

### Implementation

In this module:

- The order-blind behavior of self-attention was demonstrated.
- Two word embeddings were swapped.
- The resulting attention scores remained the same after row and column permutation.
- A reusable Keras `PositionalEncoding` layer was created.
- The sine and cosine encoding matrix was precomputed using NumPy.
- Positional vectors were added directly to the token embeddings.
- The output shape was verified to remain equal to the input shape.
- TensorFlow `TextVectorization` was used to convert raw text into integer token IDs.

### Training Illustration

- Vocabulary size: 5,000
- Maximum sequence length: 60
- Training subset: 200 sentiment samples
- Model dimension: 32
- Number of attention heads: 2
- Epochs: 5

This module confirmed that learnable word embeddings, positional encoding, and self-attention can be combined and trained end to end.

---

## Module 3: Encoder and Decoder Layers

This module combines attention, normalization, residual connections, and feed-forward networks into the two main building blocks of the Transformer:

- `EncoderLayer`
- `DecoderLayer`

## Encoder Layer

Each encoder layer contains:

1. Multi-head self-attention
2. Residual connection
3. Layer normalization
4. Position-wise feed-forward network
5. A second residual connection
6. A second layer normalization

The general encoder flow is:

```text
Input
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection + Layer Normalization
  ↓
Feed-Forward Network
  ↓
Residual Connection + Layer Normalization
  ↓
Encoder Output
```

## Decoder Layer

Each decoder layer contains:

1. Masked multi-head self-attention
2. Residual connection and layer normalization
3. Cross-attention over the encoder output
4. Another residual connection and layer normalization
5. Position-wise feed-forward network
6. Final residual connection and layer normalization

The decoder flow is:

```text
Target Input
  ↓
Masked Self-Attention
  ↓
Add and Normalize
  ↓
Cross-Attention with Encoder Output
  ↓
Add and Normalize
  ↓
Feed-Forward Network
  ↓
Add and Normalize
  ↓
Decoder Output
```

### Why These Components Matter

- **Residual connections** create shortcut paths that help gradients flow through the model.
- **Layer normalization** keeps token representations stable during training.
- **Feed-forward networks** transform each token representation independently.
- **Cross-attention** allows the decoder to read the encoder output.
- **Masked attention** prevents the decoder from looking at future target tokens.

### Training Illustration

A small encoder-only sentiment classifier was created to confirm that gradients flow through the custom encoder layer.

Configuration:

- Vocabulary size: 5,000
- Maximum sequence length: 60
- Samples: 200
- Model dimension: 32
- Attention heads: 2
- Feed-forward dimension: 64
- Epochs: 5

The training experiment demonstrated that the custom `EncoderLayer` could be used successfully inside a TensorFlow model.

---

## Module 4: Building the Transformer

The fourth module assembles all previously created components into a complete encoder-decoder Transformer.

The final architecture combines:

- Token embeddings
- Positional encoding
- Encoder layers
- Decoder layers
- Multi-head attention
- Cross-attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Padding masks
- Look-ahead masks
- Final vocabulary projection

## Transformer Masking

Three types of masks were implemented and visualized.

### Padding Mask

The padding mask prevents the model from paying attention to padding tokens.

Padding tokens usually have the token ID `0`.

```python
def create_padding_mask(sequence):
    return tf.not_equal(sequence, 0)[:, tf.newaxis, :]
```

### Look-Ahead Mask

The look-ahead mask prevents a decoder position from seeing future tokens.

For example, while predicting token number three, the model can only see tokens one, two, and three. It cannot see tokens four or five.

```python
def create_look_ahead_mask(size):
    rows = tf.range(size)
    columns = tf.range(size)

    return rows[:, tf.newaxis] >= columns[tf.newaxis, :]
```

### Combined Mask

The combined mask applies both restrictions:

- Future tokens are hidden.
- Padding tokens are ignored.

This prevents the decoder from cheating during training.

## Teacher Forcing

Teacher forcing is used to train the decoder.

The target sequence is shifted into two parts:

```python
tar_inp = target[:, :-1]
tar_real = target[:, 1:]
```

For example:

```text
Original target:
[START] I like machine learning [END]

Decoder input:
[START] I like machine learning

Expected output:
I like machine learning [END]
```

The decoder receives the correct previous token while learning to predict the next token.

Padding positions are excluded from the loss calculation.

## Translation Dataset

The Portuguese and English TED transcript files are loaded from Kaggle Inputs:

```text
ted_talks_pt-br.csv
ted_talks_en.csv
```

Both files are merged using the `talk_id` column so that the Portuguese and English transcripts correspond to the same TED Talk.

### Tiny Demonstration Configuration

- Encoder and decoder layers: 1
- Model dimension: 32
- Attention heads: 2
- Feed-forward dimension: 64
- Dropout rate: 0.1
- Maximum sequence length: 40
- Vocabulary size: 4,000
- Samples: 500
- Batch size: 16
- Epochs: 5

The purpose of this run was to verify:

- Encoder and decoder connectivity
- Mask creation
- Teacher forcing
- Loss calculation
- Gradient flow
- End-to-end Transformer training

It was not intended to create a production-quality translation model.

---

## Module 5: Inference and Translation

During training, teacher forcing provides the correct previous token to the decoder.

At inference time, correct target tokens are unavailable. The model must generate a translation using only its own previous predictions.

This is known as autoregressive decoding.

## Autoregressive Greedy Decoding

The `Translator` class performs the following steps:

1. Tokenize the Portuguese input sentence.
2. Initialize the decoder sequence with `[START]`.
3. Pass the source and current decoder sequence to the Transformer.
4. Read the prediction from the latest decoder position.
5. Select the token with the highest probability using `argmax`.
6. Append the predicted token to the decoder input.
7. Repeat the process.
8. Stop when `[END]` is generated or the maximum length is reached.

The greedy decoding operation is:

$$
\hat{y}_t
=
\arg\max
P(y_t \mid y_{<t},x)
$$

Where:

- `x` is the Portuguese input sequence.
- `y<t` represents previously generated English tokens.
- `ŷt` is the next predicted token.

### Inference Flow

```text
Portuguese Sentence
        ↓
Portuguese Vectorizer
        ↓
Transformer Encoder
        ↓
Decoder starts with [START]
        ↓
Predict next English token
        ↓
Append predicted token
        ↓
Repeat until [END]
        ↓
Generated English Translation
```

### Experiments

In this module:

- Tiny and larger Transformer training runs were configured.
- Sparse categorical cross-entropy was recorded across epochs.
- Training-loss graphs were produced.
- A reusable greedy-decoding `Translator` class was created.
- Portuguese test sentences were passed to the trained model.
- English translations were generated one token at a time.

Example Portuguese sentences included:

```text
este é o meu primeiro dia.
a tecnologia está mudando o mundo.
obrigado pela sua atenção.
o problema é muito difícil.
```

Because the dataset and model are intentionally small, generated translations may be repetitive, incomplete, or grammatically incorrect.

The important result is that the complete pipeline—from Portuguese input to autoregressively generated English tokens—works end to end.

---

## Repository Structure

```text
advanced-nlp-transformers-from-scratch/
├── 01-attention-revolution.ipynb
├── 02-positional-encoding.ipynb
├── 03-encoder-decoder.ipynb
├── 04-building-transformer.ipynb
├── 05-inference-translation.ipynb
└── README.md
```

Rename the downloaded Kaggle notebooks to the filenames above if their current names are different.

## Datasets and Models

This project uses locally attached Kaggle resources so that the notebooks can run without external internet access.

### Sentiment Dataset

```text
/kaggle/input/datasets/arhamsheikhdev/polarity/rt-polarity.pos
/kaggle/input/datasets/arhamsheikhdev/polarity/rt-polarity.neg
```

### Wiki-Words Embedding Model

```text
/kaggle/input/models/google/wiki-words/tensorflow2/250/1
```

### Portuguese-English TED Dataset

```text
/kaggle/input/datasets/miguelcorraljr/ted-ultimate-dataset/2020-05-01/ted_talks_pt-br.csv
/kaggle/input/datasets/miguelcorraljr/ted-ultimate-dataset/2020-05-01/ted_talks_en.csv
```

Kaggle may sometimes mount the same resources under shorter paths.

If a path is different:

1. Open the notebook's **Input** panel.
2. Locate the required file.
3. Copy its path.
4. Replace the path inside the notebook.

## Tech Stack

- Python
- NumPy
- pandas
- Matplotlib
- TensorFlow
- Keras
- TensorFlow Hub
- Kaggle Notebooks

## Running the Project

### Run on Kaggle

1. Upload or import the five notebooks into Kaggle.
2. Attach the required datasets through **Add Input**.
3. Attach the Wiki-words TensorFlow model.
4. Confirm that all local paths match the mounted files.
5. Select a GPU accelerator if available.
6. Open the notebooks in numerical order.
7. Run every notebook from top to bottom.

Running cells in order is important because later cells depend on previously defined:

- Classes
- Functions
- Constants
- Datasets
- Vectorizers
- Models

### Run Locally

Clone the repository:

```bash
git clone https://github.com/arhamsheikh17/advanced-nlp-transformers-from-scratch.git
cd advanced-nlp-transformers-from-scratch
```

Install the main dependencies:

```bash
pip install numpy pandas matplotlib tensorflow tensorflow-hub jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

The Kaggle dataset paths must be replaced with local file paths when running outside Kaggle.

## Runtime Messages

Kaggle may display CUDA-related messages such as:

```text
failed call to cuInit
Unable to register cuDNN factory
Unable to register cuBLAS factory
Unable to register cuFFT factory
```

If the notebook continues and produces model summaries, output shapes, epoch logs, or graphs, these messages are generally Kaggle environment warnings rather than failures in the model code.

A Python exception at the bottom of a cell must still be fixed before continuing.

Examples include:

- `NameError`
- `FileNotFoundError`
- `BackendError`
- `ValueError`

## Challenges Resolved

Several practical issues were handled during this project:

- The original sentiment dataset handle returned a Kaggle `BackendError`.
- External dataset downloads failed because notebook internet access was unavailable.
- The Cornell polarity dataset was uploaded manually to Kaggle.
- The Wiki-words model was loaded through a local Kaggle Input path.
- Missing definitions caused `NameError` exceptions after notebook session restarts.
- Dataset variables such as `df_sample` and `N_SAMPLES` were redefined where required.
- TensorFlow CUDA warnings were separated from actual Python exceptions.

These issues demonstrate why reproducible file paths, ordered execution, and explicit configuration variables are important in machine-learning notebooks.

## Limitations

- The models are intentionally small and designed for learning.
- The training datasets are limited.
- Greedy decoding keeps only one token choice at every step and may miss a better complete translation.
- Word-level vocabularies produce more unknown tokens than modern subword tokenizers.
- Full TED Talk transcripts are much longer than the configured maximum sequence length.
- Truncating transcripts limits the amount of translation context available to the model.
- TED transcripts merged by `talk_id` may not provide clean sentence-level alignment.
- Translation quality should not be compared with production translation systems.
- No BLEU, ROUGE, or other formal translation metric was calculated.
- The small experiments do not include extensive hyperparameter tuning.

## Possible Improvements

Future versions of this project could include:

- Sentence-aligned Portuguese-English data
- Byte Pair Encoding
- WordPiece or SentencePiece tokenization
- Larger training datasets
- Deeper encoder and decoder stacks
- More attention heads
- Transformer learning-rate scheduling
- Beam-search decoding
- Length penalties
- BLEU score evaluation
- Training and validation splits
- Model checkpointing
- Early stopping
- TensorBoard monitoring
- Saved-model deployment
- A Streamlit translation interface

## Key Learning Outcomes

After completing this project, I gained practical experience with:

- Calculating attention weights
- Understanding Query, Key, and Value projections
- Building multi-head attention models
- Creating sinusoidal positional encodings
- Implementing custom Keras layers
- Building Transformer encoder layers
- Building Transformer decoder layers
- Creating causal and padding masks
- Training sequence-to-sequence models
- Applying teacher forcing
- Implementing masked loss functions
- Building autoregressive inference loops
- Generating translations token by token
- Debugging TensorFlow and Kaggle notebook issues

## Conclusion

Across five modules, this project develops a Transformer from its smallest mathematical component into a working sequence-to-sequence system.

The project begins with attention scores, adds positional information, builds reusable encoder and decoder layers, introduces masking and teacher forcing, and finally generates translations through autoregressive inference.

The most important outcome is not translation accuracy. The real outcome is a practical understanding of how information flows through a Transformer.

Every major component was:

- Studied
- Implemented
- Connected
- Trained
- Visualized
- Tested

This project provides a strong foundation for understanding and working with modern Transformer-based architectures such as:

- BERT
- GPT
- T5
- Translation Transformers
- Large Language Models

## Acknowledgements

This project was completed as part of the Kaggle **Advanced NLP: Large Language Models** learning guide.

It builds on concepts introduced in the Kaggle **Intermediate NLP** course, including:

- Word embeddings
- Text preprocessing
- Recurrent neural networks
- Long Short-Term Memory networks
- Bidirectional LSTMs

## Author

**Arham Ul Haq Sheikh**

- GitHub: [@arhamsheikh17](https://github.com/arhamsheikh17)
