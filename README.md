# Transformer from Scratch (nanoGPT)

A complete, ground-up PyTorch implementation of an autoregressive character-level **Generative Pre-trained Transformer (GPT)**, inspired by Andrej Karpathy's renowned deep dive:  
▶️ **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)** (YouTube)  
Course: **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**

---

## 📖 Recommended Lecture

> *"What I cannot create, I do not understand."* — Richard Feynman

If you want the deep mathematical intuition and code derivation behind every equation and tensor shape in this repository, **watch Andrej Karpathy's full lecture**:
- **Video link:** [YouTube: Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- **Topics covered in the lecture**: Self-attention math and scaled dot-product affinities, single-head and multi-head attention, residual connections, layer normalization, scaling transformer blocks, weight initialization, and text generation.

---

## 🏗️ Architecture Overview

The architecture implements a decoder-only Transformer following the modern GPT-2 pre-layer normalization convention:

```text
Input Tokens (B, T)
        │
        ├───► Token Embedding (B, T, n_embd)
        │                 +
        └───► Position Embedding (T, n_embd)
                          │
                          ▼
             ┌─────────────────────────┐
             │    Transformer Block    │  x n_layer
             │ ┌─────────────────────┐ │
             │ │ LayerNorm (Pre-LN)  │ │
             │ │ Multi-Head Attention│ │
             │ └──────────┬──────────┘ │
             │            ▼ (+) Residual
             │ ┌─────────────────────┐ │
             │ │ LayerNorm (Pre-LN)  │ │
             │ │ Feed-Forward (MLP)  │ │
             │ └──────────┬──────────┘ │
             │            ▼ (+) Residual
             └────────────┬────────────┘
                          │
                          ▼
                Final LayerNorm
                          │
                          ▼
             Linear LM Head (n_embd -> vocab_size)
                          │
                          ▼
                     Logits (B, T, vocab_size)
```

### Components Implemented:
1. **Character-Level Tokenization**: Map raw text into discrete tokens using custom `encode` and `decode` routines.
2. **Batching**: Dynamic context chunking (`get_batch`) with random offset sampling.
3. **Scaled Dot-Product Self-Attention (`Head`)**:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
   With causal triangular masking (`tril`) to enforce the autoregressive property.
4. **Multi-Head Attention (`MultiHeadAttention`)**: Parallel attention projections concatenated and passed through a linear projection layer with dropout.
5. **Position-wise Feed-Forward Network (`FeedForward`)**: $4 \times n_{\text{embd}}$ expansion with ReLU activation and dropout.
6. **Transformer Blocks (`Block`)**: Pre-LayerNorm configuration with residual connections.
7. **Complete GPT (`GPTLanguageModel`)**: Stacks token embeddings, positional embeddings, $N$ transformer blocks, final LayerNorm, and linear projection head. Includes custom weight initialization.
8. **Iterative Text Sampling (`.generate()`)**: Autoregressive next-token prediction using multinomial sampling.

---

## 📂 Repository Structure

```text
transformer-from-scratch/
├── input.txt                      # Tiny Shakespeare training corpus (~1MB text)
├── gpt.py                         # Standalone, production-ready training script
├── transformer-from-scratch.ipynb # Step-by-step interactive Jupyter notebook
├── .gitignore                     # Git ignore rules for checkpoints/cache
└── README.md                      # Documentation and guide
```

---

## 🚀 Quick Start

### 1. Requirements

Install PyTorch and Jupyter:
```bash
pip install torch jupyter
```

*(CUDA GPU is automatically detected and used if available; otherwise runs on CPU).*

### 2. Run Standalone Script

Run the complete training pipeline directly from your terminal:
```bash
python gpt.py
```

### 3. Run the Interactive Notebook

Open and experiment cell-by-cell in Jupyter:
```bash
jupyter notebook transformer-from-scratch.ipynb
```

---

## 📊 Default Hyperparameters

| Hyperparameter | `gpt.py` (Full Scale) | `notebook` (Interactive) |
|---|---|---|
| `batch_size` | 64 | 32 |
| `block_size` (context length) | 256 | 64 |
| `n_embd` (embedding dim) | 384 | 128 |
| `n_head` | 6 | 4 |
| `n_layer` | 6 | 4 |
| `dropout` | 0.2 | 0.2 |
| `learning_rate` | 3e-4 | 1e-3 |
| `max_iters` | 5000 | 3000 |

---

## 📜 Credits & Acknowledgments

All credit to **[Andrej Karpathy](https://github.com/karpathy)** for creating the educational lecture and series that made understanding and implementing Transformers accessible to everyone.
