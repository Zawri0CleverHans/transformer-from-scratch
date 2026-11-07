# Transformer from Scratch

Building an autoregressive character-level GPT from scratch in PyTorch, inspired by Andrej Karpathy's *Zero to Hero* series.

This repository covers:
- Character-level tokenization and vocab mapping
- Positional & token embeddings
- Masked causal self-attention mechanism (`Head` module)
- Bigram and transformer language model architectures
- Cross-entropy loss estimation across train and validation splits
- Generative text sampling on the Tiny Shakespeare dataset

---

## 📺 Recommended Lecture

For detailed explanations, derivations, and intuitions behind the concepts implemented here, please watch Andrej Karpathy's complete walkthrough lecture on YouTube:

▶️ **[Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)**  
Playlist: **[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)**

---

## 📂 Project Structure

```text
transformer-from-scratch/
├── input.txt                      # Tiny Shakespeare dataset (~1MB text file)
├── transformer-from-scratch.ipynb # Step-by-step Jupyter Notebook implementation
└── README.md                      # Project documentation and guide
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ and PyTorch installed. GPU support (CUDA) is recommended but not required (will fallback to CPU).

```bash
pip install torch jupyter
```

### Running the Notebook

Clone this repository and launch Jupyter Notebook:

```bash
git clone https://github.com/Zawri0CleverHans/transformer-from-scratch.git
cd transformer-from-scratch
jupyter notebook transformer-from-scratch.ipynb
```

You can execute the cells sequentially to load the dataset, build the self-attention components, train the model, and sample generated text.

---

## 💡 Overview of Implementation Steps

1. **Dataset**: Download and inspect the Tiny Shakespeare corpus (`input.txt`).
2. **Tokenization**: Build character-to-integer (`stoi`) and integer-to-character (`itos`) encoders/decoders.
3. **Data Splitting & Batching**: Split into 90% train and 10% validation sets; implement `get_batch` to generate chunked context and target tensors (`x`, `y`).
4. **Attention Head**: Implement single-head causal self-attention with lower-triangular masking (`tril`) to prevent attending to future tokens.
5. **Language Model**: Combine token embeddings, position embeddings, and linear projection heads.
6. **Training Loop & Loss Estimation**: Train with AdamW optimizer, periodically calculating average train and validation cross-entropy loss.
7. **Text Generation**: Sample from the output probability distributions using `torch.multinomial`.

---

## 📜 Acknowledgments

Special thanks to **[Andrej Karpathy](https://github.com/karpathy)** for creating the outstanding educational series *Neural Networks: Zero to Hero*.
