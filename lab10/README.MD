## Lab 10: Sequential Data Generation

---

## 📌 Objective

Implement generative models that learn patterns from sequential text data and produce new sequences.  
Two deep-learning architectures are explored:

| Component | Architecture |
|---|---|
| I  | LSTM (Long Short-Term Memory) |
| II | Transformer Encoder |

---

## 📁 File Structure

```
├── GenAI_Lab_Week10.ipynb   ← Main notebook (all code + explanations)
└── README.md                ← This file
```

---

## 🗂️ Dataset

A 16-sentence text corpus covering AI/ML concepts (provided in the lab sheet).  
Treated as a single character stream for **character-level language modelling**.

```
artificial intelligence systems learn patterns from data
sequence models process information step by step
recurrent neural networks are useful for sequence prediction
...
```

---

## ⚙️ Environment Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| NumPy | any recent |
| Matplotlib | any recent |
| Jupyter | any recent |

Install all dependencies:
```bash
pip install torch numpy matplotlib jupyter
```

---

## 🚀 How to Run

```bash
# Clone / download this folder, then:
jupyter notebook GenAI_Lab_Week10.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`).  
Training takes ~1–3 minutes on CPU.

---

## 🔷 Component I — LSTM Sequential Generation

### Pipeline

```
Raw Text
  ↓
Character-level Vocabulary  (char ↔ integer mapping)
  ↓
Sliding Window (SEQ_LEN = 30)  →  (X, y) pairs
  ↓
Embedding Layer  →  LSTM (2 layers, 256 hidden)  →  Linear  →  Logits
  ↓
CrossEntropyLoss  +  Adam Optimizer  (150 epochs)
  ↓
Temperature Sampling  →  Generated Text
```

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Sequence Length | 30 |
| Embedding Dim | 64 |
| Hidden Dim | 256 |
| LSTM Layers | 2 |
| Dropout | 0.3 |
| Learning Rate | 0.003 |
| Epochs | 150 |
| Batch Size | 64 |

### What is Temperature Sampling?

Dividing logits by `temperature` before softmax controls the randomness of generation:

| Temperature | Effect |
|---|---|
| `< 1.0` | More focused / repetitive |
| `= 1.0` | Raw model distribution |
| `> 1.0` | More diverse / creative |

---

## 🔶 Component II — Transformer Sequential Generation

### Architecture

```
Input Sequence
  ↓
Embedding Layer
  ↓
Positional Encoding  (sinusoidal: PE(pos,2i) = sin(pos/10000^(2i/d)))
  ↓
TransformerEncoderLayer × 2
  (Multi-Head Self-Attention  +  Feed-Forward  +  LayerNorm)
  ↓
Last Token's Representation  →  Linear  →  Logits
  ↓
Temperature Sampling  →  Generated Text
```

### Key Hyperparameters

| Parameter | Value |
|---|---|
| d_model | 64 |
| Attention Heads | 4 |
| Encoder Layers | 2 |
| Feed-Forward Dim | 256 |
| Dropout | 0.1 |
| Learning Rate | 0.003 |
| Epochs | 150 |

### Why Positional Encoding?

Transformers process tokens in parallel — they have no built-in sense of order.  
Positional encoding injects position information using sine/cosine waves of different frequencies, so the model can distinguish position 1 from position 30.

---

## 📊 Model Comparison

| Aspect | LSTM | Transformer |
|---|---|---|
| Sequence Handling | Sequential (step by step) | Parallel (all at once) |
| Long-range Memory | Cell state + gating | Direct self-attention |
| Positional Info | Built-in via recurrence | Explicit positional encoding |
| Training Speed | Slower | Faster (parallelisable) |
| Typical Use | Time series, speech | NLP, LLMs (GPT, BERT) |

---

## 📖 Learning Outcomes Covered

1. ✅ Sequential data generation concepts
2. ✅ Character-level tokenisation & preprocessing
3. ✅ Sliding window input-output pair construction
4. ✅ LSTM generative model implementation
5. ✅ Transformer encoder with positional encoding
6. ✅ Temperature-based sampling for controlled generation
7. ✅ Loss curve visualisation & model comparison

---

## 🧠 Key Concepts Glossary

| Term | Definition |
|---|---|
| **Sequential Data** | Ordered data where position matters (text, speech, time series) |
| **Character-Level LM** | Model that predicts the next character given previous characters |
| **Sliding Window** | Technique to create (input, target) pairs from a stream |
| **Embedding** | Dense vector representation of a discrete token |
| **LSTM** | RNN variant with cell state and gates to handle long-range dependencies |
| **Vanishing Gradient** | Problem in deep RNNs where gradients shrink to zero, forgetting distant context |
| **Self-Attention** | Mechanism allowing each token to attend to every other token in the sequence |
| **Positional Encoding** | Sinusoidal signals added to embeddings so Transformers know token order |
| **Temperature** | Hyperparameter controlling randomness in sampling |
| **Cross-Entropy Loss** | Standard loss for multi-class classification (next-character prediction) |
| **Gradient Clipping** | Prevents exploding gradients by capping gradient norms |

---
