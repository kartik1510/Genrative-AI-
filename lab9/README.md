
## Lab 9: Generative Models for Sequential Data

---

## Objective

Understand how generative models can be applied to sequential data such as text, time-series, or language sequences. Design and implement simple generative models capable of learning patterns from sequences and generating new sequences.

---

## Learning Outcomes

After completing this lab, students will be able to:

1. Understand sequential data and its characteristics
2. Learn how generative models work for sequence prediction
3. Implement sequence-based generative models using neural networks
4. Train models to generate new sequences from learned patterns
5. Evaluate the quality of generated sequences

---

## Project Structure

```
Lab-9/
├── README.md
├── GenAI_Lab9_Component1_LSTM.ipynb        # Component I  — LSTM model
└── GenAI_Lab9_Component2_Transformer.ipynb # Component II — Transformer model
```

---

## Components

### Component I — RNN / LSTM Sequence Generation

Implements a **word-level LSTM language model** from scratch using pure NumPy.

**Tasks covered:**
1. Load and preprocess the sequential dataset
2. Convert sequences into numerical representations
3. Create input-output sequence pairs
4. Design an RNN / LSTM based generative model
5. Train the model on the sequence dataset
6. Generate new sequences using a seed input

**Architecture:**
```
Input words → Embedding (D=32) → LSTM (H=64) → Linear → Softmax → Next word
```

**LSTM Gate Equations:**

| Gate | Equation |
|------|----------|
| Input  | $i_t = \sigma(W_x x_t + W_h h_{t-1} + b_i)$ |
| Forget | $f_t = \sigma(W_x x_t + W_h h_{t-1} + b_f)$ |
| Output | $o_t = \sigma(W_x x_t + W_h h_{t-1} + b_o)$ |
| Cell   | $g_t = \tanh(W_x x_t + W_h h_{t-1} + b_g)$ |
| Cell state | $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ |
| Hidden state | $h_t = o_t \odot \tanh(c_t)$ |

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 32 |
| Hidden Units | 64 |
| Context Length | 3 words |
| Epochs | 200 |
| Learning Rate | 0.005 |
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Gradient Clipping | ±5 |

**Results:**

| Metric | Value |
|--------|-------|
| Initial Loss | 4.5095 |
| Final Loss | 0.0159 |
| Initial Perplexity | 90.88 |
| Final Perplexity | 1.02 |
| Top-1 Accuracy | 99.2% |
| Top-3 Accuracy | 100.0% |

---

### Component II — Transformer-Based Sequence Generation

Implements a **decoder-style Transformer language model** from scratch using pure NumPy, following the design of *Vaswani et al., 2017 (Attention Is All You Need)*.

**Tasks covered:**
1. Use the same dataset as Component I
2. Perform word-level tokenization
3. Implement positional encoding for sequence order
4. Design a Transformer-based architecture
5. Train the model on the sequence dataset
6. Generate sequences using the Transformer model

**Architecture:**
```
Input words
  → Token Embedding + Positional Encoding
  → Transformer Block 1  (Multi-Head Self-Attention + FFN + LayerNorm)
  → Transformer Block 2  (Multi-Head Self-Attention + FFN + LayerNorm)
  → Last token hidden state
  → Linear → Softmax → Next word probabilities
```

**Positional Encoding (Sinusoidal):**

$$PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

**Scaled Dot-Product Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Model Dimension (d_model) | 32 |
| Attention Heads | 2 |
| Feed-Forward Dim (d_ff) | 64 |
| Transformer Layers | 2 |
| Context Length | 4 words |
| Epochs | 150 |
| Learning Rate | 0.003 |
| Optimizer | Adam (β1=0.9, β2=0.999) |

**Results:**

| Metric | Value |
|--------|-------|
| Initial Loss | 5.2443 |
| Final Loss | 0.0281 |
| Initial Perplexity | 189.49 |
| Final Perplexity | 1.03 |
| Top-1 Accuracy | 100.0% |
| Top-3 Accuracy | 100.0% |

---

## Dataset

The following 16-sentence corpus is used for both components:

```
machine learning models learn patterns from data.
sequence models process data step by step.
recurrent neural networks are designed for sequential tasks.
rnn models maintain hidden states across time steps.
long short term memory networks solve long dependency problems.
lstm uses gates to control information flow.
gru models simplify the lstm architecture.
sequence prediction is useful in many applications.
language modeling predicts the next word in a sentence.
speech recognition processes audio sequences.
time series forecasting predicts future values.
music generation creates new melodies.
generative models learn probability distributions.
they generate new samples similar to training data.
sequence generation is widely used in artificial intelligence.
deep learning improves sequence modeling performance.
```

| Property | Value |
|----------|-------|
| Total Tokens | 127 |
| Vocabulary Size | 87 unique words |
| Tokenization | Word-level |

---

## LSTM vs Transformer Comparison

| Aspect | LSTM | Transformer |
|--------|------|-------------|
| Architecture | Sequential (step-by-step) | Parallel (all positions at once) |
| Context Handling | Hidden state (implicit) | Self-attention (explicit) |
| Position Info | Implicit via recurrence | Sinusoidal positional encoding |
| Context Length | 3 tokens | 4 tokens |
| Epochs | 200 | 150 |
| Final Loss | 0.0159 | 0.0281 |
| Top-1 Accuracy | 99.2% | 100.0% |
| Scalability | Limited (vanishing gradients) | Scales well to large corpora |

---

## Requirements

```
python >= 3.8
numpy
matplotlib  (optional — for plots)
jupyter     (to run .ipynb files)
```

> **No deep learning framework required.** Both models are implemented entirely in NumPy.

---

## How to Run

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install numpy matplotlib jupyter
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
4. Open and run either notebook:
   - `GenAI_Lab9_Component1_LSTM.ipynb`
   - `GenAI_Lab9_Component2_Transformer.ipynb`
5. Click **Kernel → Restart & Run All**

---

## Sample Generated Output

**LSTM — Seed:** `["lstm", "uses", "gates"]`
```
lstm uses gates to control information flow . gru models simplify
```

**Transformer — Seed:** `["generative", "models", "learn", "probability"]`
```
generative models learn probability distributions . they generate new samples similar to
```

---

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

---

*CSET419 — Lab 9 | Pure NumPy Implementation*
