# Lab 12 – Text Generation with Attention

## Objective

Implement a text generation model using attention to improve contextual understanding.

---

## Scenario

Build a chatbot that generates replies from user input. Use attention so the model focuses on important words in the sentence.

---

## Dataset

**Cornell Movie Dialog Dataset** — conversation pairs.

| Input | Output |
|-------|--------|
| "Hello" | "Hi, how are you?" |

---

## Steps

### 1. Preprocessing
- Clean raw text
- Tokenize (split into words/tokens)
- Build a vocabulary mapping words to numbers

### 2. Model Architecture

```
Embedding → LSTM Encoder → Attention → Output Layer
```

| Component | Role |
|-----------|------|
| **Embedding Layer** | Converts words into dense numerical vectors |
| **LSTM Encoder** | Processes the sequence and captures context over time |
| **Attention Layer** | Assigns weights to input words for focused context |
| **Output Layer** | Predicts the next word / response |

### 3. Training
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam

### 4. Evaluation
- Feed test inputs to the model
- Inspect whether generated replies are coherent

---

## Key Idea — Attention Mechanism

Without attention, an LSTM compresses the entire input into a single fixed vector, losing detail for longer sentences.

**Attention** assigns a weight to each input word so the model can focus on the most relevant parts when generating a response.

> When generating a reply to *"how are you"*, the model pays more attention to **"you"** than to "how" or "are".

---

## Expected Output

| Input | Output |
|-------|--------|
| "how are you" | "i am fine" |

Attention focuses more on the word **"you"**.

---

## Summary

This lab is a hands-on introduction to **seq2seq models with attention** — a foundational concept in modern NLP and the direct precursor to Transformer-based architectures.
