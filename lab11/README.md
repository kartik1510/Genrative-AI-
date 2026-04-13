# CSET419 – Introduction to Generative AI
## Lab 11: Fine-Tuning GPT-2 for Real-World Applications

---

## Overview

This lab demonstrates how to fine-tune a pre-trained GPT-2 language model for two real-world industry applications using transfer learning. Instead of training a model from scratch, we adapt GPT-2's existing language understanding to specific business domains in just a few minutes of training.

---

## Objective

Fine-tune GPT-2 on domain-specific datasets and observe how the model's output changes before and after training across two scenarios:

- **Component I** – Product Review Generator for an e-commerce platform
- **Component II** – Recipe Instruction Generator for a food-tech application

---

## Learning Outcomes

After completing this lab, students will be able to:

1. Understand how fine-tuning applies to real-world industry use cases
2. Load and configure a pre-trained GPT-2 model using Hugging Face Transformers
3. Prepare domain-specific datasets for causal language modeling
4. Fine-tune the model and compare generated output before and after training
5. Evaluate output quality using perplexity as a metric

---

## Project Structure

```
Lab11/
├── CSET419_Lab11_FineTuning.ipynb   # Main Jupyter notebook (run this)
├── CSET419_Lab11_Complete_Code.py   # Standalone Python script version
└── README.md                        # This file
```

---

## Requirements

| Library | Purpose |
|---|---|
| `transformers` | Load GPT-2 model and tokenizer |
| `datasets` | Prepare and split training data |
| `accelerate` | Optimize training on GPU/CPU |
| `torch` | Deep learning backend |

Install all dependencies with:

```bash
pip install transformers datasets accelerate -q
```

---

## How to Run (Google Colab – Recommended)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **Upload notebook** and select `CSET419_Lab11_FineTuning.ipynb`
3. Enable GPU: `Runtime → Change runtime type → T4 GPU`
4. Click `Runtime → Run all`

> Using a GPU reduces training time from ~15 minutes to under 2 minutes.

---

## Component I – Product Review Generator

**Business Scenario:** An e-commerce company wants an AI tool to auto-generate realistic product reviews.

### Dataset
20 product review sentences covering electronics (phones, laptops, headphones, smartwatches, speakers, etc.)

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | GPT-2 (117M parameters) |
| Epochs | 15 |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Max Sequence Length | 128 tokens |

### What to Observe

| | Before Fine-Tuning | After Fine-Tuning |
|---|---|---|
| Output style | Generic, Wikipedia-like text | E-commerce review language |
| Vocabulary | Random/mixed | "great value", "highly recommend", "battery life" |
| Perplexity | ~120–200 | ~15–35 |

---

## Component II – Recipe Instruction Generator

**Business Scenario:** A food-tech startup wants a smart cooking app that generates step-by-step recipe instructions.

### Dataset
20 cooking instruction sentences covering butter chicken, pasta carbonara, vegetable stir fry, and chocolate chip cookies.

### Training Configuration

| Parameter | Value |
|---|---|
| Base Model | GPT-2 (fresh instance) |
| Epochs | 15 |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Max Sequence Length | 128 tokens |

### What to Observe

| | Before Fine-Tuning | After Fine-Tuning |
|---|---|---|
| Output style | Unrelated/random text | Step-by-step cooking instructions |
| Structure | No logical flow | Marinate → Cook → Serve |
| Perplexity | ~130–180 | ~10–30 |

---

## Understanding Perplexity

Perplexity measures how "surprised" the model is by the text it generates. A lower perplexity means the model has learned the domain well.

```
Perplexity = e ^ (eval_loss)
```

A significant drop in perplexity after fine-tuning confirms that the model has successfully adapted to the target domain.

---

## Alternative Lightweight Models

If GPT-2 is too slow on your machine, you can replace `'gpt2'` with any of these in the code:

| Model | Size | Notes |
|---|---|---|
| `distilgpt2` | 82M | Smaller, trains faster — best for quick testing |
| `EleutherAI/gpt-neo-125m` | 125M | Open-source GPT-3 alternative |
| `microsoft/phi-2` | 2.7B | More powerful, needs GPU |
| `TinyLlama/TinyLlama-1.1B` | 1.1B | LLaMA architecture, very efficient |

---

## Key Concepts

**Transfer Learning** – Reusing a model trained on a large general dataset and adapting it to a specific task with a small dataset.

**Fine-Tuning** – Continuing to train a pre-trained model on new domain-specific data so it learns the patterns of that domain.

**Causal Language Modeling** – The training objective used here; the model learns to predict the next word given all previous words.

**DataCollatorForLanguageModeling** – Automatically creates input-label pairs for language model training by shifting tokens by one position.

---

## Author

**Course:** CSET419 – Introduction to Generative AI
**Lab:** 11 – Fine-Tuning Pre-Trained Models
