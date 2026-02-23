CSET419 – Introduction to Generative AI
Lab 4 – Text Generation using LSTM and Transformer
Objective

The objective of this lab is to design and implement text generation models that learn patterns from a given text corpus and generate new meaningful text sequences. Two different architectures are implemented:

LSTM-based text generation

Transformer-based text generation

Learning Outcomes

After completing this lab, the following concepts were understood and implemented:

Basics of text generation

Text preprocessing and tokenization

Sequence creation for neural networks

LSTM architecture for sequential modeling

Transformer architecture using self-attention

Text generation using trained models

Analysis of generated text quality

Dataset

The dataset consists of multiple sentences related to Artificial Intelligence, Machine Learning, NLP, and education. The text was provided in the lab manual.

The dataset was converted to lowercase and used directly for training.

Component I: LSTM-Based Text Generation
Methodology

Text Preprocessing

Converted text to lowercase

Split text line by line

Tokenization

Word-level tokenization using Keras Tokenizer

Created vocabulary index

Sequence Creation

Generated n-gram sequences

Example:
"machine learning allows"
becomes training samples for next-word prediction

Padding

All sequences padded to same length

Model Architecture

Embedding Layer

LSTM Layer

Dense Output Layer with Softmax

Training

Loss: Categorical Crossentropy

Optimizer: Adam

Epochs: 100

Text Generation

Provided seed text

Predicted next words iteratively

Used sampling instead of argmax to reduce repetition

LSTM Model Architecture

Embedding → LSTM → Dense (Softmax)

The LSTM captures sequential dependencies by maintaining memory of previous tokens.

Sample Output (LSTM)

Example generated text:

machine learning allows systems to improve automatically with experience and enhance intelligent systems

The generated text shows coherence but may contain repetition due to small dataset size.

Component II: Transformer-Based Text Generation
Methodology

Used same dataset and tokenization

Created input-output sequences

Built custom Transformer block using:

MultiHeadAttention

Feed Forward Network

Layer Normalization

Residual connections

Global Average Pooling applied

Softmax output layer for next word prediction

Transformer Model Architecture

Embedding → Transformer Block → GlobalAveragePooling → Dense (Softmax)

The Transformer model uses self-attention to focus on relevant parts of the sequence instead of processing strictly in order like LSTM.

Important Implementation Notes

key_dim in MultiHeadAttention was set to embed_dim // num_heads

training=None added in call() method to prevent runtime errors

Sampling used instead of argmax to reduce repetitive outputs

Temperature sampling implemented to improve diversity

Sample Output (Transformer)

Example generated text:

machine learning allows systems to improve automatically with experience generation and learning systems improve education technology

Repetition may occur due to limited dataset size.

Comparison Between LSTM and Transformer

LSTM:

Processes data sequentially

Captures long-term dependencies using memory cells

Slower for long sequences

Transformer:

Uses self-attention mechanism

Processes input in parallel

More scalable for larger datasets

Performs better with large-scale data

Since the dataset in this lab is small, both models show similar performance.

Observations

Small datasets lead to repetitive or limited text generation.

Sampling strategies significantly improve output diversity.

Transformers require correct dimensional handling to avoid runtime errors.

Text generation is fundamentally probabilistic.

Conclusion

In this lab, two different architectures for text generation were successfully implemented and tested. Both LSTM and Transformer models were able to learn patterns from the dataset and generate new text sequences.

The experiment demonstrates how modern language models work at a basic level. While performance is limited due to small dataset size, the lab provides strong foundational understanding of generative text models.
