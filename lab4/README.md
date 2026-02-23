# CSET419 -- Introduction to Generative AI

## Lab 4 -- Text Generation using LSTM and Transformer

## Objective

The objective of this lab is to design and implement text generation
models that learn patterns from a given text corpus and generate new
meaningful text sequences. Two architectures were implemented:

1.  LSTM-based text generation\
2.  Transformer-based text generation

------------------------------------------------------------------------

## Learning Outcomes

After completing this lab, the following concepts were understood and
implemented:

-   Basics of text generation\
-   Text preprocessing and tokenization\
-   Sequence creation for neural networks\
-   LSTM architecture for sequential modeling\
-   Transformer architecture using self-attention\
-   Text generation using trained models\
-   Analysis of generated text quality

------------------------------------------------------------------------

## Dataset

The dataset consists of multiple sentences related to Artificial
Intelligence, Machine Learning, NLP, and education. The text was
converted to lowercase and used directly for training.

------------------------------------------------------------------------

# Component I: LSTM-Based Text Generation

## Methodology

1.  Text Preprocessing (lowercasing and cleaning)\
2.  Word-level tokenization using Keras Tokenizer\
3.  Creation of n-gram input-output sequences\
4.  Sequence padding\
5.  Model design using Embedding → LSTM → Dense (Softmax)\
6.  Model training using Adam optimizer and categorical crossentropy\
7.  Text generation using seed text

## Sample Output

Example:

machine learning allows systems to improve automatically with experience
and enhance intelligent systems

The LSTM model captures sequential dependencies effectively but may
generate repetition due to small dataset size.

------------------------------------------------------------------------

# Component II: Transformer-Based Text Generation

## Methodology

1.  Used same tokenized dataset\
2.  Built custom Transformer block including:
    -   MultiHeadAttention\
    -   Feed Forward Network\
    -   Layer Normalization\
    -   Residual connections\
3.  Global Average Pooling\
4.  Softmax output layer for next-word prediction\
5.  Training and text generation

## Sample Output

Example:

machine learning allows systems to improve automatically with experience
generation and intelligent systems

Repetition may occur due to limited training data.

------------------------------------------------------------------------

# Comparison

LSTM: - Sequential processing\
- Memory-based long-term dependency handling\
- Slower for long sequences

Transformer: - Uses self-attention\
- Processes tokens in parallel\
- Scales better for larger datasets

For this small dataset, both models performed similarly.

------------------------------------------------------------------------

# Observations

-   Text generation is probabilistic.\
-   Small datasets lead to limited creativity and repetition.\
-   Sampling strategies improve output diversity.\
-   Proper dimension handling is critical in Transformer models.

------------------------------------------------------------------------

# Conclusion

This lab demonstrates foundational concepts behind modern language
models. Both LSTM and Transformer architectures successfully generated
text based on learned patterns. While performance is limited by dataset
size, the experiment provides strong understanding of generative AI
principles.
