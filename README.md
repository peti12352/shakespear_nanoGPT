# Overview {#overview .unnumbered}

This repository contains the implementation of a transformer-based
language model trained on the *Tiny Shakespeare* dataset. The model
predicts the next character in a sequence given a context, showcasing
the power of the transformer architecture for sequence modeling tasks.
The notebook provides a step-by-step explanation of the model, data
processing, training, and text generation.

# Table of Contents {#table-of-contents .unnumbered}

1.  [Overview](#sec:overview)

2.  [Key Concepts](#sec:key-concepts)

3.  [Model Architecture](#sec:model-architecture)

4.  [Training](#sec:training)

5.  [Text Generation](#sec:text-generation)

6.  [Requirements](#sec:requirements)

7.  [Usage](#sec:usage)

8.  [Acknowledgments](#sec:acknowledgments)

# Overview

This repository demonstrates the use of a **transformer architecture**
to generate text based on the famous *Tiny Shakespeare* dataset. It
includes all the necessary components for:

-   Data preprocessing

-   Model creation (using a simplified transformer)

-   Training on GPU/TPU

-   Text generation after training

The model is based on the transformer architecture, which was first
introduced in the paper *Attention Is All You Need*
[@vaswani2017attention] and uses multi-head self-attention for capturing
complex dependencies in the text.

# Key Concepts

-   **Self-Attention**: A mechanism to model relationships within a
    sequence of tokens by computing weighted interactions between
    tokens.

-   **Multi-Head Attention**: Combines multiple attention heads to allow
    the model to focus on different parts of the sequence in parallel.

-   **Positional Encoding**: Encodes the position of tokens in a
    sequence, allowing the model to capture order-dependent
    relationships.

-   **Residual Connections**: Skip connections that help gradients flow
    smoothly through deep networks.

-   **Layer Normalization**: A technique to normalize the activations of
    neurons, aiding in stabilizing training and speeding up convergence.

-   **Cross-Entropy Loss**: A loss function used for classification
    tasks, minimizing the difference between predicted probabilities and
    true labels.

# Model Architecture

The model follows a simple **transformer architecture**:

1.  **Embedding Layers**: Converts the input tokens into dense vectors
    of fixed size.

2.  **Multi-Head Attention**: Computes the attention scores between all
    pairs of tokens and aggregates the values.

3.  **Feed-Forward Networks**: A two-layer neural network applied to the
    output of the attention layer.

4.  **Residual Connections and Layer Normalization**: Helps prevent
    vanishing/exploding gradient problems in deeper models.

5.  **Output Layer**: A softmax layer that predicts the next character
    in the sequence.

## Model Components: {#model-components .unnumbered}

-   **Token Embeddings**: Embeds the input sequence into continuous
    space.

-   **Positional Embeddings**: Encodes the relative position of tokens.

-   **Transformer Blocks**: Stacked blocks consisting of multi-head
    attention, feed-forward networks, and residual connections.

-   **Final Linear Layer**: Maps the output of the last transformer
    block to the vocabulary size.

# Training

The model is trained using the **AdamW optimizer** and the
**cross-entropy loss** function. Training involves sampling batches of
sequences, predicting the next character, and updating model weights
using backpropagation.

-   **Batch size**: 16

-   **Learning rate**: 0.001

-   **Max Iterations**: 5000

-   **Evaluation Interval**: Every 100 iterations

-   **Training Split**: 90% training data, 10% validation data

The model is trained on a **CUDA-enabled GPU** or **TPU**, significantly
speeding up the training process.

# Text Generation

Once trained, the model is capable of generating text by predicting one
character at a time. The generation process begins with a seed (initial
token) and iteratively generates the next token based on the previous
ones.

The generation method uses **multinomial sampling** to choose the next
token from the predicted probabilities and append it to the sequence.

## Sample Text Generation: {#sample-text-generation .unnumbered}

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
    print(generated_text)

# Requirements

-   Python 3.x

-   PyTorch (CUDA enabled)

-   NVIDIA GPU (or TPU for faster training)

-   Additional Python packages:

    -   torch

    -   torch.nn

    -   numpy

# Usage

1.  Clone this repository:

            git clone https://github.com/yourusername/transformer-text-generation.git
            cd transformer-text-generation

2.  Install dependencies:

            pip install -r requirements.txt

3.  Open the notebook and execute the cells to preprocess data, train
    the model, and generate text.

4.  You can experiment with the hyperparameters (e.g., number of layers,
    embedding size, batch size) and observe how they affect the model's
    performance.

# Acknowledgments

-   **Andrej Karpathy**: The original tutorial on char-RNN, which
    inspired this implementation.

-   **Attention Is All You Need Paper**: The foundational paper behind
    transformer-based models, providing the theoretical basis for the
    architecture used here.

Feel free to fork, contribute, or use the model for your own text
generation tasks!

