# [*Bigram*](https://web.stanford.edu/~jurafsky/slp3/3.pdf)-Based (nano) Language Model: <br /> Tiny Shakespeare Text Generation

This is a simple language model for character-level text generation trained on the *Tiny Shakespeare* dataset. The model predicts the next character based on context using multi-head self-attention. Based on Andrej Karpathy's *"Zero to Hero"* series on Youtube, [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762) paper and [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385v1) paper. It's important emphasize to though that this model is not even close to state-of-the-art solutions for text generation, but it helped me understand the fundamental architecture of transformer-based LLMs.

Here I found a few Shakespear datasets: https://github.com/cobanov/shakespeare-dataset/blob/main/text/romeo-and-juliet_TXT_FolgerShakespeare.txt
The current notebook contains and the model was trained on *Romeo and Juliet*

## Key Concepts
(roughly speaking)
1. **Self-Attention**: Computes *affinities* between tokens by weighing their interactions.
2. **Multi-Head Attention**: Uses *multiple attention heads* for parallel focus on different sequence parts.
3. **Positional Encoding**: Encodes the *position of tokens* (order matters, but it gets lost without this - this problem first appeared in transformer architectures)
4. **Residual Connections**: Prevents *vanishing/exploding gradients* in deep networks and *speeds up* computation (see [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385v1)).
5. **Cross-Entropy Loss**: Measures prediction error for classification tasks.

## Model Architecture

1. **Embedding Layers**
2. **Multi-Head Attention**
3. **Feed-Forward Network**
4. **Residual + LayerNorm**
5. **Final Linear Layer** (outputs logits for next-token prediction)

## Training

- **Optimizer**: AdamW
- **Loss**: Cross-entropy
- **Batch Size**: 16
- **Max Iterations**: 5000
- **Learning Rate**: 1e-3
- **Training Split**: 90\% training, 10\% validation

Trained on CUDA-enabled TPU on Kaggle

## Text Generation

**Process**: Generate text by predicting one character at a time using multinomial sampling.
```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
print(generated_text)
```

## Requirements

- Python 3.x
- Dependencies: `torch`, `torch.nn`, `numpy`

## Usage
Clone the repository:
   ```bash
   git clone https://github.com/peti12352/shakespear_nanoGPT.git
   cd shakespear_nanoGPT


