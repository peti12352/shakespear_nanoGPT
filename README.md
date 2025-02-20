# Transformer-Based Language Model: Tiny Shakespeare Text Generation

A transformer-based language model for character-level text generation trained on the *Tiny Shakespeare* dataset. The model predicts the next character based on context using multi-head self-attention. Based on Andrej Karpathy's *"Zero to Hero"* series on Youtube, [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762) paper and [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385v1) paper.

## Key Concepts
(roughly speaking)
1. **Self-Attention**: Computes *affinities* between tokens by weighing their interactions.
2. **Multi-Head Attention**: Uses *multiple attention heads* for parallel focus on different sequence parts.
3. **Positional Encoding**: Encodes the *position of tokens* to capture order-dependent relationships.
4. **Residual Connections**: Prevents *vanishing/exploding gradients* in deep networks and *speeds up* computation (see [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385v1)).
5. **Cross-Entropy Loss**: Measures prediction error for classification tasks.

## Model Architecture

1. **Embedding Layers**: Token to vector transformation.
2. **Multi-Head Attention**: Captures token dependencies.
3. **Feed-Forward Network**: Two fully connected layers.
4. **Residual + LayerNorm**: Stabilizes and speeds up training.
5. **Final Linear Layer**: Outputs logits for next-token prediction.

## Training

- **Optimizer**: AdamW
- **Loss**: Cross-entropy
- **Batch Size**: 16
- **Max Iterations**: 5000
- **Learning Rate**: 0.001
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
- PyTorch (CUDA enabled)
- NVIDIA GPU/TPU
- Dependencies: `torch`, `torch.nn`, `numpy`

## Usage
Clone the repository:
   ```bash
   git clone https://github.com/peti12352/shakespear_nanoGPT.git
   cd shakespear_nanoGPT


