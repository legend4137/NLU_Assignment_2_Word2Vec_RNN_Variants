# Character-Level Name Generation using RNN Variants

This repository contains a PyTorch implementation of sequence models for character-level name generation. It trains neural networks to learn the morphological patterns of Indian names and generates novel names character by character. 

The project explores three distinct recurrent architectures, evaluated on their generative realism, novelty, and diversity.

## Best Hyperparameters

Through experimentation, the following hyperparameters yielded the best balance of novelty, diversity, and realism for each specific architecture:

* **Vanilla RNN**
  * Embedding Dimension: 64
  * Hidden Size: 128
  * Learning Rate: 0.001
  * Epochs: 100

* **BLSTM (Bidirectional LSTM)**
  * Embedding Dimension: 32
  * Hidden Size: 128
  * Learning Rate: 0.001
  * Epochs: 10

* **RNN with Attention**
  * Embedding Dimension: 16
  * Hidden Size: 128
  * Learning Rate: 0.001
  * Epochs: 10

## Models Implemented

1. **Vanilla RNN (From Scratch):** A standard recurrent neural network implementing the core `h_t = tanh(W_x * x_t + W_h * h_{t-1})` recurrence relation.
2. **Bidirectional LSTM (BLSTM):** Utilizes PyTorch's built-in `nn.LSTM` module. *(Note: Autoregressive text generation requires strict left-to-right causal masking. Evaluating this model bidirectionally demonstrates the "lookahead bias" phenomenon).*
3. **RNN with Attention (From Scratch):** Extends the Vanilla RNN with an additive attention mechanism. It computes alignment scores over past hidden states to form a context vector, utilizing a causal mask to prevent information leakage from future timesteps.

## Evaluation Metrics

The script automatically generates a batch of 1,000 names post-training to compute:
* **Novelty Rate:** The percentage of generated names that do not appear in the training dataset.
* **Diversity Rate:** The ratio of unique names generated compared to the total generated batch.

## Requirements

* Python 3.7+
* PyTorch

## Usage

### 1. Dataset Preparation
Ensure you have a text file named `TrainingNames.txt` in the same directory as the script. The file should contain comma-separated names (e.g., `Aarav, Vivaan, Aditya...`).

### 2. Running the Script
The script uses command-line arguments to select the model and tune hyperparameters. 

**Basic Usage:**
```bash
python train.py --model_name attention
```

**Custom Hyperparameters:**
```bash
python train.py --model_name vanilla --embedding_dim 32 --hidden_size 256 --learning_rate 0.005 --epochs 20
```

### 3. Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model_name` | `str` | **Required** | Model to train. Choices: `vanilla`, `blstm`, `attention`. |
| `--embedding_dim` | `int` | `16` | Size of the character embedding vectors. |
| `--hidden_size` | `int` | `128` | Number of features in the hidden state. |
| `--num_layers` | `int` | `1` | Number of recurrent layers (applicable to BLSTM). |
| `--learning_rate` | `float`| `0.001` | Learning rate for the Adam optimizer. |
| `--epochs` | `int` | `10` | Number of training epochs. |
| `--batch_size` | `int` | `32` | Number of sequences per training batch. |

## Model Saving
After training, the script automatically saves the model weights (state dict) to the local directory using the format `{model_name}_best_model.pth`. You can use this file to load the trained model for inference without retraining.