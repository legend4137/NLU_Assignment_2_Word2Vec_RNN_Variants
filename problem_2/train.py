import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# TASK-0: DATASET PREPARATION
# =====================================================================

with open("TrainingNames.txt", "r") as f:
    content = f.read()

# Clean and lowercase the names, splitting by comma
names = [name.strip().lower() for name in content.split(',') if name.strip()]
print("Total names:", len(names))

chars = set()
for name in names:
    chars.update(list(name))

chars = sorted(list(chars))

# Add special tokens required for sequence modeling:
# <PAD>: Ensures uniform batch sizes
# <SOS>: Start of Sequence (tells the model to begin generating)
# <EOS>: End of Sequence (tells the model to stop generating)
chars = ['<PAD>', '<SOS>', '<EOS>'] + chars

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

vocab_size = len(chars)
print("Vocab size:", vocab_size)

def encode(name):
    return [char2idx['<SOS>']] + [char2idx[c] for c in name] + [char2idx['<EOS>']]

encoded_names = [encode(name) for name in names]
# To determine the padding length
max_len = max(len(seq) for seq in encoded_names)

def pad(seq):
    return seq + [char2idx['<PAD>']] * (max_len - len(seq))

padded_data = [pad(seq) for seq in encoded_names]

class NameDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]   # input, target

# =====================================================================
# TASK-1: MODEL IMPLEMENTATIONS (From Scratch)
# Implemented Vanilla RNN, BLSTM, and RNN with Attention.
# =====================================================================

class VanillaRNN(nn.Module):
    """
    Trainable Parameters: 
    Embedding: (vocab_size * embed_dim)
    W_x: (embed_dim * hidden_size) + hidden_size
    W_h: (hidden_size * hidden_size) + hidden_size
    FC: (hidden_size * vocab_size) + vocab_size
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN weights
        self.W_x = nn.Linear(embedding_dim, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            # RNN equation: h_t = tanh(W_x * x_t + W_h * h_{t-1})
            h = torch.tanh(self.W_x(x_t) + self.W_h(h))

            # output
            out = self.fc(h)
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs
    

class BLSTM(nn.Module):
    """
    Trainable Parameters: 
    Contains 4 gates * 2 directions
    FC layer is (hidden_size * 2 * vocab_size) due to concatenation
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True # Change to False for unidirectional
        )

        # If bidirectional=False -> hidden_size
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


class RNNAttention(nn.Module):
    """
    Trainable Parameters:
    Base RNN params
    W_a (hidden_size * hidden_size)
    modified FC layer taking combined context and hidden state.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN weights
        self.W_x = nn.Linear(embedding_dim, hidden_size)
        self.W_h = nn.Linear(hidden_size, hidden_size)

        # Attention weight matrix
        self.W_a = nn.Linear(hidden_size, hidden_size)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)
        h = torch.zeros(batch_size, self.hidden_size)

        hidden_states = []

        # RNN pass
        for t in range(seq_len):
            x_t = x[:, t, :]
            h = torch.tanh(self.W_x(x_t) + self.W_h(h))
            hidden_states.append(h.unsqueeze(1))

        H = torch.cat(hidden_states, dim=1)  # (B, T, H)

        outputs = []

        # -------- ATTENTION PER TIMESTEP --------
        for t in range(seq_len):
            h_t = H[:, t, :].unsqueeze(1)  # (B, 1, H)

            # score against all hidden states
            scores = torch.bmm(self.W_a(H),h_t.transpose(1, 2))

            # Prevents attention from 'cheating' by looking at future tokens
            scores[:, t+1:, :] = float('-inf')

            weights = torch.softmax(scores, dim=1)
            context = torch.sum(weights * H, dim=1)  # (B, H)

            # combine current + context
            combined = torch.cat([h_t.squeeze(1), context], dim=1)
            out = self.fc(combined)  # (B, V)
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)

        return outputs


# Helper Functions

def count_params(model):
    # Calculate total trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_name(model, max_len=20):

    model.eval()
    x = torch.tensor([[char2idx['<SOS>']]])
    name = ""

    for _ in range(max_len):
        out = model(x)
        # Extract probabilities for the final timestep
        probs = torch.softmax(out[0, -1], dim=0)
        idx = torch.multinomial(probs, 1).item()

        if idx == char2idx['<EOS>']:
            break

        name += idx2char[idx]
        x = torch.cat([x, torch.tensor([[idx]])], dim=1)

    return name

# =====================================================================
# TASK-2: QUANTITATIVE EVALUATION
# =====================================================================

def generate_batch(model, n=500):
    generated = []
    for _ in range(n):
        name = generate_name(model)
        if name:  
            generated.append(name)
    return generated

def compute_novelty(generated, train_set):
    # Novelty: Percentage of generated names not in the training set
    novel = [name for name in generated if name not in train_set]
    return len(novel) / len(generated) if generated else 0

def compute_diversity(generated):
    # Diversity: Number of unique generated names / Total generated names
    return len(set(generated)) / len(generated) if generated else 0

def evaluate_model(model, name, n=500):
    print(f"\n--- Running Task-2 Evaluation for {name} ---")
    generated = generate_batch(model, n)
    
    novelty = compute_novelty(generated, names)
    diversity = compute_diversity(generated)

    print(f"Novelty Rate: {novelty:.4f} ({(novelty*100):.2f}%)")
    print(f"Diversity Rate: {diversity:.4f} ({(diversity*100):.2f}%)")
    
    return novelty, diversity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task-1 & Task-2 Model Training")
    
    parser.add_argument('--model_name', type=str, choices=['vanilla', 'blstm', 'attention'], required=True)
    parser.add_argument('--embedding_dim', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()

    # Load dataloader
    dataset = NameDataset(padded_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"\nInitializing {args.model_name.upper()} Model")
    print(f"Hyperparameters -> Embed: {args.embedding_dim}, Hidden: {args.hidden_size}, LR: {args.learning_rate}, Epochs: {args.epochs}")

    # Model Selection
    if args.model_name == 'vanilla':
        model = VanillaRNN(vocab_size, args.embedding_dim, args.hidden_size)
        display_name = "Vanilla RNN"
    elif args.model_name == 'blstm':
        model = BLSTM(vocab_size, args.embedding_dim, args.hidden_size, args.num_layers)
        display_name = "Bidirectional LSTM"
    elif args.model_name == 'attention':
        model = RNNAttention(vocab_size, args.embedding_dim, args.hidden_size)
        display_name = "RNN with Attention"

    print(f"Total Trainable Parameters: {count_params(model)}\n")

    # Optimizer and Loss
    # ignore_index ensures the model isn't penalized for failing to predict <PAD> tokens
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx['<PAD>'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training Loop
    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for x, y in dataloader:
            outputs = model(x)

            loss = criterion(outputs.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}: Cumulative Loss = {total_loss:.4f}")

    save_path = f"{args.model_name}_best_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n--- Model weights saved locally to {save_path} ---")

    # Sample Generation Test
    print("\n--- Example Generated Names ---")
    for _ in range(10):
        print(generate_name(model))

    # Task-2 Execution
    evaluate_model(model, display_name, n=1000)